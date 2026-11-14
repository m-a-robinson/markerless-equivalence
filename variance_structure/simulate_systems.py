"""Forward simulation: one ground-truth movement, two measurement processes.

A 3-D linked-segment chain (pelvis -> thigh -> shank; 9 joint angles) is given
known "true" kinematics for a population of subjects and strides. It is then
measured by

  * a DIRECT-KINEMATICS, marker-based (MB) pipeline: each segment is tracked
    independently (6 DoF) from its own skin markers, which carry landmark
    placement error (fixed per subject), soft-tissue artefact (subject-specific,
    driven by joint flexion) and white noise;
  * an INVERSE-KINEMATICS, markerless-like (ML) pipeline: a template model with
    fixed joint centres and defined degrees of freedom is fitted by least squares
    to noisy keypoints. Three switchable mechanisms: keypoint noise, "template
    blindness" (the model cannot see a subject's individual bony alignment) and
    a prior that pulls poorly observed degrees of freedom toward the template.

The same variance statistics used on the experimental data are then computed,
with the truth known. Parameter values are plausible magnitudes, NOT calibrated
to Theia or to this laboratory.

Usage:  python simulate_systems.py     (~2 min; writes results/simulation, figures/simulation)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from variance_structure import components, fpca_stats

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results" / "simulation", HERE / "figures" / "simulation"
T = 101
PH = np.linspace(0, 2 * np.pi, T)
GC = np.arange(T)
NAMES = ["Pelvis Tilt", "Pelvis Obliquity", "Pelvis Rotation", "Hip Flexion", "Hip Adduction",
         "Hip Rotation", "Knee Flexion", "Knee Varus", "Knee Rotation"]
D2R = np.pi / 180

DEFAULTS = dict(
    n_subj=60, n_stride=8, seed=1,
    # --- truth (degrees) ---
    sd_between=[2.0, 1.5, 2.5, 4.0, 2.5, 3.0, 4.0, 1.5, 3.0],   # functional between-subject SD
    sd_stride=[0.7, 0.5, 1.5, 1.3, 0.9, 1.0, 1.7, 0.6, 1.2],    # stride-to-stride SD
    # individual bony alignment: fixed rotation of each anatomical frame (x, y, z), SD in degrees
    sd_anat=dict(pelvis=[4.0, 1.0, 1.0], thigh=[1.0, 1.5, 6.0], shank=[1.0, 1.5, 5.0]),
    # --- MB, direct kinematics (metres) ---
    mb_placement_sd=0.008,                                       # landmark placement, per subject
    mb_sta_gain_sd=dict(pelvis=0.004, thigh=0.010, shank=0.004), # STA at full driver excursion
    mb_sta_stride_sd=0.0015,                                     # stride-specific smooth STA
    mb_noise_sd=0.0003,
    mb_smooth=2.5,                                               # Gaussian sigma, nodes (~6 Hz)
    # --- ML, inverse kinematics ---
    ml_kp_noise_sd=0.002,                                        # temporally smooth keypoint noise
    ml_kp_bias_sd=0.004,                                         # subject-level keypoint bias
    ml_scale_sd=0.02,                                            # segment-length scaling error
    ml_alpha=1.0,            # template blindness: 1 = model cannot see bony alignment, 0 = sees it
    ml_prior_sd=4.0,         # deg; prior pulling knee varus/rotation toward 0 (np.inf = off)
    ml_smooth=0.8,                                               # (~20 Hz)
)

# marker / keypoint templates in segment coordinates (x medio-lateral, y anterior, z up), metres
LT, LS = 0.42, 0.40
MARKERS = dict(
    pelvis=np.array([[0.12, 0.08, 0.05], [-0.12, 0.08, 0.05], [0.05, -0.10, 0.07], [-0.05, -0.10, 0.07]]),
    thigh=np.array([[0.05, 0, -LT], [-0.05, 0, -LT], [0.08, 0.02, -0.5 * LT], [0.09, 0, -0.05]]),
    shank=np.array([[0.035, 0, -LS], [-0.035, 0, -LS], [0, 0.04, -0.1 * LS], [0.06, 0, -0.5 * LS]]),
)
KEYPTS = dict(
    pelvis=np.array([[0, 0, 0], [-0.20, 0, 0], [-0.10, -0.08, 0.08], [-0.10, 0.08, 0.08]]),
    thigh=np.array([[0, 0, -LT], [0.05, 0, -LT], [-0.05, 0, -LT], [0, 0.06, -0.5 * LT]]),
    shank=np.array([[0, 0, -LS], [0.035, 0, -LS], [-0.035, 0, -LS], [0, 0.04, -0.4 * LS]]),
)


# ------------------------------------------------------------------ rotations
def euler(a):
    """Cardan X-Y-Z rotation matrices. a: (..., 3) radians (flexion, ab/adduction, axial)."""
    ca, sa = np.cos(a[..., 0]), np.sin(a[..., 0])
    cb, sb = np.cos(a[..., 1]), np.sin(a[..., 1])
    cc, sc = np.cos(a[..., 2]), np.sin(a[..., 2])
    R = np.empty(a.shape[:-1] + (3, 3))
    R[..., 0, 0] = cb * cc;                 R[..., 0, 1] = -cb * sc;                R[..., 0, 2] = sb
    R[..., 1, 0] = sa * sb * cc + ca * sc;  R[..., 1, 1] = -sa * sb * sc + ca * cc; R[..., 1, 2] = -sa * cb
    R[..., 2, 0] = -ca * sb * cc + sa * sc; R[..., 2, 1] = ca * sb * sc + sa * cc;  R[..., 2, 2] = ca * cb
    return R


def cardan(R):
    """Inverse of euler(): (..., 3, 3) -> (..., 3) radians."""
    return np.stack([np.arctan2(-R[..., 1, 2], R[..., 2, 2]),
                     np.arcsin(np.clip(R[..., 0, 2], -1, 1)),
                     np.arctan2(-R[..., 0, 1], R[..., 0, 0])], -1)


def chain_angles(Rp, Rt, Rs):
    """9 joint angles (deg) from pelvis, thigh, shank orientations."""
    Tm = lambda R: np.swapaxes(R, -1, -2)
    return np.concatenate([cardan(Rp), cardan(Tm(Rp) @ Rt), cardan(Tm(Rt) @ Rs)], -1) / D2R


def smooth_rand(shape, rng, nharm=3, offset=True):
    """Smooth random functions over the cycle with unit pointwise variance: shape + (T,)."""
    w = 1.0 / (nharm + offset)
    out = rng.normal(size=shape + (1,)) * np.sqrt(w) * offset
    for h in range(1, nharm + 1):
        out = out + np.sqrt(w) * (rng.normal(size=shape + (1,)) * np.sqrt(2) * np.cos(h * PH)
                                  + rng.normal(size=shape + (1,)) * np.sqrt(2) * np.sin(h * PH))
    return out


def mean_curves():
    g = lambda mu, sd, amp: amp * np.exp(-((GC - mu) / sd) ** 2)
    return np.stack([
        10 + 1.5 * np.cos(2 * PH), 4 * np.sin(PH + 0.6), 5 * np.cos(PH),              # pelvis
        12 + 23 * np.cos(PH + 0.15), 5 * np.sin(PH + 0.3) - 1, 4 * np.sin(PH - 0.8),  # hip
        3 + g(15, 9, 18) + g(73, 11, 62), 2 * np.sin(PH), 5 * np.cos(PH + 1.0),       # knee
    ], 0)                                                                             # (9, T)


# ---------------------------------------------------------------------- truth
def make_truth(p, rng):
    N, K = p["n_subj"], p["n_stride"]
    sb, sw = np.array(p["sd_between"]), np.array(p["sd_stride"])
    subj = smooth_rand((N, 1, 9), rng, nharm=2) * sb[None, None, :, None]
    stride = smooth_rand((N, K, 9), rng, nharm=3) * sw[None, None, :, None]
    q = mean_curves()[None, None] + subj + stride                 # functional angles, deg (N,K,9,T)
    q = np.moveaxis(q, 2, 3) * D2R                                # (N,K,T,9)
    F = dict(pelvis=euler(q[..., 0:3]))
    F["thigh"] = F["pelvis"] @ euler(q[..., 3:6])
    F["shank"] = F["thigh"] @ euler(q[..., 6:9])
    # individual bony alignment (rotation vectors), fixed per subject and segment
    rv = {s: rng.normal(size=(N, 3)) * np.array(p["sd_anat"][s]) * D2R for s in F}
    Dm = lambda s, k: Rotation.from_rotvec(rv[s] * k).as_matrix()[:, None, None]
    A = {s: F[s] @ Dm(s, 1.0) for s in F}                         # anatomical (bone) frames
    o = dict(pelvis=np.zeros((N, K, T, 3)))                       # segment origins (joint centres)
    o["thigh"] = o["pelvis"]
    o["shank"] = o["thigh"] + A["thigh"] @ np.array([0, 0, -LT])
    truth = chain_angles(A["pelvis"], A["thigh"], A["shank"])
    return dict(q=q, F=F, A=A, o=o, Dm=Dm, truth=truth)


# -------------------------------------------------- MB: direct kinematics
def kabsch(P, Y):
    """Rotation R minimising |R P - Y| for template P (J,3) and observations Y (..., J, 3)."""
    Pc = P - P.mean(0)
    Yc = Y - Y.mean(-2, keepdims=True)
    H = np.swapaxes(Yc, -1, -2) @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    U[..., :, 2] *= d[..., None]
    return U @ Vt


def measure_mb(tr, p, rng, placement=True, sta=True):
    N, K = p["n_subj"], p["n_stride"]
    q = tr["q"]
    driver = dict(pelvis=q[..., 3] / (30 * D2R), thigh=q[..., 6] / (60 * D2R), shank=q[..., 6] / (60 * D2R))
    R = {}
    for s, P in MARKERS.items():
        J = len(P)
        local = np.broadcast_to(P, (N, K, T, J, 3)).copy()
        if placement:   # landmark misplaced once per subject -> anatomical frame mis-defined
            local += (rng.normal(size=(N, 1, 1, J, 3)) * p["mb_placement_sd"])
        if sta:         # skin markers slide as the joint flexes; pattern fixed within a subject
            gain = rng.normal(size=(N, 1, 1, J, 3)) * p["mb_sta_gain_sd"][s]
            local += gain * driver[s][..., None, None]
            local += np.moveaxis(smooth_rand((N, K, J, 3), rng, nharm=6, offset=False), -1, 2) * p["mb_sta_stride_sd"]
        Y = (tr["A"][s][..., None, :, :] @ local[..., None])[..., 0] + tr["o"][s][..., None, :]
        Y += rng.normal(size=Y.shape) * p["mb_noise_sd"]
        Y = gaussian_filter1d(Y, p["mb_smooth"], axis=2, mode="nearest")
        R[s] = kabsch(P, Y)                                       # each segment on its own: 6 DoF
    return chain_angles(R["pelvis"], R["thigh"], R["shank"])


# -------------------------------------------------- ML: inverse kinematics
def ik_forward(x, lt, ls):
    """Model keypoints for generalised coordinates x: (B, 12) -> (B, 36)."""
    Rp = euler(x[:, 3:6])
    Rt = Rp @ euler(x[:, 6:9])
    Rs = Rt @ euler(x[:, 9:12])
    t = x[:, None, 0:3]
    kt = KEYPTS["thigh"] * np.array([1, 1, 1])[None] * np.stack([np.ones_like(lt), np.ones_like(lt), lt / LT], -1)[:, None]
    ks = KEYPTS["shank"][None] * np.stack([np.ones_like(ls), np.ones_like(ls), ls / LS], -1)[:, None]
    knee = t + (Rt @ np.stack([0 * lt, 0 * lt, -lt], -1)[..., None])[..., 0][:, None]
    kp_p = t + np.einsum("bij,kj->bki", Rp, KEYPTS["pelvis"])
    kp_t = t + np.einsum("bij,bkj->bki", Rt, kt)
    kp_s = knee + np.einsum("bij,bkj->bki", Rs, ks)
    return np.concatenate([kp_p, kp_t, kp_s], 1).reshape(len(x), -1)


def ik_solve(obs, x0, lt, ls, kp_sd, prior_sd, iters=12):
    """Batched Levenberg-Marquardt. obs (B, 36); the model has fixed joint centres (no joint
    translation) and a zero-mean prior on knee varus / rotation (x[10], x[11])."""
    x = x0.copy()
    wp = 0.0 if np.isinf(prior_sd) else 1.0 / (prior_sd * D2R)

    def resid(x):
        return np.concatenate([(ik_forward(x, lt, ls) - obs) / kp_sd, wp * x[:, 10:12]], 1)

    for _ in range(iters):
        r = resid(x)
        Jm = np.empty(r.shape + (12,))
        for j in range(12):
            dx = np.zeros(12); dx[j] = 1e-6
            Jm[..., j] = (resid(x + dx) - r) / 1e-6
        JT = np.swapaxes(Jm, 1, 2)
        H = JT @ Jm
        H += 1e-6 * np.eye(12) * np.trace(H, axis1=1, axis2=2)[:, None, None] / 12
        x = x - np.linalg.solve(H, (JT @ r[..., None]))[..., 0]
    return x


def measure_ml(tr, p, rng, noise=True, alpha=None, prior_sd=None):
    N, K = p["n_subj"], p["n_stride"]
    alpha = p["ml_alpha"] if alpha is None else alpha
    prior_sd = p["ml_prior_sd"] if prior_sd is None else prior_sd
    obs = []
    for s, C in KEYPTS.items():
        # keypoints follow the body surface; alpha = how much of the bony alignment they miss
        G = tr["F"][s] @ tr["Dm"](s, 1.0 - alpha)
        Y = (G[..., None, :, :] @ C[..., None])[..., 0] + tr["o"][s][..., None, :]
        if noise:
            Y = Y + rng.normal(size=(N, 1, 1, len(C), 3)) * p["ml_kp_bias_sd"]
            Y = Y + np.moveaxis(smooth_rand((N, K, len(C), 3), rng, nharm=5, offset=False), -1, 2) * p["ml_kp_noise_sd"]
        obs.append(Y)
    obs = np.concatenate(obs, 3).reshape(N, K * T, 36)
    scale = 1 + (rng.normal(size=(N, 2)) * p["ml_scale_sd"] if noise else np.zeros((N, 2)))
    x0 = np.concatenate([np.zeros((T, 3)), mean_curves().T * D2R], 1)        # template pose
    out = np.empty((N, K * T, 12))
    for i in range(N):                                                       # per subject: memory
        B = K * T
        out[i] = ik_solve(obs[i], np.tile(x0, (K, 1)), np.full(B, LT * scale[i, 0]),
                          np.full(B, LS * scale[i, 1]), p["ml_kp_noise_sd"], prior_sd)
    ang = out[..., 3:].reshape(N, K, T, 9) / D2R
    return gaussian_filter1d(ang, p["ml_smooth"], axis=2, mode="nearest")


# ------------------------------------------------------------------ statistics
def to_arr(a):
    """(N, K, T, 9) -> (subj, var, T, trial), the layout used by variance_structure."""
    return a.transpose(0, 3, 2, 1)


def summarise(truth, mb, ml):
    rows = []
    tr, mb, ml = to_arr(truth), to_arr(mb), to_arr(ml)
    for v, name in enumerate(NAMES):
        c = components(mb[:, v], ml[:, v])
        ct_mb = components(tr[:, v], mb[:, v])     # 'mb' slot = truth, 'ml' slot = system
        ct_ml = components(tr[:, v], ml[:, v])
        g = lambda d, k: np.mean(d[k])
        fp, _ = fpca_stats(mb[:, v], ml[:, v])
        ratio_w = np.sqrt(g(c, "w_ml") / g(c, "w_mb"))
        ratio_b = np.sqrt(g(c, "b_ml") / g(c, "b_mb"))
        rows.append(dict(
            variable=name,
            SDb_true=np.sqrt(g(ct_mb, "b_mb")), SDb_MB=np.sqrt(g(c, "b_mb")), SDb_ML=np.sqrt(g(c, "b_ml")),
            SDw_true=np.sqrt(g(ct_mb, "w_mb")), SDw_MB=np.sqrt(g(c, "w_mb")), SDw_ML=np.sqrt(g(c, "w_ml")),
            ratio_w=ratio_w, ratio_b=ratio_b, contrast=np.log(ratio_w) - np.log(ratio_b),
            r_w=g(c, "w_c") / np.sqrt(g(c, "w_mb") * g(c, "w_ml")),
            r_b=g(c, "b_c") / np.sqrt(g(c, "b_mb") * g(c, "b_ml")),
            frisch_upper_b=g(c, "b_ml") / g(c, "b_c"),
            # known-truth quantities: scale (slope on truth) and error SD of each system
            lam_b_MB=g(ct_mb, "b_c") / g(ct_mb, "b_mb"), lam_b_ML=g(ct_ml, "b_c") / g(ct_ml, "b_mb"),
            lam_w_MB=g(ct_mb, "w_c") / g(ct_mb, "w_mb"), lam_w_ML=g(ct_ml, "w_c") / g(ct_ml, "w_mb"),
            errSDb_MB=np.sqrt(max(g(ct_mb, "b_mb") + g(ct_mb, "b_ml") - 2 * g(ct_mb, "b_c"), 0)),
            errSDb_ML=np.sqrt(max(g(ct_ml, "b_mb") + g(ct_ml, "b_ml") - 2 * g(ct_ml, "b_c"), 0)),
            errSDw_MB=np.sqrt(max(g(ct_mb, "w_mb") + g(ct_mb, "w_ml") - 2 * g(ct_mb, "w_c"), 0)),
            errSDw_ML=np.sqrt(max(g(ct_ml, "w_mb") + g(ct_ml, "w_ml") - 2 * g(ct_ml, "w_c"), 0)),
            grubbs_shared_w=g(c, "w_c"), true_var_w=g(ct_mb, "w_mb"),
            rough_MB=fp["rough_MB"], rough_ML=fp["rough_ML"],
        ))
    return pd.DataFrame(rows)


SCENARIOS = {
    # name: (MB kwargs, ML kwargs)
    "1 full: all mechanisms": (dict(), dict()),
    "2 MB placement error only": (dict(sta=False), dict(noise=False, alpha=0.0, prior_sd=np.inf)),
    "3 MB soft-tissue artefact only": (dict(placement=False), dict(noise=False, alpha=0.0, prior_sd=np.inf)),
    "4 ML template blindness only": (dict(placement=False, sta=False), dict(noise=False, alpha=1.0, prior_sd=np.inf)),
    "5 ML model prior only": (dict(placement=False, sta=False), dict(noise=False, alpha=0.0, prior_sd=4.0)),
    "6 ML keypoint noise only": (dict(placement=False, sta=False), dict(noise=True, alpha=0.0, prior_sd=np.inf)),
}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    p = DEFAULTS
    tr = make_truth(p, np.random.default_rng(p["seed"]))
    tabs, keep = [], {}
    for name, (kmb, kml) in SCENARIOS.items():
        rng = np.random.default_rng(p["seed"] + 100)
        mb, ml = measure_mb(tr, p, rng, **kmb), measure_ml(tr, p, rng, **kml)
        t = summarise(tr["truth"], mb, ml)
        t.insert(0, "scenario", name)
        tabs.append(t)
        if name.startswith("1"):
            keep = dict(mb=mb, ml=ml)
        print(f"\n=== {name}")
        print(t[["variable", "SDb_true", "SDb_MB", "SDb_ML", "SDw_true", "SDw_MB", "SDw_ML", "ratio_w", "ratio_b",
                 "contrast", "r_w", "r_b", "frisch_upper_b", "lam_b_MB", "lam_b_ML", "lam_w_ML",
                 "rough_MB", "rough_ML"]].round(2).to_string(index=False))
    tab = pd.concat(tabs)
    tab.to_csv(RES / "simulation_summary.csv", index=False)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    C_MB, C_ML = "#1f4fd8", "#d62728"

    # Fig S1: one subject, all strides: truth vs both systems
    f, ax = plt.subplots(3, 3, figsize=(9, 7), sharex=True)
    for v, a in enumerate(ax.ravel()):
        a.plot(GC, keep["mb"][0, :, :, v].T, color=C_MB, lw=0.6, alpha=0.7)
        a.plot(GC, keep["ml"][0, :, :, v].T, color=C_ML, lw=0.6, alpha=0.7)
        a.plot(GC, tr["truth"][0, :, :, v].T, color="k", lw=0.5, alpha=0.6)
        a.set_title(NAMES[v], fontweight="bold")
    f.suptitle("Fig S1. Simulated subject, 8 strides: truth (black), direct kinematics / MB (blue), "
               "inverse kinematics / ML (red)", fontsize=9, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "figS1_example_subject.png", dpi=170); plt.close(f)

    # Fig S2: ML:MB SD ratio at each level, full scenario vs experiment
    full = tab[tab.scenario.str.startswith("1")].reset_index(drop=True)
    exp = pd.read_csv(HERE / "results" / "primary_clean" / "variance_components_summary.csv")
    emap = {"Pelvis Tilt": "Pelvis Anterior Tilt", "Knee Rotation": None}
    f, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(9)
    for a, (lab, rw, rb) in zip(ax, (
            ("Simulation (all mechanisms)", full.ratio_w, full.ratio_b),
            ("Experiment (Study 2, clean)",
             [exp.loc[exp.variable == emap.get(n, n), "ratio_w"].squeeze() if emap.get(n, n) else np.nan for n in NAMES],
             [exp.loc[exp.variable == emap.get(n, n), "ratio_b"].squeeze() if emap.get(n, n) else np.nan for n in NAMES]))):
        a.bar(x - 0.2, np.asarray(rw, float), 0.4, color="0.75", label="stride-to-stride")
        a.bar(x + 0.2, np.asarray(rb, float), 0.4, color="0.25", label="between-subject")
        a.axhline(1, color="k", lw=0.6)
        a.set_xticks(x); a.set_xticklabels(NAMES, rotation=45, ha="right"); a.set_title(lab, fontweight="bold")
    ax[0].set_ylabel("SD ratio ML / MB"); ax[0].legend(frameon=False)
    f.suptitle("Fig S2. A scaled copy would give equal bars within each variable", fontsize=9, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "figS2_ratio_by_level.png", dpi=170); plt.close(f)

    # Fig S3: which mechanism produces what (ablations)
    cols = [("contrast", "log ratio: stride − between\n(0 = scaled copy)", "RdBu_r", -1.2, 1.2),
            ("lam_b_ML", "ML scale of true between-\nsubject differences (1 = faithful)", "PuOr", 0, 2),
            ("lam_b_MB", "MB scale of true between-\nsubject differences (1 = faithful)", "PuOr", 0, 2),
            ("frisch_upper_b", "upper bound on ML:MB scale\n(from the two systems only)", "PuOr", 0, 2)]
    f, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, (col, lab, cmap, lo, hi) in zip(ax, cols):
        M = tab.pivot(index="scenario", columns="variable", values=col).loc[list(SCENARIOS), NAMES]
        im = a.imshow(M.values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for (i, j), val in np.ndenumerate(M.values):
            a.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5)
        a.set_xticks(range(9)); a.set_xticklabels(NAMES, rotation=60, ha="right", fontsize=6)
        a.set_yticks(range(len(SCENARIOS)))
        a.set_yticklabels(list(SCENARIOS) if a is ax[0] else [], fontsize=6.5)
        a.set_title(lab, fontsize=7.5, fontweight="bold")
        plt.colorbar(im, ax=a, fraction=0.046)
    f.suptitle("Fig S3. Mechanism ablations", fontsize=9, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "figS3_ablations.png", dpi=170); plt.close(f)


if __name__ == "__main__":
    main()
