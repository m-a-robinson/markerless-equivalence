"""Preliminary analyses: the structure of variance in simultaneous markerless (ML)
and marker-based (MB) gait kinematics (Study 2 data, n=24, 8 strides each).

Model, per variable and gait-cycle node t, system s, subject i, stride k:
    Y_sik(t) = mu_s(t) + b_si(t) + w_sik(t)
Both systems observed the *same* stride k, so the ML-MB cross-covariance at each
level estimates the variance the systems share (true movement variation plus any
common error); the remainder is system-specific (Grubbs 1948 two-instrument
estimator, assuming unit scale and independent errors).

Primary analysis: gross-error strides removed (rule in `gross_error_flags`).
Sensitivity analysis: all data, as used in the equivalence manuscript.
Outputs go to ./results/<tag>/ (CSV) and ./figures/<tag>/ (PNG).

Usage:  python variance_structure.py          (both analyses, ~3 min)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.stats import kurtosis
import spm1d

from load_data import load, TITLES, SHORT, PLANE

HERE = Path(__file__).resolve().parent
NBOOT = 1000
NPERM = 1000
T = np.arange(101)
C_MB, C_ML = "#1f4fd8", "#d62728"
KF, KV, HR = SHORT.index("KneeFlex"), SHORT.index("KneeVar"), SHORT.index("HipRot")
AD, AI, AR = SHORT.index("AnkDF"), SHORT.index("AnkInv"), SHORT.index("AnkRot")


# ------------------------------------------------------------- gross errors
def _runs(b, run=3, axis=2):
    return (uniform_filter1d(b.astype(float), run, axis=axis) >= 1 - 1e-9).any(axis)


def gross_error_flags(X, thr=8, absmin=10):
    """Rule fixed before running the main analysis.
    Stride flag: |deviation from the subject's median waveform| > thr robust SD
    (1.4826*MAD pooled over subjects/strides at that node) AND > absmin degrees,
    for >= 3 consecutive nodes. Subject flag: same rule applied to the subject-mean
    waveform relative to the group median. X: (subj, var, 101, trial)."""
    d = X - np.nanmedian(X, 3, keepdims=True)
    sc = 1.4826 * np.nanmedian(np.abs(d), axis=(0, 3), keepdims=True)
    stride = _runs((np.abs(d) > thr * sc) & (np.abs(d) > absmin))           # subj, var, trial
    m = np.nanmean(X, 3)
    dm = m - np.nanmedian(m, 0, keepdims=True)
    scm = 1.4826 * np.nanmedian(np.abs(dm), 0, keepdims=True)
    subj = _runs((np.abs(dm) > thr * scm) & (np.abs(dm) > absmin), axis=2)  # subj, var
    return stride, subj


def clean(MB, ML, ids):
    """Drop flagged strides (both systems, all variables, keeping pairing) and any
    subject with a subject-level flag or < 4 clean strides."""
    log = []
    s_mb, j_mb = gross_error_flags(MB)
    s_ml, j_ml = gross_error_flags(ML)
    bad_stride = s_mb.any(1) | s_ml.any(1)                                   # subj, trial
    for i, k in zip(*np.where(bad_stride)):
        vs = [SHORT[v] for v in range(MB.shape[1]) if s_mb[i, v, k] or s_ml[i, v, k]]
        sysn = "MB" if s_mb[i, :, k].any() else "ML"
        log.append(dict(subject=ids[i], trial=k + 1, level="stride", system=sysn, variables=";".join(vs)))
    mb, ml = MB.copy(), ML.copy()
    mb[np.broadcast_to(bad_stride[:, None, None, :], mb.shape)] = np.nan
    ml[np.broadcast_to(bad_stride[:, None, None, :], ml.shape)] = np.nan
    n_ok = np.sum(~np.isnan(mb[:, 0, 0, :]), 1)
    keep = ~(j_mb.any(1) | j_ml.any(1)) & (n_ok >= 4)
    for i in np.where(~keep)[0]:
        vs = [SHORT[v] for v in range(MB.shape[1]) if j_mb[i, v] or j_ml[i, v]]
        log.append(dict(subject=ids[i], trial="all", level="subject",
                        system="MB" if j_mb[i].any() else "ML",
                        variables=";".join(vs) + f" ({n_ok[i]} clean strides)"))
    return mb[keep], ml[keep], [x for x, k in zip(ids, keep) if k], pd.DataFrame(log)


# ---------------------------------------------------------------- components
def components(mb, ml):
    """Variance components for one variable. mb, ml: (subj, 101, trial), NaN-padded,
    with identical missingness. One-way random-effects moment estimators."""
    n = np.sum(~np.isnan(mb[:, 0, :]), axis=1)
    m_mb, m_ml = np.nanmean(mb, 2), np.nanmean(ml, 2)
    d_mb, d_ml = mb - m_mb[..., None], ml - m_ml[..., None]
    vw_mb_i = np.nansum(d_mb ** 2, 2) / (n[:, None] - 1)
    vw_ml_i = np.nansum(d_ml ** 2, 2) / (n[:, None] - 1)
    cw_i = np.nansum(d_mb * d_ml, 2) / (n[:, None] - 1)
    ns = len(n)
    return dict(
        w_mb=vw_mb_i.mean(0), w_ml=vw_ml_i.mean(0), w_c=cw_i.mean(0),
        b_mb=np.var(m_mb, 0, ddof=1) - np.mean(vw_mb_i / n[:, None], 0),
        b_ml=np.var(m_ml, 0, ddof=1) - np.mean(vw_ml_i / n[:, None], 0),
        b_c=np.sum((m_mb - m_mb.mean(0)) * (m_ml - m_ml.mean(0)), 0) / (ns - 1)
            - np.mean(cw_i / n[:, None], 0),
        sdw_mb_i=np.sqrt(vw_mb_i.mean(1)), sdw_ml_i=np.sqrt(vw_ml_i.mean(1)),
        m_mb=m_mb, m_ml=m_ml,
    )


def scalar_summary(c):
    """Cycle-averaged summaries (variances averaged over nodes; SD = sqrt)."""
    g = {k: np.mean(c[k]) for k in ("w_mb", "w_ml", "w_c", "b_mb", "b_ml", "b_c")}
    pos = lambda x: max(x, 1e-9)
    s = dict(
        SDw_MB=np.sqrt(g["w_mb"]), SDw_ML=np.sqrt(g["w_ml"]),
        SDb_MB=np.sqrt(pos(g["b_mb"])), SDb_ML=np.sqrt(pos(g["b_ml"])),
        ratio_w=np.sqrt(g["w_ml"] / g["w_mb"]),
        ratio_b=np.sqrt(pos(g["b_ml"]) / pos(g["b_mb"])),
        r_w=g["w_c"] / np.sqrt(g["w_mb"] * g["w_ml"]),
        r_b=g["b_c"] / np.sqrt(pos(g["b_mb"]) * pos(g["b_ml"])),
        shared_w=g["w_c"], uniq_w_MB=g["w_mb"] - g["w_c"], uniq_w_ML=g["w_ml"] - g["w_c"],
        shared_b=g["b_c"], uniq_b_MB=g["b_mb"] - g["b_c"], uniq_b_ML=g["b_ml"] - g["b_c"],
        icc_MB=g["b_mb"] / (g["b_mb"] + g["w_mb"]), icc_ML=g["b_ml"] / (g["b_ml"] + g["w_ml"]),
        # Frisch bounds on the ML:MB scale of the shared stride signal
        lam_lo=g["w_c"] / g["w_mb"], lam_hi=g["w_ml"] / g["w_c"] if g["w_c"] > 0 else np.nan,
    )
    # a rescaled MB predicts the same ratio at both levels -> this contrast = 0
    s["log_ratio_w_minus_b"] = np.log(s["ratio_w"]) - np.log(s["ratio_b"])
    # does individual stride variability rank the same in both systems?
    s["r_indiv_var"] = np.corrcoef(c["sdw_mb_i"], c["sdw_ml_i"])[0, 1]
    return s


def stride_devs(x):
    d = x - np.nanmean(x, 2, keepdims=True)
    d = d.transpose(0, 2, 1).reshape(-1, 101)
    return d[~np.isnan(d[:, 0])]


def fpca_stats(mb, ml, q=3):
    out, vecs = {}, {}
    for name, x in (("MB", mb), ("ML", ml)):
        d = stride_devs(x)
        lam, vec = np.linalg.eigh(np.cov(d.T))
        lam, vec = lam[::-1].clip(0), vec[:, ::-1]
        vecs[name] = vec[:, :q]
        out[f"PR_{name}"] = lam.sum() ** 2 / np.sum(lam ** 2)          # participation ratio
        out[f"n90_{name}"] = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.9) + 1)
        out[f"pc1_{name}"] = lam[0] / lam.sum()
        # normalised roughness: mean sq. 2nd difference / mean sq. deviation
        out[f"rough_{name}"] = 1e4 * np.mean(np.diff(d, 2, axis=1) ** 2) / np.mean(d ** 2)
        # tail weight of standardised stride deviations (excess kurtosis, node-averaged)
        z = (d - d.mean(0)) / d.std(0)
        out[f"kurt_{name}"] = np.mean(kurtosis(z, axis=0))
    # Krzanowski (1979) similarity of leading-q PC subspaces (1 = identical)
    sv = np.linalg.svd(vecs["MB"].T @ vecs["ML"], compute_uv=False)
    out["krz_q3"] = np.sum(sv ** 2) / q
    return out, vecs


def pooled_corr_matrix(x):
    """Pooled within-subject stride correlation between variables at every node.
    x: (subj, var, 101, trial) -> (var, var, 101)."""
    nv = x.shape[1]
    d = x - np.nanmean(x, 3, keepdims=True)
    d = d.transpose(1, 2, 0, 3).reshape(nv, 101, -1)
    d = d[:, :, ~np.isnan(d).any(axis=(0, 1))]
    z = (d - d.mean(2, keepdims=True)) / d.std(2, keepdims=True)
    return np.einsum("atn,btn->abt", z, z) / z.shape[2]


def best_shift(mbk, mlk, grid=np.arange(-6, 6.01, 0.05)):
    """Time shift (% gait cycle) of the ML waveform that best matches MB, offset removed."""
    sl = slice(8, 93)
    sse = []
    for g in grid:
        s = np.interp(T + g, T, mlk)
        e = (s[sl] - s[sl].mean()) - (mbk[sl] - mbk[sl].mean())
        sse.append(np.sum(e ** 2))
    return grid[int(np.argmin(sse))]


# ====================================================================== run
def run(MB, ML, MDC, tag, rng):
    RES, FIG = HERE / "results" / tag, HERE / "figures" / tag
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    NS, NV = MB.shape[:2]
    mdc = MDC.copy()
    mdc[8] = 5.0  # foot progression: no Study 1 MDC, manuscript uses ±5°

    # ---------------- R1/R2: variance components, shared vs specific
    rows, curves = [], {}
    for v in range(NV):
        c = components(MB[:, v], ML[:, v])
        curves[v] = c
        s = scalar_summary(c)
        bs = []
        for _ in range(NBOOT):
            idx = rng.integers(0, NS, NS)
            bs.append(scalar_summary(components(MB[idx, v], ML[idx, v])))
        bs = pd.DataFrame(bs)
        row = dict(variable=TITLES[v], plane=PLANE[v])
        for k, val in s.items():
            lo, hi = np.nanpercentile(bs[k], [2.5, 97.5])
            row[k], row[k + "_lo"], row[k + "_hi"] = val, lo, hi
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "variance_components_summary.csv", index=False)

    # ---------------- R1b: is the ML:MB stride-SD ratio constant over the cycle?
    def logratio_sd(mb, ml):
        c = components(mb, ml)
        return np.std(np.log(np.sqrt(c["w_ml"] / c["w_mb"])))

    perm_rows = []
    for v in range(NV):
        mb, ml = MB[:, v], ML[:, v]
        k = np.sqrt(np.mean(curves[v]["w_ml"]) / np.mean(curves[v]["w_mb"]))
        obs = logratio_sd(mb, ml)
        # H0: ML stride deviations = k * (exchangeable copy of MB deviations):
        # rescale ML by 1/k and swap system labels within each stride pair.
        d_mb = mb - np.nanmean(mb, 2, keepdims=True)
        d_ml = (ml - np.nanmean(ml, 2, keepdims=True)) / k
        null = []
        for _ in range(NPERM):
            flip = rng.random((NS, 1, mb.shape[2])) < 0.5
            null.append(logratio_sd(np.where(flip, d_ml, d_mb), np.where(flip, d_mb, d_ml) * k))
        perm_rows.append(dict(variable=TITLES[v], global_ratio=k, sd_log_ratio=obs,
                              null_median=np.median(null),
                              p_perm=(1 + np.sum(np.array(null) >= obs)) / (NPERM + 1)))
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(RES / "temporal_proportionality_test.csv", index=False)

    # ---------------- R3: functional PCA / temporal covariance structure
    fp_rows = []
    for v in range(NV):
        s, _ = fpca_stats(MB[:, v], ML[:, v])
        bs = []
        for _ in range(NBOOT // 2):
            idx = rng.integers(0, NS, NS)
            bs.append(fpca_stats(MB[idx, v], ML[idx, v])[0])
        bs = pd.DataFrame(bs)
        row = dict(variable=TITLES[v])
        for k, val in s.items():
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            row[k], row[k + "_lo"], row[k + "_hi"] = val, lo, hi
        # reference: MB-vs-MB subspace similarity between random halves of subjects
        half = []
        for _ in range(200):
            p = rng.permutation(NS)
            _, va = fpca_stats(MB[p[: NS // 2], v], MB[p[: NS // 2], v])
            _, vb = fpca_stats(MB[p[NS // 2:], v], MB[p[NS // 2:], v])
            sv = np.linalg.svd(va["MB"].T @ vb["MB"], compute_uv=False)
            half.append(np.sum(sv ** 2) / 3)
        row["krz_MB_splithalf_median"] = np.median(half)
        fp_rows.append(row)
    fp = pd.DataFrame(fp_rows)
    fp.to_csv(RES / "fpca_structure.csv", index=False)

    # ---------------- R4: inter-joint coupling
    R_MB, R_ML = pooled_corr_matrix(MB), pooled_corr_matrix(ML)
    pairs = [(a, b) for a in range(NV) for b in range(a + 1, NV)]
    cm = pd.DataFrame({
        "pair": [f"{SHORT[a]}-{SHORT[b]}" for a, b in pairs],
        "r_MB_cycle_mean": [R_MB[a, b].mean() for a, b in pairs],
        "r_ML_cycle_mean": [R_ML[a, b].mean() for a, b in pairs],
    })
    cm["diff"] = cm.r_MB_cycle_mean - cm.r_ML_cycle_mean
    # bootstrap CI for the headline coupling contrasts
    key = [(HR, KV), (KF, KV), (AI, AR), (AD, AI)]
    bsd = {p: [] for p in key}
    for _ in range(NBOOT // 2):
        idx = rng.integers(0, NS, NS)
        rb_mb, rb_ml = pooled_corr_matrix(MB[idx]), pooled_corr_matrix(ML[idx])
        for (a, b) in key:
            bsd[(a, b)].append(rb_mb[a, b].mean() - rb_ml[a, b].mean())
    cm["diff_lo"] = np.nan
    cm["diff_hi"] = np.nan
    for (a, b) in key:
        i = pairs.index((a, b))
        cm.loc[i, ["diff_lo", "diff_hi"]] = np.percentile(bsd[(a, b)], [2.5, 97.5])
    cm = cm.reindex(cm["diff"].abs().sort_values(ascending=False).index)
    cm.to_csv(RES / "interjoint_coupling.csv", index=False)

    # ---------------- R5: what drives the between-system difference variance?
    # var(d) = (sd_ML - sd_MB)^2 + 2 sd_ML sd_MB (1 - r)   [scale mismatch + decorrelation]
    eq_rows, eq_curves = [], {}
    for v in range(NV):
        a, b = curves[v]["m_mb"], curves[v]["m_ml"]
        sa, sb = a.std(0, ddof=1), b.std(0, ddof=1)
        r = np.array([np.corrcoef(a[:, t], b[:, t])[0, 1] for t in T])
        vd = np.var(b - a, 0, ddof=1)
        scale_part, decor_part = (sb - sa) ** 2, 2 * sa * sb * (1 - r)
        ci = spm1d.stats.ci_pairedsample(a, b, 0.1, datum="difference", mu=0)
        lo, hi = ci.ci
        hw = hi - ci.mean
        # counterfactual: ML a perfectly correlated rescaling of MB (r = 1),
        # same RFT critical value (approximation: smoothness held fixed)
        hw_cf = hw * np.sqrt(scale_part / vd)
        eq_curves[v] = dict(scale=scale_part, decor=decor_part, bias=ci.mean, hw=hw, hw_cf=hw_cf)
        eq_rows.append(dict(
            variable=TITLES[v], MDC=mdc[v], mean_abs_bias=np.mean(np.abs(ci.mean)),
            sd_diff=np.sqrt(vd.mean()),
            frac_var_from_decorrelation=decor_part.mean() / vd.mean(),
            frac_var_from_scale_mismatch=scale_part.mean() / vd.mean(),
            mean_r_subject_means=r.mean(),
            pct_equiv_observed=100 * np.mean((np.abs(lo) <= mdc[v]) & (np.abs(hi) <= mdc[v])),
            pct_equiv_if_r1=100 * np.mean(np.abs(ci.mean) + hw_cf <= mdc[v]),
            pct_equiv_bias_only=100 * np.mean(np.abs(ci.mean) <= mdc[v])))
    eqt = pd.DataFrame(eq_rows)
    eqt.to_csv(RES / "difference_variance_decomposition.csv", index=False)

    # ---------------- R6: relative event timing between systems
    shifts = np.full((NS, MB.shape[3]), np.nan)
    for i in range(NS):
        for k in range(MB.shape[3]):
            if not np.isnan(MB[i, KF, 0, k]):
                shifts[i, k] = best_shift(MB[i, KF, :, k], ML[i, KF, :, k])
    within_sd_shift = np.sqrt(np.nanmean(np.nanvar(shifts, 1, ddof=1)))
    dmu = np.gradient(np.nanmean(MB[:, KF], (0, 2)))
    pred_timing_var = within_sd_shift ** 2 * dmu ** 2
    dd = ML[:, KF] - MB[:, KF]
    dd = dd - np.nanmean(dd, 2, keepdims=True)
    n = np.sum(~np.isnan(dd[:, 0, :]), 1)
    obs_diff_var = np.mean(np.nansum(dd ** 2, 2) / (n[:, None] - 1), 0)
    # sensitivity: re-register each ML stride by its shift and recompute stride-level r
    MLr = ML.copy()
    for i in range(NS):
        for k in range(MB.shape[3]):
            if not np.isnan(shifts[i, k]):
                for v in range(NV):
                    MLr[i, v, :, k] = np.interp(T + shifts[i, k], T, ML[i, v, :, k])
    reg_r = [scalar_summary(components(MB[:, v], MLr[:, v]))["r_w"] for v in range(NV)]
    timing = dict(
        mean_shift_pctGC=np.nanmean(shifts),
        between_subj_sd_of_mean_shift=np.std(np.nanmean(shifts, 1), ddof=1),
        within_subj_sd_shift=within_sd_shift,
        timing_share_of_stride_diff_var_kneeflex_swing=pred_timing_var[60:90].sum() / obs_diff_var[60:90].sum(),
        timing_share_of_stride_diff_var_kneeflex_cycle=pred_timing_var.sum() / obs_diff_var.sum())
    pd.DataFrame([timing]).to_csv(RES / "event_timing.csv", index=False)
    # How much of the *mean* ML-MB difference waveform is a pure time offset?
    # ML(t + g) ~ MB(t)  =>  predicted ML(t) - MB(t) = MB(t - g) - MB(t), plus a constant offset
    g = timing["mean_shift_pctGC"]
    tb_rows = []
    for v in range(NV):
        mb_m, ml_m = np.nanmean(MB[:, v], (0, 2)), np.nanmean(ML[:, v], (0, 2))
        obs = ml_m - mb_m
        pred = np.interp(T - g, T, mb_m) - mb_m
        tb_rows.append(dict(variable=TITLES[v], max_abs_pred_timing_bias=np.abs(pred).max(),
                            frac_bias_shape_explained_by_timing=1 - np.var(obs - pred) / np.var(obs)))
    pd.DataFrame(tb_rows).to_csv(RES / "timing_explained_bias.csv", index=False)
    pd.DataFrame(dict(variable=TITLES, r_w=tab.r_w, r_w_after_registration=reg_r)
                 ).to_csv(RES / "registration_sensitivity.csv", index=False)

    # ================================================================ FIGURES
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 8.5, "axes.titleweight": "bold"})

    def grid_fig(title):
        f, ax = plt.subplots(4, 3, figsize=(8.5, 9.5), sharex=True)
        f.suptitle(title, fontsize=9.5, fontweight="bold")
        return f, ax.ravel()

    def label(a, v, ylab):
        a.set_xlim(0, 100)
        if v % 3 == 0: a.set_ylabel(ylab)
        if v >= 9: a.set_xlabel("Gait cycle (%)")

    # Fig 1: variance components over the gait cycle
    f, ax = grid_fig("Fig 1. Between-subject (solid) and within-subject stride (dashed) SD")
    for v in range(NV):
        c, a = curves[v], ax[v]
        a.plot(T, np.sqrt(c["b_mb"].clip(0)), color=C_MB, lw=1.6, label="MB between")
        a.plot(T, np.sqrt(c["b_ml"].clip(0)), color=C_ML, lw=1.6, label="ML between")
        a.plot(T, np.sqrt(c["w_mb"]), color=C_MB, lw=1.2, ls="--", label="MB within")
        a.plot(T, np.sqrt(c["w_ml"]), color=C_ML, lw=1.2, ls="--", label="ML within")
        a.set_title(TITLES[v]); a.set_ylim(bottom=0); label(a, v, "SD (°)")
    ax[0].legend(fontsize=6, frameon=False)
    f.tight_layout(); f.savefig(FIG / "fig1_variance_components.png", dpi=200); plt.close(f)

    # Fig 2: ML/MB SD ratio at each level; scalar rescaling predicts two coincident flat lines
    f, ax = grid_fig("Fig 2. ML:MB SD ratio. A rescaled MB would give two identical flat lines")
    for v in range(NV):
        c, a = curves[v], ax[v]
        a.axhline(1, color="0.6", lw=0.8)
        a.plot(T, np.sqrt(c["w_ml"] / c["w_mb"]), color="k", ls="--", lw=1.2, label="within-subject (stride)")
        a.plot(T, np.sqrt(c["b_ml"].clip(1e-6) / c["b_mb"].clip(1e-6)), color="k", lw=1.6, label="between-subject")
        a.set_yscale("log"); a.set_ylim(0.1, 4)
        a.set_yticks([0.125, 0.25, 0.5, 1, 2, 4]); a.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4"])
        a.minorticks_off()
        a.set_title(TITLES[v]); label(a, v, "SD ratio ML/MB")
        a.text(98, 2.9, f"flat-ratio test p={perm.p_perm[v]:.3f}", ha="right", fontsize=6)
    ax[0].legend(fontsize=6, frameon=False, loc="lower left")
    f.tight_layout(); f.savefig(FIG / "fig2_sd_ratio.png", dpi=200); plt.close(f)

    # Fig 3: stride variance budget
    f, ax = grid_fig("Fig 3. Stride-to-stride variance: shared by both systems (grey) vs system-specific")
    for v in range(NV):
        c, a = curves[v], ax[v]
        a.fill_between(T, 0, c["w_c"].clip(0), color="0.8", label="shared (cross-cov.)")
        a.plot(T, c["w_mb"], color=C_MB, lw=1.4, label="MB total")
        a.plot(T, c["w_ml"], color=C_ML, lw=1.4, label="ML total")
        a.set_title(f"{TITLES[v]}  r={tab.r_w[v]:.2f}"); a.set_ylim(bottom=0); label(a, v, "Variance (°²)")
    ax[0].legend(fontsize=6, frameon=False)
    f.tight_layout(); f.savefig(FIG / "fig3_shared_vs_specific.png", dpi=200); plt.close(f)

    # Fig 4: functional PCs of stride deviations
    f, ax = grid_fig("Fig 4. First two functional PCs of stride deviations (PC × √eigenvalue)")
    for v in range(NV):
        a = ax[v]
        for name, x, col in (("MB", MB, C_MB), ("ML", ML, C_ML)):
            d = stride_devs(x[:, v])
            lam, vec = np.linalg.eigh(np.cov(d.T)); lam, vec = lam[::-1], vec[:, ::-1]
            for j, ls in ((0, "-"), (1, ":")):
                sgn = np.sign(vec[np.argmax(np.abs(vec[:, j])), j])
                a.plot(T, sgn * vec[:, j] * np.sqrt(lam[j]), color=col, ls=ls, lw=1.3,
                       label=f"{name} PC{j+1} {100*lam[j]/lam.sum():.0f}%")
        a.axhline(0, color="0.7", lw=0.6)
        a.set_title(f"{TITLES[v]}  K={fp.krz_q3[v]:.2f}")
        a.legend(fontsize=5, frameon=False, ncol=2, loc="lower center"); label(a, v, "°")
    f.tight_layout(); f.savefig(FIG / "fig4_fpca.png", dpi=200); plt.close(f)

    # Fig 5: inter-joint coupling
    f, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for a, (p, q, lab) in zip(ax[:2], ((HR, KV, "Hip rotation ↔ knee varus"), (AI, AR, "Ankle inversion ↔ ankle rotation"))):
        a.plot(T, R_MB[p, q], color=C_MB, label="MB"); a.plot(T, R_ML[p, q], color=C_ML, label="ML")
        a.axhline(0, color="0.7", lw=0.6); a.set_ylim(-1, 1); a.set_xlim(0, 100)
        a.set_title(lab); a.set_xlabel("Gait cycle (%)"); a.set_ylabel("within-subject stride r")
    ax[0].legend(frameon=False)
    im = ax[2].imshow(R_MB.mean(2) - R_ML.mean(2), cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax[2].set_xticks(range(NV)); ax[2].set_xticklabels(SHORT, rotation=90, fontsize=6)
    ax[2].set_yticks(range(NV)); ax[2].set_yticklabels(SHORT, fontsize=6)
    ax[2].set_title("Coupling r, MB − ML (cycle mean)")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    f.tight_layout(); f.savefig(FIG / "fig5_coupling.png", dpi=200); plt.close(f)

    # Fig 6: decomposition of Var(ML−MB) and its effect on equivalence
    f, ax = grid_fig("Fig 6. Var(ML−MB) of subject means: scale mismatch (amber) vs decorrelation (purple).\n"
                     "Black: |bias| + 90% CI half-width; dotted: same if r=1; green: MDC bound")
    for v in range(NV):
        e, a = eq_curves[v], ax[v]
        a.stackplot(T, e["scale"], e["decor"], colors=["#f2b134", "#7a5195"], alpha=0.75)
        a2 = a.twinx()
        a2.plot(T, np.abs(e["bias"]) + e["hw"], color="k", lw=1.1)
        a2.plot(T, np.abs(e["bias"]) + e["hw_cf"], color="k", ls=":", lw=1.1)
        a2.axhline(mdc[v], color="g", ls="--", lw=0.9)
        a2.set_ylim(0, max(mdc[v] * 2.2, (np.abs(e["bias"]) + e["hw"]).max() * 1.05))
        a2.spines["right"].set_visible(True)
        if v % 3 == 2: a2.set_ylabel("°")
        a.set_title(f"{TITLES[v]}  equiv. {eqt.pct_equiv_observed[v]:.0f}% → {eqt.pct_equiv_if_r1[v]:.0f}% if r=1")
        label(a, v, "Var(diff) (°²)")
    f.tight_layout(); f.savefig(FIG / "fig6_difference_decomposition.png", dpi=200); plt.close(f)

    # Fig 7: event timing
    f, ax = plt.subplots(1, 2, figsize=(8.5, 3))
    ax[0].hist(shifts[~np.isnan(shifts)], bins=40, color="0.4")
    ax[0].set_xlabel("ML shift that best aligns knee flexion with MB (% GC)"); ax[0].set_ylabel("strides")
    ax[1].plot(T, obs_diff_var, color="k", label="observed stride-level Var(ML−MB)")
    ax[1].plot(T, pred_timing_var, color="#d95f02", label="predicted from relative timing jitter")
    ax[1].set_xlabel("Gait cycle (%)"); ax[1].set_ylabel("°²"); ax[1].set_title("Knee flexion")
    ax[1].legend(frameon=False, fontsize=7)
    f.tight_layout(); f.savefig(FIG / "fig7_event_timing.png", dpi=200); plt.close(f)

    return dict(tab=tab, perm=perm, fp=fp, cm=cm, eqt=eqt, timing=timing, reg_r=reg_r)


def fig0_gross_errors(MB, ML, ids, flags_log):
    """Raw strides for subjects with flagged MB gross errors."""
    FIG = HERE / "figures"
    subs = list(dict.fromkeys(flags_log.subject))[:4]
    f, ax = plt.subplots(len(subs), 3, figsize=(9, 2.1 * len(subs)), sharex=True)
    for r, sid in enumerate(subs):
        i = ids.index(sid)
        for c, v in enumerate((AD, AI, AR)):
            a = ax[r, c]
            a.plot(T, MB[i, v], color=C_MB, lw=0.7); a.plot(T, ML[i, v], color=C_ML, lw=0.7)
            a.set_title(f"{sid} {TITLES[v]}", fontsize=8)
            a.spines[["top", "right"]].set_visible(False)
    for a in ax[-1]: a.set_xlabel("Gait cycle (%)")
    f.suptitle("Fig 0. Gross-error strides (MB blue, ML red); all 8 strides shown", fontsize=9, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "fig0_gross_errors.png", dpi=170); plt.close(f)


if __name__ == "__main__":
    MB, ML, IDS, MDC, _ = load()
    MBc, MLc, IDSc, log = clean(MB, ML, IDS)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    log.to_csv(HERE / "results" / "gross_error_log.csv", index=False)
    fig0_gross_errors(MB, ML, IDS, log)
    print(log.to_string(index=False))
    print(f"clean: {MBc.shape[0]} subjects, {np.sum(~np.isnan(MBc[:, 0, 0, :]))} strides")

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    for tag, (mb, ml) in (("primary_clean", (MBc, MLc)), ("sensitivity_all", (MB, ML))):
        out = run(mb, ml, MDC, tag, np.random.default_rng(20261029))
        print(f"\n=================== {tag}")
        t = out["tab"]
        print(t[["variable", "SDw_MB", "SDw_ML", "SDb_MB", "SDb_ML", "ratio_w", "ratio_w_lo", "ratio_w_hi",
                 "ratio_b", "ratio_b_lo", "ratio_b_hi", "log_ratio_w_minus_b", "log_ratio_w_minus_b_lo",
                 "log_ratio_w_minus_b_hi"]])
        print(t[["variable", "r_w", "r_w_lo", "r_w_hi", "r_b", "r_b_lo", "r_b_hi", "uniq_w_MB", "uniq_w_ML",
                 "lam_lo", "lam_hi", "icc_MB", "icc_ML", "r_indiv_var", "r_indiv_var_lo", "r_indiv_var_hi"]])
        print(out["perm"])
        print(out["fp"][["variable", "PR_MB", "PR_ML", "PR_MB_lo", "PR_MB_hi", "PR_ML_lo", "PR_ML_hi", "pc1_MB",
                         "pc1_ML", "rough_MB", "rough_ML", "rough_MB_lo", "rough_MB_hi", "rough_ML_lo", "rough_ML_hi",
                         "kurt_MB", "kurt_ML", "krz_q3", "krz_q3_lo", "krz_q3_hi", "krz_MB_splithalf_median"]])
        print(out["cm"].head(10))
        print(out["eqt"])
        print(out["timing"])
        print(np.round(out["reg_r"], 2))
