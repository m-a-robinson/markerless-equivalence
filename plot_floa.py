"""Functional limits of agreement (FLoA; Roislien, Rennie & Skaaret 2012, Gait Posture 36:495-499)
between marker-based (MB) and markerless (ML) gait kinematics, with +/-5 deg bounds.

Following the paper, per variable:
  d_i(t) = MB_i(t) - ML_i(t),  A_i(t) = (MB_i(t) + ML_i(t)) / 2        (participant means)
  1. independence of d and A: concurrent functional regression d(t) = b0(t) + b1(t) A(t),
     pointwise slope CI and a functional permutation F-test (max-F over the gait cycle)
  2. FLoA(t) = mean d(t) +/- 1.96 SD d(t)
  3. estimation uncertainty: bootstrap (participants resampled) 95% CI of each FLoA limit

Extra (not in the paper): a simultaneous band built with the spm1d random-field-theory
critical threshold, mean +/- zstar * SD * sqrt(1 + 1/n)  (functional prediction band), checked
by leave-one-participant-out coverage.

The sampled curves have 101 nodes (as many basis functions as data), so the functional
objects reproduce the data at the nodes and FLoA can be evaluated directly at the nodes.

Usage: python3 plot_floa.py [bound_deg]      (default 5)
"""
import sys, csv
import numpy as np
import scipy.io as sio
from scipy import stats as ss
import spm1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BOUND = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
NBOOT, NPERM = 5000, 5000
rng = np.random.default_rng(2026)

m = sio.loadmat('stydy2data_20260226_withMDC.mat')
gm = m['s2data'][0, 0]['grandmeans'][0, 0]
plt_order_r = [11, 12, 13, 18, 19, 20, 21, 22, 17, 14, 15, 16]
titles = ['Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',
          'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
          'Knee Flexion', 'Knee Varus', 'Foot Progression',
          'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation']
x = np.arange(101)
GREEN, RED, BLUE = '#33ee33', '#b00020', '#4a7fb5'


def floa(d, k=1.96):
    """d: nodes x participants -> (mean, lower, upper)."""
    md, sd = d.mean(1), d.std(1, ddof=1)
    return md, md - k * sd, md + k * sd


def regression_independence(d, A):
    """Concurrent functional regression of d(t) on A(t) across participants.
    Returns slope(t), 95% CI of slope, overall permutation p (max-F), pointwise F."""
    n = d.shape[1]
    Ac = A - A.mean(1, keepdims=True)
    Sxx = (Ac ** 2).sum(1)

    def fstat(dd):
        dc = dd - dd.mean(1, keepdims=True)
        b1 = (Ac * dc).sum(1) / Sxx
        sse = (dc ** 2).sum(1) - b1 ** 2 * Sxx
        F = b1 ** 2 * Sxx / (sse / (n - 2))
        return b1, F, sse

    b1, F, sse = fstat(d)
    se = np.sqrt(sse / (n - 2) / Sxx)
    tcrit = ss.t.ppf(0.975, n - 2)
    perm_max = np.empty(NPERM)
    for p in range(NPERM):                       # shuffle d-curves against A-curves
        perm_max[p] = fstat(d[:, rng.permutation(n)])[1].max()
    p_overall = (1 + (perm_max >= F.max()).sum()) / (1 + NPERM)
    return b1, b1 - tcrit * se, b1 + tcrit * se, p_overall


def bootstrap_floa(d):
    n = d.shape[1]
    idx = rng.integers(0, n, size=(NBOOT, n))
    db = d[:, idx]                               # nodes x NBOOT x n
    md, sd = db.mean(2), db.std(2, ddof=1)
    return md - 1.96 * sd, md + 1.96 * sd        # nodes x NBOOT each


def rft_band(d, alpha=0.05):
    """Simultaneous prediction band: mean +/- zstar*SD*sqrt(1+1/n) (zstar = spm1d RFT threshold)."""
    n = d.shape[1]
    zstar = spm1d.stats.ttest(d.T).inference(alpha, two_tailed=True, interp=True).zstar
    md, sd = d.mean(1), d.std(1, ddof=1)
    h = zstar * sd * np.sqrt(1 + 1 / n)
    return md - h, md + h, zstar


def loo_coverage(d):
    """Leave-one-participant-out: does the held-out curve lie inside the band built from the others?
    Returns whole-curve coverage and node-wise coverage for FLoA (1.96 SD) and the RFT band."""
    n = d.shape[1]
    res = np.zeros((2, 2))
    for i in range(n):
        rest, held = np.delete(d, i, 1), d[:, i]
        _, lo, hi = floa(rest)
        inside = (held >= lo) & (held <= hi)
        res[0] += [inside.all(), inside.mean()]
        lo2, hi2, _ = rft_band(rest)
        inside = (held >= lo2) & (held <= hi2)
        res[1] += [inside.all(), inside.mean()]
    return res / n


results = []
fig, axes = plt.subplots(4, 3, figsize=(8, 9.8), sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, top=0.915, bottom=0.06, hspace=0.3, wspace=0.22)
figb, axesb = plt.subplots(4, 3, figsize=(8, 9.8), sharex=True)
figb.subplots_adjust(left=0.08, right=0.98, top=0.915, bottom=0.06, hspace=0.3, wspace=0.22)

for jj in range(12):
    k = plt_order_r[jj] - 1
    mb, ml = gm['mb'][k, 0], gm['ml'][k, 0]              # nodes x participants
    d, A = mb - ml, (mb + ml) / 2
    n = d.shape[1]
    md, lo, hi = floa(d)
    w = hi - lo
    inside = (lo >= -BOUND) & (hi <= BOUND)

    b1, b1lo, b1hi, p_ind = regression_independence(d, A)
    slope_sig = 100 * np.mean((b1lo > 0) | (b1hi < 0))
    blo, bhi = bootstrap_floa(d)
    lo_ci = np.percentile(blo, [2.5, 97.5], axis=1)       # CI of lower limit  (2 x nodes)
    hi_ci = np.percentile(bhi, [2.5, 97.5], axis=1)       # CI of upper limit
    inside_ci = (lo_ci[0] >= -BOUND) & (hi_ci[1] <= BOUND)  # whole bootstrap CI inside bounds
    slo, shi, zstar = rft_band(d)
    cov = loo_coverage(d)

    i = int(np.argmax(w))
    lim = max(10, np.ceil(max(np.abs(lo_ci).max(), np.abs(hi_ci).max(), np.abs(slo).max(), np.abs(shi).max(), BOUND) / 5) * 5)

    # ---------- Figure A: FLoA + bootstrap CI + bounds ----------
    ax = axes.ravel()[jj]
    ax.plot(x, d, color='0.82', lw=0.5, zorder=1)
    ax.fill_between(x, lo_ci[0], lo_ci[1], color=BLUE, alpha=0.25, lw=0, zorder=2)
    ax.fill_between(x, hi_ci[0], hi_ci[1], color=BLUE, alpha=0.25, lw=0, zorder=2)
    ax.plot(x, lo, '--', color='k', lw=1.2, zorder=4)
    ax.plot(x, hi, '--', color='k', lw=1.2, zorder=4)
    ax.plot(x, md, 'k', lw=2.2, zorder=5)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axhline(BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    ax.axhline(-BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    ax.annotate('', xy=(i, lo[i]), xytext=(i, hi[i]),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.4, shrinkA=0, shrinkB=0), zorder=7)
    ax.text(i + (3 if i < 70 else -3), md[i], f'{w[i]:.0f}°', color=RED, fontsize=7.5, fontweight='bold',
            ha='left' if i < 70 else 'right', va='center', zorder=7,
            bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
    ax.text(0.02, 0.97, f'FLoA width {w.mean():.1f}° (mean)\n{100 * inside.mean():.0f}% of cycle within ±{BOUND:g}°',
            transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#333')
    ax.set_ylim(-lim, lim * 1.2)

    # ---------- Figure B: pointwise FLoA vs simultaneous RFT prediction band ----------
    axb = axesb.ravel()[jj]
    axb.fill_between(x, slo, shi, color='#f7d58b', alpha=0.9, lw=0, zorder=2)
    axb.plot(x, lo, '--', color='k', lw=1.1, zorder=4)
    axb.plot(x, hi, '--', color='k', lw=1.1, zorder=4)
    axb.plot(x, md, 'k', lw=2.2, zorder=5)
    axb.axhline(0, color='k', lw=0.5, ls=':')
    axb.axhline(BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    axb.axhline(-BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    axb.text(0.02, 0.97, f'z* = {zstar:.2f} (vs 1.96)\nwidth {np.mean(shi - slo):.1f}° vs {w.mean():.1f}°',
             transform=axb.transAxes, ha='left', va='top', fontsize=7, color='#333')
    axb.set_ylim(-lim, lim * 1.2)

    for a in (ax, axb):
        a.set_xlim(0, 100)
        a.set_title(titles[jj], fontsize=9, fontweight='bold')
        a.tick_params(labelsize=7)
        if jj % 3 == 0: a.set_ylabel('MB − ML (°)', fontsize=8)
        if jj >= 9: a.set_xlabel('Gait Cycle (%)', fontsize=8)

    results.append([titles[jj], n, round(w.mean(), 2), round(w.min(), 2), round(w.max(), 2),
                    round(100 * inside.mean(), 1), round(100 * inside_ci.mean(), 1),
                    round(float(np.abs(md).max()), 2), round(p_ind, 4), round(slope_sig, 1),
                    round(float(np.mean(b1)), 3), round(zstar, 2), round(float(np.mean(shi - slo)), 2),
                    round(100 * cov[0, 0], 1), round(100 * cov[0, 1], 1),
                    round(100 * cov[1, 0], 1), round(100 * cov[1, 1], 1)])
    print(titles[jj], 'done', flush=True)

hA = [Line2D([0], [0], color='k', lw=2, label='Mean difference'),
      Line2D([0], [0], color='k', ls='--', lw=1.2, label='FLoA (mean ± 1.96 SD)'),
      Patch(color=BLUE, alpha=0.25, label='Bootstrap 95% CI of FLoA'),
      Line2D([0], [0], color=GREEN, ls='--', lw=1.6, label=f'±{BOUND:g}° bounds'),
      Line2D([0], [0], color='0.8', lw=1, label='Individual participants'),
      Line2D([0], [0], color=RED, marker='$\\leftrightarrow$', ls='', ms=9, label='Max FLoA width')]
fig.legend(handles=hA, loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.savefig(f'mlmb_floa_{BOUND:g}deg.jpg', dpi=300)
hB = [Line2D([0], [0], color='k', lw=2, label='Mean difference'),
      Line2D([0], [0], color='k', ls='--', lw=1.1, label='Pointwise FLoA (± 1.96 SD)'),
      Patch(color='#f7d58b', label='Simultaneous RFT band (z* · SD · √(1+1/n))'),
      Line2D([0], [0], color=GREEN, ls='--', lw=1.6, label=f'±{BOUND:g}° bounds')]
figb.legend(handles=hB, loc='upper center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
figb.savefig(f'mlmb_floa_simultaneous_{BOUND:g}deg.jpg', dpi=300)

hdr = ['Variable', 'N', 'MeanFLoAWidth_deg', 'MinWidth_deg', 'MaxWidth_deg',
       f'PctCycleFLoAWithin{BOUND:g}', f'PctCycleBootCIWithin{BOUND:g}', 'MaxAbsMeanDiff_deg',
       'IndepPermP_dvsA', 'PctCycleSlopeCIexcludes0', 'MeanSlope', 'RFT_zstar', 'MeanRFTBandWidth_deg',
       'LOO_WholeCurveCoverage_FLoA_pct', 'LOO_NodeCoverage_FLoA_pct',
       'LOO_WholeCurveCoverage_RFT_pct', 'LOO_NodeCoverage_RFT_pct']
with open(f'floa_summary_{BOUND:g}deg.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(hdr); wr.writerows(results)
for r in results:
    print(f'{r[0]:20s} width {r[2]:5.1f} in{r[5]:4.0f}% bootCIin{r[6]:4.0f}% | indep p={r[8]:.3f} slopeSig{r[9]:4.0f}% | z*={r[11]:.2f} RFTw {r[12]:5.1f} | LOO whole FLoA {r[13]:4.0f}% RFT {r[15]:4.0f}%')
