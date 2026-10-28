"""Equivalence figure (MDC bounds) with three-way classification and SPM paired t-test bar.

Python counterpart of the 'mlmb_equivalence_mdc_feb2026_percent.jpg' figure in
ML_MB_EquivalenceAnalysis_MDC.m (uses the spm1d python port; reproduces the
saved MATLAB ci_data percentages).

Usage: python3 plot_equivalence_mdc_classified.py
"""
import numpy as np
import scipy.io as sio
import spm1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ALPHA_CI = 0.1    # 90% CI (two one-sided tests at 0.05)
ALPHA_T = 0.05    # SPM paired t-test, two-tailed

m = sio.loadmat('stydy2data_20260226_withMDC.mat')
gm = m['s2data'][0, 0]['grandmeans'][0, 0]
plt_order_r = [11, 12, 13, 18, 19, 20, 21, 22, 17, 14, 15, 16]
titles = ['Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',
          'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
          'Knee Flexion', 'Knee Varus', 'Foot Progression',
          'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation']
bounds = [float(np.squeeze(m['ci_data'][j, 1])) for j in range(12)]  # foot progression = 5 default
ylim_max = [10] * 10 + [20, 20]
ylim_min = [-v for v in ylim_max]

C_EQ, C_NE, C_INC = '#2ca25f', '#de5a3c', '#d9d9d9'
x = np.arange(101)

def runs(mask):
    """Contiguous [start, end] node ranges where mask is True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(idx[np.r_[0, br + 1]], idx[np.r_[br, idx.size - 1]]))

fig = plt.figure(figsize=(8, 10))
gs = fig.add_gridspec(8, 3, height_ratios=[10, 1.1] * 4, hspace=0.35, wspace=0.22,
                      left=0.08, right=0.98, top=0.93, bottom=0.06)
rows = []
for jj in range(12):
    r, c = divmod(jj, 3)
    ax = fig.add_subplot(gs[2 * r, c])
    axb = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)

    k = plt_order_r[jj] - 1
    yA, yB = gm['mb'][k, 0].T, gm['ml'][k, 0].T            # subjects x 101
    b = bounds[jj]

    ci = spm1d.stats.ci_pairedsample(yA, yB, ALPHA_CI, datum='difference', mu=0)
    lo, hi = np.asarray(ci.ci).min(0), np.asarray(ci.ci).max(0)
    md = np.asarray(ci.mean if hasattr(ci, 'mean') else (yA - yB).mean(0)).ravel()
    if md.shape != (101,):
        md = (yA - yB).mean(0)

    eq = (lo >= -b) & (hi <= b)
    ne = (lo > b) | (hi < -b)
    inc = ~eq & ~ne
    pct_eq = 100 * eq.mean()

    # classification shading (each node spans +/-0.5)
    for mask, col in ((eq, C_EQ), (ne, C_NE), (inc, C_INC)):
        for s, e in runs(mask):
            ax.axvspan(s - 0.5, e + 0.5, color=col, alpha=0.35, lw=0)

    ax.fill_between(x, lo, hi, color='k', alpha=0.25, lw=0)
    ax.plot(x, md, 'k', lw=1)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axhline(b, color='g', ls='--', lw=1.3)
    ax.axhline(-b, color='g', ls='--', lw=1.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(ylim_min[jj], ylim_max[jj] + 5)
    ax.set_title(titles[jj], fontsize=9)
    ax.text(0.98, 0.97, f'{round(pct_eq)}% Equiv.', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#00682f', fontweight='bold')
    ax.tick_params(labelsize=7, labelbottom=False)
    if c == 0:
        ax.set_ylabel('Difference MB − ML (°)', fontsize=8)

    # SPM{t} paired t-test, markerbased vs markerless
    t = spm1d.stats.ttest_paired(yA, yB).inference(ALPHA_T, two_tailed=True, interp=True)
    sig = np.abs(np.asarray(t.z)) > t.zstar
    axb.set_ylim(0, 1); axb.set_yticks([])
    for s, e in runs(sig):
        axb.axvspan(s - 0.5, e + 0.5, color='k', lw=0)
    for sp in ('top', 'right', 'left'):
        axb.spines[sp].set_visible(False)
    axb.set_xticks([0, 20, 40, 60, 80, 100])
    axb.tick_params(labelsize=7, length=2)
    if r < 3:
        axb.tick_params(labelbottom=False)
    else:
        axb.set_xlabel('Gait Cycle (%)', fontsize=8)
    if c == 0:
        axb.set_ylabel('SPM', fontsize=7, rotation=0, ha='right', va='center')

    rows.append((titles[jj], b, pct_eq, 100 * ne.mean(), 100 * inc.mean(), 100 * sig.mean()))

handles = [Patch(color=C_EQ, alpha=0.35, label='Equivalent (CI inside bounds)'),
           Patch(color=C_NE, alpha=0.35, label='Not equivalent (CI outside bounds)'),
           Patch(color=C_INC, alpha=0.35, label='Inconclusive (CI crosses bound)'),
           Line2D([0], [0], color='g', ls='--', label='MDC bounds'),
           Patch(color='k', label='SPM paired t-test p<0.05')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, 0.995))
fig.savefig('mlmb_equivalence_mdc_feb2026_percent_classified.jpg', dpi=300)

print(f"{'Variable':22s}{'bound':>6s}{'%Eq':>6s}{'%NotEq':>8s}{'%Inc':>6s}{'%SPMsig':>9s}")
for n, b, e, ne_, i, s in rows:
    print(f'{n:22s}{b:6.2f}{e:6.0f}{ne_:8.0f}{i:6.0f}{s:9.0f}')
