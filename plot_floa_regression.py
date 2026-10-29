"""Regression-based functional limits of agreement (after Carstensen 2010 / Bland & Altman 1999,
applied functionally as in Roislien et al. 2012).

At every node of the gait cycle, across participants:
    d(t) = a(t) + b(t) * A(t) + e(t),   d = MB - ML,  A = (MB + ML) / 2
    mean difference given A:   a(t) + b(t) * A
    limits:                    a(t) + b(t) * A  +/- 1.96 * s_res(t)
Plotted for A at the 10th, 50th and 90th percentile of the participants' A(t), against the
conventional FLoA (grey) and +/-5 deg bounds.  b(t) = 0 recovers the conventional FLoA.
cov(d, A) = (var(MB) - var(ML)) / 2, so b > 0 means MB is more variable between participants than ML.

Usage: python3 plot_floa_regression.py [bound_deg]   (default 5)
"""
import sys, csv
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BOUND = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
m = sio.loadmat('stydy2data_20260226_withMDC.mat')
gm = m['s2data'][0, 0]['grandmeans'][0, 0]
plt_order_r = [11, 12, 13, 18, 19, 20, 21, 22, 17, 14, 15, 16]
titles = ['Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',
          'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
          'Knee Flexion', 'Knee Varus', 'Foot Progression',
          'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation']
perm_p = {r['Variable']: float(r['IndepPermP_dvsA']) for r in csv.DictReader(open('floa_summary_5deg.csv'))}
x = np.arange(101)
GREEN = '#33ee33'
LEVELS = [(10, '#2b6cb0', 'A low (10th percentile)'), (50, 'k', 'A median'), (90, '#d9731a', 'A high (90th percentile)')]

fig, axes = plt.subplots(4, 3, figsize=(8, 9.8), sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, top=0.915, bottom=0.06, hspace=0.3, wspace=0.22)
rows = []
for jj, ax in enumerate(axes.ravel()):
    k = plt_order_r[jj] - 1
    mb, ml = gm['mb'][k, 0], gm['ml'][k, 0]               # nodes x participants
    d, A = mb - ml, (mb + ml) / 2
    n = d.shape[1]
    Ac = A - A.mean(1, keepdims=True); dc = d - d.mean(1, keepdims=True)
    Sxx = (Ac ** 2).sum(1)
    b = (Ac * dc).sum(1) / Sxx
    a = d.mean(1) - b * A.mean(1)
    sres = np.sqrt(((dc - b[:, None] * Ac) ** 2).sum(1) / (n - 2))
    sraw = d.std(1, ddof=1)
    r2 = 1 - sres ** 2 * (n - 2) / ((dc ** 2).sum(1))

    # conventional FLoA for reference
    ax.fill_between(x, d.mean(1) - 1.96 * sraw, d.mean(1) + 1.96 * sraw, color='0.88', lw=0, zorder=1)

    within50 = None
    for pc, col, _ in LEVELS:
        Ap = np.percentile(A, pc, axis=1)
        mu = a + b * Ap
        lo, hi = mu - 1.96 * sres, mu + 1.96 * sres
        ax.plot(x, lo, '--', color=col, lw=1.0, zorder=3)
        ax.plot(x, hi, '--', color=col, lw=1.0, zorder=3)
        ax.plot(x, mu, color=col, lw=2.0 if pc == 50 else 1.5, zorder=4)
        if pc == 50:
            within50 = 100 * np.mean((lo >= -BOUND) & (hi <= BOUND))
            w50 = (hi - lo).mean()
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axhline(BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    ax.axhline(-BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    allv = [np.abs(a + b * np.percentile(A, p, axis=1)).max() + 1.96 * sres.max() for p in (10, 50, 90)]
    lim = max(10, np.ceil(max(max(allv), 1.96 * sraw.max() + np.abs(d.mean(1)).max(), BOUND) / 5) * 5)
    ax.set_ylim(-lim, lim * 1.25); ax.set_xlim(0, 100)
    ax.set_title(titles[jj], fontsize=9, fontweight='bold')
    ax.text(0.02, 0.97, f'slope {b.mean():.2f}, R² {r2.mean():.2f}\nwidth {2 * 1.96 * sres.mean():.1f}° vs {2 * 1.96 * sraw.mean():.1f}°',
            transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#333')
    ax.tick_params(labelsize=7)
    if jj % 3 == 0: ax.set_ylabel('MB − ML (°)', fontsize=8)
    if jj >= 9: ax.set_xlabel('Gait Cycle (%)', fontsize=8)

    s_mb, s_ml = mb.std(1, ddof=1), ml.std(1, ddof=1)
    rows.append([titles[jj], n, round(s_mb.mean(), 2), round(s_ml.mean(), 2), round(s_mb.mean() / s_ml.mean(), 2),
                 round(100 * np.mean(s_mb > s_ml), 0), round(float(np.mean([np.corrcoef(mb[i], ml[i])[0, 1] for i in range(101)])), 2),
                 round(b.mean(), 3), round(perm_p[titles[jj]], 4), round(r2.mean(), 2),
                 round(2 * 1.96 * sraw.mean(), 2), round(2 * 1.96 * sres.mean(), 2),
                 round(100 * (1 - sres.mean() / sraw.mean()), 0), round(within50, 1)])

handles = [Patch(color='0.88', label='Conventional FLoA'),
           *[Line2D([0], [0], color=c, lw=1.6, label=l) for _, c, l in LEVELS],
           Line2D([0], [0], color='k', ls='--', lw=1, label='Regression LoA (dashed)'),
           Line2D([0], [0], color=GREEN, ls='--', lw=1.6, label=f'±{BOUND:g}° bounds')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.savefig(f'mlmb_floa_regression_{BOUND:g}deg.jpg', dpi=300)

hdr = ['Variable', 'N', 'MeanBetweenSubjSD_MB', 'MeanBetweenSubjSD_ML', 'SDratio_MB_over_ML', 'PctNodesMBmoreVariable',
       'MeanCorr_MB_ML', 'MeanSlope_dOnA', 'IndepPermP', 'MeanR2', 'ConvFLoAwidth_deg', 'RegLoAwidth_deg',
       'PctWidthReduction', f'PctCycleRegLoA_medianA_within{BOUND:g}']
with open(f'floa_regression_summary_{BOUND:g}deg.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print(f"{'':20s}{'SDmb':>6s}{'SDml':>6s}{'ratio':>6s}{'r':>6s}{'slope':>7s}{'R2':>6s}{'convW':>7s}{'regW':>7s}{'red%':>6s}{'in5%':>6s}")
for r in rows: print(f'{r[0]:20s}{r[2]:6.2f}{r[3]:6.2f}{r[4]:6.2f}{r[6]:6.2f}{r[7]:7.2f}{r[9]:6.2f}{r[10]:7.1f}{r[11]:7.1f}{r[12]:6.0f}{r[13]:6.0f}')
