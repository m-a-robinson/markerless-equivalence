"""Pointwise (functional) 95% limits of agreement between marker-based and markerless
kinematics, with +/-5 deg bounds and LoA widths annotated.

At each of the 101 gait-cycle nodes: LoA = mean(d) +/- 1.96*SD(d), where d is the
paired MB - ML difference across participants (participant means over trials).
Usage: python3 plot_loa_pointwise.py [bound_deg]   (default 5)
"""
import sys, csv
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BOUND = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
m = sio.loadmat('stydy2data_20260226_withMDC.mat')
gm = m['s2data'][0, 0]['grandmeans'][0, 0]
plt_order_r = [11, 12, 13, 18, 19, 20, 21, 22, 17, 14, 15, 16]
titles = ['Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',
          'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
          'Knee Flexion', 'Knee Varus', 'Foot Progression',
          'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation']
x = np.arange(101)
C_IN, C_OUT = '#a6dcbd', '#f7d58b'      # LoA band: inside / not inside the bounds
GREEN = '#33ee33'

fig, axes = plt.subplots(4, 3, figsize=(8, 9.6), sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, top=0.925, bottom=0.06, hspace=0.3, wspace=0.22)
rows = []
for jj, ax in enumerate(axes.ravel()):
    k = plt_order_r[jj] - 1
    d = gm['mb'][k, 0] - gm['ml'][k, 0]          # 101 x n participants (MB - ML)
    n = d.shape[1]
    md, sd = d.mean(1), d.std(1, ddof=1)
    lo, hi = md - 1.96 * sd, md + 1.96 * sd
    w = hi - lo                                   # LoA width at each node
    inside = (lo >= -BOUND) & (hi <= BOUND)

    ax.plot(x, d, color='0.8', lw=0.5, zorder=1)  # individual participant differences
    ax.fill_between(x, lo, hi, color=C_OUT, alpha=0.9, lw=0, zorder=2)
    ax.fill_between(x, lo, hi, where=inside | np.r_[inside[1:], False] | np.r_[False, inside[:-1]],
                    color=C_IN, lw=0, zorder=3)
    ax.plot(x, lo, color='0.35', lw=0.8, zorder=4)
    ax.plot(x, hi, color='0.35', lw=0.8, zorder=4)
    ax.plot(x, md, 'k', lw=2.2, zorder=5)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axhline(BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)
    ax.axhline(-BOUND, color=GREEN, ls='--', lw=1.6, zorder=6)

    # LoA width: double-headed arrow at the node of maximum width
    i = int(np.argmax(w))
    ax.annotate('', xy=(i, lo[i]), xytext=(i, hi[i]),
                arrowprops=dict(arrowstyle='<->', color='#b00020', lw=1.4, shrinkA=0, shrinkB=0), zorder=7)
    ax.text(i + (3 if i < 70 else -3), md[i], f'{w[i]:.0f}°', color='#b00020', fontsize=7.5,
            fontweight='bold', ha='left' if i < 70 else 'right', va='center', zorder=7,
            bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))

    lim = max(10, np.ceil(max(np.abs(lo).max(), np.abs(hi).max(), BOUND) / 5) * 5)
    ax.set_xlim(0, 100); ax.set_ylim(-lim, lim + 0.45 * lim * 0.5)
    ax.set_title(titles[jj], fontsize=9, fontweight='bold')
    ax.text(0.02, 0.97, f'width {w.mean():.1f}° (mean)\n{100 * inside.mean():.0f}% within ±{BOUND:g}°',
            transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#333')
    ax.tick_params(labelsize=7)
    if jj % 3 == 0: ax.set_ylabel('MB − ML (°)', fontsize=8)
    if jj >= 9: ax.set_xlabel('Gait Cycle (%)', fontsize=8)
    rows.append([titles[jj], n, round(w.mean(), 2), round(w.min(), 2), round(w.max(), 2),
                 round(100 * inside.mean(), 1), round(np.abs(md).max(), 2)])

handles = [Line2D([0], [0], color='k', lw=2, label='Mean difference'),
           Patch(color=C_IN, label=f'LoA within ±{BOUND:g}°'),
           Patch(color=C_OUT, label=f'LoA exceeds ±{BOUND:g}°'),
           Line2D([0], [0], color=GREEN, ls='--', label=f'±{BOUND:g}° bounds'),
           Line2D([0], [0], color='0.8', lw=1, label='Individual participants'),
           Line2D([0], [0], color='#b00020', marker='$\\leftrightarrow$', ls='', ms=9, label='Max LoA width')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.savefig(f'mlmb_loa_pointwise_{BOUND:g}deg.jpg', dpi=300)

with open(f'loa_pointwise_summary_{BOUND:g}deg.csv', 'w', newline='') as f:
    w_ = csv.writer(f)
    w_.writerow(['Variable', 'N', 'MeanLoAWidth_deg', 'MinLoAWidth_deg', 'MaxLoAWidth_deg',
                 f'PctGaitCycleLoAWithin{BOUND:g}deg', 'MaxAbsMeanDiff_deg'])
    w_.writerows(rows)
print(f"{'Variable':22s}{'N':>3s}{'meanW':>7s}{'minW':>6s}{'maxW':>6s}{'%in':>6s}{'max|bias|':>10s}")
for r in rows: print(f'{r[0]:22s}{r[1]:3d}{r[2]:7.1f}{r[3]:6.1f}{r[4]:6.1f}{r[5]:6.0f}{r[6]:10.1f}')
