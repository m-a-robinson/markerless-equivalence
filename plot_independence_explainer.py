"""Explainer figure: why d = MB - ML is not independent of A = (MB + ML)/2, and why the extra MB
variance is not simply a scaled-up copy of ML.

Key identity:   cov(d, A) = (var(MB) - var(ML)) / 2     -> d depends on A iff the variances differ.
Scaled-copy counterfactual: ML' = mean(ML) + k (MB - mean(MB)),  k = SD(ML)/SD(MB).  Then
r(MB, ML') = 1 and d' is an exact linear function of A' (all points on one line, zero residual).

Columns: one case where independence holds and three where it fails.  Rows 2-3 use a representative
node: the one whose MB-ML correlation is closest to its cycle average (avoids cherry-picking).   Usage: python3 plot_independence_explainer.py
"""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

m = sio.loadmat('stydy2data_20260226_withMDC.mat')
gm = m['s2data'][0, 0]['grandmeans'][0, 0]
plt_order_r = [11, 12, 13, 18, 19, 20, 21, 22, 17, 14, 15, 16]
names = ['Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation', 'Hip Flexion', 'Hip Adduction',
         'Hip Rotation', 'Knee Flexion', 'Knee Varus', 'Foot Progression', 'Ankle Dorsiflexion',
         'Ankle Inversion', 'Ankle Rotation']
cases = [('Pelvis Obliquity', 'Equal spread:\nindependence holds'),
         ('Foot Progression', 'MB more variable,\nsystems highly correlated'),
         ('Hip Rotation', 'MB 4× more variable,\nsystems weakly correlated'),
         ('Ankle Rotation', 'MB 2× more variable,\nsystems uncorrelated')]
MBC, MLC, GREY = '#2b6cb0', '#d9473a', '0.55'
x = np.arange(101)

fig, axes = plt.subplots(4, 4, figsize=(10.5, 12.5))
fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.075, hspace=0.5, wspace=0.34)
for c, (nm, sub) in enumerate(cases):
    k = plt_order_r[names.index(nm)] - 1
    mb, ml = gm['mb'][k, 0], gm['ml'][k, 0]            # nodes x participants
    n = mb.shape[1]
    s_mb, s_ml = mb.std(1, ddof=1), ml.std(1, ddof=1)
    r_t = np.array([np.corrcoef(mb[i], ml[i])[0, 1] for i in range(101)])
    node = int(np.argmin(np.abs(r_t - r_t.mean())))

    # ---- row 1: participant curves
    ax = axes[0, c]
    ax.plot(x, mb, color=MBC, lw=0.6, alpha=0.6); ax.plot(x, ml, color=MLC, lw=0.6, alpha=0.6)
    ax.axvline(node, color='k', lw=0.8, ls=':')
    ax.set_title(f'{nm}\n{sub}', fontsize=9, fontweight='bold', pad=6)
    ax.set_xlabel(f'Gait Cycle (%)\nmean SD  MB {s_mb.mean():.1f}°, ML {s_ml.mean():.1f}°  (×{s_mb.mean() / s_ml.mean():.1f})', fontsize=8)
    if c == 0: ax.set_ylabel('Angle (°)', fontsize=8)

    # ---- row 2: ML vs MB at the node, vs scaled-copy prediction
    a, b_ = mb[node], ml[node]
    kk = s_ml[node] / s_mb[node]
    ax = axes[1, c]
    lim = [min(a.min(), b_.min()) - 2, max(a.max(), b_.max()) + 2]
    xs = np.array(lim)
    ax.plot(xs, xs, ls='--', color=GREY, lw=1)                                          # identity
    ax.plot(xs, ml[node].mean() + kk * (xs - a.mean()), color='k', lw=1.6)              # scaled copy
    ax.scatter(a, b_, s=16, color='#444', zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('MB angle (°)', fontsize=8); 
    if c == 0: ax.set_ylabel('ML angle (°)', fontsize=8)
    ax.text(0.04, 0.96, f'r = {r_t[node]:.2f},  r² = {r_t[node] ** 2:.2f}\nML SD/MB SD = {kk:.2f}',
            transform=ax.transAxes, fontsize=7.5, va='top', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5), zorder=5)
    ax.set_title(f'representative node: {node}% of gait cycle', fontsize=8)

    # ---- row 3: d vs A at the node
    d, A = a - b_, (a + b_) / 2
    sl, ic = np.polyfit(A, d, 1)
    ax = axes[2, c]
    ax.scatter(A, d, s=16, color='#444', zorder=3)
    xa = np.array([A.min() - 1, A.max() + 1])
    ax.plot(xa, ic + sl * xa, color=MBC, lw=1.6)                                        # observed regression
    sl_sc = 2 * (1 - kk) / (1 + kk)                                                     # scaled-copy slope
    A_c = (a.mean() + b_.mean()) / 2; d_c = a.mean() - b_.mean()
    ax.plot(xa, d_c + sl_sc * (xa - A_c), color='k', lw=1.6)                            # scaled-copy line
    ax.axhline(0, color=GREY, lw=0.6, ls=':')
    ax.set_xlabel('Average A = (MB+ML)/2 (°)', fontsize=8)
    if c == 0: ax.set_ylabel('Difference d = MB − ML (°)', fontsize=8)
    res = d - (ic + sl * A)
    ax.text(0.04, 0.96, f'observed slope {sl:.2f}\nresidual SD {res.std(ddof=2):.1f}°',
            transform=ax.transAxes, fontsize=7.5, va='top', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5), zorder=5)

    # ---- row 4: MB-ML correlation across the cycle
    ax = axes[3, c]
    ax.axhline(1, color='k', lw=1.6)
    ax.plot(x, r_t, color=MBC, lw=2)
    ax.axvline(node, color='k', lw=0.8, ls=':')
    ax.set_ylim(-0.4, 1.05); ax.set_xlim(0, 100)
    ax.set_xlabel('Gait Cycle (%)', fontsize=8)
    if c == 0: ax.set_ylabel('r (MB, ML) across participants', fontsize=8)
    ax.text(0.04, 0.05, f'mean r = {r_t.mean():.2f}', transform=ax.transAxes, fontsize=7.5)
    ax.text(0.97, 0.97, 'scaled copy', transform=ax.transAxes, fontsize=7, ha='right', va='top')
    for r_ in range(4): axes[r_, c].tick_params(labelsize=7)

fig.text(0.5, 0.985, 'Why d depends on A, and why the extra MB spread is not a scaled-up copy of ML', ha='center',
         fontsize=11, fontweight='bold')
fig.text(0.5, 0.962, 'cov(d, A) = (var MB − var ML) / 2   →   d is independent of A only if the two systems are equally variable across participants',
         ha='center', fontsize=8.5)
fig.legend(handles=[Line2D([0], [0], color=MBC, label='MB curves (row 1); observed regression of d on A (row 3); MB–ML correlation (row 4)'),
                    Line2D([0], [0], color=MLC, label='ML (row 1)'),
                    Line2D([0], [0], color='k', lw=1.6, label='Scaled-copy prediction: ML = mean + (SD ML/SD MB)·(MB − mean); r = 1'),
                    Line2D([0], [0], color=GREY, ls='--', label='Identity (ML = MB)')],
           loc='lower center', ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig('mlmb_independence_explainer.jpg', dpi=300)
