"""Visual check of the flagged marker-based strides.
One panel per flagged subject x variable.  Every MB stride of that subject is drawn; flagged strides are
coloured and numbered.  Reference lines: MB grand mean +/- 1 SD across participants
(s2data.grandmeans.mb{jj,2}, {jj,3}), the participant mean of all strides, the participant mean with the
flagged strides removed, and the participant's ML mean.
Location of a stride:  s2data.subj{ii,1}.mb{jj,1}(:,trial)
Usage: python3 plot_flagged_trials.py
"""
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

m = sio.loadmat('stydy2data_20260226_withMDC.mat')
fp = m['fullPaths']
ids = [os.path.basename(str(fp[i, 0][0]).replace('\\', '/')) for i in range(24)]
VAR = {'AnkDF': (14, 'Ankle Dorsiflexion'), 'AnkInv': (15, 'Ankle Inversion'), 'AnkRot': (16, 'Ankle Rotation'),
       'FootProg': (17, 'Foot Progression'), 'HipRot': (20, 'Hip Rotation'), 'KneeFlex': (21, 'Knee Flexion'),
       'KneeVar': (22, 'Knee Varus')}
flagged = """P011,4,AnkDF;AnkInv;AnkRot
P013,1,HipRot;KneeVar
P013,8,HipRot;KneeVar
P021,3,KneeFlex;FootProg;AnkDF;AnkInv;AnkRot
P024,1,FootProg;AnkDF;AnkRot
P024,2,FootProg;AnkRot
P024,3,FootProg;AnkRot
P024,4,AnkDF
P024,5,FootProg;AnkDF;AnkRot
P024,6,AnkDF;AnkRot
P024,8,AnkDF"""
panels = {}
for line in flagged.split('\n'):
    p, t, vs = line.split(',')
    for v in vs.split(';'):
        panels.setdefault((p, v), []).append(int(t))

s2 = m['s2data'][0, 0]
subj, gm = s2['subj'], s2['grandmeans'][0, 0]
x = np.arange(101)
FLAGCOL = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#1f77b4']

keys = sorted(panels, key=lambda k: (k[0], list(VAR).index(k[1])))
nr = int(np.ceil(len(keys) / 3))
fig, axes = plt.subplots(nr, 3, figsize=(11, 3.1 * nr + 1.0), squeeze=False)
fig.subplots_adjust(left=0.06, right=0.985, top=1 - 0.9 / (3.1 * nr + 1.0), bottom=0.03, hspace=0.38, wspace=0.18)
for ax, (p, v) in zip(axes.ravel(), keys):
    ii, (jj, vname) = ids.index(p), VAR[v]
    trials = subj[ii, 0]['mb'][0, 0][jj - 1, 0]                      # 101 x 8
    ml_mean = subj[ii, 0]['ml'][0, 0][jj - 1, 0].mean(1)
    gmean = gm['mb'][jj - 1, 1].ravel(); gsd = gm['mb'][jj - 1, 2].ravel()
    bad = [t - 1 for t in panels[(p, v)]]
    good = [t for t in range(trials.shape[1]) if t not in bad]
    ax.fill_between(x, gmean - gsd, gmean + gsd, color='0.85', lw=0, zorder=1)
    ax.plot(x, gmean, color='0.45', lw=1.6, zorder=2)
    for t in good:                                                   # unflagged strides
        ax.plot(x, trials[:, t], color='#4c72b0', lw=0.8, alpha=0.7, zorder=3)
        ax.text(100.8, trials[-1, t], str(t + 1), fontsize=6, color='#4c72b0', va='center')
    for c, t in zip(FLAGCOL, bad):                                   # flagged strides
        ax.plot(x, trials[:, t], color=c, lw=1.5, zorder=5)
        ax.text(100.8, trials[-1, t], str(t + 1), fontsize=7, color=c, va='center', fontweight='bold')
    ax.plot(x, trials.mean(1), color='k', lw=2, zorder=6)
    if good:
        ax.plot(x, trials[:, good].mean(1), color='#2ca02c', lw=2, ls='--', zorder=7)
    ax.plot(x, ml_mean, color='#d9473a', lw=1.3, ls=':', zorder=6)
    ax.set_xlim(0, 100); ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=7)
    ax.set_title(f'{p} (ii={ii + 1})  {vname} (jj={jj})\nflagged strides: {", ".join(str(t + 1) for t in bad)}  of {trials.shape[1]}',
                 fontsize=8.5, fontweight='bold')
    ax.set_ylabel('Angle (°)', fontsize=7.5)
for ax in axes.ravel()[len(keys):]:
    ax.axis('off')
for ax in axes[-1]:
    ax.set_xlabel('Gait Cycle (%)', fontsize=7.5)

fig.legend(handles=[Patch(color='0.85', label='MB grand mean ± 1 SD'),
                    Line2D([0], [0], color='0.45', lw=1.6, label='MB grand mean'),
                    Line2D([0], [0], color='#4c72b0', lw=1, label='Unflagged MB strides'),
                    Line2D([0], [0], color='#d62728', lw=1.5, label='Flagged MB strides (coloured, numbered)'),
                    Line2D([0], [0], color='k', lw=2, label='Participant mean, all strides'),
                    Line2D([0], [0], color='#2ca02c', lw=2, ls='--', label='Participant mean, flagged removed'),
                    Line2D([0], [0], color='#d9473a', lw=1.3, ls=':', label='Participant ML mean')],
           loc='upper center', ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.savefig('flagged_trials_visual_check.jpg', dpi=200)
fig.savefig('flagged_trials_visual_check.pdf')
print(len(keys), 'panels,', sum(len(v) for v in panels.values()), 'flagged strides')
