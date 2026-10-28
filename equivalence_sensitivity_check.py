"""Python cross-check of ML_MB_EquivalenceAnalysis_Sensitivity.m (uses spm1d python port).
Usage: python3 equivalence_sensitivity_check.py  -> writes equivalence_sensitivity_summary.csv
"""
import csv
import numpy as np
import scipy.io as sio
import spm1d

m = sio.loadmat('stydy2data_20260226_withMDC.mat')
gm = m['s2data'][0, 0]['grandmeans'][0, 0]
plt_order_r = [11, 12, 13, 18, 19, 20, 21, 22, 17, 14, 15, 16]
names = ['Pelvis Anterior Tilt', 'Pelvis Obliquity', 'Pelvis Rotation',
         'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
         'Knee Flexion', 'Knee Varus', 'Foot Progression',
         'Ankle Dorsiflexion', 'Ankle Inversion', 'Ankle Rotation']
mdc = [float(np.squeeze(m['ci_data'][j, 1])) for j in range(12)]  # Foot progression already set to 5

rows = []
for j in range(12):
    k = plt_order_r[j] - 1
    ci = np.asarray(spm1d.stats.ci_pairedsample(
        gm['mb'][k, 0].T, gm['ml'][k, 0].T, 0.1, datum='difference', mu=0).ci)
    lo, hi = ci.min(0), ci.max(0)
    for label, b in (('MDC', mdc[j]), ('fixed2', 2.0), ('fixed5', 5.0)):
        eq = (lo >= -b) & (hi <= b)
        ne = (lo > b) | (hi < -b)
        rows.append([names[j], label, round(b, 3), *(round(100 * x.mean(), 2) for x in (eq, ne, ~eq & ~ne))])

with open('equivalence_sensitivity_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Variable', 'BoundSet', 'Bound_deg', 'PctEquivalent', 'PctNotEquivalent', 'PctInconclusive'])
    w.writerows(rows)
