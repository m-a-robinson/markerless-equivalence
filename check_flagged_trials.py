"""Check flagged marker-based trials against the study 2 database.
Trial-level data: s2data.subj{ii,1}.mb{jj,1}(:, trial)   (== data.subj{ii,1}.rawdata{jj,5}),
101 nodes x 8 trials.  For every subject x system x variable x trial the statistic is the RMS
difference between that trial and the median of the subject's other trials; flags are judged against
the robust (median/MAD) distribution of that statistic over all subjects and trials for the variable.
"""
import os
import numpy as np
import scipy.io as sio

m = sio.loadmat('stydy2data_20260226_withMDC.mat')
fp = m['fullPaths']
ids = [os.path.basename(str(fp[i, 0][0]).replace('\\', '/')) for i in range(24)]
fn = [str(x[0]).replace('.txt', '') for x in m['filenames'][0]]            # 1-based variable index = position
VAR = {'AnkDF': 14, 'AnkInv': 15, 'AnkRot': 16, 'FootProg': 17, 'HipRot': 20, 'KneeVar': 22, 'KneeFlex': 21}
subj = m['s2data'][0, 0]['subj']

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
flags = {(p, int(t), v) for l in flagged.split('\n') for p, t, vs in [l.split(',')] for v in vs.split(';')}


def stat(sys_):
    """dict (subject idx, jj) -> (rmsd per trial, max abs dev per trial)"""
    out = {}
    for ii in range(24):
        for jj in range(1, 24):
            a = subj[ii, 0][sys_][0, 0][jj - 1, 0]                       # 101 x 8
            nt = a.shape[1]
            r, mx = np.full(nt, np.nan), np.full(nt, np.nan)
            for t in range(nt):
                others = np.nanmedian(np.delete(a, t, 1), 1)
                dv = a[:, t] - others
                r[t], mx[t] = np.sqrt(np.nanmean(dv ** 2)), np.nanmax(np.abs(dv))
            out[(ii, jj)] = (r, mx)
    return out

S = {s: stat(s) for s in ('mb', 'ml')}


def robust(sys_, jj):
    allr = np.concatenate([S[sys_][(ii, jj)][0] for ii in range(24)]); allr = allr[~np.isnan(allr)]
    med = np.median(allr); mad = 1.4826 * np.median(np.abs(allr - med))
    return med, mad, allr

print('LOCATION: s2data.subj{ii,1}.mb{jj,1}(:,trial)  [= data.subj{ii,1}.rawdata{jj,5}(:,trial)], 101 x 8')
print('trials per subject (MB, AnkDF): ' + ', '.join(f'{ids[i]}={subj[i,0]["mb"][0,0][13,0].shape[1]}' for i in range(24) if subj[i,0]["mb"][0,0][13,0].shape[1] != 8) + ' (all others 8)')
print('subject index ii: ' + ', '.join(f'{p}={ids.index(p) + 1}' for p in ('P011', 'P013', 'P021', 'P024')))
print('variable index jj: ' + ', '.join(f'{k}={v}({fn[v - 1]})' for k, v in VAR.items()) + '\n')
print(f"{'subj':5s}{'ii':>3s}{'trial':>6s}{'var':>9s}{'jj':>4s}{'RMSD°':>7s}{'maxDev°':>8s}{'robustZ':>8s}{'pctile':>7s}  {'ML robustZ':>10s}  verdict")
for p, t, v in sorted(flags, key=lambda k: (k[0], k[1], k[2])):
    ii, jj = ids.index(p), VAR[v]
    r, mx = S['mb'][(ii, jj)]
    if t > len(r):
        print(f'{p} trial {t} {v}: subject has only {len(r)} trials in database'); continue
    med, mad, allr = robust('mb', jj)
    z = (r[t - 1] - med) / mad
    pc = 100 * np.mean(allr <= r[t - 1])
    mlr = S['ml'][(ii, jj)][0][t - 1]; mmed, mmad, _ = robust('ml', jj)
    verdict = 'CONFIRMED outlier' if z > 3.5 else ('elevated' if z > 2 else 'NOT unusual')
    print(f'{p:5s}{ii + 1:3d}{t:6d}{v:>9s}{jj:4d}{r[t - 1]:7.1f}{mx[t - 1]:8.1f}{z:8.1f}{pc:7.0f}  {(mlr - mmed) / mmad:10.1f}  {verdict}')

# Outliers in the whole database that were NOT flagged (MB and ML), robust z > 4
print('\nUnflagged trials with robust z > 4 (any system), right-side variables 14-23:')
rows = []
for sys_ in ('mb', 'ml'):
    for jj in range(14, 24):
        med, mad, _ = robust(sys_, jj)
        for ii in range(24):
            r = S[sys_][(ii, jj)][0]
            for t in range(len(r)):
                z = (r[t] - med) / mad
                name = {v: k for k, v in VAR.items()}.get(jj, fn[jj - 1])
                if z > 4 and not (sys_ == 'mb' and (ids[ii], t + 1, name) in flags):
                    rows.append((z, ids[ii], ii + 1, t + 1, sys_.upper(), name, jj, r[t]))
for z, p, ii, t, s_, v, jj, r in sorted(rows, reverse=True)[:25]:
    print(f'  {p} (ii={ii}) trial {t} {s_} {v} (jj={jj}) RMSD {r:.1f}°  robustZ {z:.1f}')
print(f'  ... {len(rows)} total')

# P024 FootProg: how many strides are flagged
n_fp = sum(1 for (p, t, v) in flags if p == 'P024' and v == 'FootProg')
print(f'\nP024 FootProg flagged in {n_fp}/8 strides (trials ' + ','.join(str(t) for (p, t, v) in sorted(flags) if p == 'P024' and v == 'FootProg') + ')')

# ---------------- Impact of removing the flagged strides ----------------
import spm1d
BOUND = 5.0
BASE_DROP = {'FootProg': ('P024',)}      # saved grandmeans already exclude P024 for foot progression (n = 23)

def curves(jj, v, drop_cells=()):
    mb, ml = [], []
    for ii in range(24):
        if ids[ii] in BASE_DROP.get(v, ()):
            continue
        a = subj[ii, 0]['mb'][0, 0][jj - 1, 0]
        bad = [t - 1 for (p, t, vv) in drop_cells if p == ids[ii] and VAR[vv] == jj]
        good = [t for t in range(a.shape[1]) if t not in bad]
        mb.append(a[:, good].mean(1)); ml.append(subj[ii, 0]['ml'][0, 0][jj - 1, 0].mean(1))
    return np.array(mb), np.array(ml)

def summarise(mb, ml):
    ci = np.asarray(spm1d.stats.ci_pairedsample(mb, ml, 0.1, datum='difference', mu=0).ci)
    lo, hi = ci.min(0), ci.max(0)
    e = (lo >= -BOUND) & (hi <= BOUND); ne = (lo > BOUND) | (hi < -BOUND)
    d = mb - ml
    return (100 * e.mean(), 100 * ne.mean(), 100 * (1 - e.mean() - ne.mean()),
            mb.std(0, ddof=1).mean(), 2 * 1.96 * d.std(0, ddof=1).mean(), np.abs(d.mean(0)).max())

print('\nEffect of removing flagged strides (participant means rebuilt from remaining strides)')
print(f"{'variable':9s}{'':>7s}{'%E/%NE/%I (±5)':>18s}{'MB between-subj SD':>20s}{'FLoA width':>12s}{'max|mean diff|':>16s}")
for v, jj in VAR.items():
    for lab, cells in (('before', ()), ('after', flags)):
        r = summarise(*curves(jj, v, cells))
        print(f'{v:9s}{lab:>7s}{r[0]:6.0f}/{r[1]:3.0f}/{r[2]:3.0f}{r[3]:18.2f}{r[4]:14.1f}{r[5]:14.1f}')

# ---------------- CSV of flagged locations ----------------
import csv
with open('flagged_trials_check.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['subject', 'subj_index_ii', 'trial', 'variable', 'var_index_jj', 'matlab_location',
                'rmsd_vs_median_of_other_trials_deg', 'max_abs_dev_deg', 'robust_z_MB', 'robust_z_ML_same_trial'])
    for p, t, v in sorted(flags):
        ii, jj = ids.index(p), VAR[v]
        r, mx = S['mb'][(ii, jj)]; med, mad, _ = robust('mb', jj); mmed, mmad, _ = robust('ml', jj)
        w.writerow([p, ii + 1, t, v, jj, f's2data.subj{{{ii + 1},1}}.mb{{{jj},1}}(:,{t})',
                    round(r[t - 1], 2), round(mx[t - 1], 1), round((r[t - 1] - med) / mad, 1),
                    round((S['ml'][(ii, jj)][0][t - 1] - mmed) / mmad, 1)])
