"""Study 3 (WU vs UOB, markerless) raw-data quality check.

Usage (from this folder):  python3 run_qc.py
Writes
    results/qc_issues.csv          one row per issue (condition, subject, side, joint, file, stride, check, ...)
    results/stride_counts.csv      strides per condition x subject x file (raw and after removing exact duplicates)
    results/stride_metrics.csv     every stride: RMSD from the participant median curve + robust z
    figures/subjects_<COND>.pdf    one page per participant: all strides L (red) / R (blue), flagged strides numbered
    figures/subjects_WU_vs_UOB.pdf one page per participant in both conditions: WU vs UOB mean +/- SD, L and R
    figures/overview_<COND>.jpg    every participant mean (L and R) on the group band; outlying participants labelled
    figures/group_means_WU_vs_UOB.jpg  group mean +/- SD, WU vs UOB (matched participants), L and R
    figures/flagged_strides.pdf    close-up of every variable with a 'major' stride flag
    results/stride_recommendations.csv  every flagged stride with the variables it is flagged in + recommendation

Checks (thresholds in CFG)
    model         variable without signal (flat / zero) for some participants but not others -> different
                  model / degrees of freedom used in processing
    structure     file layout, V3D mean/SD columns vs strides, L/R/pelvis stride counts, low stride count,
                  missing participants
    duplicate     strides that are exact copies of another stride (same participant, or across conditions)
    stride        stride outlier: RMSD from the participant median curve, robust z (median/MAD) within the
                  participant x file; also implausible frame-to-frame jumps (gimbal / wrap)
    participant   participant mean far from the group (robust z across participants); within-participant
                  variability far above the group
    LR            L-R difference of the participant means far above that of other participants
    identity      WU participant mean does not best match the same participant in UOB (label check)
    WU_UOB        WU-UOB difference of the participant means far above that of other participants
"""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import s3common as c  # noqa: E402

CFG = dict(
    stride_z=4.0, stride_min_rmsd=3.0, stride_major_rmsd=8.0,                        # stride outliers
    flat_ptp=0.1,                                                                      # 'no signal' (deg)
    jump_deg_per_pct=10.0,                                                             # frame-to-frame jump
    subj_z=3.5, subj_min_rmsd=5.0,                                                     # participant mean vs group
    var_z=3.5, var_min=3.0,                                                            # within-participant variability
    lr_z=3.5, lr_min=5.0,                                                              # L-R difference
    cond_z=3.5, cond_min=5.0,                                                          # WU-UOB difference
    min_strides=10,
)
X = np.arange(c.N)
SEV_ORDER = {'major': 0, 'minor': 1, 'info': 2}


def robust_z(v, floor):
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    return (v - med) / max(mad, floor), med


def rmsd(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def files_for(joint):
    return [joint + 'V'] if joint.startswith('PE') else ['L' + joint + 'V', 'R' + joint + 'V']


# ------------------------------------------------------------------------------------------------
def main():
    c.FIG_DIR.mkdir(exist_ok=True)
    c.RES_DIR.mkdir(exist_ok=True)
    cells, info = c.load()
    files = info['files']
    subjects = info['subjects']
    issues = []

    def add(cond, subj, cell_or_file, stride, check, severity, metric, value, desc):
        if isinstance(cell_or_file, c.Cell):
            side, joint, f, title = cell_or_file.side, cell_or_file.joint, cell_or_file.file, cell_or_file.title
        elif cell_or_file in (None, ''):
            side = joint = f = title = ''
        else:
            f = cell_or_file
            side, joint = c._parse(f + '.txt')
            title = c.JOINTS[joint] + ('' if side == '-' else f' ({side})')
        issues.append(dict(condition=cond, subject=subj, side=side, joint=title, file=f, stride=stride, check=check,
                           severity=severity, metric=metric,
                           value='' if value is None else (f'{value:.2f}' if isinstance(value, float) else value),
                           description=desc))

    # ---------------- structure ----------------------------------------------------------------
    all_ids = sorted(set(subjects['UOB']) | set(subjects['WU']) | {'P020'})
    for cond in c.CONDS:
        if info['n_listed'][cond] != len(subjects[cond]):
            add(cond, '', '', '', 'structure', 'major', 'participants imported', len(subjects[cond]),
                f'{cond}: {cond.lower()}_fullPaths lists {info["n_listed"][cond]} participants but only '
                f'{len(subjects[cond])} were imported ({subjects[cond][0]}-{subjects[cond][-1]}); missing: '
                + ', '.join(s for s in all_ids if s not in subjects[cond] and s != 'P020'))
        add(cond, 'P020', '', '', 'structure', 'info', '', None,
            f'P020 absent from {cond} (noted in import script as missing from UOB; also absent from WU)')
        add(cond, '', 'PEXV', '', 'structure', 'info', '', None,
            'Pelvis files (PEX/PEY/PEZ) are 102 rows: row 1 is all zeros and col 1 is the SD (the V3D mean column '
            'was removed by the import). Other files: col 1 = mean, col 2 = SD. Handled in s3common.load(); the '
            'import script should drop the first 2 columns (and row 1 for pelvis) before analysis.')

    # V3D column notes, stride counts, duplicates within participant
    dup_sets = {}          # (cond, subj, file) -> set of 0-based stride indices that are copies of an earlier stride
    count_rows = []
    for (cond, subj, f), cell in cells.items():
        for n in cell.notes:
            if n.startswith('pelvis file'):
                continue
            add(cond, subj, cell, '', 'structure', 'minor', 'V3D column check', None,
                n + ' -> the stride columns do not reproduce the exported V3D summary; check the export '
                    '(stride missing from or extra in the pelvis export)')
        st = cell.strides
        dups = set()
        groups = defaultdict(list)
        for t in range(st.shape[1]):
            groups[np.round(st[:, t], 4).tobytes()].append(t)
        for g in groups.values():
            if len(g) > 1:
                dups.update(g[1:])
        dup_sets[(cond, subj, f)] = dups
        count_rows.append(dict(condition=cond, subject=subj, file=f, n_strides=st.shape[1],
                               n_exact_duplicates=len(dups), n_unique=st.shape[1] - len(dups)))
    # summarise duplicates once per participant x side (they are the same strides in every file of a side)
    for cond in c.CONDS:
        for subj in subjects[cond]:
            for side, fl in (('L', [f for f in files if f[0] == 'L']), ('R', [f for f in files if f[0] == 'R']),
                             ('-', [f for f in files if f.startswith('PE')])):
                per_file = {f: dup_sets[(cond, subj, f)] for f in fl}
                union = sorted(set().union(*per_file.values()))
                if not union:
                    continue
                cell = cells[(cond, subj, fl[0])]
                groups = defaultdict(list)
                for t in range(cell.strides.shape[1]):
                    groups[np.round(cell.strides[:, t], 4).tobytes()].append(t + 1)
                pairs = ['=' .join(map(str, g)) for g in groups.values() if len(g) > 1]
                allf = all(per_file[f] == per_file[fl[0]] for f in fl)
                sname = {'L': 'left', 'R': 'right', '-': 'pelvis'}[side]
                add(cond, subj, '', ';'.join(str(t + 1) for t in union), 'duplicate',
                    'major' if len(union) >= 3 else 'minor', 'duplicate strides', len(union),
                    f'{sname} files ({"all" if allf else "some"} {len(fl)}): identical strides {", ".join(pairs)} '
                    f'(of {cell.strides.shape[1]}) - the same stride was exported more than once (duplicate gait events '
                    f'or the same trial loaded twice). Copies are excluded from the stride statistics below.')
                issues[-1]['side'] = side
                issues[-1]['joint'] = 'all ' + sname + ' joints'

    # stride counts: L vs R vs pelvis, and low n
    for cond in c.CONDS:
        for subj in subjects[cond]:
            nL, nR, nP = (cells[(cond, subj, f)].strides.shape[1] for f in ('LKXV', 'RKXV', 'PEXV'))
            if nP != nR:
                add(cond, subj, 'PEXV', '', 'structure', 'minor', 'pelvis vs right strides', f'{nP} vs {nR}',
                    f'pelvis has {nP} strides but right side has {nR} (pelvis normally follows the right-side events); '
                    'pelvis strides cannot be matched to a side/stride number - check events in the affected trials')
            uL = nL - len(dup_sets[(cond, subj, 'LKXV')])
            uR = nR - len(dup_sets[(cond, subj, 'RKXV')])
            if min(uL, uR) < CFG['min_strides']:
                add(cond, subj, '', '', 'structure', 'minor', 'unique strides L/R', f'{uL}/{uR}',
                    f'only {uL} left / {uR} right unique strides (< {CFG["min_strides"]})')

    # cross-condition / cross-participant identical strides
    seen = defaultdict(list)
    for (cond, subj, f), cell in cells.items():
        for t in range(cell.strides.shape[1]):
            seen[(f, np.round(cell.strides[:, t], 3).tobytes())].append((cond, subj, t + 1))
    cross = defaultdict(set)
    for (f, _), v in seen.items():
        if len({(a, b) for a, b, _ in v}) > 1:
            cross[tuple(sorted(set(v)))].add(f)
    cross_by_pair = defaultdict(list)
    for key, fs in cross.items():
        cross_by_pair[tuple(sorted({(a, b) for a, b, _ in key}))].append((key, fs))
    for pair, lst in cross_by_pair.items():
        desc = []
        for key, fs in lst:
            desc.append(' = '.join(f'{a} {b} stride {t}' for a, b, t in key) + f' ({len(fs)} files)')
        (c1, s1), (c2, s2) = pair
        add(f'{c1}+{c2}', s1 if s1 == s2 else f'{s1}/{s2}', '', '', 'duplicate', 'major',
            'identical strides across conditions/participants', len(lst),
            '; '.join(desc) + ' - the same trial appears in both data sets (a trial copied into the wrong folder). '
            'Remove it from the condition it does not belong to.')
        issues[-1]['joint'] = 'all joints'

    # ---------------- model consistency: flat (zero) signals ---------------------------------------
    flat_vars = set()                  # files with no signal in most participants (skipped by later checks)
    for f in files:
        state = {}
        for cond in c.CONDS:
            for subj in subjects[cond]:
                ptp = np.ptp(cells[(cond, subj, f)].strides, axis=0)
                state[(cond, subj)] = ptp < CFG['flat_ptp']
        nflat = sum(v.all() for v in state.values())
        tot = len(state)
        if nflat > tot / 2:
            flat_vars.add(f)
        if nflat == tot:
            add('WU+UOB', 'all', f, '', 'model', 'info', 'participants with flat signal', f'{nflat}/{tot}',
                'no signal in any participant (constrained degree of freedom); exclude this variable')
        elif nflat > tot / 2:
            live = sorted({f'{k[0]} {k[1]}' for k, v in state.items() if not v.all()})
            for k, v in state.items():
                if not v.all():
                    ptp = np.ptp(cells[k + (f,)].strides, axis=0)
                    add(k[0], k[1], cells[k + (f,)], '', 'model', 'major', 'median stride ROM (deg)',
                        float(np.median(ptp)),
                        f'{nflat}/{tot} participant-condition sets have no signal (flat at 0) for this variable, '
                        f'but {", ".join(live)} do (ROM {ptp.min():.1f}-{ptp.max():.1f} deg). This participant was '
                        'processed with a different model / joint degrees of freedom - check the processing of '
                        'ALL its variables and re-process with the same model as the others')
        elif nflat:
            for k, v in state.items():
                if v.all():
                    add(k[0], k[1], cells[k + (f,)], '', 'model', 'major', 'stride ROM', '<0.1',
                        'flat (no signal) for this participant only: missing/failed variable in the export')
        for k, v in state.items():
            if v.any() and not v.all():
                for t in np.where(v)[0]:
                    add(k[0], k[1], cells[k + (f,)], int(t) + 1, 'stride', 'major', 'stride ROM', '<0.1',
                        'flat stride (no signal) - missing data in this stride')

    # ---------------- stride outliers ----------------------------------------------------------
    flags = defaultdict(dict)          # (cond, subj, file) -> {stride0: severity}
    metric_rows = []
    for (cond, subj, f), cell in cells.items():
        st = cell.strides
        keep = [t for t in range(st.shape[1]) if t not in dup_sets[(cond, subj, f)]]
        med = np.median(st[:, keep], 1)
        r = np.array([rmsd(st[:, t], med) for t in range(st.shape[1])])
        z, rmed = robust_z(r[keep], 0.25)
        zfull = (r - rmed) / max(1.4826 * np.median(np.abs(r[keep] - rmed)), 0.25)
        jumps = np.abs(np.diff(st, axis=0)).max(0)
        for t in range(st.shape[1]):
            metric_rows.append(dict(condition=cond, subject=subj, file=f, stride=t + 1,
                                    duplicate=int(t in dup_sets[(cond, subj, f)]), rmsd_from_median=round(r[t], 3),
                                    robust_z=round(zfull[t], 2), max_jump=round(jumps[t], 2)))
            if t in dup_sets[(cond, subj, f)]:
                continue
            if zfull[t] > CFG['stride_z'] and r[t] > max(CFG['stride_min_rmsd'], 2 * rmed):
                sev = 'major' if r[t] > CFG['stride_major_rmsd'] else 'minor'
                flags[(cond, subj, f)][t] = sev
                dev = st[:, t] - med
                k = int(np.argmax(np.abs(dev)))
                off = float(dev.mean())
                kind = 'offset' if abs(off) > 0.7 * r[t] else 'shape/peak'
                add(cond, subj, cell, t + 1, 'stride', sev, 'RMSD from participant median (deg) | z', float(r[t]),
                    f'z={zfull[t]:.1f}; {kind}: mean offset {off:+.1f} deg, max dev {dev[k]:+.1f} deg at {k}% '
                    f'(participant median RMSD {rmed:.1f})')
            if jumps[t] > CFG['jump_deg_per_pct']:
                add(cond, subj, cell, t + 1, 'stride', 'major', 'max jump deg/% cycle', float(jumps[t]),
                    'implausible frame-to-frame jump (gimbal lock / angle wrap / event error)')

    # cross-stride consistency: same stride flagged in many files of one side
    for cond in c.CONDS:
        for subj in subjects[cond]:
            for side in 'LR':
                cnt = defaultdict(list)
                for f in files:
                    if f[0] == side:
                        for t in flags[(cond, subj, f)]:
                            cnt[t].append(c.JOINTS[c._parse(f + '.txt')[1]])
                for t, js in cnt.items():
                    if len(js) >= 3:
                        add(cond, subj, '', t + 1, 'stride', 'major', 'files flagged for this stride', len(js),
                            f'{side} stride {t + 1} is an outlier in {len(js)} variables ({", ".join(js)}) - '
                            'likely a tracking/event error for the whole stride; consider removing it from every '
                            'variable')
                        issues[-1]['side'] = side
                        issues[-1]['joint'] = 'multiple'

    # per-stride summary across variables -> recommendation
    rec_rows = []
    for cond in c.CONDS:
        for subj in subjects[cond]:
            for side, lab in (('L', 'left'), ('R', 'right'), ('P', 'pelvis')):
                fl = [f for f in files if (f.startswith('PE') if side == 'P' else f[0] == side)]
                per = defaultdict(list)
                for f in fl:
                    st = cells[(cond, subj, f)].strides
                    med = np.median(np.delete(st, sorted(dup_sets[(cond, subj, f)]), axis=1), 1)
                    for t, sev in flags[(cond, subj, f)].items():
                        per[t].append((cells[(cond, subj, f)].title, sev, rmsd(st[:, t], med)))
                for t, lst in sorted(per.items()):
                    nmaj = sum(sev == 'major' for _, sev, _ in lst)
                    worst = max(r for _, _, r in lst)
                    action = ('REMOVE stride from all variables' if (len(lst) >= 3 or worst > 15)
                              else 'remove from flagged variable(s) / inspect' if nmaj else 'inspect')
                    rec_rows.append(dict(condition=cond, subject=subj, side=lab, stride=t + 1,
                                         n_strides=cells[(cond, subj, fl[0])].strides.shape[1],
                                         n_variables_flagged=len(lst), n_major=nmaj, worst_rmsd=round(worst, 1),
                                         variables='; '.join(f'{v} ({r:.1f})' for v, _, r in lst),
                                         recommendation=action))

    # ---------------- participant-level --------------------------------------------------------
    means = {k: np.delete(v.strides, sorted(dup_sets[k]), axis=1).mean(1) for k, v in cells.items()}
    sds = {k: np.delete(v.strides, sorted(dup_sets[k]), axis=1).std(1, ddof=1) for k, v in cells.items()}
    subj_flags = defaultdict(set)      # (cond, file) -> subjects
    for cond in c.CONDS:
        S = subjects[cond]
        for f in [f for f in files if f not in flat_vars]:
            M = np.array([means[(cond, s, f)] for s in S])
            gmed = np.median(M, 0)
            r = np.array([rmsd(m, gmed) for m in M])
            z, rmed = robust_z(r, 0.5)
            for i, s in enumerate(S):
                if z[i] > CFG['subj_z'] and r[i] > CFG['subj_min_rmsd']:
                    off = float((M[i] - gmed).mean())
                    cc = float(np.corrcoef(M[i], gmed)[0, 1]) if np.ptp(gmed) > 1 else np.nan
                    kind = 'offset' if abs(off) > 0.7 * r[i] else 'shape'
                    subj_flags[(cond, f)].add(s)
                    add(cond, s, cells[(cond, s, f)], '', 'participant', 'minor',
                        'RMSD of participant mean from group median (deg)', float(r[i]),
                        f'z={z[i]:.1f}; {kind}: mean offset {off:+.1f} deg, r with group shape {cc:.2f} '
                        f'(group median RMSD {rmed:.1f}). Check model/static calibration for this participant; '
                        'may be genuine.')
            v = np.array([np.mean(sds[(cond, s, f)]) for s in S])
            zv, vmed = robust_z(v, 0.25)
            for i, s in enumerate(S):
                if zv[i] > CFG['var_z'] and v[i] > CFG['var_min']:
                    add(cond, s, cells[(cond, s, f)], '', 'participant', 'minor',
                        'mean within-participant SD (deg)', float(v[i]),
                        f'z={zv[i]:.1f}; stride-to-stride variability well above other participants '
                        f'(median {vmed:.1f}) - inspect strides for mixed/bad trials')

    # ---------------- L vs R --------------------------------------------------------------------
    joints_lr = [j for j in c.JOINTS if not j.startswith('PE')]
    for cond in c.CONDS:
        S = subjects[cond]
        for j in [j for j in joints_lr if 'L' + j + 'V' not in flat_vars]:
            d = np.array([rmsd(means[(cond, s, 'L' + j + 'V')], means[(cond, s, 'R' + j + 'V')]) for s in S])
            z, dmed = robust_z(d, 0.5)
            for i, s in enumerate(S):
                if z[i] > CFG['lr_z'] and d[i] > CFG['lr_min']:
                    diff = means[(cond, s, 'L' + j + 'V')] - means[(cond, s, 'R' + j + 'V')]
                    add(cond, s, '', '', 'LR', 'minor', 'RMSD L-R mean (deg)', float(d[i]),
                        f'z={z[i]:.1f}; L-R mean offset {diff.mean():+.1f} deg (group median L-R RMSD {dmed:.1f}). '
                        'Large asymmetry for this participant only: check segment/model fit on one side.')
                    issues[-1]['side'] = 'L vs R'
                    issues[-1]['joint'] = c.JOINTS[j]
                    issues[-1]['file'] = f'L{j}V/R{j}V'

    # ---------------- WU vs UOB -----------------------------------------------------------------
    common = [s for s in subjects['WU'] if s in subjects['UOB']]
    # identity check: is each WU participant closest to the same participant in UOB? (all variables)
    live = [f for f in files if f not in flat_vars]
    U = np.array([np.concatenate([means[('UOB', s, f)] for f in live]) for s in subjects['UOB']])
    for s in common:
        v = np.concatenate([means[('WU', s, f)] for f in live])
        d = np.sqrt(((U - v) ** 2).mean(1))
        order = list(np.argsort(d))
        rank = order.index(subjects['UOB'].index(s)) + 1
        if rank > 3:
            near = ', '.join(f'{subjects["UOB"][k]} ({d[k]:.1f})' for k in order[:3])
            add('WU vs UOB', s, '', '', 'identity', 'minor', 'rank of same ID among UOB participants', rank,
                f'WU {s} is only the #{rank} closest match to UOB {s} (RMSD {d[subjects["UOB"].index(s)]:.1f} deg '
                f'over all variables; closest: {near}). Check the participant labels and the quality of both '
                'data sets for this participant.')
            issues[-1]['joint'] = 'all joints'
    for f in live:
        d = np.array([rmsd(means[('WU', s, f)], means[('UOB', s, f)]) for s in common])
        z, dmed = robust_z(d, 0.5)
        for i, s in enumerate(common):
            if z[i] > CFG['cond_z'] and d[i] > CFG['cond_min']:
                diff = means[('WU', s, f)] - means[('UOB', s, f)]
                add('WU vs UOB', s, cells[('WU', s, f)], '', 'WU_UOB', 'minor', 'RMSD WU-UOB mean (deg)', float(d[i]),
                    f'z={z[i]:.1f}; WU-UOB mean offset {diff.mean():+.1f} deg (median across participants '
                    f'{dmed:.1f}). Participant differs between conditions much more than others.')

    # ---------------- write tables -------------------------------------------------------------
    issues.sort(key=lambda r: (SEV_ORDER[r['severity']], r['subject'], r['condition'], r['check'], r['file'],
                               str(r['stride'])))
    with open(c.RES_DIR / 'qc_issues.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(issues[0]))
        w.writeheader()
        w.writerows(issues)
    for name, rows in (('stride_counts.csv', count_rows), ('stride_metrics.csv', metric_rows),
                       ('stride_recommendations.csv', rec_rows)):
        with open(c.RES_DIR / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---------------- figures ------------------------------------------------------------------
    group = {}
    for cond in c.CONDS:
        for j in c.JOINTS:
            M = np.array([means[(cond, s, f)] for s in subjects[cond] for f in files_for(j)])
            group[(cond, j)] = (M.mean(0), M.std(0, ddof=1))
    plot_subject_pages(cells, subjects, flags, dup_sets, group, means)
    plot_paired_pages(common, means, sds, group, issues)
    for cond in c.CONDS:
        plot_overview(cond, subjects[cond], means, group, subj_flags)
    plot_group_conditions(common, means)
    plot_flagged(cells, flags, dup_sets, means)

    # summary to stdout
    by = defaultdict(int)
    for r in issues:
        by[(r['severity'], r['check'])] += 1
    for k in sorted(by, key=lambda k: (SEV_ORDER[k[0]], k[1])):
        print(f'{k[0]:6s} {k[1]:12s} {by[k]}')
    print(len(issues), 'issues ->', c.RES_DIR / 'qc_issues.csv')


# ------------------------------------------------------------------------------------------------
def _style(ax, row_last, col_first):
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=6.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='0.3', lw=0.5, zorder=0)
    if col_first:
        ax.set_ylabel('Angle (°)', fontsize=7)
    if row_last:
        ax.set_xlabel('Gait Cycle (%)', fontsize=7)


def _grid(title, figsize=(8.27, 11.69)):
    fig, axes = plt.subplots(5, 3, figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.905, bottom=0.05, hspace=0.5, wspace=0.28)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.985)
    return fig, axes


def plot_subject_pages(cells, subjects, flags, dup_sets, group, means):
    FL = {'L': ['#b2182b', '#ef8a62', '#67001f', '#f4a582'], 'R': ['#2166ac', '#67a9cf', '#053061', '#4393c3']}
    for cond in c.CONDS:
        with PdfPages(c.FIG_DIR / f'subjects_{cond}.pdf') as pdf:
            for subj in subjects[cond]:
                nL, nR, nP = (cells[(cond, subj, f)].strides.shape[1] for f in ('LKXV', 'RKXV', 'PEXV'))
                fig, axes = _grid(f'Study 3  {cond}  {subj}  -  all strides   (L n={nL}, R n={nR}, pelvis n={nP})')
                fig.legend(handles=[Patch(color='0.85', label=f'{cond} group mean ± 1 SD (L+R)'),
                                    Line2D([0], [0], color=c.SIDE_COL['L'], lw=0.7, alpha=.6, label='Left strides'),
                                    Line2D([0], [0], color=c.SIDE_COL['R'], lw=0.7, alpha=.6, label='Right strides'),
                                    Line2D([0], [0], color='0.25', lw=0.7, alpha=.6, label='Pelvis strides'),
                                    Line2D([0], [0], color='k', lw=1.6, label='Participant mean (L solid / R dashed)'),
                                    Line2D([0], [0], color='#b2182b', lw=1.6, label='Flagged stride (numbered)'),
                                    Line2D([0], [0], color='0.5', lw=1, ls=':', label='Exact duplicate stride')],
                           loc='upper center', ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.965))
                for r, row in enumerate(c.GRID):
                    for col, j in enumerate(row):
                        ax = axes[r, col]
                        if j is None:
                            ax.axis('off')
                            continue
                        gm, gs = group[(cond, j)]
                        ax.fill_between(X, gm - gs, gm + gs, color='0.87', lw=0, zorder=1)
                        for f in files_for(j):
                            cell = cells[(cond, subj, f)]
                            side = cell.side
                            col_ = '0.25' if side == '-' else c.SIDE_COL[side]
                            fl = flags[(cond, subj, f)]
                            for t in range(cell.strides.shape[1]):
                                if t in fl:
                                    continue
                                ls = ':' if t in dup_sets[(cond, subj, f)] else '-'
                                ax.plot(X, cell.strides[:, t], color=col_, lw=0.55, alpha=0.45, ls=ls, zorder=2)
                            ax.plot(X, means[(cond, subj, f)], color='k', lw=1.4, ls='--' if side == 'R' else '-',
                                    zorder=5)
                            pal = FL['R' if side in 'R-' else 'L']
                            for i, (t, sev) in enumerate(sorted(fl.items())):
                                cc = pal[i % len(pal)]
                                ax.plot(X, cell.strides[:, t], color=cc, lw=1.5 if sev == 'major' else 1.1, zorder=6)
                                ax.text(101, cell.strides[-1, t], f'{side if side != "-" else ""}{t + 1}', fontsize=5.5,
                                        color=cc, va='center', fontweight='bold', clip_on=False)
                        nflag = sum(len(flags[(cond, subj, f)]) for f in files_for(j))
                        ax.set_title(c.JOINTS[j] + (f'   [{nflag} flagged]' if nflag else ''), fontsize=8,
                                     fontweight='bold', color='#b2182b' if nflag else 'k')
                        _style(ax, r == 4 or (r == 3 and col > 0), col == 0)
                pdf.savefig(fig)
                plt.close(fig)


def plot_paired_pages(common, means, sds, group, issues):
    with PdfPages(c.FIG_DIR / 'subjects_WU_vs_UOB.pdf') as pdf:
        for subj in common:
            fig, axes = _grid(f'Study 3  {subj}  -  WU vs UOB participant mean ± 1 SD   (L solid, R dashed)')
            fig.legend(handles=[Line2D([0], [0], color=c.COND_COL['WU'], lw=1.6, label='WU'),
                                Line2D([0], [0], color=c.COND_COL['UOB'], lw=1.6, label='UOB'),
                                Line2D([0], [0], color='0.3', lw=1.2, label='Left / pelvis'),
                                Line2D([0], [0], color='0.3', lw=1.2, ls='--', label='Right')],
                       loc='upper center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.965))
            flagged = {r['file'] for r in issues if r['check'] == 'WU_UOB' and r['subject'] == subj}
            for r, row in enumerate(c.GRID):
                for col, j in enumerate(row):
                    ax = axes[r, col]
                    if j is None:
                        ax.axis('off')
                        continue
                    for cond in c.CONDS:
                        for f in files_for(j):
                            m, s = means[(cond, subj, f)], sds[(cond, subj, f)]
                            ls = '--' if f[0] == 'R' else '-'
                            ax.fill_between(X, m - s, m + s, color=c.COND_COL[cond], alpha=0.12, lw=0)
                            ax.plot(X, m, color=c.COND_COL[cond], lw=1.3, ls=ls)
                    hit = [f for f in files_for(j) if f in flagged]
                    ax.set_title(c.JOINTS[j] + (f'   [WU≠UOB: {",".join(h[0] for h in hit)}]' if hit else ''),
                                 fontsize=8, fontweight='bold', color='#b2182b' if hit else 'k')
                    _style(ax, r == 4 or (r == 3 and col > 0), col == 0)
            pdf.savefig(fig)
            plt.close(fig)


def plot_overview(cond, S, means, group, subj_flags):
    fig, axes = _grid(f'Study 3  {cond}  -  every participant mean (n={len(S)}); outlying participants labelled',
                      figsize=(8.27, 11.0))
    fig.legend(handles=[Patch(color='0.85', label='Group mean ± 1 SD (L+R)'),
                        Line2D([0], [0], color=c.SIDE_COL['L'], lw=0.8, label='Left participant means'),
                        Line2D([0], [0], color=c.SIDE_COL['R'], lw=0.8, label='Right participant means'),
                        Line2D([0], [0], color='0.25', lw=0.8, label='Pelvis participant means'),
                        Line2D([0], [0], color='k', lw=2.0, label='Flagged participant (bold, labelled)')],
               loc='upper center', ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.965))
    for r, row in enumerate(c.GRID):
        for col, j in enumerate(row):
            ax = axes[r, col]
            if j is None:
                ax.axis('off')
                continue
            gm, gs = group[(cond, j)]
            ax.fill_between(X, gm - gs, gm + gs, color='0.87', lw=0, zorder=1)
            for f in files_for(j):
                side = c._parse(f + '.txt')[0]
                colr = '0.25' if side == '-' else c.SIDE_COL[side]
                for s in S:
                    bad = s in subj_flags[(cond, f)]
                    ax.plot(X, means[(cond, s, f)], color=colr, lw=1.8 if bad else 0.6, alpha=1 if bad else 0.5,
                            zorder=4 if bad else 2)
                    if bad:
                        ax.text(101, means[(cond, s, f)][-1], f'{s}{"" if side == "-" else side}', fontsize=5.5,
                                color=colr, va='center', fontweight='bold', clip_on=False)
            ax.plot(X, gm, color='k', lw=1.2, zorder=3)
            ax.set_title(c.JOINTS[j], fontsize=8, fontweight='bold')
            _style(ax, r == 4 or (r == 3 and col > 0), col == 0)
    fig.savefig(c.FIG_DIR / f'overview_{cond}.jpg', dpi=200)
    plt.close(fig)


def plot_group_conditions(common, means):
    fig, axes = plt.subplots(5, 6, figsize=(15, 11.5))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.05, hspace=0.5, wspace=0.3)
    fig.suptitle(f'Study 3  -  WU vs UOB group mean ± 1 SD  (matched participants, n={len(common)})',
                 fontsize=12, fontweight='bold', y=0.985)
    fig.legend(handles=[Line2D([0], [0], color=c.COND_COL['WU'], lw=2.5, label='WU'),
                        Line2D([0], [0], color=c.COND_COL['UOB'], lw=2.5, label='UOB')],
               loc='upper center', ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
    for side_i, side in enumerate(['L', 'R']):
        fig.text(0.27 + side_i * 0.49, 0.905, 'LEFT (pelvis repeated)' if side == 'L' else 'RIGHT (pelvis repeated)', ha='center', fontsize=11,
                 fontweight='bold', color=c.SIDE_COL[side])
        for r, row in enumerate(c.GRID):
            for col, j in enumerate(row):
                ax = axes[r, side_i * 3 + col]
                if j is None:
                    ax.axis('off')
                    continue
                f = j + 'V' if j.startswith('PE') else side + j + 'V'
                for cond in c.CONDS:
                    M = np.array([means[(cond, s, f)] for s in common])
                    m, sd = M.mean(0), M.std(0, ddof=1)
                    ax.fill_between(X, m - sd, m + sd, color=c.COND_COL[cond], alpha=0.2, lw=0)
                    ax.plot(X, m, color=c.COND_COL[cond], lw=2)
                ax.set_title(c.JOINTS[j], fontsize=8, fontweight='bold')
                _style(ax, r == 4 or (r == 3 and col > 0), col == 0)
    fig.savefig(c.FIG_DIR / 'group_means_WU_vs_UOB.jpg', dpi=200)
    plt.close(fig)


def plot_flagged(cells, flags, dup_sets, means):
    keys = [k for k, v in flags.items() if any(s == 'major' for s in v.values())]
    keys.sort(key=lambda k: (k[1], k[0], k[2]))
    if not keys:
        return
    with PdfPages(c.FIG_DIR / 'flagged_strides.pdf') as pdf:
        for i0 in range(0, len(keys), 20):
            _flag_page(pdf, keys[i0:i0 + 20], cells, flags, dup_sets, means, i0 // 20 + 1,
                       int(np.ceil(len(keys) / 20)))


def _flag_page(pdf, keys, cells, flags, dup_sets, means, page, npage):
    fig, axes = plt.subplots(5, 4, figsize=(11.69, 8.27 * 1.4), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.03, hspace=0.6, wspace=0.22)
    fig.suptitle(f'Study 3 - variables with a major stride flag (page {page}/{npage})', fontsize=11,
                 fontweight='bold', y=0.99)
    COL = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
    for ax, (cond, subj, f) in zip(axes.ravel(), keys):
        cell = cells[(cond, subj, f)]
        fl = flags[(cond, subj, f)]
        good = [t for t in range(cell.strides.shape[1]) if t not in fl and t not in dup_sets[(cond, subj, f)]]
        for t in good:
            ax.plot(X, cell.strides[:, t], color='#4c72b0', lw=0.6, alpha=0.6)
        for i, t in enumerate(sorted(fl)):
            ax.plot(X, cell.strides[:, t], color=COL[i % 6], lw=1.5 if fl[t] == 'major' else 1.0)
            ax.text(101, cell.strides[-1, t], str(t + 1), fontsize=6.5, color=COL[i % 6], fontweight='bold',
                    va='center', clip_on=False)
        ax.plot(X, means[(cond, subj, f)], color='k', lw=1.8)
        ax.plot(X, cell.strides[:, good].mean(1), color='#2ca02c', lw=1.6, ls='--')
        ax.set_title(f'{subj} {cond}  {cell.title}\nflagged stride(s) {", ".join(str(t + 1) for t in sorted(fl))}'
                     f'  (n={cell.strides.shape[1]})', fontsize=7.5, fontweight='bold')
        _style(ax, False, False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    fig.legend(handles=[Line2D([0], [0], color='#4c72b0', lw=1, label='Unflagged strides'),
                        Line2D([0], [0], color='#d62728', lw=1.5, label='Flagged strides (numbered)'),
                        Line2D([0], [0], color='k', lw=1.8, label='Participant mean (duplicates removed)'),
                        Line2D([0], [0], color='#2ca02c', lw=1.6, ls='--', label='Mean with flagged strides removed')],
               loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == '__main__':
    main()
