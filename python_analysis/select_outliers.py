#!/usr/bin/env python3
"""Choose which flagged (outlier) data are included in the analysis.

The choice is stored in ``outlier_selection.csv``; ``02_equivalence_analysis.py`` and ``03_loa_analysis.py`` read
that file, so run this script *before* them whenever you change your mind.

What can be excluded
--------------------
* a **stride** (``trial`` = 1-8): one MB stride is left out of that participant's MB mean for one variable
  (the ML mean always uses all ML strides);
* a **participant** (``trial`` = ``all``): the participant is left out of that variable for both systems,
  because the equivalence and LoA analyses are paired.

Catalogue
---------
The cells below are the 26 stride-level flags supplied for the study plus participant-level options for P024
(whose foot variables are corrupted in almost every stride, see ``../flagged_trials_visual_check.jpg``).
Any other cell can be added with ``--exclude SUBJECT:TRIAL:VARIABLE``.

Presets (``--preset``)
----------------------
manuscript     THE FINAL DECISION (default): every flagged stride below is excluded and P024 is excluded from
               ankle dorsiflexion, ankle rotation and foot progression - see "Manuscript selection" below
as_published   nothing excluded except P024 foot progression (reproduces the grand means saved in the .mat file)
include_all    everything included, even P024 foot progression
user_flagged   all 26 flagged strides + P024 foot progression participant (P024 ankle DF/rotation strides only)
recommended    P011 stride 4, P021 stride 3 and P024's ankle/foot variables excluded; P013 strides 1 and 8 kept
               (they look like a set-up change, not noise - check the lab notes before removing them)

Manuscript selection (record of the decision and the evidence, from ``../flagged_trials_visual_check.jpg``)
----------------------------------------------------------------------------------------------------------
Stride numbers are the column of ``s2data.subj{ii,1}.mb{jj,1}`` (101 x 8).  Peak deviations are from the cohort mean.

* P011 stride 4 (AnkDF, AnkInv, AnkRot): spike at ~70 % of the gait cycle (69-110 deg), ML unaffected.
* P013 strides 1 and 8 (HipRot, KneeVar): ~15 deg (hip rotation) / ~6 deg (knee varus) away from the participant's
  other six strides, which agree closely with each other.  Excluded as flagged; this may reflect a set-up change in the
  session, so the lab notes should be checked.
* P021 stride 3 (AnkDF, AnkInv, AnkRot, FootProg, KneeFlex): spikes of up to ~80 deg near 68 % of the cycle in the
  ankle variables and constant offsets in foot progression and knee flexion; the other 7 strides agree.
* P024 flagged strides (FootProg, AnkDF, AnkRot) and P024 excluded from AnkDF, AnkRot and FootProg for both systems:
  large spikes at 55-75 % of the cycle in nearly every stride (foot progression 8/8 strides, 60-119 deg from the
  cohort; ankle rotation 6/8; ankle dorsiflexion 5/8), so no reliable participant mean can be formed.

Result: 26 strides from four participants excluded, and n = 23 for ankle dorsiflexion, ankle rotation and foot
progression.  The reasons are written to the ``note`` column of ``outlier_selection.csv``.

Examples
--------
    python select_outliers.py --preset manuscript          # the final decision (also the default if no file exists)
    python select_outliers.py --list                       # show the current choice with deviation metrics
    python select_outliers.py --preset recommended
    python select_outliers.py --preset user_flagged --include P013:*:HipRot --include P013:*:KneeVar
    python select_outliers.py --exclude P024:all:AnkDF     # participant-level exclusion
    python select_outliers.py --interactive                # answer questions group by group

``SUBJECT:TRIAL:VARIABLE`` accepts ``*`` for the trial and the variable, e.g. ``P011:*:*``.
Variable codes: PelTilt PelObl PelRot HipFlex HipAdd HipRot KneeFlex KneeVar FootProg AnkDF AnkInv AnkRot.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from common import DATA_FILE, SELECTION_FILE, VAR_BY_CODE, load_dataset

FLAGGED = """P011,4,AnkDF;AnkInv;AnkRot
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
P024_PARTICIPANT_VARS = ('AnkDF', 'AnkRot', 'FootProg')   # participant-level options for P024


def build_catalogue():
    """List of (subject, trial, variable, source) - trial is an int or 'all'."""
    cat = []
    for line in FLAGGED.splitlines():
        s, t, vs = line.split(',')
        cat += [(s, int(t), v, 'flagged stride') for v in vs.split(';')]
    cat += [('P024', 'all', v, 'participant-level option') for v in P024_PARTICIPANT_VARS]
    return cat


CATALOGUE = build_catalogue()

# reason written to the ``note`` column for every excluded cell (see the docstring for the evidence)
RATIONALE = {
    'P011': 'stride 4: spike at ~70% of the cycle (69-110 deg from cohort); other strides agree, ML unaffected',
    'P013': 'strides 1 and 8: offset from the participant\'s other six strides (possible set-up change); '
            'excluded as flagged',
    'P021': 'stride 3: spikes up to ~80 deg near 68% of the cycle plus constant offsets; other strides agree',
    'P024': 'large spike at 55-75% of the cycle in nearly all strides of the ankle/foot variables; '
            'participant excluded for AnkDF, AnkRot and FootProg',
}


def preset(name):
    """Return {(subject, trial, variable): 'include' | 'exclude'} for a named preset."""
    sel = {(s, t, v): 'include' for s, t, v, _ in CATALOGUE}
    if name == 'manuscript':
        for s, t, v, src in CATALOGUE:                 # all 26 flagged strides + P024 participant-level options
            sel[(s, t, v)] = 'exclude'
    elif name == 'as_published':
        sel[('P024', 'all', 'FootProg')] = 'exclude'
    elif name == 'include_all':
        pass
    elif name == 'user_flagged':
        for s, t, v, src in CATALOGUE:
            if src == 'flagged stride':
                sel[(s, t, v)] = 'exclude'
        sel[('P024', 'all', 'FootProg')] = 'exclude'
    elif name == 'recommended':
        for s, t, v, src in CATALOGUE:
            if s in ('P011', 'P021') or (s == 'P024' and v in P024_PARTICIPANT_VARS):
                sel[(s, t, v)] = 'exclude'
    else:
        raise SystemExit(f'unknown preset {name!r}')
    return sel


def read_selection(path):
    sel = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            t = r['trial'].strip()
            sel[(r['subject'].strip(), 'all' if t.lower() == 'all' else int(t), r['variable'].strip())] = \
                r['action'].strip().lower()
    return sel


def write_selection(sel, path):
    src = {(s, t, v): k for s, t, v, k in CATALOGUE}
    rows = sorted(sel, key=lambda k: (k[0], list(VAR_BY_CODE).index(k[2]), 0 if k[1] == 'all' else k[1]))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'trial', 'variable', 'action', 'source', 'note'])
        for k in rows:
            w.writerow([k[0], k[1], k[2], sel[k], src.get(k, 'user added'),
                        RATIONALE.get(k[0], '') if sel[k] == 'exclude' and k in src else ''])
    n_x = sum(a == 'exclude' for a in sel.values())
    print(f'Wrote {path}  ({n_x} exclusions, {len(sel) - n_x} kept)')


def expand(token, subjects):
    """Expand ``SUBJECT:TRIAL:VARIABLE`` (with ``*`` wildcards) into concrete keys."""
    try:
        s, t, v = token.split(':')
    except ValueError:
        raise SystemExit(f'bad cell {token!r}; use SUBJECT:TRIAL:VARIABLE')
    subs = subjects if s == '*' else [s]
    vars_ = list(VAR_BY_CODE) if v == '*' else [v]
    for vv in vars_:
        if vv not in VAR_BY_CODE:
            raise SystemExit(f'unknown variable {vv!r}')
    keys = []
    for ss in subs:
        for vv in vars_:
            if t == '*':
                found = [k for k in CATALOGUE if k[0] == ss and k[2] == vv] or None
                if found is None:
                    raise SystemExit(f'{ss}/{vv} is not in the catalogue; give a trial number or "all"')
                keys += [(k[0], k[1], k[2]) for k in found]
            else:
                keys.append((ss, 'all' if t.lower() == 'all' else int(t), vv))
    return keys


def metrics(ds):
    """Deviation metrics (deg) for every catalogue cell: peak |stride - cohort mean| and RMS vs the median of
    the participant's other strides.  Large values (> ~15 deg peak) are obvious artefacts."""
    out = {}
    for s, t, v, _ in CATALOGUE:
        if t == 'all':
            continue
        ii, jj = ds.subjects.index(s), ds.var_index(v)
        others = [i for i in range(len(ds.subjects)) if i != ii and ds.in_published_grandmean[i, jj]]
        cohort = np.nanmean(np.array([np.nanmean(ds.mb[i, jj, :ds.n_trials[i, jj]], axis=0) for i in others]), 0)
        a = ds.mb[ii, jj, :ds.n_trials[ii, jj]]
        ref = np.nanmedian(np.delete(a, t - 1, 0), 0)
        out[(s, t, v)] = (np.abs(a[t - 1] - cohort).max(), np.sqrt(np.mean((a[t - 1] - ref) ** 2)))
    return out


def show(sel, ds=None):
    m = metrics(ds) if ds is not None else {}
    print(f"{'subject':8s}{'trial':>6s}{'variable':>10s}  {'action':8s}{'peak dev. from cohort (deg)':>29s}{'RMS vs own median (deg)':>26s}")
    for k in sorted(sel, key=lambda k: (k[0], list(VAR_BY_CODE).index(k[2]), 0 if k[1] == 'all' else k[1])):
        pk, rms = m.get(k, (None, None))
        print(f'{k[0]:8s}{k[1]!s:>6s}{k[2]:>10s}  {sel[k]:8s}' +
              (f'{pk:29.1f}{rms:26.1f}' if pk is not None else ''))
    n_x = sum(a == 'exclude' for a in sel.values())
    print(f'\n{n_x} exclusions, {len(sel) - n_x} kept')


def ask(prompt):
    """``input`` that returns None at end of input (so a scripted or interrupted session stops cleanly)."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return None


def interactive(sel):
    """Walk through each subject/variable group of the catalogue (Ctrl-D stops and keeps the answers so far)."""
    groups = sorted({(s, v) for s, t, v, _ in CATALOGUE}, key=lambda g: (g[0], list(VAR_BY_CODE).index(g[1])))
    print('For each participant/variable: [k]eep all, [e]xclude all strides, [p]articipant (drop for this variable),'
          ' [s]elect strides one by one, Enter = leave unchanged.')
    for s, v in groups:
        keys = [k for k in sel if k[0] == s and k[2] == v]
        strides = sorted(k[1] for k in keys if k[1] != 'all')
        status = ', '.join(f"{k[1]}:{sel[k][0]}" for k in sorted(keys, key=lambda k: (0 if k[1] == 'all' else k[1])))
        ans = ask(f'\n{s} {VAR_BY_CODE[v].title} ({v})  [{status}]\n  choice [k/e/p/s/Enter]: ')
        if ans is None:
            print('\n  input ended - keeping the answers given so far')
            break
        if ans == 'k':
            for k in keys:
                sel[k] = 'include'
        elif ans == 'e':
            for k in keys:
                sel[k] = 'exclude' if k[1] != 'all' else 'include'
        elif ans == 'p':
            sel[(s, 'all', v)] = 'exclude'                 # participant dropped for this variable
        elif ans == 's':
            for t in strides:
                a = ask(f'    stride {t}: exclude? [y/N]: ')
                sel[(s, t, v)] = 'exclude' if a == 'y' else 'include'
            if (s, 'all', v) in sel:
                sel[(s, 'all', v)] = 'include'
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', choices=['manuscript', 'as_published', 'include_all', 'user_flagged', 'recommended'],
                    help='start from a preset (replaces the current selection)')
    ap.add_argument('--include', action='append', default=[], metavar='S:T:V', help='keep this cell (repeatable)')
    ap.add_argument('--exclude', action='append', default=[], metavar='S:T:V', help='exclude this cell (repeatable)')
    ap.add_argument('--interactive', action='store_true', help='answer questions group by group')
    ap.add_argument('--list', action='store_true', help='print the selection (no changes are written)')
    ap.add_argument('--selection', type=Path, default=SELECTION_FILE, help='selection file to read/write')
    args = ap.parse_args()

    ds = load_dataset() if DATA_FILE.exists() else None
    subjects = ds.subjects if ds else sorted({c[0] for c in CATALOGUE})
    if args.preset:
        sel = preset(args.preset)
    elif args.selection.exists():
        sel = read_selection(args.selection)
        for s, t, v, _ in CATALOGUE:                       # add catalogue cells missing from an older file
            sel.setdefault((s, t, v), 'include')
    else:
        sel = preset('manuscript')

    if args.list and not (args.include or args.exclude or args.interactive or args.preset):
        show(sel, ds)
        return
    for tok in args.include:
        for k in expand(tok, subjects):
            sel[k] = 'include'
    for tok in args.exclude:
        for k in expand(tok, subjects):
            sel[k] = 'exclude'
    if args.interactive:
        sel = interactive(sel)
    write_selection(sel, args.selection)
    if args.list:
        show(sel, ds)


if __name__ == '__main__':
    main()
