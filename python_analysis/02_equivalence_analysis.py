#!/usr/bin/env python3
"""Equivalence analysis of markerless (ML) vs marker-based (MB) gait kinematics with +/-5 degree bounds.

For each of the 12 variables:

1. participant mean curves are built from the strides that survive the outlier selection
   (``outlier_selection.csv``, see ``select_outliers.py``);
2. the 90 % simultaneous paired CI of the mean difference MB - ML is computed with spm1d
   (random-field theory; two one-sided tests at 5 %, Dixon et al. 2018);
3. every gait-cycle node is classified as

   * **equivalent**      - the CI lies entirely inside the bounds,
   * **not equivalent**  - the CI lies entirely beyond one bound,
   * **inconclusive**    - the CI crosses a bound;

4. an spm1d paired t-test (two-tailed, p < 0.05) marks where the systems differ statistically.

Outputs (in ``results/``)
-------------------------
``equivalence_<tag>.npz``       per-variable curves (CI limits, mean difference, class codes, SPM mask) + metadata,
                                read by ``04_plot_equivalence.py``
``equivalence_<tag>_summary.csv``  percentage of the gait cycle in each class and where SPM is significant
``table1_<tag>.csv`` / ``.docx``   manuscript Table 1 (``.docx`` only if python-docx is installed)
``equivalence_<tag>_selection.csv``  copy of the outlier selection that was used

Usage
-----
    python 02_equivalence_analysis.py [--bound 5] [--selection outlier_selection.csv] [--tag 5deg]
    python 02_equivalence_analysis.py --bound mdc      # per-variable Study 1 MDC bounds instead of a fixed bound
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from common import (ANALYSED, CLASS_LABELS, EQUIVALENT, INCONCLUSIVE, NOT_EQUIVALENT, RESULTS_DIR, SELECTION_FILE,
                    classify_equivalence, load_dataset, load_selection, paired_ci, participant_means, runs,
                    spm_paired_ttest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bound', default='5', help='equivalence bound in degrees (e.g. 5) or "mdc" (default 5)')
    ap.add_argument('--selection', type=Path, default=SELECTION_FILE, help='outlier selection file')
    ap.add_argument('--alpha-ci', type=float, default=0.1, help='alpha of the CI (0.1 -> 90 %% CI; default)')
    ap.add_argument('--alpha-spm', type=float, default=0.05, help='alpha of the SPM paired t-test')
    ap.add_argument('--tag', default=None, help='label used in output file names (default derived from the bound)')
    ap.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    use_mdc = args.bound.lower() == 'mdc'
    tag = args.tag or ('mdc' if use_mdc else f'{float(args.bound):g}deg')
    ds = load_dataset()
    sel = load_selection(args.selection)
    print(f'Outlier selection: {sel.source}  ({len(sel.strides)} strides, {len(sel.participants)} participants excluded)')

    arrays, meta, rows = {}, {'tag': tag, 'alpha_ci': args.alpha_ci, 'alpha_spm': args.alpha_spm,
                              'variables': [], 'titles': [], 'bounds': [], 'n': []}, []
    for k, var in enumerate(ANALYSED):
        bound = float(ds.mdc_bounds[k]) if use_mdc else float(args.bound)
        mb, ml, subs = participant_means(ds, var.code, sel)
        lo, hi, md = paired_ci(mb, ml, args.alpha_ci)
        cls = classify_equivalence(lo, hi, bound)
        sig, tstat, zstar = spm_paired_ttest(mb, ml, args.alpha_spm)

        for name, arr in (('lo', lo), ('hi', hi), ('md', md), ('cls', cls), ('spm_sig', sig), ('spm_t', tstat)):
            arrays[f'{var.code}_{name}'] = arr
        meta['variables'].append(var.code); meta['titles'].append(var.title)
        meta['bounds'].append(bound); meta['n'].append(len(subs))

        pct = {c: 100 * np.mean(cls == c) for c in (EQUIVALENT, NOT_EQUIVALENT, INCONCLUSIVE)}
        rows.append(dict(variable=var.code, title=var.title, n=len(subs), bound=bound,
                         pct_equivalent=pct[EQUIVALENT], pct_not_equivalent=pct[NOT_EQUIVALENT],
                         pct_inconclusive=pct[INCONCLUSIVE], pct_spm_significant=100 * sig.mean(),
                         fully_equivalent=bool(np.all(cls == EQUIVALENT)),
                         equivalent_intervals='; '.join(f'{a}-{b}' for a, b in runs(cls == EQUIVALENT)) or '-',
                         spm_threshold=zstar))
        print(f'{var.title:20s} n={len(subs):2d} bound={bound:5.2f}  E/NE/I = '
              f'{pct[EQUIVALENT]:3.0f}/{pct[NOT_EQUIVALENT]:3.0f}/{pct[INCONCLUSIVE]:3.0f} %   SPM sig. {100 * sig.mean():3.0f} %')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / f'equivalence_{tag}.npz', meta=json.dumps(meta), **arrays)
    if args.selection.exists():                      # keep a record of which data were excluded
        shutil.copy(args.selection, args.out_dir / f'equivalence_{tag}_selection.csv')

    with open(args.out_dir / f'equivalence_{tag}_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- Table 1 (intervals are gait-cycle %, node index 0-100) ----
    hdr = ['Joint variable', 'Bound ±(°)', 'n', 'Fully equivalent', '% Equivalent', '% Not equivalent',
           '% Inconclusive', '% SPM p<0.05', 'Equivalent gait cycle intervals (%)']
    body = [[r['title'], f"{r['bound']:.2f}" if use_mdc else f"{r['bound']:g}", r['n'],
             'Yes' if r['fully_equivalent'] else 'No', f"{r['pct_equivalent']:.0f}", f"{r['pct_not_equivalent']:.0f}",
             f"{r['pct_inconclusive']:.0f}", f"{r['pct_spm_significant']:.0f}",
             r['equivalent_intervals'].replace('-', '–')] for r in sorted(rows, key=lambda r: -r['pct_equivalent'])]
    with open(args.out_dir / f'table1_{tag}.csv', 'w', newline='') as f:
        csv.writer(f).writerows([hdr] + body)
    try:
        import docx
        d = docx.Document()
        t = d.add_table(rows=1, cols=len(hdr)); t.style = 'Table Grid'
        for c, h in zip(t.rows[0].cells, hdr):
            c.text = h
        for r in body:
            for c, v in zip(t.add_row().cells, r):
                c.text = str(v)
        d.save(args.out_dir / f'table1_{tag}.docx')
    except ImportError:
        print('python-docx not installed - Table 1 written as .csv only')
    print(f'Results written to {args.out_dir}')


if __name__ == '__main__':
    main()
