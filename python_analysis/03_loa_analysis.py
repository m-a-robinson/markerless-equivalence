#!/usr/bin/env python3
"""Functional limits of agreement (LoA) between markerless (ML) and marker-based (MB) gait kinematics.

Two bands are computed for each of the 12 variables from the participant differences ``d(t) = MB(t) - ML(t)``:

**Pointwise functional LoA** (Roislien, Rennie & Skaaret 2012, Gait & Posture 36:495-499)
    ``mean d(t) +/- 1.96 SD d(t)`` at every node.  At each node about 95 % of participants' differences are
    expected inside, but the probability that a participant's *whole curve* stays inside is much lower.

**Simultaneous prediction band** (random-field theory)
    ``mean d(t) +/- z* SD d(t) sqrt(1 + 1/n)`` where ``z*`` is the spm1d random-field-theory critical threshold
    (two-tailed, family-wise alpha = 0.05) of the one-sample t field of ``d``.  A new participant's entire
    difference curve is expected inside this band with 95 % probability.  This is a prediction band (it allows
    for estimation of the mean) and is wider than the pointwise LoA; it is *not* the Bland-Altman LoA
    itself.  It assumes smooth, approximately Gaussian differences.

Both are compared with the +/-5 degree bounds.  Optionally (``--coverage``) a leave-one-participant-out check
reports how often a held-out participant's curve lies inside each band built from the others.

Participant means use the strides retained by the outlier selection (``outlier_selection.csv``).

Outputs (in ``results/``): ``loa_<tag>.npz`` (curves, read by ``05_plot_loa.py``) and ``loa_<tag>_summary.csv``.

Usage
-----
    python 03_loa_analysis.py [--bound 5] [--selection outlier_selection.csv] [--coverage]
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from common import (ANALYSED, RESULTS_DIR, SELECTION_FILE, functional_loa, load_dataset, load_selection,
                    participant_means, rft_prediction_band)


def loo_coverage(d):
    """Leave-one-participant-out coverage of whole curves and of single nodes for both bands (fractions)."""
    n = d.shape[0]
    res = np.zeros((2, 2))
    for i in range(n):
        rest, held = np.delete(d, i, 0), d[i]
        _, lo, hi = functional_loa(rest)
        ins = (held >= lo) & (held <= hi)
        res[0] += [ins.all(), ins.mean()]
        lo2, hi2, _ = rft_prediction_band(rest)
        ins = (held >= lo2) & (held <= hi2)
        res[1] += [ins.all(), ins.mean()]
    return res / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bound', type=float, default=5.0, help='bound in degrees compared with the bands (default 5)')
    ap.add_argument('--selection', type=Path, default=SELECTION_FILE, help='outlier selection file')
    ap.add_argument('--alpha', type=float, default=0.05, help='family-wise alpha of the simultaneous band')
    ap.add_argument('--coverage', action='store_true', help='add leave-one-out coverage check (slower)')
    ap.add_argument('--tag', default=None, help='label used in output file names (default derived from the bound)')
    ap.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    tag = args.tag or f'{args.bound:g}deg'
    ds = load_dataset()
    sel = load_selection(args.selection)
    print(f'Outlier selection: {sel.describe()}')

    arrays = {}
    meta = {'tag': tag, 'bound': args.bound, 'alpha': args.alpha, 'variables': [], 'titles': [], 'n': []}
    rows = []
    for var in ANALYSED:
        mb, ml, subs = participant_means(ds, var.code, sel)
        d = mb - ml
        md, lo, hi = functional_loa(d)
        slo, shi, zstar = rft_prediction_band(d, args.alpha)
        for name, arr in (('d', d), ('md', md), ('lo', lo), ('hi', hi), ('slo', slo), ('shi', shi)):
            arrays[f'{var.code}_{name}'] = arr
        meta['variables'].append(var.code); meta['titles'].append(var.title); meta['n'].append(len(subs))

        row = dict(variable=var.code, title=var.title, n=len(subs),
                   pointwise_width_mean=float(np.mean(hi - lo)), pointwise_width_max=float(np.max(hi - lo)),
                   pct_pointwise_within=100 * float(np.mean((lo >= -args.bound) & (hi <= args.bound))),
                   rft_zstar=zstar, simultaneous_width_mean=float(np.mean(shi - slo)),
                   pct_simultaneous_within=100 * float(np.mean((slo >= -args.bound) & (shi <= args.bound))),
                   max_abs_mean_difference=float(np.abs(md).max()))
        msg = (f'{var.title:20s} n={len(subs):2d}  pointwise width {row["pointwise_width_mean"]:5.1f} deg '
               f'({row["pct_pointwise_within"]:3.0f} % within ±{args.bound:g})  simultaneous width '
               f'{row["simultaneous_width_mean"]:5.1f} deg (z*={zstar:.2f}, {row["pct_simultaneous_within"]:3.0f} % within)')
        if args.coverage:
            c = loo_coverage(d)
            row.update(loo_whole_curve_pointwise=100 * c[0, 0], loo_node_pointwise=100 * c[0, 1],
                       loo_whole_curve_simultaneous=100 * c[1, 0], loo_node_simultaneous=100 * c[1, 1])
            msg += f'  LOO whole-curve coverage {100 * c[0, 0]:3.0f} % / {100 * c[1, 0]:3.0f} %'
        rows.append(row)
        print(msg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / f'loa_{tag}.npz', meta=json.dumps(meta), **arrays)
    if args.selection.exists():
        shutil.copy(args.selection, args.out_dir / f'loa_{tag}_selection.csv')
    with open(args.out_dir / f'loa_{tag}_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'Results written to {args.out_dir}')


if __name__ == '__main__':
    main()
