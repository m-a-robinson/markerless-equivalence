#!/usr/bin/env python3
"""Convert the Study 2 MATLAB database into a plain NumPy archive.

Input
-----
``stydy2data_20260226_withMDC.mat`` (MATLAB v5 file in the repository root).  The parts used are

* ``s2data.subj{ii,1}.mb{jj,1}`` / ``.ml{jj,1}``  - trial-level gait-cycle curves, 101 nodes x trials
* ``s2data.grandmeans.mb{jj,1}``                  - the participant means that entered the published grand means
* ``gait_allsubj_sw_mdc``                         - Study 1 minimal detectable change (MDC) curves and means
* ``fullPaths``                                   - folder paths, the last element is the participant ID
* ``filenames``                                   - the 23 joint-angle file names (variable codes)

Output
------
``data/study2_gait_data.npz`` (loadable with ``numpy.load(..., allow_pickle=False)``):

====================== =========================== ==================================================
key                    shape / dtype               meaning
====================== =========================== ==================================================
mb_trials, ml_trials   (24, 23, 8, 101) float64   marker-based / markerless angle (deg); axes are
                                                   participant, variable, trial, gait-cycle node.
                                                   Missing trials are padded with NaN.
n_trials               (24, 23) int                number of stored trials per participant/variable
subjects               (24,) str                   participant IDs (P005 ... P029)
variable_files         (23,) str                   MATLAB file codes, e.g. ``RAX`` = right ankle X
in_published_grandmean (24, 23) bool               whether the participant's mean entered the grand
                                                   means saved in the .mat file (P024 is missing
                                                   for foot progression, ``RFP``)
analysed_codes         (12,) str                   the 12 variables analysed in the manuscript
analysed_index         (12,) int                   their 0-based position along the variable axis
mdc_bounds             (12,) float64               Study 1 mean MDC for the analysed variables
                                                   (foot progression defaults to 5 deg)
mdc_curves             (12, 101) float64           time-varying MDC for the analysed variables
                                                   (NaN for foot progression)
====================== =========================== ==================================================

Usage
-----
    python 01_convert_mat_to_npz.py [--mat PATH] [--out PATH]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

from common import ANALYSED, DATA_FILE, DEFAULT_MAT


def convert(mat_path: Path, out_path: Path) -> None:
    m = sio.loadmat(str(mat_path))

    # participant IDs: last element of the Windows folder path
    full_paths = m['fullPaths']
    subjects = np.array([os.path.basename(str(full_paths[i, 0][0]).replace('\\', '/'))
                         for i in range(full_paths.shape[0])])
    variable_files = np.array([str(x[0]).replace('.txt', '') for x in m['filenames'][0]])
    n_subj, n_var = len(subjects), len(variable_files)

    s2 = m['s2data'][0, 0]
    subj, grand = s2['subj'], s2['grandmeans'][0, 0]
    max_trials, n_nodes = 8, 101
    mb = np.full((n_subj, n_var, max_trials, n_nodes), np.nan)
    ml = np.full_like(mb, np.nan)
    n_trials = np.zeros((n_subj, n_var), dtype=int)
    in_grand = np.zeros((n_subj, n_var), dtype=bool)

    for ii in range(n_subj):
        for jj in range(n_var):
            a = subj[ii, 0]['mb'][0, 0][jj, 0]          # nodes x trials
            b = subj[ii, 0]['ml'][0, 0][jj, 0]
            nt = a.shape[1]
            mb[ii, jj, :nt, :] = a.T
            ml[ii, jj, :b.shape[1], :] = b.T
            n_trials[ii, jj] = nt
            # was this participant's mean part of the saved grand mean?
            pm = subj[ii, 0]['mb'][0, 0][jj, 1].ravel()
            g = grand['mb'][jj, 0]                       # nodes x participants included
            in_grand[ii, jj] = any(np.allclose(g[:, c], pm, equal_nan=True) for c in range(g.shape[1]))

    # Study 1 MDC (rows follow the Study 1 order; ANALYSED stores the mapping)
    mdc = m['gait_allsubj_sw_mdc']
    analysed_codes = np.array([a.code for a in ANALYSED])
    analysed_index = np.array([int(np.flatnonzero(variable_files == a.file)[0]) for a in ANALYSED])
    mdc_bounds = np.array([float(np.squeeze(mdc[a.mdc_row, 1])) if a.mdc_row is not None else 5.0
                           for a in ANALYSED])
    mdc_curves = np.array([np.squeeze(mdc[a.mdc_row, 0]) if a.mdc_row is not None
                           else np.full(n_nodes, np.nan) for a in ANALYSED])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mb_trials=mb, ml_trials=ml, n_trials=n_trials, subjects=subjects,
                        variable_files=variable_files, in_published_grandmean=in_grand,
                        analysed_codes=analysed_codes, analysed_index=analysed_index,
                        mdc_bounds=mdc_bounds, mdc_curves=mdc_curves)

    # ---- sanity checks against the saved grand means ----
    for a, jj in zip(ANALYSED, analysed_index):
        keep = in_grand[:, jj]
        part = np.nanmean(mb[keep, jj], axis=1)         # participant means over trials
        g = grand['mb'][jj, 0]
        assert part.shape[0] == g.shape[1] and np.allclose(part.T, g, equal_nan=True), a.code
    print(f'Wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)')
    print(f'  participants: {n_subj}; variables: {n_var}; analysed: {len(ANALYSED)}')
    short = [(subjects[i], variable_files[j], int(n_trials[i, j])) for i in range(n_subj)
             for j in range(n_var) if n_trials[i, j] != 8]
    print('  cells with fewer than 8 trials:', short)
    excl = [(subjects[i], variable_files[j]) for i in range(n_subj) for j in range(n_var)
            if not in_grand[i, j]]
    print('  participant/variable cells absent from the saved grand means:', excl)
    print('  grand-mean cross-check passed for all analysed variables')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mat', type=Path, default=DEFAULT_MAT, help='input .mat file')
    ap.add_argument('--out', type=Path, default=DATA_FILE, help='output .npz file')
    args = ap.parse_args()
    convert(args.mat, args.out)
