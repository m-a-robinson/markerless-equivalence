"""Loader for the Study 3 raw markerless gait curves (``20261113_WU_UOB_Import.mat``).

Layout of every ``<cond>_data.subj{ii}.rawdata{kk}`` cell (Visual3D export after ``Import_and_Organise_study3.m``
has dropped the first text column):

* non-pelvis files (101 x N):  col 1 = V3D mean, col 2 = V3D SD (n-1), cols 3..N = strides
* pelvis files PEX/PEY/PEZ (102 x N): row 1 = all zeros (a header row read as data), col 1 = V3D SD,
  cols 2..N = strides; the mean column is missing (it was the column removed by the import script)

``load()`` strips those columns/rows and checks that the reported mean/SD reproduce the strides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
MAT = HERE.parent / '20261113_WU_UOB_Import.mat'
FIG_DIR = HERE / 'figures'
RES_DIR = HERE / 'results'
CONDS = ['WU', 'UOB']
COND_COL = {'WU': '#1f5fbf', 'UOB': '#d9473a'}
SIDE_COL = {'L': '#d9473a', 'R': '#1f5fbf'}

# joint code -> (title, row of the 4x3 panel grid, column) ; side is added from the file name
JOINTS = {'PEX': 'Pelvis Anterior Tilt', 'PEY': 'Pelvis Obliquity', 'PEZ': 'Pelvis Rotation',
          'HX': 'Hip Flexion', 'HY': 'Hip Adduction', 'HZ': 'Hip Rotation',
          'KX': 'Knee Flexion', 'KY': 'Knee Varus', 'KZ': 'Knee Rotation',
          'AX': 'Ankle Dorsiflexion', 'AY': 'Ankle Inversion', 'AZ': 'Ankle Rotation',
          'FP': 'Foot Progression'}
# plotting order (5 rows x 3 cols); pelvis has no side
GRID = [['PEX', 'PEY', 'PEZ'], ['HX', 'HY', 'HZ'], ['KX', 'KY', 'KZ'], ['AX', 'AY', 'AZ'], ['FP', None, None]]
N = 101


@dataclass
class Cell:
    cond: str
    subject: str
    side: str            # 'L', 'R' or '-' (pelvis)
    joint: str           # key of JOINTS
    file: str            # e.g. LKXV
    strides: np.ndarray  # 101 x n
    v3d_mean: np.ndarray | None
    v3d_sd: np.ndarray
    notes: list = field(default_factory=list)

    @property
    def title(self):
        return JOINTS[self.joint] + ('' if self.side == '-' else f' ({self.side})')


def _parse(name):
    code = name.replace('V.txt', '')
    if code.startswith('PE'):
        return '-', code
    return code[0], code[1:]


def subject_ids(m, cond):
    fp = m[cond.lower() + '_fullPaths']
    return [str(fp[i, 0][0]).replace('\\', '/').rstrip('/').split('/')[-2] for i in range(fp.shape[0])]


def load(path: Path = MAT):
    """Return ``(cells, info)``; ``cells[(cond, subject, file)] -> Cell``."""
    m = sio.loadmat(path)
    files = [str(x[0]) for x in m['filenames'][0]]
    cells, info = {}, {'files': [f.replace('.txt', '') for f in files], 'subjects': {}, 'n_listed': {}}
    for cond in CONDS:
        ids = subject_ids(m, cond)
        subj = m[cond.lower() + '_data'][0, 0]['subj']
        n = subj.shape[0]
        info['n_listed'][cond] = len(ids)
        info['subjects'][cond] = ids[:n]          # cells are filled in fullPaths order; see report for the check
        for ii in range(n):
            raw = subj[ii, 0]['rawdata'][0, 0]
            for kk, fname in enumerate(files):
                a = np.asarray(raw[kk, 0], float)
                side, joint = _parse(fname)
                f = fname.replace('.txt', '')
                notes = []
                if a.shape[0] == N + 1:                      # pelvis layout
                    if not np.allclose(a[0], 0):
                        notes.append('first row of 102-row file is not all zero')
                    a = a[1:]
                    mean, sd, st = None, a[:, 0], a[:, 1:]
                    notes.append('pelvis file: extra zero row and no V3D mean column (import offset)')
                else:
                    mean, sd, st = a[:, 0], a[:, 1], a[:, 2:]
                    if np.abs(mean - st.mean(1)).max() > 1e-3:
                        notes.append(f'V3D mean column does not match strides (max diff {np.abs(mean - st.mean(1)).max():.2f})')
                if st.shape[1] > 1 and np.abs(sd - st.std(1, ddof=1)).max() > 1e-3:
                    notes.append(f'V3D SD column does not match strides (max diff {np.abs(sd - st.std(1, ddof=1)).max():.2f})')
                cells[(cond, ids[ii], f)] = Cell(cond, ids[ii], side, joint, f, st, mean, sd, notes)
    return cells, info
