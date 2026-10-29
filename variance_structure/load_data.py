"""Load Study 2 simultaneous ML/MB gait data into tidy NumPy arrays.

Arrays are shaped (subject, variable, time[101], trial[8]); a missing trial is NaN.
"""
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "stydy2data_20260226_withMDC.mat"

# Right-side variables in the plotting order used in the MATLAB scripts (plt_order_r, 1-based there)
RIGHT_IDX = [10, 11, 12, 17, 18, 19, 20, 21, 16, 13, 14, 15]
TITLES = ["Pelvis Anterior Tilt", "Pelvis Obliquity", "Pelvis Rotation",
          "Hip Flexion", "Hip Adduction", "Hip Rotation",
          "Knee Flexion", "Knee Varus", "Foot Progression",
          "Ankle Dorsiflexion", "Ankle Inversion", "Ankle Rotation"]
SHORT = ["PelTilt", "PelObl", "PelRot", "HipFlex", "HipAdd", "HipRot",
         "KneeFlex", "KneeVar", "FootProg", "AnkDF", "AnkInv", "AnkRot"]
PLANE = ["sag", "fro", "tra"] * 4


def load(right_only=True):
    m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    s = m["s2data"]
    nsub, nvar = len(s.subj), 23
    out = {}
    for sysname in ("mb", "ml"):
        arr = np.full((nsub, nvar, 101, 8), np.nan)
        for i in range(nsub):
            cells = getattr(s.subj[i], sysname)
            for j in range(nvar):
                x = np.asarray(cells[j, 0], float)
                arr[i, j, :, :x.shape[1]] = x
        out[sysname] = arr[:, RIGHT_IDX] if right_only else arr
    ids = [p.split("\\")[-1] for p in m["fullPaths"]]
    mdc = np.array([float(m["gait_allsubj_sw_mdc"][k, 1]) for k in [0, 1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5]])
    mdc[8] = np.nan  # foot progression not in Study 1 (MATLAB code reuses slot 9 = knee Z)
    mdc_curve = np.stack([np.asarray(m["gait_allsubj_sw_mdc"][k, 0], float) for k in [0, 1, 2, 6, 7, 8, 9, 10, 11, 3, 4, 5]])
    return out["mb"], out["ml"], ids, mdc, mdc_curve
