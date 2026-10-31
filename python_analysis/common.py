"""Shared helpers for the markerless (ML) vs marker-based (MB) gait analysis.

Contents
--------
* ``ANALYSED``               - the 12 right-side joint angles analysed in the manuscript, in plotting order
* ``load_dataset``           - read the NumPy archive written by ``01_convert_mat_to_npz.py``
* ``load_selection``         - read the outlier selection written by ``select_outliers.py``
* ``participant_means``      - participant mean curves after applying the outlier selection
* ``paired_ci``              - spm1d simultaneous (random-field-theory) CI of the paired mean difference
* ``spm_paired_ttest``       - spm1d paired t-test (statistical difference between systems)
* ``classify_equivalence``   - equivalent / not equivalent / inconclusive for each gait-cycle node
* ``functional_loa``         - pointwise functional limits of agreement (Roislien et al. 2012)
* ``classify_loa``           - LoA outcome at each node: acceptable / group-level only / systematic bias
* ``rft_prediction_band``    - simultaneous prediction band using the spm1d random-field-theory threshold

Conventions
-----------
* Curves are 101 nodes (0-100 % of the gait cycle).  ``d = MB - ML`` everywhere.
* Equivalence classes are coded ``EQUIVALENT = 1``, ``NOT_EQUIVALENT = 2``, ``INCONCLUSIVE = 3``.
"""
from __future__ import annotations

import csv
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MAT = HERE.parent / 'stydy2data_20260226_withMDC.mat'
DATA_FILE = HERE / 'data' / 'study2_gait_data.npz'
SELECTION_FILE = HERE / 'outlier_selection.csv'
RESULTS_DIR = HERE / 'results'
FIGURES_DIR = HERE / 'figures'

N_NODES = 101
EQUIVALENT, NOT_EQUIVALENT, INCONCLUSIVE = 1, 2, 3
LOA_ACCEPTABLE, LOA_GROUP_ONLY, LOA_BIAS = 1, 2, 3     # LoA outcome codes (green / amber / red)
CLASS_LABELS = {EQUIVALENT: 'Equivalent', NOT_EQUIVALENT: 'Not equivalent', INCONCLUSIVE: 'Inconclusive'}

# code      short name used in the outlier lists and in all result tables
# file      MATLAB file code (R = right side; PE = pelvis; A/H/K = ankle/hip/knee; X/Y/Z = sagittal/frontal/transverse,
#           RFP = right foot progression)
# mdc_row   0-based row of ``gait_allsubj_sw_mdc`` (Study 1 order); None = not examined in Study 1
Variable = namedtuple('Variable', 'code file title mdc_row')
ANALYSED = [
    Variable('PelTilt', 'PEX', 'Pelvis Anterior Tilt', 0),
    Variable('PelObl', 'PEY', 'Pelvis Obliquity', 1),
    Variable('PelRot', 'PEZ', 'Pelvis Rotation', 2),
    Variable('HipFlex', 'RHX', 'Hip Flexion', 6),
    Variable('HipAdd', 'RHY', 'Hip Adduction', 7),
    Variable('HipRot', 'RHZ', 'Hip Rotation', 8),
    Variable('KneeFlex', 'RKX', 'Knee Flexion', 9),
    Variable('KneeVar', 'RKY', 'Knee Varus', 10),
    Variable('FootProg', 'RFP', 'Foot Progression', None),
    Variable('AnkDF', 'RAX', 'Ankle Dorsiflexion', 3),
    Variable('AnkInv', 'RAY', 'Ankle Inversion', 4),
    Variable('AnkRot', 'RAZ', 'Ankle Rotation', 5),
]
VAR_BY_CODE = {v.code: v for v in ANALYSED}


# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------
@dataclass
class Dataset:
    """Trial-level data; axes of ``mb`` and ``ml`` are (participant, variable, trial, node)."""
    mb: np.ndarray
    ml: np.ndarray
    n_trials: np.ndarray
    subjects: list
    variable_files: list
    in_published_grandmean: np.ndarray
    analysed_index: np.ndarray
    mdc_bounds: np.ndarray
    mdc_curves: np.ndarray

    def var_index(self, code: str) -> int:
        """Position of an analysed variable along the variable axis of ``mb`` / ``ml``."""
        return int(self.analysed_index[[a.code for a in ANALYSED].index(code)])


def load_dataset(path: Path = DATA_FILE) -> Dataset:
    """Load the archive created by ``01_convert_mat_to_npz.py``."""
    z = np.load(path, allow_pickle=False)
    return Dataset(mb=z['mb_trials'], ml=z['ml_trials'], n_trials=z['n_trials'],
                   subjects=[str(s) for s in z['subjects']],
                   variable_files=[str(s) for s in z['variable_files']],
                   in_published_grandmean=z['in_published_grandmean'],
                   analysed_index=z['analysed_index'], mdc_bounds=z['mdc_bounds'], mdc_curves=z['mdc_curves'])


# --------------------------------------------------------------------------------------------------
# outlier selection
# --------------------------------------------------------------------------------------------------
@dataclass
class Selection:
    """Data to leave out.  Everything not listed is kept.  Each row of ``outlier_selection.csv`` has a ``scope``:

    ``scope = variable`` (one variable only; the earlier behaviour)
        ``strides``       MB strides removed from one participant's MB mean for one variable:
                          {(subject, trial 1-based, variable code)}
        ``participants``  participants removed from one variable, both systems: {(subject, variable code)}

    ``scope = all`` (every variable and both systems; the rule the variance-structure analysis requires)
        ``strides_all``       a stride is removed from every variable in MB *and* ML: {(subject, trial): variables}
        ``participants_all``  a participant is removed from every variable: {subject: variables}

    The dictionaries of the ``all`` scope remember which variables triggered the flag (used for logging only).
    """
    strides: set = field(default_factory=set)
    participants: set = field(default_factory=set)
    strides_all: dict = field(default_factory=dict)
    participants_all: dict = field(default_factory=dict)
    source: str = ''

    def only_scope_all(self) -> bool:
        """True if no exclusion is restricted to a single variable (required by the variance analysis)."""
        return not (self.strides or self.participants)

    def describe(self) -> str:
        parts = []
        if self.strides_all or self.participants_all:
            parts.append(f'scope all: {len(self.strides_all)} strides removed from every variable and both systems, '
                         f'{len(self.participants_all)} participants dropped entirely')
        if self.strides or self.participants:
            parts.append(f'scope variable: {len(self.strides)} MB strides, '
                         f'{len(self.participants)} participant/variable cells')
        return f'{self.source}  (' + ('; '.join(parts) or 'nothing excluded') + ')'


def load_selection(path: Path = SELECTION_FILE) -> Selection:
    """Read ``outlier_selection.csv`` (columns: subject, trial, variable, action, ..., scope).

    Rows with ``action == exclude`` are applied; ``trial == all`` excludes the participant.  ``scope`` (``variable``
    or ``all``) says whether the exclusion applies to that variable only or to every variable (see ``Selection``);
    files without a ``scope`` column are read as ``variable``.  If the file does not exist the published behaviour is
    used (only P024 foot progression is left out).
    """
    sel = Selection(source=str(path))
    if not Path(path).exists():
        sel.participants.add(('P024', 'FootProg'))
        sel.source = 'default (as published: P024 foot progression excluded)'
        return sel
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row['action'].strip().lower() != 'exclude':
                continue
            subj, var = row['subject'].strip(), row['variable'].strip()
            scope = (row.get('scope') or 'variable').strip().lower()
            if scope not in ('variable', 'all'):
                raise ValueError(f"scope must be 'variable' or 'all', got {scope!r}")
            if row['trial'].strip().lower() == 'all':
                if scope == 'all':
                    sel.participants_all.setdefault(subj, set()).add(var)
                else:
                    sel.participants.add((subj, var))
            else:
                t = int(row['trial'])
                if scope == 'all':
                    sel.strides_all.setdefault((subj, t), set()).add(var)
                else:
                    sel.strides.add((subj, t, var))
    return sel


def participant_means(ds: Dataset, code: str, sel: Selection):
    """Participant mean curves for one variable after applying the selection.

    Returns ``(mb, ml, subjects)`` with ``mb`` and ``ml`` of shape (n participants, 101).  Rules:

    * ``scope = variable`` strides are left out of the MB mean only (ML keeps all its strides);
    * ``scope = all`` strides are left out of both the MB and the ML mean, for every variable;
    * participants listed with either scope are dropped from both systems (``scope = all``: from every variable);
    * a participant left with no strides is dropped with a warning.
    """
    jj = ds.var_index(code)
    mb_out, ml_out, subs = [], [], []
    for ii, s in enumerate(ds.subjects):
        if s in sel.participants_all or (s, code) in sel.participants:
            continue
        nt = int(ds.n_trials[ii, jj])
        drop_all = {t for (ss, t) in sel.strides_all if ss == s}                  # every variable, both systems
        keep_ml = [t for t in range(nt) if (t + 1) not in drop_all]
        keep_mb = [t for t in keep_ml if (s, t + 1, code) not in sel.strides]
        if not keep_mb or not keep_ml:
            print(f'  warning: all strides of {s} {code} excluded - participant dropped')
            continue
        mb_out.append(np.nanmean(ds.mb[ii, jj, keep_mb], axis=0))
        ml_out.append(np.nanmean(ds.ml[ii, jj, keep_ml], axis=0))
        subs.append(s)
    return np.array(mb_out), np.array(ml_out), subs


# --------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------
def runs(mask):
    """Contiguous (start, end) node ranges (inclusive, 0-based = % gait cycle) where ``mask`` is True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(idx[np.r_[0, br + 1]], idx[np.r_[br, idx.size - 1]]))


def paired_ci(mb: np.ndarray, ml: np.ndarray, alpha: float = 0.1):
    """spm1d paired-sample CI of the mean difference MB - ML.

    ``alpha = 0.1`` gives the 90 % simultaneous (random-field-theory) CI that corresponds to two one-sided
    tests at 5 % (Dixon et al. 2018).  Returns ``(lower, upper, mean_difference)``, each of length 101.
    """
    import spm1d
    ci = spm1d.stats.ci_pairedsample(mb, ml, alpha, datum='difference', mu=0)
    c = np.asarray(ci.ci)
    return c.min(0), c.max(0), (mb - ml).mean(0)


def spm_paired_ttest(mb: np.ndarray, ml: np.ndarray, alpha: float = 0.05):
    """spm1d paired t-test (two-tailed, random-field-theory family-wise error control).

    Returns ``(significant_mask, t_statistic, critical_threshold)``.
    """
    import spm1d
    t = spm1d.stats.ttest_paired(mb, ml).inference(alpha, two_tailed=True, interp=True)
    z = np.asarray(t.z)
    return np.abs(z) > t.zstar, z, float(t.zstar)


def classify_equivalence(lo: np.ndarray, hi: np.ndarray, bound: float) -> np.ndarray:
    """Three-way outcome for each node from the CI and the equivalence bounds +/- ``bound``.

    EQUIVALENT      CI entirely inside  [-bound, bound]
    NOT_EQUIVALENT  CI entirely outside [-bound, bound] (beyond one bound)
    INCONCLUSIVE    CI crosses a bound
    """
    eq = (lo >= -bound) & (hi <= bound)
    ne = (lo > bound) | (hi < -bound)
    out = np.full(lo.shape, INCONCLUSIVE, dtype=int)
    out[eq], out[ne] = EQUIVALENT, NOT_EQUIVALENT
    return out


def functional_loa(d: np.ndarray, k: float = 1.96):
    """Pointwise functional limits of agreement (Roislien et al. 2012): mean(d) +/- k SD(d) at each node.

    ``d`` has shape (participants, 101).  Returns ``(mean, lower, upper)``.
    """
    md, sd = d.mean(0), d.std(0, ddof=1)
    return md, md - k * sd, md + k * sd


def rft_prediction_band(d: np.ndarray, alpha: float = 0.05):
    """Simultaneous 95 % prediction band for a new participant's whole difference curve.

    band(t) = mean(d) +/- z* SD(d) sqrt(1 + 1/n), where ``z*`` is the spm1d random-field-theory critical
    threshold (two-tailed, family-wise ``alpha``) of the one-sample t field of ``d``.  The band covers an
    entire new curve (not just each node) with probability ``1 - alpha`` if the differences are smooth and
    approximately Gaussian.  Returns ``(lower, upper, z*)``.
    """
    import spm1d
    n = d.shape[0]
    zstar = float(spm1d.stats.ttest(d).inference(alpha, two_tailed=True, interp=True).zstar)
    md, sd = d.mean(0), d.std(0, ddof=1)
    h = zstar * sd * np.sqrt(1 + 1 / n)
    return md - h, md + h, zstar


def classify_loa(md: np.ndarray, lo: np.ndarray, hi: np.ndarray, bound: float) -> np.ndarray:
    """Outcome of the pointwise LoA at each node, judged against the bounds +/- ``bound``.

    LOA_ACCEPTABLE   (1, green)  the LoA lie inside the bounds: individual differences are expected to be acceptable
    LOA_GROUP_ONLY   (2, amber)  the LoA exceed a bound but the mean difference is inside: agreement at group level only
    LOA_BIAS         (3, red)    the mean difference itself is beyond a bound: systematic bias

    ``md``, ``lo``, ``hi`` are the mean difference and the lower/upper LoA (e.g. from ``functional_loa``).  The classes
    are nested: LoA inside the bounds implies the mean is inside.
    """
    out = np.full(md.shape, LOA_GROUP_ONLY, dtype=int)
    out[(lo >= -bound) & (hi <= bound)] = LOA_ACCEPTABLE
    out[np.abs(md) > bound] = LOA_BIAS
    return out
