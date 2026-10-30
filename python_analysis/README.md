# Python analysis: markerless (ML) vs marker-based (MB) gait kinematics

Python re-implementation of the equivalence and limits-of-agreement (LoA) analyses, built from the MATLAB database
`../stydy2data_20260226_withMDC.mat`.  Everything is plain NumPy/SciPy/Matplotlib plus
[spm1d](https://spm1d.org) (random-field-theory inference).

```
pip install -r requirements.txt
```

## Workflow

```
python 01_convert_mat_to_npz.py        # .mat -> data/study2_gait_data.npz       (once)
python select_outliers.py --list       # choose which flagged data to keep       (any time)
python 02_equivalence_analysis.py      # equivalence tests, +/-5 deg bounds
python 03_loa_analysis.py              # pointwise + simultaneous LoA bands
python 04_plot_equivalence.py          # figures/equivalence_5deg.jpg
python 05_plot_loa.py                  # figures/loa_5deg.jpg
```

Each script has a full docstring (`python <script> --help`).  Analysis scripts read `outlier_selection.csv`, so
re-run `02`, `03` and the plots after changing the selection.  Results are written to `results/`, figures to
`figures/`; `*_selection.csv` in `results/` records which data were excluded for that run.

| file | purpose |
|---|---|
| `common.py` | shared helpers: variable list, data loading, outlier selection, equivalence classes, LoA bands |
| `01_convert_mat_to_npz.py` | convert the MATLAB structure to a NumPy archive and verify it against the saved grand means |
| `select_outliers.py` | choose which flagged strides / participants are included (presets, CLI, interactive) |
| `02_equivalence_analysis.py` | 90 % spm1d CI of MB - ML, three-way classification, SPM paired t-test, Table 1 |
| `03_loa_analysis.py` | pointwise functional LoA and simultaneous prediction band, optional leave-one-out coverage |
| `04_plot_equivalence.py` | equivalence figure (CI coloured by outcome, SPM bar) |
| `05_plot_loa.py` | LoA figure (pointwise + simultaneous bands, +/-5 deg bounds, width annotations) |

## Data format (`data/study2_gait_data.npz`)

Load with `numpy.load(path, allow_pickle=False)`, or use `common.load_dataset()`.

| key | shape | meaning |
|---|---|---|
| `mb_trials`, `ml_trials` | (24, 23, 8, 101) | angle (deg); axes = participant, variable, trial, gait-cycle node (0-100 %). Missing trials are NaN |
| `n_trials` | (24, 23) | strides stored per participant/variable (8, except P011 foot progression = 7) |
| `subjects` | (24,) | participant IDs P005 ... P029 |
| `variable_files` | (23,) | MATLAB file codes (`RAX` = right ankle X, ...); `PE*` pelvis, `RH*` hip, `RK*` knee, `RA*` ankle, `RFP` foot progression |
| `in_published_grandmean` | (24, 23) | whether the participant entered the saved grand means (only P024 foot progression is False) |
| `analysed_codes`, `analysed_index` | (12,) | the 12 analysed variables and their position on the variable axis |
| `mdc_bounds`, `mdc_curves` | (12,), (12, 101) | Study 1 MDC for the analysed variables (foot progression defaults to 5 deg) |

Trial N in the outlier lists is stored at index N-1 on the trial axis.  The left-side variables of P024 stride 4
contain NaNs; they are not analysed.

## Outlier selection

`outlier_selection.csv` (columns `subject, trial, variable, action, source`).  `action = exclude` is applied:

* `trial = 1-8`: that MB stride is left out of the participant's MB mean for that variable (ML always uses all strides);
* `trial = all`: the participant is dropped from that variable for both systems (paired analyses).

The default (`as_published`) reproduces the saved grand means: only P024 foot progression is excluded.  Other
presets: `include_all`, `user_flagged` (all 26 supplied strides + P024 foot progression) and `recommended`
(P011 stride 4, P021 stride 3, P024 ankle/foot variables; P013 strides 1 and 8 kept).  See
`../flagged_trials_visual_check.jpg` for the evidence behind the recommendation, and `select_outliers.py --help`.

## Default results (`as_published`, +/-5 deg)

Equivalent / not equivalent / inconclusive (% of gait cycle); LoA widths are mean widths of the bands (deg).

| variable | n | equiv. | not equiv. | inconcl. | SPM sig. | pointwise LoA | simultaneous band |
|---|---|---|---|---|---|---|---|
| Pelvis Anterior Tilt | 24 | 64 | 0 | 36 | 29 | 21.5 | 29.4 |
| Pelvis Obliquity | 24 | 90 | 0 | 10 | 63 | 6.7 | 11.0 |
| Pelvis Rotation | 24 | 85 | 0 | 15 | 57 | 11.0 | 17.7 |
| Hip Flexion | 24 | 57 | 0 | 43 | 26 | 23.4 | 33.8 |
| Hip Adduction | 24 | 72 | 0 | 28 | 84 | 11.2 | 17.7 |
| Hip Rotation | 24 | 15 | 0 | 85 | 28 | 27.6 | 45.1 |
| Knee Flexion | 24 | 77 | 0 | 23 | 45 | 11.4 | 18.3 |
| Knee Varus | 24 | 81 | 0 | 19 | 30 | 12.3 | 20.8 |
| Foot Progression | 23 | 0 | 19 | 81 | 100 | 11.6 | 18.1 |
| Ankle Dorsiflexion | 24 | 62 | 7 | 31 | 83 | 9.4 | 15.9 |
| Ankle Inversion | 24 | 7 | 63 | 30 | 81 | 16.6 | 29.3 |
| Ankle Rotation | 24 | 18 | 38 | 45 | 61 | 26.3 | 42.5 |

## Methods and caveats

* **Equivalence**: 90 % simultaneous paired CI (spm1d, random-field theory) of MB - ML, equivalent to two one-sided
  tests at 5 % (Dixon et al. 2018).  Not equivalent = CI entirely beyond a bound; inconclusive = CI crosses a bound.
  Use `--bound mdc` in `02_equivalence_analysis.py` for the Study 1 MDC bounds instead.
* **SPM**: `spm1d.stats.ttest_paired`, two-tailed, family-wise p < 0.05.
* **Pointwise LoA** (Roislien et al. 2012, Gait & Posture 36:495): mean difference +/- 1.96 SD at each node.
* **Simultaneous band**: mean +/- z* SD sqrt(1 + 1/n) with z* the spm1d random-field-theory threshold of the
  one-sample t field of the differences.  It is a *prediction band* for a new participant's whole curve (not the
  Bland-Altman LoA, and not a CI of the LoA limits).  It is a pragmatic construction: it assumes smooth,
  approximately Gaussian differences, and its coverage has only been checked by leave-one-out on n = 24
  (`03_loa_analysis.py --coverage`: roughly 92-100 % whole-curve coverage against 50-92 % for the pointwise LoA).
* **Independence of the difference and the average** (a requirement of the conventional LoA) fails for most
  variables because MB is more variable between participants than ML; the regression-based LoA and independence
  test are in the repository root (`plot_floa.py`, `plot_floa_regression.py`) and are not part of this folder.
* Equivalence intervals are reported in gait-cycle % (node index 0-100).
