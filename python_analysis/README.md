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
python 06_plot_means.py                # figures/means_sd.jpg (grand mean +/- SD, as in the manuscript)
python 07_plot_conceptual_figure.py    # figures/conceptual_outcomes.png (schematic of the equivalence and LoA outcomes)
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
| `05_plot_loa.py` | LoA figure (pointwise + simultaneous bands, +/-5 deg bounds, width annotations; green shading marks where the LoA lie inside the bounds, i.e. acceptable agreement) |
| `06_plot_means.py` | grand mean +/- SD figure of MB and ML (Python version of the manuscript means figure) |
| `07_plot_conceptual_figure.py` | conceptual figure of the possible equivalence and LoA outcomes (synthetic data; no results needed) |

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

`outlier_selection.csv` (columns `subject, trial, variable, action, source, note, scope`) is the **single source of
truth for excluded data**.  `02`, `03` and `06` read it, and so does the variance analysis in `../variance_structure`,
so the ML-vs-MB comparison, the LoA analysis and the variance analysis always use the same strides and participants.
`action = exclude` is applied; `trial = all` excludes the participant.  The `scope` column sets how far an exclusion
reaches:

| scope | stride exclusion | participant exclusion |
|---|---|---|
| `all` (default) | removed from **every variable, MB and ML** | dropped from **every variable** |
| `variable` | removed from the MB mean of the named variable only | dropped from the named variable only |

The variance analysis needs identical strides in both systems and across variables, so it accepts only `scope = all`
(it stops with an explanation otherwise).  The default `manuscript` selection therefore uses `scope = all`: P011
stride 4, P013 strides 1 and 8 and P021 stride 3 are removed from every variable and both systems, and P024 is dropped
entirely (n = 23 throughout).  Switch scope with `python select_outliers.py --scope variable` (per-variable, MB-only,
as in the earlier analyses; n = 23 only for ankle dorsiflexion, ankle rotation and foot progression).

The `note` column and the docstring of `select_outliers.py` record the reason for each exclusion (evidence in
`../flagged_trials_visual_check.jpg`).  Other presets: `as_published` (only P024 foot progression excluded, scope
`variable`, reproducing the grand means saved in the .mat file), `include_all`, `user_flagged` and `recommended`.
After changing the selection rerun `02`-`06` and, if needed, `python ../variance_structure/variance_structure.py`.

## Default results (`manuscript` selection, scope all, +/-5 deg)

Equivalent / not equivalent / inconclusive (% of gait cycle); LoA widths are mean widths of the bands (deg).

| variable | n | equiv. | not equiv. | inconcl. | SPM sig. | pointwise LoA | simultaneous band |
|---|---|---|---|---|---|---|---|
| Pelvis Anterior Tilt | 23 | 66 | 0 | 34 | 26 | 21.6 | 29.7 |
| Pelvis Obliquity | 23 | 90 | 0 | 10 | 64 | 6.7 | 11.1 |
| Pelvis Rotation | 23 | 82 | 0 | 18 | 58 | 11.2 | 18.2 |
| Hip Flexion | 23 | 55 | 0 | 45 | 28 | 23.2 | 33.9 |
| Hip Adduction | 23 | 70 | 0 | 30 | 83 | 11.4 | 18.0 |
| Hip Rotation | 23 | 11 | 4 | 85 | 21 | 28.8 | 47.2 |
| Knee Flexion | 23 | 76 | 0 | 24 | 45 | 11.4 | 18.6 |
| Knee Varus | 23 | 77 | 0 | 23 | 30 | 12.9 | 22.0 |
| Foot Progression | 23 | 0 | 17 | 83 | 100 | 11.2 | 17.5 |
| Ankle Dorsiflexion | 23 | 63 | 7 | 30 | 86 | 8.6 | 14.5 |
| Ankle Inversion | 23 | 7 | 63 | 30 | 84 | 14.4 | 25.4 |
| Ankle Rotation | 23 | 15 | 38 | 48 | 64 | 23.6 | 38.4 |

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
  (`03_loa_analysis.py --coverage`: roughly 91-100 % whole-curve coverage (mean about 96 %) against 43-96 % for the pointwise LoA).
* **Independence of the difference and the average** (a requirement of the conventional LoA) fails for most
  variables because MB is more variable between participants than ML.  The independence test and the
  regression-based LoA are not implemented in this folder.
* Equivalence intervals are reported in gait-cycle % (node index 0-100).
