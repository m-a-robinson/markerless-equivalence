# Variance-structure analyses (ML vs MB)

Preliminary analyses supporting the draft study outline in [`STUDY_OUTLINE.md`](STUDY_OUTLINE.md).

```bash
pip install numpy scipy pandas matplotlib spm1d
python variance_structure.py      # ~5 min; writes results/ and figures/
```

* `load_data.py` reads `../stydy2data_20260226_withMDC.mat` into arrays (subject × variable × 101 × stride) for the 12 right-side variables, in manuscript order.
* `variance_structure.py` runs everything twice: `primary_clean` (flagged strides removed; by default the exclusions are read from the shared `../python_analysis/outlier_selection.csv`, so this analysis uses exactly the same trials as the ML-vs-MB equivalence and LoA analyses; the applied exclusions are logged in `results/gross_error_log.csv`; `--exclusions detect` re-derives them with the fixed gross-error rule instead) and `sensitivity_all` (all data, as in the Chapter 5 equivalence analysis, whose Table 1 it reproduces).

| Output | Content |
|---|---|
| `variance_components_summary.csv` | between-/within-subject SDs, ML:MB ratios, shared vs specific variance, ICCs, bootstrap CIs |
| `temporal_proportionality_test.csv` | permutation test that the ML:MB stride-SD ratio is flat over the gait cycle |
| `fpca_structure.csv` | participation ratio, PC1 share, roughness, kurtosis, Krzanowski subspace similarity |
| `interjoint_coupling.csv` | pooled within-stride correlations between all variable pairs, MB vs ML |
| `difference_variance_decomposition.csv` | Var(ML−MB) split into SD mismatch and decorrelation; equivalence under counterfactuals |
| `event_timing.csv`, `timing_explained_bias.csv`, `registration_sensitivity.csv` | relative ML/MB gait-event timing and its effect |
