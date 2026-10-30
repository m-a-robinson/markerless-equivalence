# The structure of variance in markerless and marker-based gait kinematics

**Draft study outline and preliminary results.** These are exploratory analyses of the Study 2 simultaneous ML/MB data set (n = 24, 8 strides each).

*Working title:* **Not a scaled copy: the structure of variance in markerless and marker-based kinematics, and why it determines agreement between systems.**

---

## 1. Working thesis

Validation studies compare markerless (ML) and marker-based (MB) kinematics through mean differences, RMSE, limits of agreement or equivalence tests. Each of these summaries depends on the variance of the between-system difference: Var(ML − MB) = σ²_ML + σ²_MB − 2·cov. That variance in turn depends on how each system's variance is structured:

* how it splits into between-subject, between-session and stride-to-stride levels;
* how it is distributed across the gait cycle;
* how much of it the two systems share (true movement) and how much is specific to each system (error);
* its temporal smoothness and dimensionality;
* how it couples across joints and planes;
* the shape of its error distribution (Gaussian vs gross errors).

The usual working assumption is that ML is "MB with less variance". That treats ML variance as a scaled-down copy of MB variance, Σ_ML = c²·Σ_MB, with one scale factor c at every level, gait-cycle node and joint. **The thesis is that this assumption is false.** The two systems have qualitatively different error structures because they fail for different physical reasons:

* **MB:** marker placement, soft-tissue artefact, axis-definition crosstalk, and tracking or labelling failures.
* **ML:** model priors and shape templates, inverse-kinematics constraints, smoothing, and pose-estimation error.

Because of this, a between-system comparison can only be interpreted, and a "smaller ML variance" can only be called better, once the variance structure of each system has been characterised. The study proposed here does that, and it identifies which components of each system's variance are true movement and which are measurement.

---

## 2. Model and terminology

For variable *v*, gait-cycle node *t*, system *s* ∈ {MB, ML}, participant *i* and stride *k*:

  Y_sik(t) = μ_s(t) + b_si(t) + w_sik(t)

* **b** is the between-subject deviation. It mixes true inter-individual differences with system-specific subject-level error, such as marker placement and body-shape-dependent model fitting.
* **w** is the stride deviation. It mixes true stride-to-stride variability with stride-level error, such as soft-tissue artefact, pose-estimation noise and gait-event timing.

The two systems recorded **the same strides at the same time**. Their cross-covariance at each level therefore estimates the variance they share, and the remainder is system-specific. This is the two-instrument Grubbs (1948) estimator, which assumes unit scale and independent errors. The proposed study adds a third independent instrument so that neither assumption is needed (§5).

**Scalar-scaling null hypothesis (H₀-scale):** σ_ML/σ_MB = c, the same at the between-subject and stride levels, at every node and for the shared component, with an identical covariance shape. The preliminary results test parts of this directly.

---

## 3. Preliminary results (Study 2, existing data)

**Code:** `variance_structure/variance_structure.py`. It reproduces the manuscript's Table 1 equivalence percentages exactly from the `.mat` file. **Primary analysis:** gross-error strides were removed with a fixed rule (§3.1), leaving 23 participants and 180 strides. **Sensitivity analysis:** all data, as in the manuscript. Results are in `results/primary_clean/` and `results/sensitivity_all/`, figures in `figures/`. 95% CIs come from a participant-cluster bootstrap (1,000 resamples). All results are exploratory and are not corrected for 12 variables; they are intended to generate hypotheses and to inform planning.

### 3.1 R0: MB has a gross-error class that ML does not

The flag rule: deviation from the participant's median stride above 8 robust SD **and** above 10°, sustained for at least 3 consecutive nodes. It flagged **11 MB strides and 0 ML strides**.

* Nine of them (P011, P021, P024) peak at 64–72% of the gait cycle, the swing phase with peak plantarflexion and fast foot rotation.
* They appear as coupled spikes of up to ~110° in ankle dorsiflexion, inversion and rotation (`figures/fig0_gross_errors.png`). This pattern fits a foot-segment tracking failure or a Cardan-sequence singularity. The flagged P021 stride is also offset by about 30° in ankle rotation across the whole cycle, which looks like a marker swap.
* The other two (P013) are hip rotation and knee varus excursions in terminal swing (77–80% GC). That is a knee-axis crosstalk pattern (§3.5).
* P024 has the ankle artefact on all 8 strides (7 flagged), so it is a subject-level model or marker-placement problem.

Excess kurtosis of MB stride deviations (all data) reaches 16–51 at the ankle, against <1 for ML. **The two systems' error distributions differ in shape, not just in scale.**

> *Manuscript note:* P024's MB foot data, and the single strides from P011/P013/P021, feed into the current Study 2 equivalence results. Removing them changes equivalence little; for example, ankle rotation goes from 9% to 3% (see `difference_variance_decomposition.csv` in each results folder). They should still be checked in QTM/Visual3D.

### 3.2 R1: the ML:MB variance ratio depends on the level (main result)

| Variable | SD_w MB / ML (°) | ML:MB ratio, stride | ML:MB ratio, between-subject | log-ratio contrast (w − b) | flat-ratio p |
|---|---|---|---|---|---|
| Pelvis Anterior Tilt | 0.80 / 0.66 | 0.83 [0.77, 0.90] | 0.36 [0.21, 0.54] | 0.85 [0.44, 1.40] * | 0.066 |
| Pelvis Obliquity | 0.55 / 0.58 | 1.05 [0.99, 1.13] | 0.96 [0.72, 1.26] | 0.10 [-0.19, 0.39] | 0.001 |
| Pelvis Rotation | 1.61 / 1.41 | 0.88 [0.83, 0.94] | 0.72 [0.55, 1.02] | 0.20 [-0.15, 0.46] | 0.080 |
| Hip Flexion | 1.27 / 1.32 | 1.03 [0.96, 1.11] | 0.53 [0.38, 0.71] | 0.68 [0.37, 1.00] * | 0.001 |
| Hip Adduction | 0.86 / 0.85 | 0.99 [0.89, 1.10] | 0.65 [0.50, 0.87] | 0.42 [0.11, 0.71] * | 0.086 |
| Hip Rotation | 1.31 / 0.93 | 0.71 [0.59, 0.86] | 0.24 [0.18, 0.34] | 1.09 [0.74, 1.43] * | 0.001 |
| Knee Flexion | 1.64 / 1.74 | 1.07 [1.01, 1.12] | 0.89 [0.72, 1.04] | 0.18 [0.03, 0.38] * | 0.001 |
| Knee Varus | 0.72 / 0.56 | 0.78 [0.64, 0.98] | 0.53 [0.44, 0.69] | 0.39 [0.07, 0.66] * | 0.007 |
| Foot Progression | 1.81 / 1.56 | 0.87 [0.80, 0.94] | 0.68 [0.60, 0.77] | 0.24 [0.09, 0.38] * | 0.001 |
| Ankle Dorsiflexion | 1.42 / 1.29 | 0.91 [0.87, 0.96] | 0.90 [0.81, 1.03] | 0.01 [-0.12, 0.13] | 0.001 |
| Ankle Inversion | 1.88 / 0.62 | 0.33 [0.28, 0.38] | 0.30 [0.24, 0.40] | 0.08 [-0.22, 0.35] | 0.001 |
| Ankle Rotation | 1.45 / 1.22 | 0.84 [0.73, 0.97] | 0.49 [0.37, 0.70] | 0.55 [0.14, 0.89] * | 0.001 |

*Ratios are cycle-averaged SD ratios. The contrast is log(ratio_stride) − log(ratio_between) and is 0 under scalar scaling; \* marks a 95% CI that excludes 0. "Flat-ratio p" is a permutation test of whether the stride-level ratio is constant over the gait cycle (ML rescaled by its global ratio, then system labels swapped within stride pairs; 1,000 permutations). See `fig1_variance_components.png` and `fig2_sd_ratio.png`.*

**Interpretation**

* **For 8 of 12 variables, ML compresses between-subject differences much more than stride-to-stride variation.**
  * Hip and knee flexion and hip adduction have **stride variability equal to MB or larger** (ratio 0.99–1.07). Their between-subject SD is only 53–89% of MB's.
  * Pelvic tilt and hip rotation show the most extreme compression: ML between-subject SD is 36% and 24% of MB's.
  * The lower ML "variability" in the manuscript's Fig. 3 (grand mean ± SD across participants) is therefore mostly **between-subject compression**. It does not show smoothing of stride-to-stride variation. This bears directly on the current Discussion sentence suggesting that ML smoothing "may reduce sensitivity to genuine stride-to-stride variation". For the sagittal hip and knee, the data do not support that sentence.
* **Within a level, the ratio also changes across the gait cycle** (flat-ratio test p ≤ 0.007 for 9 of 12 variables). The clearest case is ankle inversion: the ratio drops to about ⅛ at 65–75% GC (MB swing artefact) and is about 1 during stance.
* What this cannot yet say: whether MB between-subject variance is **inflated** (marker placement, soft-tissue thickness) or ML's is **shrunk** (template or prior, scaling). One session cannot separate the two; this is the central question for the new data (§5).

### 3.3 R2: how much of each system's variance is shared

| Variable | r stride (shared) | r stride after re-registration | r between-subject | r of individual stride SDs | ICC MB / ML |
|---|---|---|---|---|---|
| Pelvis Anterior Tilt | 0.57 [0.47, 0.67] | 0.57 | 0.48 [0.18, 0.68] | 0.73 [0.61, 0.86] | 0.98 / 0.92 |
| Pelvis Obliquity | 0.58 [0.53, 0.63] | 0.61 | 0.45 [0.28, 0.59] | 0.71 [0.46, 0.86] | 0.90 / 0.89 |
| Pelvis Rotation | 0.84 [0.80, 0.87] | 0.85 | 0.40 [0.10, 0.62] | 0.81 [0.54, 0.95] | 0.77 / 0.69 |
| Hip Flexion | 0.69 [0.62, 0.74] | 0.77 | 0.55 [0.35, 0.73] | 0.78 [0.65, 0.91] | 0.97 / 0.89 |
| Hip Adduction | 0.63 [0.56, 0.70] | 0.66 | 0.58 [0.46, 0.69] | 0.43 [-0.06, 0.75] | 0.95 / 0.88 |
| Hip Rotation | 0.23 [0.17, 0.30] | 0.23 | 0.24 [-0.02, 0.50] | -0.12 [-0.45, 0.25] | 0.97 / 0.79 |
| Knee Flexion | 0.78 [0.72, 0.83] | 0.94 | 0.72 [0.64, 0.81] | 0.84 [0.55, 0.93] | 0.86 / 0.81 |
| Knee Varus | 0.19 [0.13, 0.24] | 0.22 | 0.42 [0.08, 0.61] | 0.16 [-0.24, 0.70] | 0.97 / 0.93 |
| Foot Progression | 0.86 [0.76, 0.93] | 0.87 | 0.92 [0.87, 0.95] | 0.64 [0.22, 0.92] | 0.92 / 0.88 |
| Ankle Dorsiflexion | 0.82 [0.78, 0.85] | 0.88 | 0.78 [0.66, 0.87] | 0.85 [0.75, 0.92] | 0.86 / 0.86 |
| Ankle Inversion | 0.02 [-0.04, 0.10] | 0.02 | 0.07 [-0.09, 0.25] | 0.06 [-0.37, 0.48] | 0.87 / 0.84 |
| Ankle Rotation | 0.38 [0.31, 0.43] | 0.38 | 0.09 [-0.24, 0.44] | 0.36 [-0.24, 0.77] | 0.94 / 0.84 |

*See `fig3_shared_vs_specific.png`.*

* **Sagittal, foot progression and pelvic rotation:** most stride variance is shared (r = 0.69–0.86, or 0.77–0.94 after re-registration). Both systems see the same stride-to-stride changes, and they agree on **which participants are more variable** (r = 0.64–0.85). For these variables the two systems capture the same stride-to-stride signal, which supports using ML in variability research (validity against a third instrument still needs testing).
* **Hip rotation, knee varus, ankle inversion and ankle rotation:** shared stride variance is close to zero (r = 0.02–0.38). The stride-to-stride variability each system reports for these variables is largely **specific to that system**, and the two systems disagree on who is more variable. Variability or coordination studies would measure different things depending on the system. The data cannot yet tell whether either system captures the true signal. The Frisch bounds on the scale of the shared signal are uninformative here, for example 0.01–13 for ankle inversion.
* **Between-subject agreement is moderate even where means agree.** Examples: pelvic rotation r_b = 0.40, pelvic tilt 0.48, hip rotation 0.24. The systems would rank participants differently. This matters for clinical decisions about individuals, not group means.

### 3.4 R3: the temporal shape of variance differs

| Variable | Participation ratio MB / ML | PC1 share MB / ML | Roughness MB / ML | Subspace similarity K (MB split-half ref.) | Excess kurtosis MB / ML, all data |
|---|---|---|---|---|---|
| Pelvis Anterior Tilt | 2.3 / 2.6 | 64% / 59% | 3.2 / 4.2 | 0.92 [0.74, 0.97] (0.97) | 0.4 / 0.7 |
| Pelvis Obliquity | 4.3 / 4.7 | 36% / 33% | 8.7 / 6.0 | 0.95 [0.84, 0.96] (0.92) | 0.8 / 0.5 |
| Pelvis Rotation | 3.2 / 2.9 | 44% / 51% | 1.7 / 1.4 | 0.99 [0.97, 0.99] (0.98) | 0.7 / 1.3 |
| Hip Flexion | 3.6 / 3.7 | 42% / 38% | 4.1 / 2.7 | 0.96 [0.86, 0.98] (0.96) | 1.8 / 0.6 |
| Hip Adduction | 4.1 / 4.7 | 42% / 32% | 7.2 / 5.7 | 0.94 [0.71, 0.98] (0.84) | 1.9 / 0.2 |
| Hip Rotation | 6.9 / 4.0 | 29% / 44% | 51.2 / 8.9 | 0.68 [0.53, 0.75] (0.60) | 18.1 / 3.5 |
| Knee Flexion | 4.2 / 4.0 | 35% / 36% | 6.5 / 4.0 | 0.99 [0.97, 0.99] (0.95) | 5.0 / 0.5 |
| Knee Varus | 4.7 / 4.2 | 37% / 42% | 35.4 / 13.6 | 0.66 [0.53, 0.78] (0.75) | 12.0 / 2.8 |
| Foot Progression | 3.4 / 3.1 | 46% / 50% | 9.4 / 5.6 | 0.97 [0.90, 0.98] (0.96) | 9.7 / 0.6 |
| Ankle Dorsiflexion | 5.1 / 4.5 | 31% / 38% | 22.5 / 11.9 | 0.89 [0.65, 0.95] (0.89) | 26.6 / 0.7 |
| Ankle Inversion | 2.0 / 5.2 | 68% / 37% | 29.5 / 18.3 | 0.48 [0.32, 0.61] (0.90) | 15.8 / 0.6 |
| Ankle Rotation | 3.1 / 5.7 | 54% / 31% | 23.0 / 13.4 | 0.69 [0.59, 0.80] (0.87) | 51.2 / 0.3 |

*Functional PCA of stride deviations. Participation ratio = (Σλ)²/Σλ², the effective number of modes. Roughness = 10⁴ × mean squared second difference / mean squared deviation (high-frequency content). K = Krzanowski similarity of the first 3 PC subspaces; values below the MB-vs-MB split-half reference mean the systems' variance has a different temporal shape. See `fig4_fpca.png`.*

* MB stride deviations are **rougher** in 11/12 variables, with the largest differences in hip rotation (5.8×), knee varus (2.6×) and the ankle (1.6–1.9×). This is despite MB being filtered at 6 Hz and ML at 20 Hz. The likely sources are soft-tissue artefact and marker wobble. ML's smoothness most likely comes from its model and inverse kinematics rather than the output filter; this needs testing with matched filters (§5).
* For ankle inversion and rotation the **shape** differs, not just the amplitude. MB variance is concentrated in one swing-phase mode (PC1 = 68%), while ML spreads its variance over about 5 modes. Their PC subspaces are less similar (K = 0.48) than two halves of the MB sample are to each other (0.90).
* For the sagittal hip and knee, the variance has the same shape (K ≈ 0.96–0.99) and the same size. This is the only place where "ML ≈ MB" approximately holds.

### 3.5 R4: inter-joint coupling differs

*See `fig5_coupling.png` and `interjoint_coupling.csv`.*

* **Ankle inversion ↔ ankle rotation:** the pooled within-stride correlation is −0.13 in MB and +0.32 in ML (difference −0.45 [−0.57, −0.32]). In ML, frontal and transverse ankle motion are coupled. That is consistent with a constrained foot/ankle model in which the two degrees of freedom are not estimated independently. The same coupling shows for foot progression ↔ ankle inversion and rotation.
* **Hip rotation ↔ knee varus:** in MB this is the classic signature of knee-axis crosstalk (r = 0.52 with all data). It falls to 0.19 once P013's two flagged strides are removed, against 0.10 in ML. MB crosstalk here therefore appears **episodic** (stride-specific axis errors) rather than constant, which is another way the error structure differs.

### 3.6 R5: event timing explains much of the sagittal "bias"

*See `fig7_event_timing.png`, `event_timing.csv`, `timing_explained_bias.csv` and `registration_sensitivity.csv`.*

* Gait events were detected separately in each system with the Zeni method. Aligning each ML knee-flexion stride to its simultaneous MB stride needs a **mean shift of −1.27% GC**: ML-normalised waveforms run early, i.e. the ML-defined cycle starts later. The within-participant SD of this shift is 0.47% GC.
* **That single time shift explains 94% of the shape of the mean knee-flexion bias waveform**, including the 61–68% non-equivalent window in the manuscript. It also explains 69% for ankle dorsiflexion and 53% for hip flexion.
* Stride-to-stride timing jitter explains ~70% of the stride-level knee-flexion ML−MB variance (76% in swing). Re-registering each stride raises knee-flexion shared stride r from 0.78 to 0.94.
* So for sagittal variables, a large part of the "angle disagreement" is really **timing disagreement**: which instant gets labelled 0% GC. The cause is event detection from different heel and toe trajectories. It is not a joint-angle measurement error. *Caveat:* shifts are estimated from the same data, which makes the registered r slightly optimistic. A common, independent event source (force plates) removes this problem (§5).

### 3.7 R6: equivalence depends on the covariance structure, not only on bias

| Variable | MDC (°) | mean abs bias (°) | Var(diff) from decorrelation | Var(diff) from SD mismatch | % equivalent: observed | if r = 1 | bias only | bias shape explained by timing offset |
|---|---|---|---|---|---|---|---|---|
| Pelvis Anterior Tilt | 4.1 | 1.8 | 48% | 52% | 49 | 62 | 96 | 0% |
| Pelvis Obliquity | 1.9 | 1.9 | 98% | 2% | 31 | 53 | 61 | 0% |
| Pelvis Rotation | 6.3 | 2.2 | 91% | 9% | 92 | 100 | 100 | 18% |
| Hip Flexion | 6.1 | 2.1 | 66% | 34% | 69 | 85 | 100 | 53% |
| Hip Adduction | 3.2 | 3.0 | 81% | 19% | 18 | 44 | 70 | 0% |
| Hip Rotation | 7.5 | 3.3 | 39% | 61% | 59 | 65 | 88 | 0% |
| Knee Flexion | 6.7 | 2.0 | 91% | 9% | 91 | 100 | 100 | 94% |
| Knee Varus | 4.0 | 2.0 | 70% | 30% | 56 | 68 | 92 | 17% |
| Foot Progression | 5.0 | 5.5 | 51% | 49% | 0 | 9 | 40 | 0% |
| Ankle Dorsiflexion | 5.1 | 3.4 | 91% | 9% | 63 | 80 | 83 | 69% |
| Ankle Inversion | 5.5 | 7.0 | 38% | 62% | 8 | 10 | 19 | 1% |
| Ankle Rotation | 4.2 | 5.9 | 77% | 23% | 3 | 26 | 38 | 8% |

*Var(ML − MB) of participant means = (SD_ML − SD_MB)² + 2·SD_ML·SD_MB·(1 − r). "If r = 1" holds both SDs fixed but makes ML a perfectly correlated rescaling of MB, and keeps the RFT critical value fixed (an approximation). "Bias only" is the percentage of the cycle where |mean difference| < MDC, the ceiling if the CI had zero width. "Bias shape explained by timing offset" is from R5 (negative values shown as 0%). See `fig6_difference_decomposition.png`.*

* The width of the equivalence CI is mostly set by **decorrelation** (participants ordered differently by the two systems) for pelvic obliquity, pelvic rotation, hip adduction, knee flexion and ankle dorsiflexion (≥80%). It is mostly set by **SD mismatch** (between-subject compression in R1) for pelvic tilt, hip rotation and ankle inversion.
* Two variables with similar bias can therefore reach very different equivalence conclusions. Hip adduction and pelvic obliquity both have about 2–3° bias; equivalence would rise from 18% to 44% and from 31% to 53% respectively if the systems were perfectly correlated.
* Foot progression, ankle inversion and ankle rotation fail mainly on **bias** (ceiling 19–40%), which points to segment or coordinate-system definitions.
* **Conclusion for the thesis:** an equivalence result combines bias, timing, scale mismatch and decorrelation. Studies that report only the result, without the variance structure, cannot say *why* systems fail to agree or *what* would fix it.

### 3.8 Sensitivity and robustness

* **Gross errors:** with all data included, R1's level-dependence holds for pelvic tilt, hip flexion and hip adduction (CIs exclude 0). Hip rotation's CI just reaches 0 (−0.01 to 1.40). The ankle variables are dominated by the gross errors: MB stride SD is inflated 1.7–2.5×, stride r for ankle dorsiflexion falls from 0.82 to 0.35, and kurtosis exceeds 15. *Conclusion:* the gross-error class is itself a central result, and every analysis should be reported with and without it, under a pre-registered rule.
* **Estimator choices:** variances are averaged over the cycle before taking ratios, so they are not dominated by low-variance nodes. Between-subject components subtract the expected stride contribution σ²_w/n. Components were also estimated pointwise (Figs 1–3).
* **Known limitations of the preliminary analysis:**
  1. One session, so between-subject true variance and subject-level error are confounded.
  2. The shared/specific split assumes unit scale and independent errors.
  3. n = 23–24 clusters gives wide between-subject CIs. The bootstrap with 23 clusters is approximate and slightly anti-conservative.
  4. Inference is pointwise or cycle-averaged; there is no family-wise control over 12 variables × 101 nodes.
  5. MDC bounds come from a different ML session/clothing study, and no MB MDC was measured on the same participants.
  6. Healthy young adults only, overground walking only.

---

## 4. Proposed study: aims and hypotheses

**Aim.** Characterise and compare the multilevel, temporal and inter-joint variance structure of simultaneous ML and MB lower-limb kinematics. Attribute each component to true movement or to system-specific error. Show how these components determine between-system agreement across tasks and conditions.

| | Hypothesis (falsifiable) | Primary test |
|---|---|---|
| **H1 Non-proportionality** | ML variance is not a scalar multiple of MB variance. The ML:MB ratio differs between levels (between-subject, between-session, stride) and across the cycle. | Bivariate multilevel model; contrast of log SD ratios between levels; functional ratio test (permutation, interval-wise). |
| **H2a Shrinkage vs H2b inflation** | (a) ML compresses true between-subject differences toward a template, or (b) MB between-subject variance is inflated by marker placement and ML is closer to the truth. | MB re-application variance removes placement error. Response to an **imposed, known change** (foot-progression feedback, wedges, brace; §5): slope ML vs MB vs third instrument. H2a predicts ML response slope < 1. |
| **H3 Timing** | A material share of sagittal ML−MB bias and stride-level disagreement comes from event timing, and it grows with joint angular velocity (speed, task). | Reprocess with force-plate events common to both systems, then compare bias and stride-level r; timing share regressed on speed and dθ/dt. |
| **H4 Error modes** | MB error is heavy-tailed and phase-specific (swing ankle, axis crosstalk). ML error is near-Gaussian and smoother but model-coupled (ankle inversion/rotation). | Kurtosis and gross-error rates; fPCA subspace similarity; coupling matrices; matched-filter reprocessing. |
| **H5 Consequence** | Agreement or equivalence outcomes are predicted by the variance-structure parameters (bias, timing, SD mismatch, r) across variables and tasks. | Hierarchical model of % equivalence (or RMSE) on these predictors across variables × tasks; out-of-task prediction. |
| **H6 Task dependence** | The ML:MB structure changes with the task's range of motion, velocity and plane. Shrinkage should be greatest for postures far from the template (deep squat, cutting); timing error greatest at high velocity (running). | Task × system interaction on each component. |

---

## 5. Data needed beyond Study 2

The existing data can answer R0–R6 but **cannot** separate truth from error at the between-subject level, cannot identify each system's error without assuming one is truth, and cannot test task dependence. Listed by priority:

1. **Reprocess existing Study 2 data (no new collection):**
   * (a) **Force-plate gait events** common to both systems, if force plates were recorded. Please confirm. This removes the event-detection component (H3).
   * (b) **Matched filtering:** ML GCVSPL and MB Butterworth at the same cut-off (e.g. both 6 Hz, and both 12 Hz). This tests whether ML smoothness comes from filtering or from the model.
   * (c) **Alternative MB models:** CGM1.1 vs CGM2.4 from the same markers, and the foot model with and without the virtual heel. This tests whether the swing ankle artefact and the crosstalk depend on the model.
   * (d) Static-trial angles from both systems (subject-level offsets).
2. **Repeated sessions with both systems recorded simultaneously.** At least 2 days, plus **within-day marker removal and re-application by ≥2 assessors.** This separates MB placement variance from true between-subject variance, gives MB and ML SEM/MDC on the same participants, and resolves H2a vs H2b. At the moment the equivalence bounds use only the ML MDC from a different study.
3. **A third, physically independent instrument** recording simultaneously, preferably IMU-based kinematics (e.g. Xsens). Its errors are unlikely to correlate with either camera system. With three instruments, a **Grubbs/triplet estimator** (equivalently a one-factor measurement model) identifies each system's error variance and scale **without assuming that either is the truth**. A second markerless pipeline (e.g. OpenCap, Pose2Sim) is useful but its errors may correlate with Theia's.
4. **Imposed, known within-person changes** as a validity criterion for the shared signal:
   * prescribed toe-in/toe-out (foot progression, hip rotation) via visual feedback;
   * lateral/medial wedges (ankle inversion);
   * a knee brace limiting flexion (sagittal range of motion);
   * at least 3 walking speeds (velocity → timing error).

   The response to each change, per system, is the direct test of shrinkage (H2) and of sensitivity to change (responsiveness = Δ/SEM).
5. **Tasks that span the variance-generating mechanisms:**
   * overground walking at 3 speeds;
   * running (high velocity, high soft-tissue artefact);
   * sit-to-stand or squat (large sagittal range of motion, slow, poses far from the walking template);
   * step-down or single-leg squat (frontal-plane knee);
   * 45° cutting or turning (transverse plane);
   * a standing static or functional calibration.
6. **Participant heterogeneity:** a deliberate spread of BMI and adiposity (soft-tissue artefact for MB, silhouette fitting for ML), recorded skinfold or body composition, and a range of foot posture. A later clinical cohort (e.g. CP or post-stroke) would test shrinkage away from typical-gait priors, where it matters most.
7. **≥10–12 clean strides per condition** for stable stride-level components and fPCA. Raw camera data should be kept so the data can be reprocessed when Theia versions change.

---

## 6. Statistical analysis plan (proposed study)

1. **Pre-registration** (OSF) of variables, gross-error rule (§3.1), primary contrasts and smallest effects of interest.
2. **Measurement model:** a multivariate (system × instrument) multilevel model at each node, and on a low-dimensional functional basis (B-splines or fPCA scores) for whole-curve inference:
   Y_s,i,d,a,c,k(t) = μ_s,c(t) + u_s,i + v_s,id + p_s,ia (MB only) + e_s,idack, with an unstructured cross-system covariance at **each** level.
   * The levels are participant, day, marker application and stride; c is the task or condition.
   * Fit with Bayesian multivariate mixed models (brms/Stan or MCMCglmm) or REML (sommer/ASReml). With the third instrument, use a latent-variable (one-factor) specification for the true angle with system-specific loadings λ_s and error variances.
3. **Primary estimands per variable, level and task, as curves and as cycle summaries:**
   * SD ratios ML:MB;
   * shared vs system-specific variance;
   * loadings λ_s (scale of the true signal: λ < 1 = shrinkage);
   * between-system correlation;
   * ICC and SEM/MDC for each system;
   * coupling matrices;
   * fPCA participation ratio, subspace similarity and roughness;
   * kurtosis and gross-error rates;
   * timing share.
4. **Inference:**
   * participant-cluster bootstrap or posterior intervals;
   * curve-level control via SPM/RFT or interval-wise testing (Pini & Vantini);
   * Holm correction across the 12 variables for primary contrasts;
   * equivalence (TOST) for the scale hypothesis itself: the stride-level ML:MB ratio is "equivalent to 1" if its 90% CI lies within [0.8, 1.25].
5. **Consequence model (H5):** decompose each variable's Var(ML − MB) and MSD into bias, timing, scale and decorrelation, then predict agreement and equivalence across variables and tasks. Report a standard **"variance-structure profile"** for each variable as a proposed reporting guideline for ML validation studies.
6. **Sensitivity analyses:** with and without gross errors; system-specific vs common events; matched filters; CGM1.1 vs CGM2.4; leave-one-participant-out influence.

**Sample size (to be finalised by simulation from the preliminary components).** With n = 23, between-subject SD ratios have CIs spanning roughly ×/÷1.3–1.7 (e.g. pelvic tilt 0.21–0.54). For a between-system correlation of 0.5, the 95% CI half-width is about ±0.31 at n = 24, ±0.24 at n = 40 and ±0.19 at n = 60 (Fisher z). **n ≈ 40–50**, with ≥2 sessions × 2 applications × ≥10 strides per task, is a reasonable working target. A simulation script that draws from the multilevel model, using the variance components in `results/primary_clean/variance_components_summary.csv`, should confirm power for H1 (level contrast) and H2 (loading < 1) before ethics submission.

---

## 7. Fit within the thesis

* **Chapter 4 (Study 1):** ML inter-session reliability and the effect of clothing.
* **Chapter 5 (Study 2):** ML–MB equivalence. *Suggested additions to the current draft:*
  * the gross-error check (§3.1);
  * in the Discussion, the distinction between between-subject compression and stride smoothing (§3.2);
  * the event-timing explanation for the swing-phase sagittal non-equivalence (§3.6).
* **Chapter 6 (new):** the structure of variance in ML and MB kinematics. Part A uses the preliminary analyses here (existing data plus reprocessing). Part B is the multi-session, multi-task, three-instrument study.
* **Chapter 7:** general discussion. It would cover when "smaller variance" means "better", a reporting framework for ML validation, and implications for clinical and variability research.

---

## 8. Key references to anchor the framework

* Grubbs FE (1948). On estimating precision of measuring instruments and product variability. *JASA* (two- and three-instrument error estimation).
* Dunn G (2004). *Statistical Evaluation of Measurement Errors.* (Method-comparison as a factor model.)
* Krzanowski WJ (1979). Between-groups comparison of principal components. *JASA.*
* Ramsay JO & Silverman BW (2005). *Functional Data Analysis.*
* Pataky TC et al. (2013–2016). SPM1D; Pini A & Vantini S (2017). Interval-wise testing for functional data.
* Della Croce U et al. (2005); Leardini A et al. (2005). Marker placement error and soft-tissue artefact.
* Zeni JA et al. (2008). Gait event detection from kinematics.
* Kanko RM et al. (2021); Wishaupt K et al. (2024); Kearney et al. (2025); Horsak B et al. (2026). Already cited in Chapter 5.
* McGinley JL et al. (2009). Reliability of 3D kinematic gait measurement.
