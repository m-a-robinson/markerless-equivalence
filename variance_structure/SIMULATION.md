# Simulation: direct kinematics (MB) vs inverse kinematics (ML-like)

`simulate_systems.py` builds a 3-D pelvis–thigh–shank chain (9 joint angles) with **known true kinematics** for 60 simulated participants × 8 strides. It then measures the same movement two ways and runs the same variance statistics as the experimental analysis (`variance_structure.py`).

```bash
python simulate_systems.py     # ~2 min; writes results/simulation/ and figures/simulation/
```

With every error source switched off, both pipelines return the true angles to within 10⁻¹² degrees.

## What is simulated

**Truth.** Each participant has their own average waveform (between-subject variation), stride-to-stride variation, and an individual bony alignment: a fixed rotation of each anatomical frame, e.g. pelvic shape, femoral anteversion, tibial torsion.

**Marker-based, direct kinematics.** Each segment is tracked on its own (6 degrees of freedom) from four skin markers, with:
- landmark placement error (8 mm SD, fixed per participant);
- soft-tissue artefact (markers slide as the joint flexes, in a pattern fixed within a participant, plus a small stride-specific part);
- white noise and a low-pass filter.

**Markerless-like, inverse kinematics.** A template model with fixed joint centres is fitted by least squares to noisy keypoints, with three switchable mechanisms:
- **keypoint noise** (smooth in time) and a small participant-level keypoint bias;
- **template blindness:** keypoints follow the body surface, so the model cannot see a participant's individual bony alignment;
- **model prior:** poorly observed degrees of freedom (knee varus and rotation) are pulled toward the template value.

Not simulated: gait-event timing, gross errors (marker swaps), cameras and 2-D detection.

**The parameter values are plausible magnitudes, not calibrated to Theia or to this laboratory.** The two noise levels were set by hand so that stride SDs are of realistic size. The simulation shows what each mechanism *does* to the variance structure; it does not show which mechanisms Theia actually has.

## Results

`results/simulation/simulation_summary.csv`; figures `figS1`–`figS3` in `figures/simulation/`.

**1. With all mechanisms on, the simulation reproduces the main experimental pattern** (`figS2_ratio_by_level.png`). The ML:MB SD ratio is near or above 1 for strides and well below 1 between participants, most strongly for pelvic tilt (1.39 vs 0.68), hip rotation (1.20 vs 0.44) and knee varus (0.98 vs 0.47). Knee flexion is close to a scaled copy (1.03 vs 0.94), as in the experiment.

**2. Each mechanism acts at a different level, which is why the result is not a scaled copy** (`figS3_ablations.png`):

| Mechanism alone | Stride-level ratio ML:MB | Between-subject ratio ML:MB | What it does |
|---|---|---|---|
| MB placement error | ≈ 1.0 | 0.63–0.97 | Inflates MB between-subject variance only |
| MB soft-tissue artefact | 0.65–1.0 | ≈ 1.0 | Inflates MB stride variance, mostly in rotations; makes MB rougher (roughness up to 26 vs ~4) |
| ML template blindness | ≈ 1.0 | 0.48–1.0 | Removes real between-subject differences from ML; strides untouched |
| ML model prior | 0.81 (knee rotation) | 0.77 | Shrinks both levels by the same factor: the only mechanism that *is* a scaled copy |
| ML keypoint noise | 1.07–2.2 | ≈ 1.0–1.5 | Inflates ML stride variance, mostly in rotations |

**3. Two different mechanisms produce the same signature.** MB placement error and ML template blindness both give "ML has less between-subject spread but the same stride variability". In the first case ML is right (its scale of true between-subject differences is 1.00) and MB is inflated. In the second, MB is right and ML keeps only 14–26% of the true differences in pelvic tilt, hip rotation and knee varus/rotation. The two systems' data alone cannot separate these; marker re-application or a third instrument can.

**4. The two-system upper bound detects proportional shrinkage but not template blindness.** With the model prior alone, the bound on the ML:MB scale (0.77 for knee rotation) equals the true ML scale (0.77). With template blindness alone the bound stays at or above 1 although ML has lost most of the true differences. So a bound below 1 in the experiment (foot progression, 0.74) indicates genuine proportional compression, while a bound at or above 1 does not rule out lost between-subject information.

**5. The shared-variance estimate is close to the true stride variance** when errors are independent, e.g. full scenario: 0.80 vs 0.88 °² pelvic tilt, 4.82 vs 5.09 knee flexion. It under-estimates where the prior shrinks ML (knee rotation 1.65 vs 2.47).

**Not reproduced:** in the experiment MB stride deviations are rougher than ML in 11/12 variables. Here the full scenario gives similar roughness (e.g. hip rotation 25 vs 22), because it depends on how smooth the assumed keypoint noise is. That parameter needs calibrating before roughness can be interpreted.

## Next steps

- Calibrate the error parameters to the experimental statistics (approximate Bayesian computation), then test which mechanisms are necessary to match them.
- Add gait-event detection from each pipeline's own heel/toe trajectories (timing offset), a foot segment, and rare gross errors.
- Use the calibrated simulator for the power analysis of the proposed study.
