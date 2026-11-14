# Simulation of direct and inverse kinematics: rationale and methods

This document describes the simulation implemented in `simulate_systems.py`. Results are in [`SIMULATION.md`](SIMULATION.md).

## 1. Rationale

Experimental comparisons of markerless (ML) and marker-based (MB) kinematics cannot attribute a difference to either system, because the true joint angles are unknown. In the Study 2 data, ML showed similar stride-to-stride variability to MB but a much smaller spread between participants. Two opposite explanations fit that result equally well: MB inflates between-participant differences through marker placement, or ML loses real between-participant differences. A single-session, two-system data set cannot separate them.

A simulation removes this limitation. The true kinematics are specified, so each system's error is known exactly, and each error mechanism can be switched on or off. The simulation therefore serves four purposes:

1. **Mechanism.** Show how the *way* a system computes joint angles shapes the structure of its variance. MB tracks each segment independently from skin markers (direct kinematics). ML fits a constrained multi-segment model to estimated keypoints (inverse kinematics). The two approaches are exposed to different error sources, which act at different levels (between participants, between strides) and on different degrees of freedom.
2. **Test of the scaled-copy assumption.** Establish which mechanisms change variance by the same factor at every level (a scaled copy) and which do not.
3. **Validation of the statistics.** Check that the variance-component, shared-variance and scale-bound estimators used on the experimental data recover known values, and identify what they cannot detect.
4. **Study design.** Identify which mechanisms are indistinguishable from two-system data, and so which additional measurements the proposed study needs. Once calibrated, the simulator can also be used for power analysis.

The model is deliberately reduced: one limb, three segments, no cameras. It is intended to show what each mechanism does to the variance structure. It is not a model of Theia3D, whose internal error processes are not public, and its parameters are plausible magnitudes that have not been calibrated.

## 2. Methods

### 2.1 Linked-segment model

The model is a three-segment chain: pelvis (P), thigh (T) and shank (S), connected by a hip and a knee, each with three rotational degrees of freedom. Segment axes are x (medio-lateral, flexion axis), y (anterior, ab/adduction axis) and z (longitudinal, axial-rotation axis). Thigh and shank lengths are L_T = 0.42 m and L_S = 0.40 m.

Joint and pelvis orientations use an X–Y–Z Cardan sequence. For angles (a, b, c) = (flexion, ab/adduction, axial rotation):

```math
\mathbf{E}(a,b,c)=\mathbf{R}_x(a)\,\mathbf{R}_y(b)\,\mathbf{R}_z(c)
```

Angles are recovered from a rotation matrix **R** with elements r_mn by

```math
a=\operatorname{atan2}(-r_{23},\,r_{33}),\qquad b=\arcsin(r_{13}),\qquad c=\operatorname{atan2}(-r_{12},\,r_{11})
```

Segment orientations follow from the nine generalised angles **q** = (**q**_P, **q**_H, **q**_K) by forward kinematics:

```math
\mathbf{F}_P=\mathbf{E}(\mathbf{q}_P),\qquad \mathbf{F}_T=\mathbf{F}_P\,\mathbf{E}(\mathbf{q}_H),\qquad \mathbf{F}_S=\mathbf{F}_T\,\mathbf{E}(\mathbf{q}_K)
```

### 2.2 True kinematics

For participant *i* (N = 60), stride *k* (K = 8) and gait-cycle node *t* (T = 101), each angle *v* is

```math
q_{ikv}(t)=\mu_v(t)+\sigma^{b}_v\,u_{iv}(t)+\sigma^{w}_v\,w_{ikv}(t)
```

where μ_v is the population mean waveform, u is a participant-specific deviation and w a stride-specific deviation. Both u and w are smooth random functions with unit variance at every node:

```math
f(t)=\sqrt{\tfrac{1}{H+1}}\Big[\xi_0+\sqrt{2}\sum_{h=1}^{H}\big(\xi_h\cos h\phi_t+\eta_h\sin h\phi_t\big)\Big],\qquad \xi,\eta\sim N(0,1),\quad \phi_t=\tfrac{2\pi t}{100}
```

with H = 2 harmonics for u and H = 3 for w.

Each participant also has an individual bony alignment: a fixed rotation **D** of each segment's anatomical frame relative to its functional frame **F**, representing for example pelvic shape, femoral anteversion and tibial torsion. With rotation vector **ρ** drawn once per participant and segment,

```math
\mathbf{A}_{s}=\mathbf{F}_{s}\,\mathbf{D}_{s},\qquad \mathbf{D}_{s}=\exp\big([\boldsymbol{\rho}_{s}]_\times\big),\qquad \boldsymbol{\rho}_{s}\sim N\big(\mathbf{0},\operatorname{diag}(\boldsymbol{\sigma}^{anat}_{s})^2\big)
```

The **true joint angles** are those between anatomical frames:

```math
\boldsymbol{\theta}_P=\operatorname{cardan}(\mathbf{A}_P),\qquad \boldsymbol{\theta}_H=\operatorname{cardan}(\mathbf{A}_P^{\top}\mathbf{A}_T),\qquad \boldsymbol{\theta}_K=\operatorname{cardan}(\mathbf{A}_T^{\top}\mathbf{A}_S)
```

Joint centres are the hip, at the pelvis origin, and the knee at **o**_S = **o**_T + **A**_T (0, 0, −L_T)ᵀ.

| Angle | Mean waveform μ(t) (°); φ = 2πt/100 | Between SD σᵇ (°) | Stride SD σʷ (°) |
|---|---|---|---|
| Pelvis tilt | 10 + 1.5 cos 2φ | 2.0 | 0.7 |
| Pelvis obliquity | 4 sin(φ + 0.6) | 1.5 | 0.5 |
| Pelvis rotation | 5 cos φ | 2.5 | 1.5 |
| Hip flexion | 12 + 23 cos(φ + 0.15) | 4.0 | 1.3 |
| Hip adduction | 5 sin(φ + 0.3) − 1 | 2.5 | 0.9 |
| Hip rotation | 4 sin(φ − 0.8) | 3.0 | 1.0 |
| Knee flexion | 3 + 18 exp(−((t−15)/9)²) + 62 exp(−((t−73)/11)²) | 4.0 | 1.7 |
| Knee varus | 2 sin φ | 1.5 | 0.6 |
| Knee rotation | 5 cos(φ + 1.0) | 3.0 | 1.2 |

Bony-alignment SDs σᵃⁿᵃᵗ (x, y, z; °): pelvis (4.0, 1.0, 1.0), thigh (1.0, 1.5, 6.0), shank (1.0, 1.5, 5.0).

### 2.3 Marker-based measurement: direct kinematics

Each segment *s* carries four skin markers with nominal anatomical-frame positions **p**_j. The measured global position of marker *j* is

```math
\mathbf{y}_{ikj}(t)=\mathbf{o}_{s}+\mathbf{A}_{s}\Big[\mathbf{p}_j+\boldsymbol{\epsilon}_{ij}+\mathbf{g}_{ij}\,d_{s}(t)+\sigma_{ss}\,\mathbf{s}_{ikj}(t)\Big]+\boldsymbol{\eta}
```

- **ε**_ij ~ N(0, σ²_place **I**) is landmark placement error, fixed for a participant.
- **g**_ij ~ N(0, σ²_g,s **I**) is a participant-specific soft-tissue artefact gain. The driver d_s(t) is the functional hip flexion divided by 30° for the pelvis, and the functional knee flexion divided by 60° for the thigh and shank.
- **s**_ikj(t) is a stride-specific smooth artefact (H = 6, no constant term).
- **η** ~ N(0, σ²_η **I**) is measurement noise.

Marker trajectories are low-pass filtered (Gaussian kernel, SD 2.5 nodes, approximating a 6 Hz cut-off). Each segment's orientation is then estimated independently by a least-squares rigid fit (Kabsch/SVD) of the measured markers to the nominal template:

```math
\hat{\mathbf{R}}_{s}=\arg\min_{\mathbf{R}\in SO(3)}\sum_{j}\big\lVert \mathbf{R}\,(\mathbf{p}_j-\bar{\mathbf{p}})-(\mathbf{y}_j-\bar{\mathbf{y}})\big\rVert^2
```

Joint angles are the Cardan angles between the estimated segment orientations, as in §2.2. No joint constraint links the segments (six degrees of freedom per segment).

### 2.4 Markerless-like measurement: inverse kinematics

Each segment has four keypoints with template positions **c**_j in the segment frame (joint centres and off-axis surface points). Keypoints follow the body surface, so they may miss part of the bony alignment. The observed position is

```math
\mathbf{k}_{ikj}(t)=\mathbf{o}_{s}+\mathbf{F}_{s}\,\mathbf{D}_{s}^{\,1-\alpha}\,\mathbf{c}_j+\mathbf{b}_{ij}+\sigma_k\,\mathbf{n}_{ikj}(t)
```

- α is **template blindness**: α = 1 means the keypoints carry no information about a participant's bony alignment; α = 0 means they carry all of it. **D**^(1−α) is the rotation with rotation vector (1 − α)**ρ**.
- **b**_ij ~ N(0, σ²_b **I**) is a participant-level keypoint bias.
- **n**_ikj(t) is temporally smooth keypoint noise (H = 5, no constant term).

A template model with fixed joint centres is fitted to the keypoints at each node. Its generalised coordinates are **x** = (**t**, **q**_P, **q**_H, **q**_K) ∈ ℝ¹², where **t** is the pelvis translation. Model segment lengths are the true lengths multiplied by (1 + ζ), with ζ ~ N(0, σ²_scale) drawn once per participant and segment. The fit minimises

```math
J(\mathbf{x})=\sum_{j=1}^{12}\frac{\lVert \hat{\mathbf{k}}_j(\mathbf{x})-\mathbf{k}_j\rVert^2}{\sigma_k^2}+\frac{q_{K,varus}^2+q_{K,rot}^2}{\sigma_p^2}
```

where the model keypoints follow from forward kinematics (§2.1). The second term is a **model prior** that pulls knee varus and knee rotation toward the template value of zero; σ_p = ∞ removes it. The problem is solved by Levenberg–Marquardt (12 iterations, finite-difference Jacobian, damping 10⁻⁶ × mean diagonal of the Gauss–Newton matrix), started from the population mean pose. The reported angles are the fitted **q**, smoothed with a Gaussian kernel (SD 0.8 nodes, approximating a 20 Hz cut-off).

### 2.5 Measurement parameters

| Parameter | Symbol | Value |
|---|---|---|
| **Marker-based** | | |
| Landmark placement SD | σ_place | 8 mm per axis |
| Soft-tissue artefact gain SD | σ_g | pelvis 4 mm, thigh 10 mm, shank 4 mm (at full driver excursion) |
| Stride-specific artefact SD | σ_ss | 1.5 mm |
| Measurement noise SD | σ_η | 0.3 mm |
| Marker filter | | Gaussian, SD 2.5 nodes |
| **Markerless-like** | | |
| Keypoint noise SD | σ_k | 2 mm |
| Participant keypoint bias SD | σ_b | 4 mm |
| Segment-length scaling error SD | σ_scale | 2% |
| Template blindness | α | 1 (0 in ablations) |
| Prior SD on knee varus and rotation | σ_p | 4° (∞ in ablations) |
| Angle filter | | Gaussian, SD 0.8 nodes |

Marker and keypoint template coordinates are listed in `simulate_systems.py` (`MARKERS`, `KEYPTS`). Off-axis lever arms are 35–120 mm, so axial rotations are the least well determined in both pipelines.

The placement and soft-tissue values are of the order reported for palpation error and skin-marker movement in the lower limb, but they were not taken from specific sources and should be checked against the literature (e.g. Della Croce et al., 2005; Leardini et al., 2005) or calibrated before publication. σ_k and σ_η were set by hand so that stride-level SDs were of realistic size. Gait-event detection, a foot segment, gross errors (marker swaps) and camera geometry are not modelled.

### 2.6 Scenarios

| Scenario | Marker-based errors | Markerless-like errors |
|---|---|---|
| 1 Full | placement, soft tissue, noise | keypoint noise and bias, scaling, α = 1, σ_p = 4° |
| 2 MB placement only | placement, noise | none (α = 0, σ_p = ∞) |
| 3 MB soft tissue only | soft tissue, noise | none |
| 4 ML template blindness only | noise | α = 1 |
| 5 ML model prior only | noise | σ_p = 4° |
| 6 ML keypoint noise only | noise | keypoint noise and bias, scaling |

All scenarios use the same true kinematics (random seed 1) and the same error seed. With all error sources off and no filtering, both pipelines reproduce the true angles to within 10⁻¹²°.

### 2.7 Outcome measures

Outcomes are computed for each angle with the functions used for the experimental data (`variance_structure.py`). For system *s*, at each node, with n strides per participant:

```math
\sigma^2_{w,s}=\frac{1}{N}\sum_i \operatorname{var}_k\big(Y_{sik}\big),\qquad \sigma^2_{b,s}=\operatorname{var}_i\big(\bar{Y}_{si}\big)-\frac{\sigma^2_{w,s}}{n}
```

with cross-system covariances c_w and c_b defined in the same way. Variances are averaged over the gait cycle before ratios are taken. The outcomes are:

- **SD ratios** ML:MB at the stride and between-participant levels, and their log difference (zero for a scaled copy).
- **Between-system correlations** r_w and r_b.
- **Upper bound on the ML:MB scale**, σ²_b,ML / c_b, available from the two systems alone.
- **True scale of each system**, λ = cov(system, truth) / var(truth), at each level; λ = 1 is faithful and λ < 1 is shrinkage. This is only available because the truth is known.
- **Error SD of each system** against the truth at each level.
- **Shared stride variance** c_w compared with the true stride variance.
- **Roughness** of stride deviations (10⁴ × mean squared second difference / mean squared deviation).
