# Study 3 raw-data quality check: WU (warm-up) vs UOB, markerless, left and right

Source: `study3_UOB/20261113_WU_UOB_Import.mat` (commit `0f6633f`, with WU P021–P029 included).
Re-run everything with `python3 run_qc.py && python3 make_report_table.py > results/participant_summary.md`
(needs numpy, scipy, matplotlib).

**Data checked:** 24 participants (P005–P029; P020 is absent from both conditions) × 2 conditions × 23 variables.
That is L/R ankle, knee, hip (X/Y/Z), L/R foot progression and pelvis X/Y/Z.
The export holds 1974 WU and 765 UOB L+R strides, of which 1858 and 741 are unique.

## 1. Priority issues (fix in the export or processing before analysis)

| # | Participant / condition | Problem | Action |
|---|---|---|---|
| 1 | **WU P024** | Every stride exported **4 times** (L 1=2=3=4, 5=6=7=8 …). Only 12 L / 13 R / 13 pelvis unique strides out of 48/52/52. | Re-export (most likely each trial was added to the V3D workspace 4 times or the events were duplicated). |
| 2 | **WU P011, UOB P011** | Many duplicate strides, some repeated 3–4 times. WU has 31/46 L, 33/48 R and 29/42 pelvis unique; UOB has 16/23 L and 17/23 R. | Re-export or check events. |
| 3 | **P013, UOB and WU** | **The same trial is in both conditions.** UOB P013 strides 15–16 are identical to WU P013 strides 1–2 in all 23 files. | Remove the trial from the folder it doesn't belong to (probably UOB). |
| 4 | **P005, both conditions** | **Different model.** Knee rotation is exactly 0 for the other 23 participants (a constrained DOF) but has 4–20° ROM for P005. P005's ankle rotation is also offset +8–10° from the group in both conditions. | Re-process P005 with the same model/settings as the others and re-check all its variables. |
| 5 | **WU P023 pelvis rotation** stride 12 | 139° jump within 1% of the cycle (angle wrap or gimbal lock). It pulls the participant mean off by about 13°. | Remove the stride, or re-process with a different rotation sequence. |
| 6 | Pelvis files: **WU P011, UOB P013, WU P022, WU P029** | The pelvis stride count differs from the right side (42 vs 48, 16 vs 18, 53 vs 56, 35 vs 39), and the exported V3D SD column doesn't match the stride columns. Pelvis strides can't be paired with right strides. | Check events or the pelvis export for those trials. |
| 7 | Smaller duplicate sets | UOB P005 (L 1=2, 6=7, 10=11; R/pelvis 1=2); WU P005 (L 1=2, 3=4, 12=13, 14=15); UOB P006 L3=4; WU P006 L18=19; UOB P007 L5=6; WU P008 (L24=25, R17=18); WU P009 (L17=18, 35=36); UOB P010 (L12=13, 15=16, R11=12); UOB P015 R13=15=16; WU P015 R2=4=5. | Copies are excluded from all statistics here; remove them in the analysis too. |
| 8 | Import script (`Import_and_Organise_study3.m`) | Each file's first two columns are the V3D **mean** and **SD**, not strides. In pelvis files the mean column has already been dropped and row 1 is all zeros (102 rows). | Drop columns 1–2 (and row 1 for pelvis) before any analysis. `s3common.load()` already does this. |
| 9 | Knee rotation (all participants) | No signal (constrained DOF). | Leave this variable out of Study 3. |

## 2. Participant-level flags (check, may be genuine)

* **UOB P028:** very high stride-to-stride variability in 10 variables (early-stance knee flexion ranges 4–19°). The WU P028 mean is only the 17th-closest match to UOB P028, while every other participant matches their own ID (P009/P010 rank 3).
  This looks like poor-quality UOB data for P028 rather than a label swap. Confirm the participant label anyway.
* **P019 right foot progression:** offset −11° (UOB) and −13° (WU) from the group. WU P019 right hip and ankle rotation are also offset. Because it's consistent across conditions, it's likely the static or segment definition, or genuine.
* **WU P026:** left hip adduction +10° vs the right side (group median L–R RMSD 1.6°), plus a pelvis obliquity offset. Check the WU model fit; UOB P026 is fine.
* **P010 pelvis rotation:** WU−UOB offset of +9°, and high variability in UOB (also UOB P014).
* **P014 knee flexion:** WU vs UOB differ more than for other participants (RMSD 5–6°).
* **P013 knee varus:** offset about −4.5° in both conditions (consistent, so probably the model or genuine).
* **WU P023:** high pelvis variability in all three planes (see priority issue 5).

## 3. Stride outliers

Each participant × condition × variable was compared with the participant median curve (exact copies excluded).
A stride is flagged when its RMSD is a robust z > 4 above the other strides **and** > 3° (and > 2× the participant's median).
A flag is **major** when the RMSD is > 8°.
The flags are then combined across the variables of each side in `results/stride_recommendations.csv`:

* **REMOVE stride from all variables**: flagged in ≥ 3 variables, or RMSD > 15° in any one. That's 101 WU and 13 UOB L/R strides (about 5% of WU and 2% of UOB).
* **remove from flagged variable(s) / inspect**: one or two variables with a major flag. That's 32 WU and 31 UOB.
* **inspect**: minor flags only. That's 130 WU and 63 UOB.

WU has many more flags because it has 2–4× more strides per participant and visibly more heterogeneous strides.
Several participants have 2 clusters of curves (e.g. a pelvis tilt offset of about 5° between subsets of strides), consistent with different trials or walking directions.
About a quarter of the WU removals are the first or last stride of the exported set, so gait initiation/termination or the capture-volume edge are likely sources.
Trial identities aren't in the export, so this can't be confirmed here. Exporting the trial name per stride would make this check much sharper.

These thresholds are a screening tool. **Confirm each removal on the participant pages before excluding it.**

## 4. Figures (in `figures/`)

* `subjects_WU.pdf` and `subjects_UOB.pdf`: one page per participant, 13 joints. Every left stride is red and every right stride blue, pelvis strides grey.
  The participant mean is drawn with L solid and R dashed, on the condition's group mean ± 1 SD band. Flagged strides are bold and numbered (L12, R5, …), and duplicates are dotted.
* `subjects_WU_vs_UOB.pdf`: one page per participant, WU vs UOB participant mean ± SD (L solid, R dashed). Panel titles are red where WU≠UOB was flagged.
* `overview_WU.jpg` and `overview_UOB.jpg`: every participant mean on the group band, with outlying participants bold and labelled.
* `group_means_WU_vs_UOB.jpg`: group mean ± SD for WU vs UOB, left and right, in the same style as `python_analysis/figures/means_sd.jpg`.
* `flagged_strides.pdf`: a close-up of every variable with a major stride flag, in the same layout as `flagged_trials_visual_check.jpg`. It shows unflagged strides, flagged strides (numbered), the mean, and the mean with the flagged strides removed.

## 5. Result tables (in `results/`)

* `qc_issues.csv` has one row per issue with these columns: condition, subject, side, joint, file, stride, check (structure / duplicate / model / stride / participant / LR / identity / WU_UOB), severity (major / minor / info), metric, value and description. Sort or filter it in Excel.
* `stride_recommendations.csv` has one row per flagged stride: the variables it is flagged in (with RMSD) and the recommendation.
* `stride_metrics.csv` has every stride's RMSD from the participant median, its robust z, max jump and duplicate status.
* `stride_counts.csv` has strides per participant × condition × file, exported vs unique.
* `participant_summary.md` is the table below.

Stride numbers are 1-based and count strides only (the V3D mean/SD columns are removed).
Stride *k* is column *k* + 2 of the imported `rawdata{kk}` array, or column *k* + 1 for pelvis files.

## 6. Per-participant summary

| Participant | Cond | Strides L, R, pelvis (unique/exported) | Strides to remove from all variables | Participant-level flags |
|---|---|---|---|---|
| P005 | UOB | 11/14, 13/14, 13/14 | – | **different model (knee rot)**; duplicate strides (left); duplicate strides (pelvis); duplicate strides (right); mean outlier: Ankle Rotation (L); mean outlier: Ankle Rotation (R) |
| P005 | WU | 22/26, 26, 26 | – | **different model (knee rot)**; duplicate strides (left); mean outlier: Ankle Rotation (L); mean outlier: Ankle Rotation (R) |
| P006 | UOB | 13/14, 14, 14 | R10 | duplicate strides (left); mean outlier: Ankle Dorsiflexion (R) |
| P006 | WU | 23/24, 24, 24 | – | duplicate strides (left); mean outlier: Ankle Dorsiflexion (L) |
| P007 | UOB | 12/13, 13, 13 | – | duplicate strides (left) |
| P007 | WU | 35, 35, 35 | L1, L10 | – |
| P008 | UOB | 15, 15, 15 | L15 | – |
| P008 | WU | 47/48, 48/49, 48/49 | R1, R9, R36, P12 | duplicate strides (left); duplicate strides (pelvis); duplicate strides (right) |
| P009 | UOB | 11, 11, 11 | – | high variability: Knee Varus (L); high variability: Knee Varus (R) |
| P009 | WU | 41/43, 43, 43 | – | duplicate strides (left) |
| P010 | UOB | 14/16, 15/16, 15/16 | L12, P7, P11, P13 | WU≠UOB: Pelvis Rotation; duplicate strides (left); duplicate strides (pelvis); duplicate strides (right); high variability: Ankle Dorsiflexion (R); high variability: Foot Progression (R); high variability: Pelvis Rotation; mean outlier: Pelvis Rotation |
| P010 | WU | 36, 36, 36 | L36, R9, R19, R25, R36, P25 | WU≠UOB: Pelvis Rotation |
| P011 | UOB | 16/23, 17/23, 17/23 | R1 | duplicate strides (left); duplicate strides (pelvis); duplicate strides (right) |
| P011 | WU | 31/46, 33/48, 29/42 | R6, R26, R27 | duplicate strides (left); duplicate strides (pelvis); duplicate strides (right); pelvis export mismatch |
| P012 | UOB | 15, 15, 15 | R6 | – |
| P012 | WU | 38, 38, 38 | L19, L20, R19, R34, P19 | – |
| P013 | UOB | 18, 18, 16 | P7 | **copied trial (both conds)**; mean outlier: Knee Varus (L); mean outlier: Knee Varus (R); pelvis export mismatch |
| P013 | WU | 55, 55, 55 | L1, L47, L55, R1, R4, R16, R32, R55 | **copied trial (both conds)**; mean outlier: Knee Varus (R) |
| P014 | UOB | 19, 19, 19 | L17, P16, P17, P18 | WU≠UOB: Knee Flexion (L); WU≠UOB: Knee Flexion (R); high variability: Foot Progression (R); high variability: Pelvis Rotation |
| P014 | WU | 59, 58, 58 | L1, L2, L8, L21, L43, R1, R8, R21, R28, R52, P52 | WU≠UOB: Knee Flexion (L); WU≠UOB: Knee Flexion (R) |
| P015 | UOB | 15, 15/17, 15/17 | L7, L15 | duplicate strides (pelvis); duplicate strides (right) |
| P015 | WU | 58, 59/61, 59/61 | L1, L2, L52, R1, R2, R3, R7, R31, P3 | duplicate strides (pelvis); duplicate strides (right); high variability: Knee Flexion (L) |
| P016 | UOB | 19, 19, 19 | – | L-R asymmetry: Knee Flexion |
| P016 | WU | 58, 57, 57 | L10, L27, L45, L58, R26, R44, R57, P26, P27 | high variability: Pelvis Rotation |
| P017 | UOB | 16, 16, 16 | – | – |
| P017 | WU | 66, 66, 66 | L9, L22, R9, P10 | – |
| P018 | UOB | 14, 14, 14 | R10 | – |
| P018 | WU | 12, 12, 12 | R4, P10 | high variability: Pelvis Rotation |
| P019 | UOB | 12, 12, 12 | – | mean outlier: Foot Progression (R) |
| P019 | WU | 49, 49, 49 | L28, L29, R18, R29, R30, P29, P30 | mean outlier: Ankle Rotation (R); mean outlier: Foot Progression (R); mean outlier: Hip Rotation (R) |
| P021 | UOB | 19, 19, 19 | P12 | – |
| P021 | WU | 46, 46, 46 | L18, L23, L41, R35, R37, R40, R41, P11, P46 | – |
| P022 | UOB | 17, 17, 17 | L14, R1, P17 | – |
| P022 | WU | 56, 56, 53 | L11, L18, L46, R56 | pelvis export mismatch |
| P023 | UOB | 15, 15, 15 | L1, R2 | high variability: Ankle Dorsiflexion (R); high variability: Hip Adduction (R); high variability: Hip Rotation (L) |
| P023 | WU | 16, 16, 16 | L12, R12, P8, P11, P12 | high variability: Foot Progression (R); high variability: Pelvis Anterior Tilt; high variability: Pelvis Obliquity; high variability: Pelvis Rotation; mean outlier: Pelvis Rotation |
| P024 | UOB | 20, 20, 20 | – | – |
| P024 | WU | 12/48, 13/52, 13/52 | L9, R9 | duplicate strides (left); duplicate strides (pelvis); duplicate strides (right); high variability: Foot Progression (L); mean outlier: Knee Flexion (R) |
| P025 | UOB | 15, 16, 16 | – | – |
| P025 | WU | 38, 38, 38 | L1, L5, R1, R5, R32, R38 | – |
| P026 | UOB | 15, 15, 15 | – | – |
| P026 | WU | 23, 23, 23 | L1, L15, L16, R1, R16, R21 | L-R asymmetry: Hip Adduction; mean outlier: Hip Adduction (L); mean outlier: Pelvis Obliquity |
| P027 | UOB | 16, 16, 16 | – | – |
| P027 | WU | 38, 38, 38 | L20, L38, R20, R25, R34, R38, P20 | – |
| P028 | UOB | 14, 14, 14 | P6, P7 | WU/UOB ID mismatch; WU≠UOB: Ankle Dorsiflexion (L); WU≠UOB: Ankle Dorsiflexion (R); WU≠UOB: Ankle Rotation (R); WU≠UOB: Hip Flexion (L); WU≠UOB: Hip Flexion (R); WU≠UOB: Knee Flexion (R); high variability: Ankle Dorsiflexion (L); high variability: Ankle Dorsiflexion (R); high variability: Foot Progression (R); high variability: Hip Flexion (L); high variability: Hip Flexion (R); high variability: Knee Flexion (L); high variability: Knee Flexion (R); high variability: Knee Varus (R); high variability: Pelvis Anterior Tilt; mean outlier: Ankle Dorsiflexion (L); mean outlier: Ankle Dorsiflexion (R) |
| P028 | WU | 26, 26, 26 | L1, L8, L12, R12 | WU/UOB ID mismatch; WU≠UOB: Ankle Dorsiflexion (L); WU≠UOB: Ankle Dorsiflexion (R); WU≠UOB: Ankle Rotation (R); WU≠UOB: Hip Flexion (L); WU≠UOB: Hip Flexion (R); WU≠UOB: Knee Flexion (R); high variability: Ankle Dorsiflexion (L); high variability: Hip Flexion (L); high variability: Knee Flexion (L) |
| P029 | UOB | 16, 16, 16 | – | – |
| P029 | WU | 39, 39, 35 | L6, R6, R16, R17, R39 | pelvis export mismatch |
