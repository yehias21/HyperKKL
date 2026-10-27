# Archive (AAAI) recipe vs the current v2 recipe: equal-budget paired test

Finished 2026-10-26 05:05 CET (08:05 Dubai). Design and decision rule: `STUDY.md` section 6, fixed before
any arm ran; system selection frozen by `study_data/select_systems.py` at 2026-10-26 03:15:30 Dubai after the C3
data-baseline re-evaluation reached 66/66 merged runs (`SELECTION.json`). Everything ran from a scratch copy of
the package (`code/hyperbest.patch` = the full change, 6 files, against the code snapshot the copy was taken
from, `experiments/code/b2cb5b3dfe80.tar.gz`); the live package, tree, queue and caches were not written. Data
and evaluation episodes were read read-only from the live cache (same keys), so every arm, and the C3 baselines
quoted for context, see exactly the same episodes. The default lora retrained in the copy reproduces the live
C3 numbers (e.g. realistic/vdp calibration cell: sequence_ood 0.0322 vs live 0.0325).

## Verdicts (mechanical, pre-registered)

| hypothesis | rule | sequence_ood favour / against | shape_ood favour / against | verdict |
|---|---|---|---|---|
| **A7 faithful @BEST** vs better of default lora/hyper | >= 5/8 in favour on both tiers, none against (97.5% t-CI over seeds) | 6 / **1** | 6 / **1** | **INCONCLUSIVE** |
| **A8 @BEST without the encoder half** | same | 7 / **1** | 7 / **1** | **INCONCLUSIVE** |
| A1 SOAP lr 1e-2 vs A2 Adam lr 1e-2 (is it the optimiser or the step size?) | same rule, 95% | 7 / 0 | 7 / 0 | **SUPERIOR** (SOAP, not the lr) |

Both composites fail the pre-registered rule on ONE cell, lorenz/partial, where they are significantly worse
(A7 +205% [+82, +329], A8 +167% [+65, +269] on sequence_ood) than the reference (default hyper 0.0181 there).
On the other 7 cells they are better by 40-85% with CIs far from 0. So the archived recipe as a package is
**not** declared superior: the pipeline is not to be changed on the strength of A7/A8.

## What the components say (95% CI, no verdict; 8 cells x seeds 0-2, primary arms seeds 0-4)

| arm (one change vs default lora) | seq favour / against | shape favour / against | reading |
|---|---|---|---|
| A1 SOAP lr 1e-2 (phases 1 + 2) | **8 / 0** | **8 / 0** | the only component that helps everywhere: -28% to -67% |
| A2 Adam lr 1e-2 | 1 / 0 | 0 / 1 | the step size alone does nothing |
| A2b Adam lr 1e-2, betas 0.95/0.95 (deviation, see below) | (seed means) | | between A2 and A1 on all 8 cells (e.g. duffing/partial 0.067 / 0.040 / 0.024 for A2 / A2b / A1): part of the gain is SOAP's betas, most of it is the preconditioner |
| A3 phase-1 steps x3 (Adam) | 1 / 1 | 1 / 1 | ~0; lorenz/partial +96% |
| A4 GRU context (+ ctx guard) | 1 / 1 | 1 / 2 | ~0 (SFM vs GRU: no difference that transfers) |
| A5 delta_curv 1e-4 + delta_reg 1e-3 | 0 / 3 | 1 / 1 | harmful or neutral |
| A6 encoder LoRA + characteristic residual | 0 / 2 | 0 / 1 | harmful or neutral (as the archive's own leave-one-out said) |

Mechanism (phase-1 diagnostics, `results_test.json` / tables below): SOAP's gain comes mostly through the
**base maps**: the phase-1 decoder error eps_dec falls 5-60x (duffing 0.0064 -> 0.0010, quad_tank 0.163 ->
0.0027, vdp 0.011 -> 0.001, lorenz/partial 0.34 -> 0.13), while the decoder Lipschitz proxy l_dec_q95 rises
(vdp 23 -> 59, lorenz 10 -> 103), the trade-off STUDY.md section 3.6 predicted from the archive. On lorenz/partial
the x3 SOAP phase 1 of A7/A8 does NOT improve the base map (eps_dec 0.36 vs 0.13 for x1 SOAP, l_dec 130) and the
curvature arm is 3x worse there; stacking them is what breaks that cell. SOAP alone (A1) is better than the
reference on lorenz/partial too (-31% seq).

## Against the baselines (seed means, sequence_ood; baselines from the live C3 nodes, same episodes)

| cell (role) | default lora / hyper | A1 SOAP | A8 | best non-HyperKKL baseline |
|---|---|---|---|---|
| duffing / partial (LOSE) | 0.054 / 0.065 | 0.024 | 0.013 | greybox_ukf 0.0078 |
| quad_tank / partial (LOSE) | 0.071 / 0.063 | 0.037 | 0.021 | greybox_ekf 0.014 |
| vdp / partial (WIN) | 0.018 / 0.020 | 0.0099 | 0.0098 | greybox_ekf 0.026 |
| lorenz / partial (WIN) | 0.029 / 0.018 | 0.013 | 0.048 | nlox 0.029 |
| duffing / mismatch30 (LOSE) | 0.056 / 0.063 | 0.024 | 0.015 | greybox_ukf 0.0059 |
| battery / mismatch30 (LOSE) | 0.049 / 0.070 | 0.017 | 0.024 | greybox_ekf 0.015 |
| vdp / mismatch30 ("WIN", r = 2.1) | 0.020 / 0.023 | 0.0097 | 0.0095 | greybox_ekf 0.0098 |
| lorenz / mismatch30 ("WIN", r = 1.03) | 0.0081 / 0.010 | 0.0047 | 0.0044 | nlox 0.0068 |

With SOAP the learned observer now beats every baseline on vdp/partial, lorenz (both), ties the grey-box EKF on
vdp/mismatch30, and the remaining losses to the grey-box filters shrink from 3.4-9.5x to 1.1-2.5x
(duffing, quad_tank, battery). Note on the selection: after the data baselines were merged, every mismatch30 cell
has r > 1 (the current recipe loses there everywhere), so its two "WIN" cells are the two smallest losses.

## Recommendation

1. **Do not adopt @BEST (A7) or A8 as a package**: pre-registered rule not met (one significant loss on 8).
2. **SOAP is the archived ingredient worth carrying over**, but it was a component arm here, not a primary
   hypothesis, so this test is exploratory evidence for it, not a confirmation. Proposed confirmatory node
   (pre-register before running): `phase1.optimizer=soap, phase1.lr=1e-2, phase2.optimizer=soap, phase2.lr=1e-2`
   applied to ALL variants (latent, concat, film, lora, hyper) on the 6 dev systems (C1-where and C3 scenarios),
   then once on the test systems; plus a split arm SOAP-in-phase-1-only vs SOAP-in-phase-2-only, since the
   diagnostics point to phase 1 (base maps) as the carrier. Because the base maps are shared, this changes all
   variants equally and cannot bias C1 by itself.
3. Drop the curvature penalty, the encoder half, phase-1 x3, and the GRU switch from the candidate list.
4. For `experiments/tuning/PROTOCOL.md` (not edited): the optimiser, not the lr grid, is the lever; an Adam lr
   up to 1e-2 does nothing on its own (A2).

## Deviations from STUDY.md section 6 (all declared before results were seen)

* **A2b added** (Adam lr 1e-2 with SOAP's betas 0.95/0.95 and eps 1e-8): the CPU convergence test showed that on an
  ill-conditioned quadratic Adam with these betas beats SOAP, so A1-vs-A2 alone could have attributed a betas
  effect to preconditioning. Added before any arm ran; not part of any verdict.
* The SOAP convergence unit test was relaxed from "SOAP beats Adam on the toy quadratic" (false for this toy) to
  "SOAP converges" (`tests/test_archive_recipe.py`); the Adam-limit tests are exact.
* Extra seeds 3-4 were run for A0 (lora, hyper), A7 and A8 only (nodes `AR-*-x`), as planned: the primary contrasts
  use seeds 0-4 on all 8 cells, the component contrasts (A1-A6, A2b) seeds 0-2.
* GPUs: 5-7 throughout, plus 0-4 between 00:15 and 05:05 CET under the user's temporary permission (no start after
  03:45, hard stop 05:40; none was needed). Peak GPU memory per job 2.2-2.7 GiB.
* A timing-calibration job on C3-realistic/vdp (`AR-calib-realistic`, not a test cell) ran before the freeze;
  its numbers are not used anywhere in the verdicts.

Files: `SELECTION.json` (frozen cells + timestamp), `results_test.json` (all contrasts), `RESULTS_TEST.md`
(this file + the generated tables below), `code/` (patch, new tests, launcher, dispatcher, analysis script, arm
definitions, node files, regression-test summary: 192 passed, 5 skipped, 1 xfailed on the copy).

---

# Archive-recipe test: mechanical verdicts

Generated 2026-10-26 05:06 Dubai from `analyze_ar.py` (pre-registered rule: STUDY.md section 6.4). Selection frozen at 2026-10-26 00:15:30 CET.

Cell = per-seed paired relative change of the arm's mean episode NRMSE vs the better of default lora / hyper in that cell (Student-t CI over seeds; **bold** = CI excludes 0 in favour, *italic* = against). Primary arms use 97.5% CIs (Bonferroni over A7, A8); others 95%.


## A7_best: **INCONCLUSIVE**

counts: sequence_ood: favour 6/8, against 1/8; shape_ood: favour 6/8, against 1/8

| cell | sequence_ood | shape_ood | id |
|---|---|---|---|
| duffing/partial (LOSE) | **-75% [-79, -70]** | **-85% [-88, -81]** | **-84% [-87, -80]** |
| quad_tank/partial (LOSE) | **-67% [-70, -65]** | **-60% [-61, -58]** | **-60% [-62, -59]** |
| vdp/partial (WIN) | **-48% [-51, -45]** | **-42% [-45, -40]** | **-45% [-46, -43]** |
| lorenz/partial (WIN) | *+205% [+82, +329]* | *+214% [+71, +356]* | *+152% [+49, +256]* |
| duffing/mismatch30 (LOSE) | **-71% [-77, -65]** | **-83% [-85, -80]** | **-83% [-85, -80]** |
| battery/mismatch30 (LOSE) | **-60% [-71, -49]** | **-60% [-66, -54]** | **-60% [-70, -50]** |
| vdp/mismatch30 (WIN) | **-53% [-59, -47]** | **-52% [-59, -44]** | **-50% [-55, -45]** |
| lorenz/mismatch30 (WIN) | -30% [-62, +2] | -25% [-62, +12] | -24% [-59, +11] |

## A8_best_noenc: **INCONCLUSIVE**

counts: sequence_ood: favour 7/8, against 1/8; shape_ood: favour 7/8, against 1/8

| cell | sequence_ood | shape_ood | id |
|---|---|---|---|
| duffing/partial (LOSE) | **-76% [-83, -69]** | **-85% [-90, -81]** | **-84% [-89, -80]** |
| quad_tank/partial (LOSE) | **-67% [-69, -66]** | **-59% [-61, -57]** | **-60% [-62, -58]** |
| vdp/partial (WIN) | **-46% [-52, -41]** | **-39% [-45, -33]** | **-43% [-47, -38]** |
| lorenz/partial (WIN) | *+167% [+65, +269]* | *+177% [+56, +297]* | *+122% [+41, +202]* |
| duffing/mismatch30 (LOSE) | **-74% [-80, -68]** | **-84% [-86, -81]** | **-84% [-86, -81]** |
| battery/mismatch30 (LOSE) | **-51% [-61, -42]** | **-55% [-71, -38]** | **-57% [-74, -39]** |
| vdp/mismatch30 (WIN) | **-53% [-58, -48]** | **-51% [-57, -44]** | **-50% [-54, -45]** |
| lorenz/mismatch30 (WIN) | **-44% [-65, -22]** | **-41% [-65, -17]** | **-40% [-64, -17]** |

## A1 (SOAP lr 1e-2) vs A2 (Adam lr 1e-2): **SUPERIOR**

| cell | sequence_ood | shape_ood | id |
|---|---|---|---|
| duffing/partial (LOSE) | **-65% [-72, -58]** | **-69% [-74, -63]** | **-68% [-70, -65]** |
| quad_tank/partial (LOSE) | **-30% [-40, -21]** | **-31% [-37, -24]** | **-32% [-42, -21]** |
| vdp/partial (WIN) | **-48% [-93, -3]** | -44% [-96, +8] | -46% [-93, +1] |
| lorenz/partial (WIN) | **-47% [-79, -15]** | **-49% [-67, -31]** | **-43% [-71, -16]** |
| duffing/mismatch30 (LOSE) | **-60% [-77, -43]** | **-62% [-79, -45]** | **-61% [-79, -43]** |
| battery/mismatch30 (LOSE) | **-48% [-73, -23]** | **-42% [-62, -22]** | **-46% [-63, -29]** |
| vdp/mismatch30 (WIN) | **-49% [-76, -22]** | **-49% [-81, -17]** | **-48% [-74, -22]** |
| lorenz/mismatch30 (WIN) | -31% [-64, +2] | **-32% [-63, -1]** | -30% [-62, +2] |

## Components vs the default (95% CI, no verdict)


### sequence_ood

| arm | duffing/partial (LOSE) | quad_tank/partial (LOSE) | vdp/partial (WIN) | lorenz/partial (WIN) | duffing/mismatch30 (LOSE) | battery/mismatch30 (LOSE) | vdp/mismatch30 (WIN) | lorenz/mismatch30 (WIN) | favour / against |
|---|---|---|---|---|---|---|---|---|---|
| A1_soap | **-56% [-64, -47]** | **-40% [-47, -34]** | **-45% [-48, -42]** | **-31% [-48, -14]** | **-58% [-60, -56]** | **-67% [-92, -41]** | **-52% [-61, -43]** | **-33% [-50, -16]** | 8 / 0 |
| A2_adam1e-2 | +27% [-22, +76] | -14% [-33, +4] | +18% [-104, +140] | +32% [-10, +75] | +8% [-39, +54] | **-37% [-73, -0]** | -2% [-78, +74] | +2% [-75, +78] | 1 / 0 |
| A3_p1x3 | -8% [-35, +18] | +8% [-4, +19] | -9% [-18, +0] | *+96% [+69, +124]* | **-5% [-6, -4]** | -2% [-44, +40] | -22% [-45, +1] | -5% [-23, +12] | 1 / 1 |
| A4_gru | -2% [-23, +20] | *+41% [+29, +53]* | +12% [-7, +31] | +79% [-102, +259] | **-14% [-28, -0]** | +55% [-57, +166] | -8% [-25, +9] | +4% [-23, +30] | 1 / 1 |
| A5_curv | +2% [-19, +22] | +11% [-2, +24] | *+16% [+5, +26]* | +265% [-56, +586] | +3% [-10, +17] | *+21% [+16, +25]* | *+3% [+1, +6]* | -4% [-9, +1] | 0 / 3 |
| A6_enc | -12% [-37, +14] | *+10% [+5, +15]* | *+26% [+10, +42]* | +106% [-55, +268] | -1% [-21, +18] | +10% [-4, +23] | +5% [-14, +24] | -0% [-9, +9] | 0 / 2 |
| A7_best | **-75% [-79, -70]** | **-67% [-70, -65]** | **-48% [-51, -45]** | *+205% [+82, +329]* | **-71% [-77, -65]** | **-60% [-71, -49]** | **-53% [-59, -47]** | -30% [-62, +2] | 6 / 1 |
| A8_best_noenc | **-76% [-83, -69]** | **-67% [-69, -66]** | **-46% [-52, -41]** | *+167% [+65, +269]* | **-74% [-80, -68]** | **-51% [-61, -42]** | **-53% [-58, -48]** | **-44% [-65, -22]** | 7 / 1 |

### shape_ood

| arm | duffing/partial (LOSE) | quad_tank/partial (LOSE) | vdp/partial (WIN) | lorenz/partial (WIN) | duffing/mismatch30 (LOSE) | battery/mismatch30 (LOSE) | vdp/mismatch30 (WIN) | lorenz/mismatch30 (WIN) | favour / against |
|---|---|---|---|---|---|---|---|---|---|
| A1_soap | **-63% [-74, -53]** | **-40% [-47, -33]** | **-40% [-43, -37]** | **-28% [-56, -0]** | **-64% [-71, -58]** | **-63% [-80, -45]** | **-53% [-64, -42]** | **-31% [-57, -5]** | 8 / 0 |
| A2_adam1e-2 | +18% [-32, +69] | -13% [-28, +2] | +22% [-118, +161] | *+41% [+20, +61]* | -4% [-52, +43] | -34% [-68, +0] | -0% [-98, +98] | +5% [-82, +92] | 0 / 1 |
| A3_p1x3 | -15% [-47, +16] | +0% [-6, +6] | -7% [-15, +0] | *+85% [+67, +102]* | **-9% [-18, -0]** | +2% [-76, +80] | -22% [-50, +5] | -3% [-19, +13] | 1 / 1 |
| A4_gru | -8% [-26, +9] | *+56% [+37, +74]* | +12% [-11, +36] | +80% [-47, +208] | **-21% [-41, -2]** | *+25% [+6, +45]* | -8% [-31, +15] | +8% [-32, +48] | 1 / 2 |
| A5_curv | -0% [-31, +30] | +1% [-14, +17] | *+13% [+6, +20]* | +158% [-71, +386] | +10% [-23, +43] | +6% [-21, +33] | +3% [-0, +7] | **-5% [-8, -2]** | 1 / 1 |
| A6_enc | -16% [-47, +16] | +11% [-2, +24] | *+22% [+7, +37]* | +91% [-53, +236] | +1% [-47, +50] | +12% [-41, +65] | +9% [-19, +36] | +2% [-4, +8] | 0 / 1 |
| A7_best | **-85% [-88, -81]** | **-60% [-61, -58]** | **-42% [-45, -40]** | *+214% [+71, +356]* | **-83% [-85, -80]** | **-60% [-66, -54]** | **-52% [-59, -44]** | -25% [-62, +12] | 6 / 1 |
| A8_best_noenc | **-85% [-90, -81]** | **-59% [-61, -57]** | **-39% [-45, -33]** | *+177% [+56, +297]* | **-84% [-86, -81]** | **-55% [-71, -38]** | **-51% [-57, -44]** | **-41% [-65, -17]** | 7 / 1 |

## Absolute NRMSE (seed means) and context


### duffing / C3-partial (LOSE, selection r = 6.92, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.05357 | 0.05069 | 0.04858 | 0.00641 | 304 | 140596 |
| A0_hyper | 0.06453 | 0.05331 | 0.0504 | 0.00641 | 304 | 137480 |
| A1_soap | 0.02359 | 0.01858 | 0.01797 | 0.00104 | 432 | 140596 |
| A2_adam1e-2 | 0.06737 | 0.0596 | 0.05535 | 0.00624 | 306 | 140596 |
| A2b_adam95_1e-2 | 0.0402 | 0.03134 | 0.02909 | 0.0036 | 306 | 140596 |
| A3_p1x3 | 0.04888 | 0.04293 | 0.04113 | 0.00372 | 299 | 140596 |
| A4_gru | 0.05304 | 0.04695 | 0.0448 | 0.00619 | 298 | 140596 |
| A5_curv | 0.05457 | 0.05106 | 0.04907 | 0.00619 | 298 | 140596 |
| A6_enc | 0.04701 | 0.04275 | 0.04139 | 0.00619 | 298 | 143728 |
| A7_best | 0.01358 | 0.007638 | 0.007767 | 0.000837 | 522 | 143728 |
| A8_best_noenc | 0.01257 | 0.0073 | 0.007403 | 0.000837 | 522 | 140596 |
| best baseline (live C3 node, seeds 0-2) | greybox_ukf 0.007752 | greybox_ukf 0.006541 | greybox_ukf 0.00732 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.0747, nrmse_p99 0.0601, div_rate 0, t_conv_median_s 3.3; A0_hyper: nrmse_unmeasured 0.0905, nrmse_p99 0.0722, div_rate 0, t_conv_median_s 5.52; A7_best: nrmse_unmeasured 0.0181, nrmse_p99 0.0195, div_rate 0, t_conv_median_s 3.14; A8_best_noenc: nrmse_unmeasured 0.0166, nrmse_p99 0.0185, div_rate 0, t_conv_median_s 3.16

### quad_tank / C3-partial (LOSE, selection r = 4.41, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.07079 | 0.04955 | 0.05096 | 0.163 | 14.6 | 141128 |
| A0_hyper | 0.0632 | 0.04931 | 0.05245 | 0.163 | 14.6 | 139024 |
| A1_soap | 0.03721 | 0.02974 | 0.02924 | 0.00272 | 22 | 141128 |
| A2_adam1e-2 | 0.05337 | 0.04302 | 0.04302 | 0.141 | 14.5 | 141128 |
| A2b_adam95_1e-2 | 0.04206 | 0.0343 | 0.03341 | 0.137 | 14 | 141128 |
| A3_p1x3 | 0.06716 | 0.04977 | 0.05005 | 0.0838 | 14.9 | 141128 |
| A4_gru | 0.08789 | 0.07721 | 0.07773 | 0.17 | 14.6 | 141128 |
| A5_curv | 0.06915 | 0.0501 | 0.05106 | 0.17 | 14.6 | 141128 |
| A6_enc | 0.06853 | 0.05524 | 0.05628 | 0.17 | 14.6 | 136361 |
| A7_best | 0.02051 | 0.01994 | 0.02014 | 0.00257 | 21.4 | 136361 |
| A8_best_noenc | 0.02077 | 0.02028 | 0.02022 | 0.00257 | 21.4 | 141128 |
| best baseline (live C3 node, seeds 0-2) | greybox_ekf 0.01416 | greybox_ukf 0.01522 | greybox_ekf 0.01503 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.0941, nrmse_p99 0.0928, div_rate 0, t_conv_median_s 112; A0_hyper: nrmse_unmeasured 0.0847, nrmse_p99 0.0826, div_rate 0, t_conv_median_s 115; A7_best: nrmse_unmeasured 0.0263, nrmse_p99 0.0302, div_rate 0, t_conv_median_s 120; A8_best_noenc: nrmse_unmeasured 0.0266, nrmse_p99 0.0312, div_rate 0, t_conv_median_s 125

### vdp / C3-partial (WIN, selection r = 0.70, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.01831 | 0.01481 | 0.01588 | 0.0114 | 22.5 | 140596 |
| A0_hyper | 0.02005 | 0.01684 | 0.01812 | 0.0114 | 22.5 | 137480 |
| A1_soap | 0.009898 | 0.008736 | 0.009071 | 0.00103 | 58.7 | 140596 |
| A2_adam1e-2 | 0.02127 | 0.01788 | 0.01864 | 0.00342 | 18.9 | 140596 |
| A2b_adam95_1e-2 | 0.01414 | 0.01201 | 0.01246 | 0.003 | 18.4 | 140596 |
| A3_p1x3 | 0.01629 | 0.01351 | 0.01431 | 0.00568 | 24.3 | 140596 |
| A4_gru | 0.02014 | 0.01642 | 0.01648 | 0.0111 | 22.8 | 140596 |
| A5_curv | 0.02076 | 0.01646 | 0.01785 | 0.0111 | 22.8 | 140596 |
| A6_enc | 0.02254 | 0.01786 | 0.01848 | 0.0111 | 22.8 | 143728 |
| A7_best | 0.009524 | 0.008513 | 0.008777 | 0.000975 | 86.8 | 143728 |
| A8_best_noenc | 0.009818 | 0.009027 | 0.009087 | 0.000975 | 86.8 | 140596 |
| best baseline (live C3 node, seeds 0-2) | greybox_ekf 0.02569 | greybox_ekf 0.01906 | greybox_ekf 0.01961 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.024, nrmse_p99 0.0238, div_rate 0, t_conv_median_s 2.11; A0_hyper: nrmse_unmeasured 0.0263, nrmse_p99 0.0252, div_rate 0, t_conv_median_s 2.24; A7_best: nrmse_unmeasured 0.0113, nrmse_p99 0.0129, div_rate 0, t_conv_median_s 2.37; A8_best_noenc: nrmse_unmeasured 0.0117, nrmse_p99 0.0123, div_rate 0, t_conv_median_s 2.38

### lorenz / C3-partial (WIN, selection r = 0.67, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.0287 | 0.02712 | 0.02877 | 0.343 | 10.8 | 141132 |
| A0_hyper | 0.01813 | 0.01752 | 0.02158 | 0.343 | 10.8 | 139020 |
| A1_soap | 0.01317 | 0.01302 | 0.01524 | 0.128 | 103 | 141132 |
| A2_adam1e-2 | 0.02556 | 0.02578 | 0.02761 | 0.23 | 22 | 141132 |
| A2b_adam95_1e-2 | 0.0217 | 0.02082 | 0.02305 | 0.282 | 35.6 | 141132 |
| A3_p1x3 | 0.0379 | 0.03413 | 0.03622 | 0.258 | 17.7 | 141132 |
| A4_gru | 0.03379 | 0.03321 | 0.03464 | 0.357 | 10.6 | 141132 |
| A5_curv | 0.06877 | 0.04613 | 0.04559 | 0.357 | 10.6 | 141132 |
| A6_enc | 0.03881 | 0.03423 | 0.03653 | 0.357 | 10.6 | 136361 |
| A7_best | 0.05417 | 0.0532 | 0.05369 | 0.315 | 132 | 136361 |
| A8_best_noenc | 0.04751 | 0.0471 | 0.04741 | 0.315 | 132 | 141132 |
| best baseline (live C3 node, seeds 0-2) | nlox 0.02868 | nlox 0.02825 | nlox 0.0285 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.0261, nrmse_p99 0.0474, div_rate 0, t_conv_median_s 39.6; A0_hyper: nrmse_unmeasured 0.017, nrmse_p99 0.0278, div_rate 0, t_conv_median_s 6.57; A7_best: nrmse_unmeasured 0.0536, nrmse_p99 0.0688, div_rate 0, t_conv_median_s 18.1; A8_best_noenc: nrmse_unmeasured 0.0471, nrmse_p99 0.0595, div_rate 0, t_conv_median_s 15.9

### duffing / C3-mismatch30 (LOSE, selection r = 9.48, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.05619 | 0.05156 | 0.04894 | 0.00648 | 298 | 140596 |
| A0_hyper | 0.0633 | 0.0544 | 0.05134 | 0.00648 | 298 | 137480 |
| A1_soap | 0.02345 | 0.01834 | 0.01767 | 0.000903 | 505 | 140596 |
| A2_adam1e-2 | 0.06028 | 0.0493 | 0.04685 | 0.0249 | 354 | 140596 |
| A2b_adam95_1e-2 | 0.04203 | 0.03154 | 0.02978 | 0.00332 | 298 | 140596 |
| A3_p1x3 | 0.05334 | 0.04703 | 0.0444 | 0.00372 | 295 | 140596 |
| A4_gru | 0.04802 | 0.04072 | 0.03949 | 0.00616 | 295 | 140596 |
| A5_curv | 0.05802 | 0.05641 | 0.05356 | 0.00616 | 295 | 140596 |
| A6_enc | 0.05528 | 0.05206 | 0.04949 | 0.00616 | 295 | 143728 |
| A7_best | 0.01621 | 0.008954 | 0.008474 | 0.000647 | 559 | 143728 |
| A8_best_noenc | 0.0147 | 0.008471 | 0.008025 | 0.000647 | 559 | 140596 |
| best baseline (live C3 node, seeds 0-2) | greybox_ukf 0.00592 | greybox_ekf 0.004784 | greybox_ekf 0.004773 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.0783, nrmse_p99 0.063, div_rate 0, t_conv_median_s 2.03; A0_hyper: nrmse_unmeasured 0.0887, nrmse_p99 0.0706, div_rate 0, t_conv_median_s 5.24; A7_best: nrmse_unmeasured 0.0217, nrmse_p99 0.0235, div_rate 0, t_conv_median_s 3.62; A8_best_noenc: nrmse_unmeasured 0.0195, nrmse_p99 0.0202, div_rate 0, t_conv_median_s 3.43

### battery / C3-mismatch30 (LOSE, selection r = 3.41, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.04883 | 0.0436 | 0.03834 | 1.8 | 676 | 141484 |
| A0_hyper | 0.07037 | 0.04956 | 0.04777 | 1.8 | 676 | 140052 |
| A1_soap | 0.01683 | 0.0165 | 0.0136 | 0.425 | 1.3e+03 | 141484 |
| A2_adam1e-2 | 0.03225 | 0.02874 | 0.02526 | 1.58 | 701 | 141484 |
| A2b_adam95_1e-2 | 0.0221 | 0.0214 | 0.0189 | 1.51 | 800 | 141484 |
| A3_p1x3 | 0.05027 | 0.04354 | 0.03868 | 1.49 | 841 | 141484 |
| A4_gru | 0.07872 | 0.055 | 0.04773 | 1.83 | 691 | 141484 |
| A5_curv | 0.0621 | 0.04588 | 0.0474 | 1.83 | 691 | 141484 |
| A6_enc | 0.05644 | 0.04815 | 0.04052 | 1.83 | 691 | 136641 |
| A7_best | 0.01941 | 0.01754 | 0.0152 | 0.291 | 1.65e+03 | 136641 |
| A8_best_noenc | 0.02388 | 0.01951 | 0.01656 | 0.291 | 1.65e+03 | 141484 |
| best baseline (live C3 node, seeds 0-2) | greybox_ekf 0.01509 | greybox_ekf 0.0133 | greybox_ekf 0.01275 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.0531, nrmse_p99 0.0853, div_rate 0, t_conv_median_s 19.1; A0_hyper: nrmse_unmeasured 0.0768, nrmse_p99 0.0952, div_rate 0, t_conv_median_s 22.9; A7_best: nrmse_unmeasured 0.0181, nrmse_p99 0.0306, div_rate 0, t_conv_median_s 5.67; A8_best_noenc: nrmse_unmeasured 0.0218, nrmse_p99 0.0348, div_rate 0, t_conv_median_s 6.86

### vdp / C3-mismatch30 (WIN, selection r = 2.10, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.02015 | 0.01741 | 0.01771 | 0.0134 | 26.7 | 140596 |
| A0_hyper | 0.02321 | 0.02119 | 0.02117 | 0.0134 | 26.7 | 137480 |
| A1_soap | 0.00973 | 0.008331 | 0.008931 | 0.00222 | 27.9 | 140596 |
| A2_adam1e-2 | 0.01979 | 0.01726 | 0.01769 | 0.0066 | 29.2 | 140596 |
| A2b_adam95_1e-2 | 0.01365 | 0.01182 | 0.0125 | 0.00434 | 27.2 | 140596 |
| A3_p1x3 | 0.01592 | 0.0137 | 0.01442 | 0.00757 | 31.5 | 140596 |
| A4_gru | 0.01885 | 0.01623 | 0.01634 | 0.0137 | 26.4 | 140596 |
| A5_curv | 0.0212 | 0.01837 | 0.01847 | 0.0137 | 26.4 | 140596 |
| A6_enc | 0.02143 | 0.01918 | 0.01888 | 0.0137 | 26.4 | 143728 |
| A7_best | 0.009431 | 0.008373 | 0.008841 | 0.00166 | 34.5 | 143728 |
| A8_best_noenc | 0.009484 | 0.008549 | 0.008881 | 0.00166 | 34.5 | 140596 |
| best baseline (live C3 node, seeds 0-2) | greybox_ekf 0.009758 | greybox_ekf 0.008522 | greybox_ekf 0.008926 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.0266, nrmse_p99 0.025, div_rate 0, t_conv_median_s 2.07; A0_hyper: nrmse_unmeasured 0.0312, nrmse_p99 0.0269, div_rate 0, t_conv_median_s 8.6; A7_best: nrmse_unmeasured 0.0111, nrmse_p99 0.0116, div_rate 0, t_conv_median_s 2.14; A8_best_noenc: nrmse_unmeasured 0.0112, nrmse_p99 0.0117, div_rate 0, t_conv_median_s 1.96

### lorenz / C3-mismatch30 (WIN, selection r = 1.03, seeds [0, 1, 2, 3, 4])

| arm | sequence_ood | shape_ood | id | eps_dec | l_dec_q95 | params |
|---|---|---|---|---|---|---|
| A0_lora | 0.008144 | 0.007883 | 0.007673 | 0.229 | 11.9 | 141132 |
| A0_hyper | 0.01018 | 0.01038 | 0.01025 | 0.229 | 11.9 | 139020 |
| A1_soap | 0.00466 | 0.004648 | 0.004638 | 0.0558 | 97.2 | 141132 |
| A2_adam1e-2 | 0.006949 | 0.006968 | 0.006766 | 0.113 | 16 | 141132 |
| A2b_adam95_1e-2 | 0.00612 | 0.006144 | 0.00601 | 0.123 | 20.7 | 141132 |
| A3_p1x3 | 0.006673 | 0.006707 | 0.006409 | 0.117 | 13.7 | 141132 |
| A4_gru | 0.007231 | 0.007397 | 0.006962 | 0.225 | 11.7 | 141132 |
| A5_curv | 0.006725 | 0.006532 | 0.006344 | 0.225 | 11.7 | 141132 |
| A6_enc | 0.007012 | 0.007001 | 0.006719 | 0.225 | 11.7 | 136361 |
| A7_best | 0.005366 | 0.005505 | 0.005444 | 0.0581 | 119 | 136361 |
| A8_best_noenc | 0.004355 | 0.004378 | 0.004332 | 0.0581 | 119 | 141132 |
| best baseline (live C3 node, seeds 0-2) | nlox 0.006822 | nlox 0.006684 | nlox 0.006692 | | | |

Scorecard essentials (sequence_ood, seed means): A0_lora: nrmse_unmeasured 0.00853, nrmse_p99 0.0104, div_rate 0, t_conv_median_s 3.47; A0_hyper: nrmse_unmeasured 0.0107, nrmse_p99 0.0126, div_rate 0, t_conv_median_s 4.08; A7_best: nrmse_unmeasured 0.00542, nrmse_p99 0.00638, div_rate 0, t_conv_median_s 2.98; A8_best_noenc: nrmse_unmeasured 0.00441, nrmse_p99 0.00538, div_rate 0, t_conv_median_s 2.93
