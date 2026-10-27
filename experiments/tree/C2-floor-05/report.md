# C2-floor-05 — Forgetting floor: SFM context forgets at >= 0.5 x alpha

*parent* `C1-where` · *contribution* C2 · *status* done · jobs 18/18

**Question.** Corollary 1: input history older than ~ln(1/eta)/alpha barely affects the map. If the context is forced to forget at rate >= c alpha (Lemma 2 then bounds its memory), does accuracy suffer?

**Hypothesis.** No loss for c <= 0.5 (memory ln(20)/(0.5 alpha) covers the theory window); a small loss is possible at c = 1 (no margin).

**Decision rule.** Memory beyond the theory window is unnecessary if C2-floor-05 is within CI of C1-where (no floor) for lora and hyper on >= 5/6 dev systems, on id and sequence_ood. The probe (on) must decay at >= the floor rate (Lemma 2 check).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (sequence_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0959 | 0.0702 | 0.0532 | 0.0285 | 0.0646 | 0.0186 |
| hyper | 0.0929 | 0.0532 | 0.0623 | 0.0173 | 0.0665 | 0.0245 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (-3%) | 0/0 (+3%) | 0/1 (+17%*) | 0/0 (-36%) | 0/0 (+3%) | 0/0 (+33%) |

## Scoreboard (shape_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0698 | 0.0233 | 0.0494 | 0.0281 | 0.0523 | 0.0157 |
| hyper | 0.0647 | 0.0284 | 0.0587 | 0.0175 | 0.0530 | 0.0226 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 1/0 (-7%) | 0/0 (+23%) | 0/4 (+19%*) | 0/0 (-36%) | 0/0 (+1%) | 0/0 (+46%) |

## Scoreboard (id / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0634 | 0.0175 | 0.0469 | 0.0290 | 0.0531 | 0.0163 |
| hyper | 0.0645 | 0.0228 | 0.0554 | 0.0201 | 0.0556 | 0.0224 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (+2%) | 0/5 (+30%*) | 0/10 (+18%*) | 2/0 (-29%) | 0/0 (+5%) | 0/0 (+39%) |

## battery

phase 1: eps_enc_rel 0.00489, eps_dec 1.83, eps_pde_rms 0.302, eps_pde_ss_rms 0.0448, l_dec_q95 806, l_dec_mean 370 · mean job wall 3.3 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0767 [0.0451, 0.1084]** | **0.0406 [0.0317, 0.0495]** | **0.0743 [0.0648, 0.0837]** | 0.0806 [0.0504, 0.1108] | **0.0641 [0.0572, 0.0710]** | 0.0453 [0.0218, 0.0687] | 0.0513 [0.0461, 0.0565] | **0.0649 [0.0530, 0.0768]** | 0.1052 [0.0932, 0.1171] | **0.0327 [0.0245, 0.0409]** | **0.0634 [0.0523, 0.0745]** | 0.00 | 141484 | 932.61 |
| hyper | 0.0786 [0.0548, 0.1024] | 0.0418 [0.0344, 0.0493] | 0.0768 [0.0679, 0.0857] | **0.0744 [0.0390, 0.1099]** | 0.0762 [0.0630, 0.0894] | **0.0408 [0.0106, 0.0711]** | **0.0511 [0.0483, 0.0539]** | 0.0742 [0.0556, 0.0927] | **0.0950 [0.0608, 0.1292]** | 0.0370 [0.0364, 0.0376] | 0.0645 [0.0523, 0.0766] | 0.00 | 140052 | 922.28 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +3.0% [-14, +20] | +3.2% [-4, +10] | +3.4% [-5, +12] | -7.9% [-26, +11] | +19.3% [-15, +53] | -10.1% [-49, +29] | -0.2% [-16, +16] | +15.1% [-30, +60] | -9.8% [-35, +15] | +14.0% [-14, +42] | +1.8% [-13, +17] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0591 [0.0508, 0.0673]** | 0.0825 [0.0575, 0.1075] | 0.0828 [0.0531, 0.1125] | 0.0547 [0.0432, 0.0662] | 0.0698 [0.0516, 0.0879] | 0.00 | 141484 | 932.61 |
| hyper | 0.0637 [0.0554, 0.0721] | **0.0661 [0.0421, 0.0901]** | **0.0765 [0.0546, 0.0984]** | **0.0524 [0.0434, 0.0615]** | **0.0647 [0.0495, 0.0799]** | 0.00 | 140052 | 922.28 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | +8.0% [-2, +18] | **-19.9% [-25, -15]** | -7.3% [-18, +3] | -3.7% [-31, +23] | -7.1% [-18, +4] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0959 [0.0826, 0.1092] | 0.0959 [0.0826, 0.1092] | 0.00 | 141484 | 932.61 |
| hyper | **0.0929 [0.0910, 0.0948]** | **0.0929 [0.0910, 0.0948]** | 0.00 | 140052 | 922.28 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | -2.9% [-14, +9] | -2.9% [-14, +9] |

## bioreactor

phase 1: eps_enc_rel 0.00825, eps_dec 0.0199, eps_pde_rms 0.00738, eps_pde_ss_rms 0.00456, l_dec_q95 44.1, l_dec_mean 18 · mean job wall 2.6 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0236 [0.0128, 0.0344]** | **0.0077 [0.0046, 0.0107]** | **0.0184 [0.0099, 0.0268]** | **0.0201 [0.0027, 0.0375]** | **0.0235 [0.0206, 0.0265]** | **0.0124 [0.0120, 0.0127]** | **0.0255 [0.0099, 0.0411]** | **0.0189 [0.0140, 0.0237]** | **0.0183 [0.0168, 0.0199]** | **0.0086 [0.0066, 0.0107]** | **0.0175 [0.0160, 0.0190]** | 0.00 | 140596 | 895.32 |
| hyper | 0.0339 [0.0305, 0.0373] | 0.0122 [0.0112, 0.0133] | 0.0198 [0.0193, 0.0204] | 0.0218 [0.0187, 0.0249] | 0.0303 [0.0265, 0.0341] | 0.0173 [0.0159, 0.0187] | 0.0314 [0.0243, 0.0384] | 0.0246 [0.0220, 0.0271] | 0.0239 [0.0222, 0.0256] | 0.0144 [0.0128, 0.0160] | 0.0228 [0.0220, 0.0235] | 0.00 | 137480 | 852.89 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +47.9% [-38, +134] | +63.0% [-10, +136] | +10.2% [-34, +54] | +16.5% [-70, +103] | **+29.1% [+6, +52]** | **+40.1% [+30, +50]** | +26.1% [-24, +76] | **+30.9% [+5, +56]** | **+30.1% [+18, +43]** | **+67.8% [+17, +119]** | **+30.3% [+20, +41]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0320 [0.0101, 0.0538]** | **0.0170 [0.0048, 0.0293]** | **0.0241 [0.0130, 0.0351]** | **0.0201 [0.0134, 0.0268]** | **0.0233 [0.0162, 0.0304]** | 0.00 | 140596 | 895.32 |
| hyper | 0.0330 [0.0302, 0.0358] | 0.0214 [0.0194, 0.0234] | 0.0339 [0.0297, 0.0380] | 0.0254 [0.0219, 0.0288] | 0.0284 [0.0259, 0.0309] | 0.00 | 137480 | 852.89 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | +7.9% [-54, +70] | +32.0% [-51, +115] | +43.3% [-11, +98] | +27.9% [-24, +80] | +23.3% [-17, +64] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0702 [-0.0577, 0.1981] | 0.0702 [-0.0577, 0.1981] | 0.00 | 140596 | 895.32 |
| hyper | **0.0532 [0.0460, 0.0604]** | **0.0532 [0.0460, 0.0604]** | 0.00 | 137480 | 852.89 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | +3.2% [-136, +143] | +3.2% [-136, +143] |

## duffing

phase 1: eps_enc_rel 0.00444, eps_dec 0.00608, eps_pde_rms 0.00261, eps_pde_ss_rms 0.000269, l_dec_q95 295, l_dec_mean 137 · mean job wall 3.7 min

- state coverage id: 0.0% of eval states outside the training box, 1.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 2.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0682 [0.0646, 0.0718]** | **0.0400 [0.0349, 0.0451]** | **0.0468 [0.0426, 0.0511]** | **0.0511 [0.0478, 0.0543]** | **0.0520 [0.0501, 0.0539]** | **0.0390 [0.0349, 0.0432]** | **0.0456 [0.0421, 0.0490]** | **0.0464 [0.0444, 0.0484]** | **0.0412 [0.0382, 0.0442]** | **0.0409 [0.0346, 0.0472]** | **0.0469 [0.0434, 0.0503]** | 0.00 | 140596 | 1241.06 |
| hyper | 0.0760 [0.0753, 0.0767] | 0.0504 [0.0500, 0.0507] | 0.0555 [0.0537, 0.0572] | 0.0605 [0.0588, 0.0623] | 0.0585 [0.0558, 0.0612] | 0.0463 [0.0454, 0.0471] | 0.0549 [0.0537, 0.0561] | 0.0548 [0.0519, 0.0577] | 0.0477 [0.0471, 0.0483] | 0.0519 [0.0510, 0.0527] | 0.0554 [0.0543, 0.0566] | 0.00 | 137480 | 1159.48 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | **+11.4% [+5, +18]** | **+26.2% [+10, +43]** | **+18.5% [+7, +30]** | **+18.6% [+10, +27]** | **+12.5% [+3, +22]** | **+18.7% [+4, +33]** | **+20.5% [+12, +29]** | **+18.2% [+15, +21]** | **+15.8% [+8, +24]** | **+27.2% [+9, +46]** | **+18.3% [+9, +28]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0490 [0.0463, 0.0517]** | **0.0465 [0.0422, 0.0507]** | **0.0514 [0.0485, 0.0543]** | **0.0509 [0.0486, 0.0531]** | **0.0494 [0.0464, 0.0524]** | 0.00 | 140596 | 1241.06 |
| hyper | 0.0577 [0.0558, 0.0596] | 0.0554 [0.0539, 0.0569] | 0.0609 [0.0591, 0.0626] | 0.0608 [0.0592, 0.0623] | 0.0587 [0.0572, 0.0601] | 0.00 | 137480 | 1159.48 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **+17.8% [+13, +23]** | **+19.3% [+6, +33]** | **+18.4% [+12, +24]** | **+19.5% [+15, +24]** | **+18.8% [+12, +26]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0532 [0.0511, 0.0553]** | **0.0532 [0.0511, 0.0553]** | 0.00 | 140596 | 1241.06 |
| hyper | 0.0623 [0.0609, 0.0637] | 0.0623 [0.0609, 0.0637] | 0.00 | 137480 | 1159.48 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **+17.1% [+10, +24]** | **+17.1% [+10, +24]** |

## lorenz

phase 1: eps_enc_rel 0.177, eps_dec 0.267, eps_pde_rms 77, eps_pde_ss_rms 8.57, l_dec_q95 10.1, l_dec_mean 3.84 · mean job wall 3.4 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.0824 [0.0486, 0.1161] | 0.0180 [0.0125, 0.0235] | 0.0289 [0.0195, 0.0382] | 0.0312 [0.0171, 0.0453] | 0.0283 [0.0113, 0.0453] | 0.0207 [0.0058, 0.0356] | 0.0224 [0.0148, 0.0300] | 0.0246 [0.0128, 0.0363] | 0.0220 [0.0132, 0.0308] | 0.0166 [0.0125, 0.0207] | 0.0290 [0.0173, 0.0407] | 0.00 | 141132 | 1190.28 |
| hyper | **0.0577 [0.0331, 0.0822]** | **0.0150 [0.0146, 0.0154]** | **0.0187 [0.0170, 0.0204]** | **0.0190 [0.0065, 0.0314]** | **0.0180 [0.0076, 0.0285]** | **0.0143 [0.0132, 0.0154]** | **0.0149 [0.0109, 0.0189]** | **0.0177 [0.0113, 0.0241]** | **0.0145 [0.0141, 0.0148]** | **0.0139 [0.0135, 0.0143]** | **0.0201 [0.0145, 0.0256]** | 0.00 | 139020 | 1658.06 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | -27.7% [-84, +28] | -15.5% [-41, +10] | **-34.4% [-62, -7]** | -36.9% [-97, +24] | -32.1% [-106, +42] | -27.3% [-72, +18] | -32.2% [-75, +10] | -25.1% [-85, +35] | **-32.9% [-62, -3]** | -15.5% [-37, +6] | -28.8% [-76, +19] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0268 [0.0141, 0.0395] | 0.0271 [0.0143, 0.0398] | 0.0265 [0.0156, 0.0375] | 0.0318 [0.0210, 0.0427] | 0.0281 [0.0166, 0.0395] | 0.00 | 141132 | 1190.28 |
| hyper | **0.0165 [0.0127, 0.0202]** | **0.0186 [0.0132, 0.0240]** | **0.0171 [0.0067, 0.0274]** | **0.0177 [0.0083, 0.0270]** | **0.0175 [0.0108, 0.0242]** | 0.00 | 139020 | 1658.06 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | -36.2% [-83, +11] | -28.7% [-81, +23] | -33.6% [-92, +24] | -42.7% [-94, +8] | -35.6% [-87, +16] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0285 [0.0138, 0.0432] | 0.0285 [0.0138, 0.0432] | 0.00 | 141132 | 1190.28 |
| hyper | **0.0173 [0.0095, 0.0252]** | **0.0173 [0.0095, 0.0252]** | 0.00 | 139020 | 1658.06 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | -36.2% [-96, +23] | -36.2% [-96, +23] |

## quad_tank

phase 1: eps_enc_rel 0.0128, eps_dec 0.139, eps_pde_rms 0.0188, eps_pde_ss_rms 0.0486, l_dec_q95 13.6, l_dec_mean 6.59 · mean job wall 3.7 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0633 [0.0518, 0.0749]** | 0.0475 [0.0447, 0.0502] | **0.0532 [0.0499, 0.0564]** | **0.0487 [0.0430, 0.0544]** | **0.0566 [0.0498, 0.0634]** | 0.0472 [0.0455, 0.0490] | **0.0492 [0.0453, 0.0530]** | **0.0606 [0.0568, 0.0643]** | **0.0534 [0.0517, 0.0552]** | **0.0516 [0.0473, 0.0558]** | **0.0531 [0.0493, 0.0569]** | 0.00 | 141128 | 1381.15 |
| hyper | 0.0713 [0.0428, 0.0998] | **0.0458 [0.0410, 0.0506]** | 0.0558 [0.0495, 0.0621] | 0.0514 [0.0408, 0.0619] | 0.0606 [0.0451, 0.0760] | **0.0456 [0.0402, 0.0511]** | 0.0502 [0.0430, 0.0574] | 0.0627 [0.0517, 0.0737] | 0.0551 [0.0435, 0.0667] | 0.0575 [0.0407, 0.0744] | 0.0556 [0.0442, 0.0669] | 0.00 | 139024 | 1004.48 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +12.3% [-21, +46] | -3.5% [-8, +1] | +5.1% [-12, +22] | +5.5% [-10, +21] | +6.8% [-8, +22] | -3.4% [-13, +6] | +2.2% [-12, +17] | +3.6% [-12, +19] | +3.1% [-16, +22] | +11.4% [-14, +37] | +4.6% [-11, +21] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0533 [0.0480, 0.0586]** | **0.0505 [0.0487, 0.0523]** | **0.0537 [0.0505, 0.0570]** | 0.0515 [0.0473, 0.0557] | **0.0523 [0.0491, 0.0554]** | 0.00 | 141128 | 1381.15 |
| hyper | 0.0533 [0.0407, 0.0660] | 0.0509 [0.0421, 0.0597] | 0.0566 [0.0499, 0.0632] | **0.0512 [0.0417, 0.0607]** | 0.0530 [0.0438, 0.0622] | 0.00 | 139024 | 1004.48 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | -0.1% [-14, +14] | +0.8% [-18, +20] | +5.2% [-1, +11] | -0.6% [-11, +10] | +1.4% [-11, +13] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0646 [0.0618, 0.0673]** | **0.0646 [0.0618, 0.0673]** | 0.00 | 141128 | 1381.15 |
| hyper | 0.0665 [0.0494, 0.0836] | 0.0665 [0.0494, 0.0836] | 0.00 | 139024 | 1004.48 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | +2.9% [-19, +25] | +2.9% [-19, +25] |

## vdp

phase 1: eps_enc_rel 0.00385, eps_dec 0.0112, eps_pde_rms 0.275, eps_pde_ss_rms 0.0566, l_dec_q95 23.2, l_dec_mean 9.73 · mean job wall 3.1 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.4% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0222 [0.0181, 0.0262]** | **0.0149 [0.0111, 0.0186]** | **0.0163 [0.0136, 0.0190]** | **0.0154 [0.0123, 0.0184]** | **0.0179 [0.0142, 0.0215]** | **0.0160 [0.0124, 0.0197]** | **0.0145 [0.0129, 0.0160]** | **0.0158 [0.0132, 0.0184]** | **0.0164 [0.0134, 0.0194]** | **0.0137 [0.0099, 0.0175]** | **0.0163 [0.0131, 0.0194]** | 0.00 | 140596 | 938.32 |
| hyper | 0.0243 [0.0237, 0.0248] | 0.0213 [0.0146, 0.0279] | 0.0228 [0.0148, 0.0307] | 0.0224 [0.0146, 0.0302] | 0.0242 [0.0176, 0.0308] | 0.0217 [0.0151, 0.0283] | 0.0206 [0.0115, 0.0297] | 0.0235 [0.0147, 0.0322] | 0.0235 [0.0139, 0.0331] | 0.0202 [0.0133, 0.0272] | 0.0224 [0.0154, 0.0295] | 0.00 | 137480 | 883.14 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +9.9% [-12, +32] | +45.1% [-32, +122] | +40.9% [-28, +109] | +47.3% [-29, +124] | +36.5% [-25, +98] | +37.1% [-32, +106] | +43.7% [-33, +120] | +49.7% [-27, +127] | +44.9% [-36, +126] | +50.1% [-36, +136] | +39.3% [-28, +107] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0162 [0.0125, 0.0199]** | **0.0153 [0.0114, 0.0193]** | **0.0162 [0.0128, 0.0196]** | **0.0152 [0.0120, 0.0184]** | **0.0157 [0.0122, 0.0193]** | 0.00 | 140596 | 938.32 |
| hyper | 0.0234 [0.0156, 0.0312] | 0.0221 [0.0150, 0.0293] | 0.0228 [0.0151, 0.0306] | 0.0222 [0.0145, 0.0299] | 0.0226 [0.0150, 0.0302] | 0.00 | 137480 | 883.14 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | +46.3% [-31, +124] | +46.6% [-33, +126] | +42.6% [-32, +117] | +47.7% [-30, +125] | +45.7% [-31, +123] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0186 [0.0149, 0.0223]** | **0.0186 [0.0149, 0.0223]** | 0.00 | 140596 | 938.32 |
| hyper | 0.0245 [0.0176, 0.0313] | 0.0245 [0.0176, 0.0313] | 0.00 | 137480 | 883.14 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | +33.1% [-28, +94] | +33.1% [-28, +94] |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0634 | 0.0698 | 0.0959 | 0.1139 | 0.1434 | 0.00 (shape_ood/clean) | 0.917 | 1.41e+03 |
| hyper | 0.0645 | 0.0647 | 0.0929 | 0.1066 | 0.1313 | 0.00 (shape_ood/clean) | 0.923 | 1.09e+03 |

### bioreactor

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0175 | 0.0233 | 0.0702 | 0.0845 | 0.4120 | 0.00 (shape_ood/clean) | 0.914 | 1.4e+03 |
| hyper | 0.0228 | 0.0284 | 0.0532 | 0.0620 | 0.0681 | 0.00 (shape_ood/clean) | 0.897 | 1.25e+03 |

### duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0469 | 0.0494 | 0.0532 | 0.0736 | 0.0603 | 0.00 (shape_ood/clean) | 0.607 | 1.17e+03 |
| hyper | 0.0554 | 0.0587 | 0.0623 | 0.0872 | 0.0692 | 0.00 (shape_ood/clean) | 0.584 | 1.04e+03 |

### lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0290 | 0.0281 | 0.0285 | 0.0267 | 0.0473 | 0.00 (shape_ood/clean) | 0.655 | 1.31e+03 |
| hyper | 0.0201 | 0.0175 | 0.0173 | 0.0162 | 0.0293 | 0.00 (shape_ood/clean) | 0.569 | 1.5e+03 |

### quad_tank

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0531 | 0.0523 | 0.0646 | 0.0833 | 0.0790 | 0.00 (shape_ood/clean) | 0.942 | 1.51e+03 |
| hyper | 0.0556 | 0.0530 | 0.0665 | 0.0876 | 0.0837 | 0.00 (shape_ood/clean) | 0.947 | 1.92e+03 |

### vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0163 | 0.0157 | 0.0186 | 0.0242 | 0.0219 | 0.00 (shape_ood/clean) | 0.634 | 1.03e+03 |
| hyper | 0.0224 | 0.0226 | 0.0245 | 0.0329 | 0.0300 | 0.00 (shape_ood/clean) | 0.623 | 1.01e+03 |
