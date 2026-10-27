# C2-floor-025 — Forgetting floor: SFM context forgets at >= 0.25 x alpha

*parent* `C1-where` · *contribution* C2 · *status* done · jobs 18/18

**Question.** Corollary 1: input history older than ~ln(1/eta)/alpha barely affects the map. If the context is forced to forget at rate >= c alpha (Lemma 2 then bounds its memory), does accuracy suffer?

**Hypothesis.** No loss for c <= 0.5 (memory ln(20)/(0.5 alpha) covers the theory window); a small loss is possible at c = 1 (no margin).

**Decision rule.** Memory beyond the theory window is unnecessary if C2-floor-05 is within CI of C1-where (no floor) for lora and hyper on >= 5/6 dev systems, on id and sequence_ood. The probe (on) must decay at >= the floor rate (Lemma 2 check).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (sequence_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0671 | 0.0767 | 0.0553 | 0.0281 | 0.0650 | 0.0181 |
| hyper | 0.0748 | 0.0523 | 0.0624 | 0.0168 | 0.0625 | 0.0248 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (+12%) | 0/0 (-23%) | 0/1 (+13%*) | 1/0 (-40%*) | 0/0 (-4%) | 0/1 (+36%*) |

## Scoreboard (shape_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0539 | 0.0296 | 0.0513 | 0.0260 | 0.0518 | 0.0151 |
| hyper | 0.0522 | 0.0273 | 0.0548 | 0.0169 | 0.0500 | 0.0225 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (-2%) | 0/0 (-2%) | 0/2 (+7%) | 1/0 (-34%) | 0/0 (-4%) | 0/3 (+49%*) |

## Scoreboard (id / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0471 | 0.0227 | 0.0489 | 0.0286 | 0.0533 | 0.0159 |
| hyper | 0.0483 | 0.0224 | 0.0517 | 0.0202 | 0.0536 | 0.0225 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (+2%) | 0/0 (+5%) | 0/2 (+6%*) | 8/0 (-29%*) | 0/0 (+1%) | 0/3 (+42%*) |

## battery

phase 1: eps_enc_rel 0.00489, eps_dec 1.83, eps_pde_rms 0.302, eps_pde_ss_rms 0.0448, l_dec_q95 806, l_dec_mean 370 · mean job wall 3.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.0748 [0.0573, 0.0923] | 0.0345 [0.0249, 0.0441] | **0.0484 [0.0398, 0.0569]** | 0.0688 [0.0572, 0.0804] | **0.0459 [0.0378, 0.0540]** | 0.0379 [0.0111, 0.0648] | **0.0340 [0.0303, 0.0376]** | **0.0527 [0.0407, 0.0646]** | **0.0543 [0.0283, 0.0802]** | **0.0248 [0.0213, 0.0283]** | **0.0471 [0.0401, 0.0542]** | 0.00 | 141484 | 1465.89 |
| hyper | **0.0703 [0.0414, 0.0992]** | **0.0328 [0.0294, 0.0362]** | 0.0518 [0.0298, 0.0737] | **0.0622 [0.0241, 0.1003]** | 0.0587 [0.0317, 0.0857] | **0.0309 [0.0174, 0.0444]** | 0.0397 [0.0163, 0.0631] | 0.0564 [0.0299, 0.0829] | 0.0554 [0.0308, 0.0800] | 0.0275 [0.0201, 0.0349] | 0.0483 [0.0280, 0.0685] | 0.00 | 140052 | 1808.09 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | -6.5% [-26, +13] | -4.5% [-22, +13] | +7.9% [-49, +64] | -9.6% [-65, +45] | +27.4% [-15, +70] | -14.6% [-80, +50] | +17.3% [-57, +92] | +8.8% [-63, +80] | +4.5% [-62, +71] | +10.9% [-19, +41] | +2.5% [-41, +46] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0491 [0.0432, 0.0549]** | 0.0620 [0.0391, 0.0849] | 0.0689 [0.0398, 0.0981] | **0.0356 [0.0242, 0.0469]** | 0.0539 [0.0372, 0.0706] | 0.00 | 141484 | 1465.89 |
| hyper | 0.0547 [0.0353, 0.0741] | **0.0487 [0.0292, 0.0683]** | **0.0670 [0.0398, 0.0941]** | 0.0386 [0.0195, 0.0576] | **0.0522 [0.0313, 0.0732]** | 0.00 | 140052 | 1808.09 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | +11.3% [-21, +44] | -19.8% [-72, +33] | -1.9% [-47, +43] | +9.0% [-46, +64] | -2.3% [-48, +44] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0671 [0.0378, 0.0963]** | **0.0671 [0.0378, 0.0963]** | 0.00 | 141484 | 1465.89 |
| hyper | 0.0748 [0.0510, 0.0986] | 0.0748 [0.0510, 0.0986] | 0.00 | 140052 | 1808.09 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | +12.4% [-15, +40] | +12.4% [-15, +40] |

## bioreactor

phase 1: eps_enc_rel 0.00825, eps_dec 0.0199, eps_pde_rms 0.00738, eps_pde_ss_rms 0.00456, l_dec_q95 44.1, l_dec_mean 18 · mean job wall 2.9 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0275 [0.0135, 0.0415]** | **0.0084 [0.0039, 0.0130]** | 0.0239 [-0.0086, 0.0565] | 0.0273 [0.0023, 0.0524] | 0.0332 [0.0071, 0.0594] | **0.0152 [0.0098, 0.0207]** | **0.0255 [0.0032, 0.0478]** | 0.0308 [-0.0119, 0.0736] | 0.0270 [0.0085, 0.0455] | **0.0097 [0.0056, 0.0138]** | 0.0227 [0.0054, 0.0399] | 0.00 | 140596 | 1937.06 |
| hyper | 0.0329 [0.0306, 0.0352] | 0.0120 [0.0111, 0.0129] | **0.0200 [0.0169, 0.0231]** | **0.0214 [0.0199, 0.0229]** | **0.0284 [0.0264, 0.0304]** | 0.0171 [0.0161, 0.0181] | 0.0305 [0.0279, 0.0330] | **0.0241 [0.0214, 0.0267]** | **0.0252 [0.0192, 0.0312]** | 0.0142 [0.0133, 0.0151] | **0.0224 [0.0209, 0.0239]** | 0.00 | 137480 | 1224.24 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +22.6% [-30, +75] | +46.2% [-26, +118] | +7.7% [-159, +175] | -15.2% [-82, +52] | -6.9% [-98, +84] | +13.6% [-23, +50] | +30.7% [-91, +153] | -6.5% [-114, +101] | -3.5% [-46, +39] | +48.4% [-4, +101] | +5.3% [-74, +84] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0364 [0.0087, 0.0642] | **0.0184 [0.0059, 0.0309]** | **0.0319 [0.0084, 0.0553]** | 0.0319 [0.0015, 0.0622] | 0.0296 [0.0075, 0.0517] | 0.00 | 140596 | 1937.06 |
| hyper | **0.0304 [0.0299, 0.0308]** | 0.0212 [0.0189, 0.0235] | 0.0330 [0.0293, 0.0367] | **0.0245 [0.0238, 0.0252]** | **0.0273 [0.0259, 0.0287]** | 0.00 | 137480 | 1224.24 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | -10.1% [-90, +70] | +21.6% [-62, +106] | +10.8% [-79, +100] | -16.1% [-87, +55] | -1.6% [-81, +77] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0767 [-0.0105, 0.1638] | 0.0767 [-0.0105, 0.1638] | 0.00 | 140596 | 1937.06 |
| hyper | **0.0523 [0.0468, 0.0578]** | **0.0523 [0.0468, 0.0578]** | 0.00 | 137480 | 1224.24 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | -23.1% [-97, +50] | -23.1% [-97, +50] |

## duffing

phase 1: eps_enc_rel 0.00444, eps_dec 0.00608, eps_pde_rms 0.00261, eps_pde_ss_rms 0.000269, l_dec_q95 295, l_dec_mean 137 · mean job wall 3.6 min

- state coverage id: 0.0% of eval states outside the training box, 1.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 2.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0696 [0.0648, 0.0744]** | **0.0427 [0.0296, 0.0558]** | **0.0486 [0.0372, 0.0600]** | **0.0533 [0.0439, 0.0627]** | **0.0549 [0.0440, 0.0658]** | **0.0401 [0.0283, 0.0519]** | **0.0473 [0.0373, 0.0573]** | **0.0490 [0.0417, 0.0564]** | **0.0423 [0.0334, 0.0511]** | **0.0437 [0.0291, 0.0583]** | **0.0489 [0.0388, 0.0591]** | 0.00 | 140596 | 1827.53 |
| hyper | 0.0759 [0.0733, 0.0784] | 0.0447 [0.0242, 0.0651] | 0.0503 [0.0348, 0.0657] | 0.0564 [0.0432, 0.0697] | 0.0564 [0.0509, 0.0618] | 0.0413 [0.0256, 0.0570] | 0.0506 [0.0373, 0.0639] | 0.0523 [0.0447, 0.0599] | 0.0455 [0.0361, 0.0548] | 0.0464 [0.0267, 0.0660] | 0.0517 [0.0395, 0.0640] | 0.00 | 137480 | 1173.46 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | **+9.0% [+3, +15]** | +4.2% [-14, +23] | +3.3% [-7, +13] | +5.7% [-2, +14] | +3.0% [-11, +17] | +2.7% [-8, +13] | +6.8% [-0, +14] | +6.7% [-1, +15] | **+7.5% [+7, +8]** | +5.8% [-5, +17] | **+5.6% [+0, +11]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0508 [0.0440, 0.0576]** | **0.0483 [0.0353, 0.0613]** | **0.0534 [0.0456, 0.0612]** | **0.0527 [0.0427, 0.0627]** | **0.0513 [0.0419, 0.0607]** | 0.00 | 140596 | 1827.53 |
| hyper | 0.0546 [0.0440, 0.0652] | 0.0499 [0.0331, 0.0668] | 0.0581 [0.0464, 0.0697] | 0.0566 [0.0413, 0.0719] | 0.0548 [0.0412, 0.0684] | 0.00 | 137480 | 1173.46 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **+7.2% [+0, +14]** | +3.2% [-6, +12] | **+8.5% [+2, +15]** | +7.3% [-3, +18] | +6.6% [-1, +15] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0553 [0.0505, 0.0600]** | **0.0553 [0.0505, 0.0600]** | 0.00 | 140596 | 1827.53 |
| hyper | 0.0624 [0.0604, 0.0645] | 0.0624 [0.0604, 0.0645] | 0.00 | 137480 | 1173.46 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **+13.1% [+7, +19]** | **+13.1% [+7, +19]** |

## lorenz

phase 1: eps_enc_rel 0.177, eps_dec 0.267, eps_pde_rms 77, eps_pde_ss_rms 8.57, l_dec_q95 10.1, l_dec_mean 3.84 · mean job wall 3.7 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.0912 [0.0744, 0.1079] | 0.0174 [0.0160, 0.0188] | 0.0197 [0.0145, 0.0250] | 0.0295 [0.0223, 0.0367] | 0.0258 [0.0122, 0.0394] | 0.0171 [0.0138, 0.0204] | 0.0244 [0.0218, 0.0270] | 0.0312 [0.0270, 0.0354] | 0.0193 [0.0182, 0.0204] | 0.0160 [0.0145, 0.0176] | 0.0286 [0.0237, 0.0335] | 0.00 | 141132 | 1109.77 |
| hyper | **0.0639 [0.0270, 0.1009]** | **0.0149 [0.0143, 0.0156]** | **0.0146 [0.0138, 0.0154]** | **0.0195 [0.0084, 0.0305]** | **0.0169 [0.0098, 0.0240]** | **0.0143 [0.0130, 0.0156]** | **0.0151 [0.0109, 0.0194]** | **0.0177 [0.0108, 0.0245]** | **0.0146 [0.0132, 0.0160]** | **0.0139 [0.0134, 0.0144]** | **0.0202 [0.0140, 0.0264]** | 0.00 | 139020 | 836.89 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | -29.0% [-81, +23] | **-14.2% [-18, -10]** | **-25.5% [-41, -10]** | **-34.3% [-62, -7]** | -32.3% [-80, +16] | **-16.0% [-29, -3]** | **-37.8% [-62, -13]** | **-43.0% [-71, -15]** | **-24.3% [-28, -21]** | **-13.5% [-18, -9]** | **-28.9% [-57, -0]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0259 [0.0213, 0.0305] | 0.0236 [0.0116, 0.0356] | 0.0259 [0.0204, 0.0314] | 0.0287 [0.0233, 0.0340] | 0.0260 [0.0193, 0.0328] | 0.00 | 141132 | 1109.77 |
| hyper | **0.0165 [0.0094, 0.0237]** | **0.0159 [0.0112, 0.0206]** | **0.0167 [0.0081, 0.0254]** | **0.0183 [0.0081, 0.0285]** | **0.0169 [0.0094, 0.0244]** | 0.00 | 139020 | 836.89 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **-35.8% [-68, -3]** | -31.0% [-66, +4] | -34.8% [-73, +4] | -35.3% [-82, +11] | -34.4% [-72, +4] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0281 [0.0246, 0.0315] | 0.0281 [0.0246, 0.0315] | 0.00 | 141132 | 1109.77 |
| hyper | **0.0168 [0.0103, 0.0232]** | **0.0168 [0.0103, 0.0232]** | 0.00 | 139020 | 836.89 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **-40.3% [-60, -21]** | **-40.3% [-60, -21]** |

## quad_tank

phase 1: eps_enc_rel 0.0128, eps_dec 0.139, eps_pde_rms 0.0188, eps_pde_ss_rms 0.0486, l_dec_q95 13.6, l_dec_mean 6.59 · mean job wall 4.4 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0663 [0.0595, 0.0730]** | 0.0440 [0.0397, 0.0483] | **0.0539 [0.0514, 0.0563]** | 0.0452 [0.0415, 0.0489] | 0.0583 [0.0481, 0.0685] | 0.0453 [0.0444, 0.0462] | 0.0478 [0.0440, 0.0516] | **0.0607 [0.0493, 0.0721]** | 0.0578 [0.0530, 0.0625] | **0.0535 [0.0464, 0.0606]** | **0.0533 [0.0513, 0.0553]** | 0.00 | 141128 | 974.10 |
| hyper | 0.0726 [0.0625, 0.0828] | **0.0426 [0.0399, 0.0454]** | 0.0542 [0.0490, 0.0595] | **0.0443 [0.0407, 0.0479]** | **0.0578 [0.0460, 0.0696]** | **0.0430 [0.0403, 0.0457]** | **0.0477 [0.0433, 0.0522]** | 0.0610 [0.0523, 0.0697] | **0.0574 [0.0496, 0.0653]** | 0.0553 [0.0461, 0.0644] | 0.0536 [0.0479, 0.0593] | 0.00 | 139024 | 1226.82 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +9.9% [-14, +33] | -3.0% [-15, +9] | +0.6% [-11, +13] | -1.9% [-10, +6] | -0.8% [-12, +10] | -5.1% [-12, +2] | -0.2% [-14, +14] | +0.6% [-9, +10] | -0.6% [-7, +6] | +3.6% [-22, +30] | +0.5% [-7, +8] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0531 [0.0494, 0.0568] | 0.0498 [0.0487, 0.0510] | 0.0530 [0.0508, 0.0553] | 0.0513 [0.0481, 0.0546] | 0.0518 [0.0496, 0.0541] | 0.00 | 141128 | 974.10 |
| hyper | **0.0505 [0.0458, 0.0553]** | **0.0480 [0.0438, 0.0522]** | **0.0519 [0.0461, 0.0578]** | **0.0495 [0.0436, 0.0553]** | **0.0500 [0.0453, 0.0547]** | 0.00 | 139024 | 1226.82 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | -4.8% [-13, +4] | -3.7% [-11, +4] | -2.1% [-11, +7] | -3.6% [-12, +5] | -3.6% [-9, +2] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0650 [0.0556, 0.0744] | 0.0650 [0.0556, 0.0744] | 0.00 | 141128 | 974.10 |
| hyper | **0.0625 [0.0564, 0.0687]** | **0.0625 [0.0564, 0.0687]** | 0.00 | 139024 | 1226.82 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | -3.7% [-10, +2] | -3.7% [-10, +2] |

## vdp

phase 1: eps_enc_rel 0.00385, eps_dec 0.0112, eps_pde_rms 0.275, eps_pde_ss_rms 0.0566, l_dec_q95 23.2, l_dec_mean 9.73 · mean job wall 2.8 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.4% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0218 [0.0175, 0.0261]** | **0.0141 [0.0125, 0.0158]** | **0.0153 [0.0138, 0.0169]** | **0.0149 [0.0141, 0.0157]** | **0.0176 [0.0157, 0.0194]** | **0.0155 [0.0134, 0.0175]** | **0.0138 [0.0134, 0.0143]** | **0.0164 [0.0138, 0.0191]** | **0.0164 [0.0155, 0.0172]** | **0.0132 [0.0123, 0.0141]** | **0.0159 [0.0144, 0.0174]** | 0.00 | 140596 | 969.18 |
| hyper | 0.0243 [0.0239, 0.0248] | 0.0213 [0.0130, 0.0296] | 0.0225 [0.0132, 0.0319] | 0.0224 [0.0132, 0.0316] | 0.0245 [0.0169, 0.0321] | 0.0220 [0.0144, 0.0295] | 0.0207 [0.0114, 0.0301] | 0.0237 [0.0143, 0.0330] | 0.0237 [0.0123, 0.0350] | 0.0203 [0.0116, 0.0289] | 0.0225 [0.0144, 0.0306] | 0.00 | 137480 | 818.92 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +12.2% [-11, +35] | **+50.2% [+7, +93]** | +46.3% [-2, +95] | +49.8% [-6, +105] | **+38.8% [+8, +70]** | **+41.6% [+10, +74]** | +50.1% [-20, +120] | +43.6% [-3, +91] | +44.3% [-18, +107] | +53.5% [-5, +112] | **+41.6% [+2, +81]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0158 [0.0140, 0.0177]** | **0.0145 [0.0128, 0.0161]** | **0.0153 [0.0145, 0.0160]** | **0.0148 [0.0132, 0.0163]** | **0.0151 [0.0137, 0.0165]** | 0.00 | 140596 | 969.18 |
| hyper | 0.0233 [0.0143, 0.0324] | 0.0219 [0.0131, 0.0308] | 0.0228 [0.0137, 0.0318] | 0.0221 [0.0135, 0.0307] | 0.0225 [0.0137, 0.0314] | 0.00 | 137480 | 818.92 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **+47.0% [+5, +89]** | **+51.2% [+4, +99]** | +48.7% [-4, +102] | **+49.3% [+5, +94]** | **+49.0% [+2, +95]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0181 [0.0160, 0.0203]** | **0.0181 [0.0160, 0.0203]** | 0.00 | 140596 | 969.18 |
| hyper | 0.0248 [0.0176, 0.0321] | 0.0248 [0.0176, 0.0321] | 0.00 | 137480 | 818.92 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **+36.5% [+10, +63]** | **+36.5% [+10, +63]** |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0471 | 0.0539 | 0.0671 | 0.0750 | 0.1439 | 0.00 (shape_ood/clean) | 0.908 | 1.4e+03 |
| hyper | 0.0483 | 0.0522 | 0.0748 | 0.0824 | 0.1201 | 0.00 (shape_ood/clean) | 0.912 | 1.27e+03 |

### bioreactor

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0227 | 0.0296 | 0.0767 | 0.0830 | 0.1560 | 0.00 (shape_ood/clean) | 0.899 | 1.37e+03 |
| hyper | 0.0224 | 0.0273 | 0.0523 | 0.0602 | 0.0661 | 0.00 (shape_ood/clean) | 0.905 | 1.16e+03 |

### duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0489 | 0.0513 | 0.0553 | 0.0772 | 0.0624 | 0.00 (shape_ood/clean) | 0.604 | 1.41e+03 |
| hyper | 0.0517 | 0.0548 | 0.0624 | 0.0875 | 0.0695 | 0.00 (shape_ood/clean) | 0.581 | 1.1e+03 |

### lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0286 | 0.0260 | 0.0281 | 0.0258 | 0.0500 | 0.00 (shape_ood/clean) | 0.605 | 1.6e+03 |
| hyper | 0.0202 | 0.0169 | 0.0168 | 0.0157 | 0.0254 | 0.00 (shape_ood/clean) | 0.579 | 1.26e+03 |

### quad_tank

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0533 | 0.0518 | 0.0650 | 0.0849 | 0.0837 | 0.00 (shape_ood/clean) | 0.946 | 1.33e+03 |
| hyper | 0.0536 | 0.0500 | 0.0625 | 0.0823 | 0.0797 | 0.00 (shape_ood/clean) | 0.954 | 1.63e+03 |

### vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0159 | 0.0151 | 0.0181 | 0.0236 | 0.0230 | 0.00 (shape_ood/clean) | 0.625 | 1.28e+03 |
| hyper | 0.0225 | 0.0225 | 0.0248 | 0.0334 | 0.0302 | 0.00 (shape_ood/clean) | 0.619 | 1.07e+03 |
