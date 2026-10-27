# C2-floor-1 — Forgetting floor: SFM context forgets at >= 1.0 x alpha

*parent* `C1-where` · *contribution* C2 · *status* done · jobs 18/18

**Question.** Corollary 1: input history older than ~ln(1/eta)/alpha barely affects the map. If the context is forced to forget at rate >= c alpha (Lemma 2 then bounds its memory), does accuracy suffer?

**Hypothesis.** No loss for c <= 0.5 (memory ln(20)/(0.5 alpha) covers the theory window); a small loss is possible at c = 1 (no margin).

**Decision rule.** Memory beyond the theory window is unnecessary if C2-floor-05 is within CI of C1-where (no floor) for lora and hyper on >= 5/6 dev systems, on id and sequence_ood. The probe (on) must decay at >= the floor rate (Lemma 2 check).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (sequence_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.1425 | 0.0412 | 0.0523 | 0.0247 | 0.0695 | 0.0181 |
| hyper | 0.1455 | 0.0583 | 0.0629 | 0.0177 | 0.0674 | 0.0267 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (+2%) | 0/0 (+44%) | 0/1 (+20%*) | 1/0 (-28%*) | 0/0 (-3%) | 0/1 (+48%*) |

## Scoreboard (shape_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.1055 | 0.0216 | 0.0505 | 0.0250 | 0.0551 | 0.0159 |
| hyper | 0.1073 | 0.0306 | 0.0605 | 0.0187 | 0.0553 | 0.0251 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/0 (+2%) | 0/2 (+42%*) | 0/4 (+20%*) | 1/0 (-23%) | 0/0 (+0%) | 0/4 (+59%*) |

## Scoreboard (id / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0984 | 0.0182 | 0.0483 | 0.0321 | 0.0573 | 0.0163 |
| hyper | 0.1066 | 0.0239 | 0.0572 | 0.0266 | 0.0603 | 0.0246 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| hyper | 0/2 (+8%) | 0/7 (+32%*) | 0/10 (+19%*) | 5/0 (-17%) | 0/3 (+5%*) | 0/9 (+52%*) |

## battery

phase 1: eps_enc_rel 0.00489, eps_dec 1.83, eps_pde_rms 0.302, eps_pde_ss_rms 0.0448, l_dec_q95 806, l_dec_mean 370 · mean job wall 3.3 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0914 [0.0693, 0.1134]** | **0.0535 [0.0436, 0.0634]** | 0.1205 [0.0930, 0.1480] | 0.1080 [0.0807, 0.1353] | **0.1057 [0.1004, 0.1110]** | **0.0692 [0.0477, 0.0906]** | **0.0938 [0.0867, 0.1009]** | **0.1204 [0.1064, 0.1343]** | **0.1646 [0.1503, 0.1790]** | **0.0589 [0.0552, 0.0625]** | **0.0984 [0.0915, 0.1053]** | 0.00 | 141484 | 1089.87 |
| hyper | 0.0981 [0.0897, 0.1065] | 0.0609 [0.0486, 0.0733] | **0.1171 [0.1001, 0.1342]** | **0.1080 [0.0925, 0.1234]** | 0.1305 [0.1071, 0.1540] | 0.0789 [0.0538, 0.1040] | 0.0978 [0.0844, 0.1112] | 0.1436 [0.1184, 0.1688] | 0.1651 [0.1464, 0.1838] | 0.0662 [0.0538, 0.0787] | 0.1066 [0.1037, 0.1095] | 0.00 | 140052 | 1090.16 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +7.8% [-8, +24] | +14.7% [-30, +59] | -2.5% [-15, +10] | +0.4% [-22, +23] | **+23.4% [+6, +41]** | **+14.1% [+8, +20]** | +4.3% [-12, +20] | +19.6% [-12, +51] | +0.4% [-14, +15] | +12.7% [-14, +39] | +8.4% [-2, +18] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0932 [0.0826, 0.1038]** | 0.1183 [0.1134, 0.1233] | 0.1182 [0.0987, 0.1378] | **0.0923 [0.0867, 0.0979]** | **0.1055 [0.0966, 0.1144]** | 0.00 | 141484 | 1089.87 |
| hyper | 0.1042 [0.0785, 0.1299] | **0.1107 [0.0933, 0.1282]** | **0.1176 [0.1169, 0.1184]** | 0.0968 [0.0864, 0.1072] | 0.1073 [0.0953, 0.1194] | 0.00 | 140052 | 1090.16 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | +12.0% [-23, +47] | -6.4% [-21, +8] | -0.2% [-16, +16] | +4.9% [-12, +22] | +1.8% [-14, +17] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.1425 [0.1296, 0.1553]** | **0.1425 [0.1296, 0.1553]** | 0.00 | 141484 | 1089.87 |
| hyper | 0.1455 [0.1248, 0.1661] | 0.1455 [0.1248, 0.1661] | 0.00 | 140052 | 1090.16 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | +2.3% [-18, +23] | +2.3% [-18, +23] |

## bioreactor

phase 1: eps_enc_rel 0.00825, eps_dec 0.0199, eps_pde_rms 0.00738, eps_pde_ss_rms 0.00456, l_dec_q95 44.1, l_dec_mean 18 · mean job wall 3.3 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0268 [0.0248, 0.0288]** | **0.0097 [0.0089, 0.0104]** | **0.0179 [0.0156, 0.0202]** | **0.0159 [0.0136, 0.0182]** | **0.0255 [0.0155, 0.0354]** | **0.0130 [0.0125, 0.0135]** | **0.0263 [0.0145, 0.0381]** | **0.0200 [0.0166, 0.0233]** | **0.0182 [0.0147, 0.0217]** | **0.0100 [0.0079, 0.0121]** | **0.0182 [0.0154, 0.0209]** | 0.00 | 140596 | 1834.74 |
| hyper | 0.0348 [0.0296, 0.0400] | 0.0131 [0.0112, 0.0150] | 0.0196 [0.0172, 0.0221] | 0.0232 [0.0200, 0.0264] | 0.0338 [0.0327, 0.0349] | 0.0182 [0.0164, 0.0200] | 0.0329 [0.0236, 0.0423] | 0.0250 [0.0223, 0.0277] | 0.0250 [0.0215, 0.0284] | 0.0152 [0.0124, 0.0181] | 0.0239 [0.0226, 0.0253] | 0.00 | 137480 | 1953.27 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | **+29.9% [+8, +51]** | **+35.3% [+18, +52]** | +9.6% [-2, +21] | **+46.3% [+5, +88]** | +35.1% [-20, +90] | **+40.1% [+25, +55]** | **+26.1% [+7, +45]** | **+25.8% [+6, +45]** | **+37.5% [+12, +63]** | +54.0% [-4, +112] | **+32.1% [+12, +52]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0264 [0.0156, 0.0372]** | **0.0148 [0.0132, 0.0165]** | **0.0257 [0.0222, 0.0292]** | **0.0195 [0.0125, 0.0265]** | **0.0216 [0.0167, 0.0265]** | 0.00 | 140596 | 1834.74 |
| hyper | 0.0361 [0.0332, 0.0390] | 0.0226 [0.0201, 0.0252] | 0.0364 [0.0348, 0.0380] | 0.0271 [0.0263, 0.0278] | 0.0306 [0.0296, 0.0315] | 0.00 | 137480 | 1953.27 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | +39.0% [-16, +94] | **+52.8% [+20, +86]** | **+42.2% [+19, +65]** | +40.6% [-9, +91] | **+42.2% [+8, +76]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0412 [0.0257, 0.0568]** | **0.0412 [0.0257, 0.0568]** | 0.00 | 140596 | 1834.74 |
| hyper | 0.0583 [0.0550, 0.0617] | 0.0583 [0.0550, 0.0617] | 0.00 | 137480 | 1953.27 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | +44.0% [-18, +106] | +44.0% [-18, +106] |

## duffing

phase 1: eps_enc_rel 0.00444, eps_dec 0.00608, eps_pde_rms 0.00261, eps_pde_ss_rms 0.000269, l_dec_q95 295, l_dec_mean 137 · mean job wall 3.8 min

- state coverage id: 0.0% of eval states outside the training box, 1.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 2.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0662 [0.0633, 0.0690]** | **0.0426 [0.0381, 0.0472]** | **0.0490 [0.0450, 0.0529]** | **0.0523 [0.0491, 0.0555]** | **0.0534 [0.0517, 0.0552]** | **0.0409 [0.0376, 0.0441]** | **0.0476 [0.0443, 0.0508]** | **0.0461 [0.0442, 0.0480]** | **0.0427 [0.0403, 0.0452]** | **0.0437 [0.0381, 0.0494]** | **0.0483 [0.0451, 0.0514]** | 0.00 | 140596 | 949.88 |
| hyper | 0.0749 [0.0730, 0.0768] | 0.0519 [0.0503, 0.0536] | 0.0579 [0.0562, 0.0596] | 0.0626 [0.0597, 0.0656] | 0.0616 [0.0571, 0.0661] | 0.0484 [0.0461, 0.0507] | 0.0573 [0.0551, 0.0596] | 0.0567 [0.0549, 0.0586] | 0.0494 [0.0480, 0.0508] | 0.0535 [0.0513, 0.0556] | 0.0572 [0.0553, 0.0591] | 0.00 | 137480 | 917.19 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | **+13.2% [+6, +21]** | **+22.0% [+6, +38]** | **+18.3% [+6, +31]** | **+19.9% [+7, +33]** | **+15.3% [+3, +28]** | **+18.5% [+4, +33]** | **+20.6% [+8, +33]** | **+23.1% [+14, +32]** | **+15.6% [+6, +25]** | **+22.6% [+5, +41]** | **+18.6% [+7, +30]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0491 [0.0479, 0.0504]** | **0.0488 [0.0453, 0.0522]** | **0.0521 [0.0494, 0.0548]** | **0.0521 [0.0499, 0.0544]** | **0.0505 [0.0482, 0.0529]** | 0.00 | 140596 | 949.88 |
| hyper | 0.0591 [0.0554, 0.0629] | 0.0581 [0.0552, 0.0609] | 0.0621 [0.0595, 0.0647] | 0.0627 [0.0602, 0.0652] | 0.0605 [0.0577, 0.0634] | 0.00 | 137480 | 917.19 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **+20.4% [+10, +31]** | **+19.2% [+5, +33]** | **+19.4% [+8, +30]** | **+20.4% [+11, +30]** | **+19.8% [+9, +31]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0523 [0.0511, 0.0535]** | **0.0523 [0.0511, 0.0535]** | 0.00 | 140596 | 949.88 |
| hyper | 0.0629 [0.0566, 0.0691] | 0.0629 [0.0566, 0.0691] | 0.00 | 137480 | 917.19 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **+20.2% [+5, +35]** | **+20.2% [+5, +35]** |

## lorenz

phase 1: eps_enc_rel 0.177, eps_dec 0.267, eps_pde_rms 77, eps_pde_ss_rms 8.57, l_dec_q95 10.1, l_dec_mean 3.84 · mean job wall 3.8 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | 0.0773 [0.0555, 0.0990] | 0.0173 [0.0137, 0.0210] | 0.0881 [0.0682, 0.1079] | 0.0252 [0.0212, 0.0292] | 0.0222 [0.0139, 0.0306] | 0.0172 [0.0133, 0.0210] | 0.0260 [0.0138, 0.0383] | 0.0222 [0.0201, 0.0243] | 0.0186 [0.0132, 0.0241] | 0.0162 [0.0119, 0.0205] | 0.0321 [0.0244, 0.0399] | 0.00 | 141132 | 2315.91 |
| hyper | **0.0586 [0.0361, 0.0811]** | **0.0152 [0.0137, 0.0167]** | **0.0843 [0.0792, 0.0893]** | **0.0178 [0.0134, 0.0221]** | **0.0178 [0.0147, 0.0209]** | **0.0145 [0.0130, 0.0160]** | **0.0186 [0.0121, 0.0251]** | **0.0170 [0.0156, 0.0184]** | **0.0151 [0.0136, 0.0166]** | **0.0141 [0.0128, 0.0154]** | **0.0266 [0.0237, 0.0294]** | 0.00 | 139020 | 1781.42 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | -23.8% [-54, +7] | **-12.2% [-22, -2]** | -3.9% [-21, +13] | **-29.2% [-58, -1]** | -18.1% [-65, +28] | **-15.4% [-26, -5]** | -25.9% [-84, +32] | **-23.5% [-28, -19]** | **-18.3% [-35, -2]** | -12.3% [-27, +3] | -16.8% [-40, +6] |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0222 [0.0161, 0.0284] | 0.0251 [0.0160, 0.0341] | 0.0243 [0.0171, 0.0314] | 0.0283 [0.0170, 0.0396] | 0.0250 [0.0169, 0.0330] | 0.00 | 141132 | 2315.91 |
| hyper | **0.0162 [0.0151, 0.0173]** | **0.0202 [0.0097, 0.0307]** | **0.0179 [0.0133, 0.0225]** | **0.0206 [0.0105, 0.0308]** | **0.0187 [0.0122, 0.0252]** | 0.00 | 139020 | 1781.42 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **-26.4% [-53, -0]** | -16.6% [-89, +56] | -25.1% [-65, +14] | -24.0% [-93, +45] | -23.1% [-75, +29] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0247 [0.0212, 0.0282] | 0.0247 [0.0212, 0.0282] | 0.00 | 141132 | 2315.91 |
| hyper | **0.0177 [0.0149, 0.0206]** | **0.0177 [0.0149, 0.0206]** | 0.00 | 139020 | 1781.42 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **-27.9% [-50, -6]** | **-27.9% [-50, -6]** |

## quad_tank

phase 1: eps_enc_rel 0.0128, eps_dec 0.139, eps_pde_rms 0.0188, eps_pde_ss_rms 0.0486, l_dec_q95 13.6, l_dec_mean 6.59 · mean job wall 4.1 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0656 [0.0486, 0.0827]** | 0.0479 [0.0451, 0.0508] | **0.0638 [0.0616, 0.0660]** | **0.0534 [0.0464, 0.0603]** | **0.0621 [0.0575, 0.0667]** | **0.0472 [0.0444, 0.0500]** | 0.0531 [0.0502, 0.0560] | **0.0651 [0.0601, 0.0702]** | **0.0612 [0.0530, 0.0693]** | **0.0542 [0.0478, 0.0606]** | **0.0573 [0.0520, 0.0626]** | 0.00 | 141128 | 1069.09 |
| hyper | 0.0737 [0.0620, 0.0854] | **0.0479 [0.0430, 0.0528]** | 0.0659 [0.0627, 0.0692] | 0.0583 [0.0502, 0.0665] | 0.0653 [0.0550, 0.0757] | 0.0477 [0.0433, 0.0520] | **0.0531 [0.0441, 0.0622]** | 0.0682 [0.0629, 0.0735] | 0.0630 [0.0558, 0.0702] | 0.0610 [0.0532, 0.0687] | 0.0603 [0.0539, 0.0668] | 0.00 | 139024 | 1411.86 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +12.7% [-5, +31] | -0.0% [-8, +8] | **+3.4% [+1, +5]** | **+9.3% [+2, +16]** | +5.1% [-4, +14] | +1.0% [-7, +9] | -0.1% [-12, +12] | +4.7% [-1, +11] | +3.0% [-1, +7] | **+12.5% [+4, +21]** | **+5.3% [+4, +7]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0564 [0.0504, 0.0623] | **0.0510 [0.0480, 0.0540]** | **0.0554 [0.0516, 0.0591]** | **0.0575 [0.0529, 0.0620]** | **0.0551 [0.0508, 0.0593]** | 0.00 | 141128 | 1069.09 |
| hyper | **0.0563 [0.0488, 0.0638]** | 0.0513 [0.0469, 0.0558] | 0.0558 [0.0510, 0.0605] | 0.0579 [0.0512, 0.0646] | 0.0553 [0.0496, 0.0610] | 0.00 | 139024 | 1411.86 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | -0.2% [-3, +3] | +0.7% [-2, +3] | +0.7% [-4, +6] | +0.6% [-3, +4] | +0.5% [-2, +3] |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | 0.0695 [0.0668, 0.0723] | 0.0695 [0.0668, 0.0723] | 0.00 | 141128 | 1069.09 |
| hyper | **0.0674 [0.0617, 0.0731]** | **0.0674 [0.0617, 0.0731]** | 0.00 | 139024 | 1411.86 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | -3.1% [-8, +2] | -3.1% [-8, +2] |

## vdp

phase 1: eps_enc_rel 0.00385, eps_dec 0.0112, eps_pde_rms 0.275, eps_pde_ss_rms 0.0566, l_dec_q95 23.2, l_dec_mean 9.73 · mean job wall 3.5 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.4% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0210 [0.0160, 0.0260]** | **0.0149 [0.0104, 0.0194]** | **0.0174 [0.0155, 0.0193]** | **0.0156 [0.0112, 0.0201]** | **0.0170 [0.0125, 0.0215]** | **0.0162 [0.0119, 0.0204]** | **0.0151 [0.0127, 0.0174]** | **0.0162 [0.0132, 0.0192]** | **0.0159 [0.0135, 0.0184]** | **0.0138 [0.0097, 0.0180]** | **0.0163 [0.0126, 0.0199]** | 0.00 | 140596 | 1191.21 |
| hyper | 0.0245 [0.0234, 0.0256] | 0.0234 [0.0228, 0.0241] | 0.0263 [0.0241, 0.0286] | 0.0244 [0.0243, 0.0246] | 0.0259 [0.0252, 0.0265] | 0.0239 [0.0233, 0.0244] | 0.0234 [0.0223, 0.0244] | 0.0256 [0.0245, 0.0268] | 0.0264 [0.0257, 0.0271] | 0.0225 [0.0213, 0.0236] | 0.0246 [0.0241, 0.0252] | 0.00 | 137480 | 1246.53 |

Paired vs `lora`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyper | +17.5% [-7, +42] | **+58.4% [+15, +102]** | **+50.9% [+41, +61]** | **+57.5% [+15, +100]** | **+53.3% [+17, +90]** | **+48.9% [+13, +85]** | **+55.5% [+28, +83]** | **+58.8% [+32, +86]** | **+66.2% [+37, +95]** | **+63.8% [+23, +105]** | **+52.2% [+21, +83]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0163 [0.0130, 0.0196]** | **0.0153 [0.0108, 0.0198]** | **0.0165 [0.0134, 0.0197]** | **0.0152 [0.0116, 0.0189]** | **0.0159 [0.0122, 0.0195]** | 0.00 | 140596 | 1191.21 |
| hyper | 0.0257 [0.0245, 0.0270] | 0.0244 [0.0237, 0.0251] | 0.0255 [0.0253, 0.0257] | 0.0248 [0.0242, 0.0254] | 0.0251 [0.0244, 0.0258] | 0.00 | 137480 | 1246.53 |

Paired vs `lora`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| hyper | **+58.1% [+32, +84]** | **+60.8% [+19, +103]** | **+54.6% [+27, +82]** | **+63.5% [+30, +97]** | **+59.1% [+27, +91]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0181 [0.0148, 0.0214]** | **0.0181 [0.0148, 0.0214]** | 0.00 | 140596 | 1191.21 |
| hyper | 0.0267 [0.0261, 0.0273] | 0.0267 [0.0261, 0.0273] | 0.00 | 137480 | 1246.53 |

Paired vs `lora`:

| method | drive | all |
|---|---|---|
| hyper | **+47.7% [+22, +73]** | **+47.7% [+22, +73]** |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0984 | 0.1055 | 0.1425 | 0.1606 | 0.1972 | 0.00 (shape_ood/clean) | 0.881 | 1.78e+03 |
| hyper | 0.1066 | 0.1073 | 0.1455 | 0.1633 | 0.2073 | 0.00 (shape_ood/clean) | 0.896 | 1.14e+03 |

### bioreactor

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0182 | 0.0216 | 0.0412 | 0.0450 | 0.0550 | 0.00 (shape_ood/clean) | 0.914 | 1.72e+03 |
| hyper | 0.0239 | 0.0306 | 0.0583 | 0.0694 | 0.0753 | 0.00 (shape_ood/clean) | 0.884 | 1.74e+03 |

### duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0483 | 0.0505 | 0.0523 | 0.0726 | 0.0608 | 0.00 (shape_ood/clean) | 0.603 | 1.16e+03 |
| hyper | 0.0572 | 0.0605 | 0.0629 | 0.0879 | 0.0706 | 0.00 (shape_ood/clean) | 0.588 | 1.21e+03 |

### lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0321 | 0.0250 | 0.0247 | 0.0230 | 0.0452 | 0.00 (shape_ood/clean) | 0.605 | 1.59e+03 |
| hyper | 0.0266 | 0.0187 | 0.0177 | 0.0169 | 0.0269 | 0.00 (shape_ood/clean) | 0.585 | 1.48e+03 |

### quad_tank

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0573 | 0.0551 | 0.0695 | 0.0929 | 0.0879 | 0.00 (shape_ood/clean) | 0.948 | 1.29e+03 |
| hyper | 0.0603 | 0.0553 | 0.0674 | 0.0903 | 0.0897 | 0.00 (shape_ood/clean) | 0.949 | 1.27e+03 |

### vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0163 | 0.0159 | 0.0181 | 0.0236 | 0.0212 | 0.00 (shape_ood/clean) | 0.636 | 1.18e+03 |
| hyper | 0.0246 | 0.0251 | 0.0267 | 0.0362 | 0.0325 | 0.00 (shape_ood/clean) | 0.616 | 1.18e+03 |
