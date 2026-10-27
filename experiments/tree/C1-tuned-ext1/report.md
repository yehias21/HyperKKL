# C1-tuned-ext1 — Tuned recipe after extension ext1 (new choices only), dev systems

*parent* `C1-tuned` · *contribution* C1 · *status* done · jobs 18/18

**Question.** Does the C1-tuned verdict change after the equal ext1 extension?

**Hypothesis.** As C1-tuned.

**Decision rule.** Verdict 'ext1': the C1-tuned rule on selection_ext1b.json (PROTOCOL.md sections 9-10).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (shape_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0684 | 0.0295 | 0.0706 | 0.0255 | 0.1635 | 0.0209 |
| concat[lr=0.016,concat_depth=3] | 0.0538 | 0.0281 | 0.0507 | 0.0169 | 0.0697 | 0.0239 |
| lora[lr=0.008,rank=16] | 0.0341 | 0.0157 | 0.0478 | 0.0209 | 0.0494 | 0.0136 |

## Scoreboard (sequence_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0750 | 0.0329 | 0.0772 | 0.0257 | 0.1973 | 0.0223 |
| concat[lr=0.016,concat_depth=3] | 0.0784 | 0.0561 | 0.0690 | 0.0174 | 0.0778 | 0.0256 |
| lora[lr=0.008,rank=16] | 0.0385 | 0.0309 | 0.0514 | 0.0196 | 0.0607 | 0.0161 |

## Scoreboard (id / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0741 | 0.0285 | 0.0684 | 0.0280 | 0.1669 | 0.0211 |
| concat[lr=0.016,concat_depth=3] | 0.0520 | 0.0228 | 0.0477 | 0.0181 | 0.0705 | 0.0243 |
| lora[lr=0.008,rank=16] | 0.0295 | 0.0132 | 0.0461 | 0.0221 | 0.0497 | 0.0144 |

## battery

phase 1: eps_enc_rel 0.00489, eps_dec 1.83, eps_pde_rms 0.302, eps_pde_ss_rms 0.0448, l_dec_q95 806, l_dec_mean 370 · mean job wall 58.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0911 [0.0722, 0.1100] | 0.0661 [0.0438, 0.0885] | 0.0732 [0.0602, 0.0862] | 0.0862 [0.0529, 0.1194] | 0.0629 [0.0569, 0.0689] | 0.0909 [0.0043, 0.1775] | 0.0613 [0.0463, 0.0762] | 0.0716 [0.0664, 0.0769] | 0.0773 [0.0462, 0.1084] | 0.0645 [0.0239, 0.1052] | 0.0741 [0.0478, 0.1004] | 0.00 | 139627 | 869.77 |
| concat[lr=0.016,concat_depth=3] | 0.0603 [0.0523, 0.0683] | 0.0426 [0.0332, 0.0519] | 0.0522 [0.0375, 0.0670] | 0.0644 [0.0505, 0.0782] | 0.0591 [0.0525, 0.0656] | 0.0606 [0.0529, 0.0683] | 0.0390 [0.0346, 0.0434] | 0.0546 [0.0497, 0.0594] | 0.0523 [0.0506, 0.0540] | 0.0378 [0.0329, 0.0427] | 0.0520 [0.0458, 0.0583] | 0.00 | 140070 | 632.11 |
| lora[lr=0.008,rank=16] | **0.0409 [0.0356, 0.0462]** | **0.0260 [0.0203, 0.0318]** | **0.0355 [0.0293, 0.0417]** | **0.0386 [0.0233, 0.0538]** | **0.0303 [0.0284, 0.0322]** | **0.0198 [0.0161, 0.0235]** | **0.0230 [0.0188, 0.0272]** | **0.0317 [0.0264, 0.0370]** | **0.0325 [0.0069, 0.0582]** | **0.0184 [0.0158, 0.0211]** | **0.0295 [0.0254, 0.0336]** | 0.00 | 143833 | 753.82 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0694 [0.0609, 0.0778] | 0.0714 [0.0428, 0.1001] | 0.0731 [0.0607, 0.0855] | 0.0596 [0.0414, 0.0779] | 0.0684 [0.0525, 0.0843] | 0.00 | 139627 | 869.77 |
| concat[lr=0.016,concat_depth=3] | 0.0525 [0.0481, 0.0569] | 0.0564 [0.0395, 0.0733] | 0.0635 [0.0496, 0.0773] | 0.0426 [0.0368, 0.0485] | 0.0538 [0.0439, 0.0636] | 0.00 | 140070 | 632.11 |
| lora[lr=0.008,rank=16] | **0.0351 [0.0300, 0.0402]** | **0.0357 [0.0208, 0.0506]** | **0.0426 [0.0298, 0.0553]** | **0.0231 [0.0183, 0.0278]** | **0.0341 [0.0260, 0.0422]** | 0.00 | 143833 | 753.82 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0750 [0.0536, 0.0965] | 0.0750 [0.0536, 0.0965] | 0.00 | 139627 | 869.77 |
| concat[lr=0.016,concat_depth=3] | 0.0784 [0.0701, 0.0867] | 0.0784 [0.0701, 0.0867] | 0.00 | 140070 | 632.11 |
| lora[lr=0.008,rank=16] | **0.0385 [0.0294, 0.0477]** | **0.0385 [0.0294, 0.0477]** | 0.00 | 143833 | 753.82 |

## bioreactor

phase 1: eps_enc_rel 0.00825, eps_dec 0.0199, eps_pde_rms 0.00738, eps_pde_ss_rms 0.00456, l_dec_q95 44.1, l_dec_mean 18 · mean job wall 70.7 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0326 [0.0308, 0.0343] | 0.0258 [0.0165, 0.0352] | 0.0268 [0.0228, 0.0308] | 0.0292 [0.0254, 0.0330] | 0.0295 [0.0240, 0.0350] | 0.0294 [0.0209, 0.0378] | 0.0271 [0.0200, 0.0342] | 0.0306 [0.0223, 0.0388] | 0.0283 [0.0209, 0.0358] | 0.0265 [0.0179, 0.0351] | 0.0285 [0.0221, 0.0350] | 0.00 | 139685 | 837.62 |
| concat[lr=0.016,concat_depth=3] | 0.0318 [0.0240, 0.0396] | 0.0137 [0.0101, 0.0172] | 0.0199 [0.0152, 0.0246] | 0.0245 [0.0184, 0.0306] | 0.0298 [0.0184, 0.0411] | 0.0185 [0.0156, 0.0215] | 0.0277 [0.0177, 0.0376] | 0.0234 [0.0169, 0.0299] | 0.0232 [0.0171, 0.0292] | 0.0167 [0.0127, 0.0208] | 0.0228 [0.0171, 0.0284] | 0.00 | 139838 | 579.47 |
| lora[lr=0.008,rank=16] | **0.0182 [0.0070, 0.0293]** | **0.0072 [0.0054, 0.0089]** | **0.0124 [0.0076, 0.0172]** | **0.0119 [0.0087, 0.0151]** | **0.0191 [0.0134, 0.0247]** | **0.0114 [0.0101, 0.0127]** | **0.0156 [0.0117, 0.0195]** | **0.0149 [0.0095, 0.0203]** | **0.0144 [0.0127, 0.0161]** | **0.0076 [0.0071, 0.0082]** | **0.0132 [0.0104, 0.0160]** | 0.00 | 142921 | 634.65 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0288 [0.0215, 0.0361] | 0.0301 [0.0249, 0.0354] | 0.0287 [0.0245, 0.0329] | 0.0305 [0.0235, 0.0375] | 0.0295 [0.0236, 0.0355] | 0.00 | 139685 | 837.62 |
| concat[lr=0.016,concat_depth=3] | 0.0298 [0.0099, 0.0496] | 0.0229 [0.0174, 0.0284] | 0.0354 [0.0257, 0.0452] | 0.0244 [0.0152, 0.0337] | 0.0281 [0.0171, 0.0391] | 0.00 | 139838 | 579.47 |
| lora[lr=0.008,rank=16] | **0.0189 [0.0168, 0.0209]** | **0.0107 [0.0078, 0.0135]** | **0.0188 [0.0138, 0.0238]** | **0.0146 [0.0105, 0.0187]** | **0.0157 [0.0124, 0.0190]** | 0.00 | 142921 | 634.65 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0329 [0.0284, 0.0373] | 0.0329 [0.0284, 0.0373] | 0.00 | 139685 | 837.62 |
| concat[lr=0.016,concat_depth=3] | 0.0561 [0.0342, 0.0780] | 0.0561 [0.0342, 0.0780] | 0.00 | 139838 | 579.47 |
| lora[lr=0.008,rank=16] | **0.0309 [0.0267, 0.0351]** | **0.0309 [0.0267, 0.0351]** | 0.00 | 142921 | 634.65 |

## duffing

phase 1: eps_enc_rel 0.00444, eps_dec 0.00608, eps_pde_rms 0.00261, eps_pde_ss_rms 0.000269, l_dec_q95 295, l_dec_mean 137 · mean job wall 73.8 min

- state coverage id: 0.0% of eval states outside the training box, 1.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 2.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0765 [0.0711, 0.0819] | 0.0629 [0.0423, 0.0835] | 0.0666 [0.0562, 0.0770] | 0.0701 [0.0625, 0.0777] | 0.0746 [0.0701, 0.0790] | 0.0616 [0.0410, 0.0823] | 0.0714 [0.0656, 0.0772] | 0.0727 [0.0650, 0.0805] | 0.0662 [0.0561, 0.0763] | 0.0628 [0.0444, 0.0813] | 0.0684 [0.0575, 0.0794] | 0.00 | 139685 | 1030.91 |
| concat[lr=0.016,concat_depth=3] | 0.0709 [0.0655, 0.0764] | **0.0350 [0.0281, 0.0419]** | **0.0438 [0.0414, 0.0463]** | 0.0515 [0.0487, 0.0542] | 0.0605 [0.0529, 0.0681] | **0.0335 [0.0284, 0.0387]** | 0.0467 [0.0460, 0.0474] | 0.0537 [0.0494, 0.0580] | 0.0464 [0.0446, 0.0481] | **0.0372 [0.0299, 0.0445]** | 0.0477 [0.0458, 0.0496] | 0.00 | 139838 | 714.08 |
| lora[lr=0.008,rank=16] | **0.0679 [0.0538, 0.0819]** | 0.0393 [0.0161, 0.0625] | 0.0453 [0.0269, 0.0638] | **0.0504 [0.0280, 0.0729]** | **0.0512 [0.0288, 0.0736]** | 0.0370 [0.0197, 0.0544] | **0.0445 [0.0230, 0.0661]** | **0.0460 [0.0274, 0.0646]** | **0.0399 [0.0241, 0.0557]** | 0.0413 [0.0225, 0.0600] | **0.0461 [0.0269, 0.0653]** | 0.00 | 142921 | 848.35 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0725 [0.0663, 0.0787] | 0.0664 [0.0549, 0.0780] | 0.0702 [0.0640, 0.0764] | 0.0735 [0.0665, 0.0804] | 0.0706 [0.0629, 0.0783] | 0.00 | 139685 | 1030.91 |
| concat[lr=0.016,concat_depth=3] | 0.0527 [0.0492, 0.0562] | **0.0424 [0.0382, 0.0466]** | 0.0542 [0.0535, 0.0549] | 0.0536 [0.0512, 0.0559] | 0.0507 [0.0488, 0.0526] | 0.00 | 139838 | 714.08 |
| lora[lr=0.008,rank=16] | **0.0469 [0.0254, 0.0684]** | 0.0450 [0.0218, 0.0682] | **0.0501 [0.0280, 0.0722]** | **0.0490 [0.0243, 0.0738]** | **0.0478 [0.0249, 0.0706]** | 0.00 | 142921 | 848.35 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0772 [0.0715, 0.0828] | 0.0772 [0.0715, 0.0828] | 0.00 | 139685 | 1030.91 |
| concat[lr=0.016,concat_depth=3] | 0.0690 [0.0575, 0.0805] | 0.0690 [0.0575, 0.0805] | 0.00 | 139838 | 714.08 |
| lora[lr=0.008,rank=16] | **0.0514 [0.0281, 0.0747]** | **0.0514 [0.0281, 0.0747]** | 0.00 | 142921 | 848.35 |

## lorenz

phase 1: eps_enc_rel 0.177, eps_dec 0.267, eps_pde_rms 77, eps_pde_ss_rms 8.57, l_dec_q95 10.1, l_dec_mean 3.84 · mean job wall 60.7 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0516 [0.0083, 0.0949] | 0.0257 [0.0111, 0.0402] | 0.0247 [0.0116, 0.0378] | 0.0252 [0.0094, 0.0410] | 0.0264 [0.0074, 0.0455] | 0.0251 [0.0127, 0.0374] | 0.0252 [0.0073, 0.0431] | 0.0272 [0.0103, 0.0441] | 0.0253 [0.0089, 0.0417] | 0.0250 [0.0109, 0.0392] | 0.0280 [0.0098, 0.0462] | 0.00 | 139627 | 788.90 |
| concat[lr=0.016,concat_depth=3] | **0.0266 [0.0142, 0.0390]** | 0.0177 [0.0156, 0.0197] | **0.0171 [0.0147, 0.0195]** | **0.0167 [0.0142, 0.0192]** | **0.0176 [0.0154, 0.0199]** | 0.0171 [0.0148, 0.0193] | **0.0168 [0.0147, 0.0190]** | **0.0185 [0.0163, 0.0208]** | **0.0169 [0.0149, 0.0189]** | 0.0170 [0.0148, 0.0192] | **0.0181 [0.0170, 0.0193]** | 0.00 | 139554 | 575.56 |
| lora[lr=0.008,rank=16] | 0.0563 [0.0431, 0.0694] | **0.0151 [0.0120, 0.0182]** | 0.0226 [-0.0032, 0.0484] | 0.0207 [0.0144, 0.0271] | 0.0200 [0.0108, 0.0293] | **0.0144 [0.0117, 0.0171]** | 0.0194 [0.0092, 0.0295] | 0.0201 [0.0137, 0.0265] | 0.0210 [0.0113, 0.0307] | **0.0144 [0.0112, 0.0175]** | 0.0221 [0.0165, 0.0277] | 0.00 | 143481 | 759.90 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0260 [0.0071, 0.0449] | 0.0252 [0.0084, 0.0420] | 0.0253 [0.0090, 0.0415] | 0.0255 [0.0097, 0.0413] | 0.0255 [0.0086, 0.0424] | 0.00 | 139627 | 788.90 |
| concat[lr=0.016,concat_depth=3] | **0.0168 [0.0150, 0.0187]** | **0.0168 [0.0153, 0.0183]** | **0.0166 [0.0149, 0.0183]** | **0.0173 [0.0155, 0.0191]** | **0.0169 [0.0152, 0.0186]** | 0.00 | 139554 | 575.56 |
| lora[lr=0.008,rank=16] | 0.0198 [0.0141, 0.0255] | 0.0190 [0.0113, 0.0268] | 0.0214 [0.0097, 0.0332] | 0.0235 [0.0139, 0.0332] | 0.0209 [0.0127, 0.0291] | 0.00 | 143481 | 759.90 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0257 [0.0089, 0.0426] | 0.0257 [0.0089, 0.0426] | 0.00 | 139627 | 788.90 |
| concat[lr=0.016,concat_depth=3] | **0.0174 [0.0155, 0.0194]** | **0.0174 [0.0155, 0.0194]** | 0.00 | 139554 | 575.56 |
| lora[lr=0.008,rank=16] | 0.0196 [0.0155, 0.0237] | 0.0196 [0.0155, 0.0237] | 0.00 | 143481 | 759.90 |

## quad_tank

phase 1: eps_enc_rel 0.0128, eps_dec 0.139, eps_pde_rms 0.0188, eps_pde_ss_rms 0.0486, l_dec_q95 13.6, l_dec_mean 6.59 · mean job wall 34.0 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.1494 [0.1321, 0.1666] | 0.1612 [0.1482, 0.1741] | 0.1741 [0.1563, 0.1920] | 0.1637 [0.1517, 0.1757] | 0.1850 [0.1658, 0.2042] | 0.1620 [0.1480, 0.1760] | 0.1503 [0.1399, 0.1606] | 0.1855 [0.1661, 0.2049] | 0.1716 [0.1513, 0.1918] | 0.1637 [0.1488, 0.1786] | 0.1669 [0.1511, 0.1826] | 0.00 | 139657 | 785.48 |
| concat[lr=0.016,concat_depth=3] | 0.0716 [0.0421, 0.1011] | 0.0650 [0.0454, 0.0846] | 0.0747 [0.0543, 0.0951] | 0.0700 [0.0406, 0.0994] | 0.0756 [0.0431, 0.1081] | 0.0643 [0.0439, 0.0846] | 0.0636 [0.0369, 0.0904] | 0.0753 [0.0517, 0.0988] | 0.0692 [0.0437, 0.0947] | 0.0756 [0.0338, 0.1174] | 0.0705 [0.0438, 0.0973] | 0.05 | 139555 | 550.71 |
| lora[lr=0.008,rank=16] | **0.0530 [0.0464, 0.0596]** | **0.0441 [0.0402, 0.0480]** | **0.0535 [0.0457, 0.0613]** | **0.0467 [0.0453, 0.0480]** | **0.0553 [0.0500, 0.0606]** | **0.0437 [0.0400, 0.0475]** | **0.0463 [0.0454, 0.0471]** | **0.0570 [0.0541, 0.0599]** | **0.0487 [0.0413, 0.0562]** | **0.0495 [0.0476, 0.0514]** | **0.0497 [0.0476, 0.0519]** | 0.00 | 143465 | 657.17 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.1700 [0.1535, 0.1864] | 0.1550 [0.1409, 0.1692] | 0.1623 [0.1461, 0.1785] | 0.1665 [0.1520, 0.1810] | 0.1635 [0.1481, 0.1788] | 0.00 | 139657 | 785.48 |
| concat[lr=0.016,concat_depth=3] | 0.0709 [0.0415, 0.1002] | 0.0675 [0.0471, 0.0879] | 0.0701 [0.0489, 0.0912] | 0.0702 [0.0432, 0.0972] | 0.0697 [0.0452, 0.0941] | 0.06 | 139555 | 550.71 |
| lora[lr=0.008,rank=16] | **0.0503 [0.0471, 0.0535]** | **0.0478 [0.0453, 0.0503]** | **0.0492 [0.0475, 0.0509]** | **0.0505 [0.0465, 0.0545]** | **0.0494 [0.0480, 0.0509]** | 0.00 | 143465 | 657.17 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.1973 [0.1717, 0.2229] | 0.1973 [0.1717, 0.2229] | 0.00 | 139657 | 785.48 |
| concat[lr=0.016,concat_depth=3] | 0.0778 [0.0573, 0.0983] | 0.0778 [0.0573, 0.0983] | 0.09 | 139555 | 550.71 |
| lora[lr=0.008,rank=16] | **0.0607 [0.0542, 0.0672]** | **0.0607 [0.0542, 0.0672]** | 0.00 | 143465 | 657.17 |

## vdp

phase 1: eps_enc_rel 0.00385, eps_dec 0.0112, eps_pde_rms 0.275, eps_pde_ss_rms 0.0566, l_dec_q95 23.2, l_dec_mean 9.73 · mean job wall 33.0 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.4% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0241 [0.0225, 0.0258] | 0.0201 [0.0161, 0.0240] | 0.0216 [0.0164, 0.0268] | 0.0202 [0.0169, 0.0236] | 0.0224 [0.0183, 0.0264] | 0.0211 [0.0170, 0.0253] | 0.0191 [0.0159, 0.0223] | 0.0213 [0.0170, 0.0255] | 0.0213 [0.0174, 0.0251] | 0.0198 [0.0151, 0.0244] | 0.0211 [0.0173, 0.0249] | 0.00 | 139685 | 677.88 |
| concat[lr=0.016,concat_depth=3] | 0.0243 [0.0229, 0.0257] | 0.0236 [0.0224, 0.0249] | 0.0255 [0.0242, 0.0268] | 0.0241 [0.0232, 0.0250] | 0.0249 [0.0226, 0.0272] | 0.0245 [0.0242, 0.0248] | 0.0227 [0.0226, 0.0228] | 0.0252 [0.0241, 0.0263] | 0.0256 [0.0250, 0.0263] | 0.0228 [0.0213, 0.0242] | 0.0243 [0.0233, 0.0253] | 0.00 | 139838 | 515.01 |
| lora[lr=0.008,rank=16] | **0.0195 [0.0146, 0.0244]** | **0.0129 [0.0128, 0.0130]** | **0.0144 [0.0139, 0.0150]** | **0.0136 [0.0133, 0.0138]** | **0.0154 [0.0133, 0.0175]** | **0.0138 [0.0131, 0.0144]** | **0.0128 [0.0125, 0.0130]** | **0.0155 [0.0128, 0.0182]** | **0.0147 [0.0137, 0.0158]** | **0.0121 [0.0114, 0.0128]** | **0.0144 [0.0135, 0.0154]** | 0.00 | 142921 | 697.74 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0218 [0.0173, 0.0263] | 0.0204 [0.0167, 0.0242] | 0.0211 [0.0170, 0.0253] | 0.0204 [0.0166, 0.0242] | 0.0209 [0.0169, 0.0250] | 0.00 | 139685 | 677.88 |
| concat[lr=0.016,concat_depth=3] | 0.0248 [0.0234, 0.0262] | 0.0234 [0.0222, 0.0246] | 0.0241 [0.0233, 0.0250] | 0.0232 [0.0222, 0.0242] | 0.0239 [0.0228, 0.0250] | 0.00 | 139838 | 515.01 |
| lora[lr=0.008,rank=16] | **0.0143 [0.0137, 0.0149]** | **0.0130 [0.0128, 0.0132]** | **0.0139 [0.0136, 0.0142]** | **0.0133 [0.0127, 0.0139]** | **0.0136 [0.0133, 0.0139]** | 0.00 | 142921 | 697.74 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0223 [0.0180, 0.0266] | 0.0223 [0.0180, 0.0266] | 0.00 | 139685 | 677.88 |
| concat[lr=0.016,concat_depth=3] | 0.0256 [0.0234, 0.0278] | 0.0256 [0.0234, 0.0278] | 0.00 | 139838 | 515.01 |
| lora[lr=0.008,rank=16] | **0.0161 [0.0149, 0.0173]** | **0.0161 [0.0149, 0.0173]** | 0.00 | 142921 | 697.74 |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0741 | 0.0684 | 0.0750 | 0.0827 | 0.1052 | 0.00 (shape_ood/clean) | 0.952 | 866 |
| concat[lr=0.016,concat_depth=3] | 0.0520 | 0.0538 | 0.0784 | 0.0931 | 0.0996 | 0.00 (shape_ood/clean) | 0.939 | 554 |
| lora[lr=0.008,rank=16] | 0.0295 | 0.0341 | 0.0385 | 0.0427 | 0.0875 | 0.00 (shape_ood/clean) | 0.893 | 766 |

### bioreactor

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0285 | 0.0295 | 0.0329 | 0.0437 | 0.0391 | 0.00 (shape_ood/clean) | 0.930 | 796 |
| concat[lr=0.016,concat_depth=3] | 0.0228 | 0.0281 | 0.0561 | 0.0685 | 0.0728 | 0.00 (shape_ood/clean) | 0.868 | 586 |
| lora[lr=0.008,rank=16] | 0.0132 | 0.0157 | 0.0309 | 0.0319 | 0.0390 | 0.00 (shape_ood/clean) | 0.896 | 771 |

### duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0684 | 0.0706 | 0.0772 | 0.1084 | 0.0834 | 0.00 (shape_ood/clean) | 0.632 | 924 |
| concat[lr=0.016,concat_depth=3] | 0.0477 | 0.0507 | 0.0690 | 0.0970 | 0.0809 | 0.00 (shape_ood/clean) | 0.601 | 646 |
| lora[lr=0.008,rank=16] | 0.0461 | 0.0478 | 0.0514 | 0.0715 | 0.0577 | 0.00 (shape_ood/clean) | 0.610 | 786 |

### lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0280 | 0.0255 | 0.0257 | 0.0241 | 0.0334 | 0.00 (shape_ood/clean) | 0.673 | 765 |
| concat[lr=0.016,concat_depth=3] | 0.0181 | 0.0169 | 0.0174 | 0.0164 | 0.0225 | 0.00 (shape_ood/clean) | 0.658 | 545 |
| lora[lr=0.008,rank=16] | 0.0221 | 0.0209 | 0.0196 | 0.0182 | 0.0411 | 0.00 (shape_ood/clean) | 0.594 | 681 |

### quad_tank

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.1669 | 0.1635 | 0.1973 | 0.2632 | 0.2742 | 0.00 (shape_ood/clean) | 0.914 | 814 |
| concat[lr=0.016,concat_depth=3] | 0.0705 | 0.0697 | 0.0778 | 0.1087 | 0.1107 | 0.09 (sequence_ood/clean) | 0.945 | 561 |
| lora[lr=0.008,rank=16] | 0.0497 | 0.0494 | 0.0607 | 0.0789 | 0.0784 | 0.00 (shape_ood/clean) | 0.947 | 698 |

### vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0211 | 0.0209 | 0.0223 | 0.0299 | 0.0264 | 0.00 (shape_ood/clean) | 0.619 | 737 |
| concat[lr=0.016,concat_depth=3] | 0.0243 | 0.0239 | 0.0256 | 0.0346 | 0.0313 | 0.00 (shape_ood/clean) | 0.617 | 542 |
| lora[lr=0.008,rank=16] | 0.0144 | 0.0136 | 0.0161 | 0.0208 | 0.0209 | 0.00 (shape_ood/clean) | 0.658 | 712 |
