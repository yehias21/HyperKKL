# TEST-C1-tuned-ext1 — Tuned recipe after extension ext1 (new choices only), test

*parent* `TEST-C1-tuned` · *contribution* C1 · *status* done · jobs 18/18

**Question.** As C1-tuned-ext1, held-out test plants.

**Hypothesis.** As C1-tuned.

**Decision rule.** Verdict 'ext1' (PROTOCOL.md sections 9-10). Run once.

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (shape_ood / clean, all families)

| method | cartpole | fhn | highway_traffic | network | rossler | two_link_arm |
|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1239 | 0.0241 | 0.7626 | 0.0567 | 1.2954 | 0.3932 |
| concat[lr=0.016,concat_depth=3] | 0.3861 | 0.0275 | 0.2691 | 0.0468 | 0.1830 | 0.2925 |
| lora[lr=0.008,rank=16] | 0.1069 | 0.0277 | 0.0617 | 0.0320 | 0.0992 | 0.1192 |

## Scoreboard (sequence_ood / clean, all families)

| method | cartpole | fhn | highway_traffic | network | rossler | two_link_arm |
|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1548 | 0.0338 | 0.7575 | 0.0719 | 1.0909 | 0.4149 |
| concat[lr=0.016,concat_depth=3] | 0.3989 | 0.0485 | 0.2681 | 0.0782 | 0.2054 | 0.3041 |
| lora[lr=0.008,rank=16] | 0.1893 | 0.0427 | 0.0699 | 0.0491 | 0.0990 | 0.1318 |

## Scoreboard (id / clean, all families)

| method | cartpole | fhn | highway_traffic | network | rossler | two_link_arm |
|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1540 | 0.0224 | 0.7649 | 0.0646 | 1.1455 | 0.4010 |
| concat[lr=0.016,concat_depth=3] | 0.3823 | 0.0253 | 0.2700 | 0.0493 | 0.1799 | 0.2992 |
| lora[lr=0.008,rank=16] | 0.0942 | 0.0253 | 0.0616 | 0.0334 | 0.0988 | 0.1203 |

## cartpole

phase 1: eps_enc_rel 0.00177, eps_dec 36.1, eps_pde_rms 1.35, eps_pde_ss_rms 0.234, l_dec_q95 80.1, l_dec_mean 37.1 · mean job wall 32.1 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.1% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.6% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.0480 [0.7521, 1.3440] | 1.2206 [1.0086, 1.4326] | 1.2507 [1.0370, 1.4644] | 1.0939 [0.8548, 1.3329] | 1.1982 [0.9266, 1.4697] | 1.1795 [0.9386, 1.4204] | 1.1278 [0.8533, 1.4024] | 1.1279 [0.8817, 1.3742] | 1.0894 [0.8391, 1.3398] | 1.1898 [0.9517, 1.4280] | 1.1540 [0.9090, 1.3989] | 0.00 | 139657 | 822.15 |
| concat[lr=0.016,concat_depth=3] | 0.5114 [0.2881, 0.7347] | 0.3340 [0.2276, 0.4403] | 0.3623 [0.2451, 0.4794] | 0.4102 [0.2951, 0.5253] | 0.4154 [0.2762, 0.5545] | 0.3140 [0.2279, 0.4002] | 0.3483 [0.2462, 0.4504] | 0.4167 [0.2103, 0.6232] | 0.3931 [0.2813, 0.5049] | 0.3278 [0.2465, 0.4091] | 0.3823 [0.2720, 0.4926] | 0.00 | 139555 | 576.26 |
| lora[lr=0.008,rank=16] | **0.2270 [0.1115, 0.3425]** | **0.0329 [0.0270, 0.0388]** | **0.0666 [0.0104, 0.1229]** | **0.0955 [0.0084, 0.1826]** | **0.1366 [0.0852, 0.1880]** | **0.0391 [0.0337, 0.0445]** | **0.0610 [0.0227, 0.0992]** | **0.1248 [0.0341, 0.2155]** | **0.1155 [0.0679, 0.1631]** | **0.0499 [0.0339, 0.0659]** | **0.0942 [0.0591, 0.1293]** | 0.00 | 143465 | 704.58 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1091 [0.8504, 1.3678] | 1.1146 [0.8140, 1.4153] | 1.1853 [0.9175, 1.4531] | 1.0867 [0.8487, 1.3246] | 1.1239 [0.8599, 1.3879] | 0.00 | 139657 | 822.15 |
| concat[lr=0.016,concat_depth=3] | 0.3543 [0.2662, 0.4424] | 0.3986 [0.2907, 0.5065] | 0.4166 [0.2958, 0.5374] | 0.3747 [0.2504, 0.4990] | 0.3861 [0.2791, 0.4931] | 0.00 | 139555 | 576.26 |
| lora[lr=0.008,rank=16] | **0.0917 [0.0733, 0.1101]** | **0.1287 [0.1273, 0.1300]** | **0.1150 [0.0587, 0.1713]** | **0.0922 [0.0439, 0.1405]** | **0.1069 [0.0834, 0.1304]** | 0.00 | 143465 | 704.58 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1548 [0.8701, 1.4395] | 1.1548 [0.8701, 1.4395] | 0.00 | 139657 | 822.15 |
| concat[lr=0.016,concat_depth=3] | 0.3989 [0.3143, 0.4836] | 0.3989 [0.3143, 0.4836] | 0.00 | 139555 | 576.26 |
| lora[lr=0.008,rank=16] | **0.1893 [0.1549, 0.2237]** | **0.1893 [0.1549, 0.2237]** | 0.00 | 143465 | 704.58 |

## fhn

phase 1: eps_enc_rel 0.0279, eps_dec 0.0532, eps_pde_rms 0.0974, eps_pde_ss_rms 0.0602, l_dec_q95 7.7, l_dec_mean 2.64 · mean job wall 25.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | **0.0280 [0.0168, 0.0393]** | **0.0158 [0.0141, 0.0176]** | **0.0207 [0.0121, 0.0293]** | **0.0222 [0.0170, 0.0275]** | **0.0276 [0.0229, 0.0324]** | **0.0158 [0.0140, 0.0175]** | **0.0239 [0.0184, 0.0294]** | **0.0252 [0.0180, 0.0324]** | **0.0266 [0.0199, 0.0334]** | **0.0181 [0.0137, 0.0226]** | **0.0224 [0.0174, 0.0274]** | 0.00 | 139685 | 905.51 |
| concat[lr=0.016,concat_depth=3] | 0.0358 [0.0279, 0.0437] | 0.0177 [0.0108, 0.0247] | 0.0207 [0.0185, 0.0230] | 0.0269 [0.0247, 0.0290] | 0.0340 [0.0329, 0.0350] | 0.0164 [0.0119, 0.0209] | 0.0259 [0.0236, 0.0282] | 0.0294 [0.0265, 0.0324] | 0.0276 [0.0270, 0.0281] | 0.0193 [0.0178, 0.0208] | 0.0253 [0.0247, 0.0259] | 0.00 | 139838 | 604.39 |
| lora[lr=0.008,rank=16] | 0.0361 [0.0324, 0.0397] | 0.0183 [0.0121, 0.0244] | 0.0226 [0.0198, 0.0254] | 0.0268 [0.0240, 0.0297] | 0.0316 [0.0297, 0.0334] | 0.0169 [0.0126, 0.0211] | 0.0266 [0.0257, 0.0276] | 0.0284 [0.0253, 0.0316] | 0.0273 [0.0234, 0.0313] | 0.0195 [0.0177, 0.0214] | 0.0253 [0.0244, 0.0263] | 0.00 | 142921 | 700.32 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | **0.0271 [0.0185, 0.0357]** | **0.0227 [0.0188, 0.0266]** | **0.0241 [0.0193, 0.0289]** | **0.0225 [0.0165, 0.0286]** | **0.0241 [0.0183, 0.0299]** | 0.00 | 139685 | 905.51 |
| concat[lr=0.016,concat_depth=3] | 0.0318 [0.0237, 0.0400] | 0.0253 [0.0229, 0.0276] | 0.0280 [0.0269, 0.0290] | 0.0251 [0.0228, 0.0273] | 0.0275 [0.0245, 0.0306] | 0.00 | 139838 | 604.39 |
| lora[lr=0.008,rank=16] | 0.0288 [0.0256, 0.0319] | 0.0274 [0.0235, 0.0312] | 0.0290 [0.0271, 0.0309] | 0.0255 [0.0242, 0.0269] | 0.0277 [0.0258, 0.0296] | 0.00 | 142921 | 700.32 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | **0.0338 [0.0244, 0.0432]** | **0.0338 [0.0244, 0.0432]** | 0.00 | 139685 | 905.51 |
| concat[lr=0.016,concat_depth=3] | 0.0485 [0.0452, 0.0519] | 0.0485 [0.0452, 0.0519] | 0.00 | 139838 | 604.39 |
| lora[lr=0.008,rank=16] | 0.0427 [0.0377, 0.0476] | 0.0427 [0.0377, 0.0476] | 0.00 | 142921 | 700.32 |

## highway_traffic

phase 1: eps_enc_rel 0.0031, eps_dec 1.39, eps_pde_rms 0.000254, eps_pde_ss_rms 6.62e-05, l_dec_q95 4.96e+04, l_dec_mean 2.69e+04 · mean job wall 49.7 min

- state coverage id: 0.0% of eval states outside the training box, 0.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.7504 [0.6787, 0.8220] | 0.7581 [0.6850, 0.8312] | 0.7557 [0.6671, 0.8443] | 0.7513 [0.6794, 0.8231] | 0.7619 [0.6772, 0.8465] | 0.7865 [0.6975, 0.8754] | 0.7794 [0.6980, 0.8607] | 0.7452 [0.6683, 0.8221] | 0.7714 [0.6901, 0.8528] | 0.7861 [0.6937, 0.8784] | 0.7649 [0.6838, 0.8460] | 0.00 | 140196 | 965.91 |
| concat[lr=0.016,concat_depth=3] | 0.2798 [0.1611, 0.3985] | 0.2635 [0.1532, 0.3738] | 0.2766 [0.1444, 0.4088] | 0.2639 [0.1502, 0.3775] | 0.2723 [0.1592, 0.3855] | 0.2710 [0.1593, 0.3827] | 0.2675 [0.1557, 0.3794] | 0.2589 [0.1490, 0.3688] | 0.2734 [0.1562, 0.3906] | 0.2732 [0.1577, 0.3887] | 0.2700 [0.1547, 0.3854] | 0.00 | 140293 | 548.34 |
| lora[lr=0.008,rank=16] | **0.0874 [0.0423, 0.1324]** | **0.0356 [0.0298, 0.0415]** | **0.0629 [0.0395, 0.0863]** | **0.0660 [0.0451, 0.0869]** | **0.0769 [0.0486, 0.1052]** | **0.0524 [0.0311, 0.0738]** | **0.0667 [0.0477, 0.0857]** | **0.0542 [0.0467, 0.0618]** | **0.0601 [0.0425, 0.0776]** | **0.0563 [0.0316, 0.0810]** | **0.0616 [0.0437, 0.0795]** | 0.00 | 144793 | 770.18 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.7612 [0.6847, 0.8377] | 0.7670 [0.6908, 0.8432] | 0.7537 [0.6810, 0.8265] | 0.7683 [0.6762, 0.8604] | 0.7626 [0.6833, 0.8419] | 0.00 | 140196 | 965.91 |
| concat[lr=0.016,concat_depth=3] | 0.2662 [0.1505, 0.3820] | 0.2684 [0.1524, 0.3844] | 0.2697 [0.1560, 0.3834] | 0.2721 [0.1521, 0.3921] | 0.2691 [0.1528, 0.3854] | 0.01 | 140293 | 548.34 |
| lora[lr=0.008,rank=16] | **0.0630 [0.0456, 0.0804]** | **0.0558 [0.0397, 0.0719]** | **0.0626 [0.0386, 0.0867]** | **0.0651 [0.0505, 0.0797]** | **0.0617 [0.0437, 0.0796]** | 0.00 | 144793 | 770.18 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.7575 [0.6746, 0.8403] | 0.7575 [0.6746, 0.8403] | 0.00 | 140196 | 965.91 |
| concat[lr=0.016,concat_depth=3] | 0.2681 [0.1534, 0.3828] | 0.2681 [0.1534, 0.3828] | 0.00 | 140293 | 548.34 |
| lora[lr=0.008,rank=16] | **0.0699 [0.0493, 0.0905]** | **0.0699 [0.0493, 0.0905]** | 0.00 | 144793 | 770.18 |

## network

phase 1: eps_enc_rel 0.0147, eps_dec 3.14, eps_pde_rms 0.386, eps_pde_ss_rms 0.168, l_dec_q95 39.8, l_dec_mean 22.6 · mean job wall 33.6 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0716 [0.0495, 0.0936] | 0.0503 [0.0462, 0.0544] | 0.0626 [0.0555, 0.0698] | 0.0543 [0.0489, 0.0597] | 0.0671 [0.0597, 0.0746] | 0.0514 [0.0491, 0.0537] | 0.0696 [0.0623, 0.0769] | 0.0709 [0.0645, 0.0772] | 0.0669 [0.0577, 0.0761] | 0.0806 [0.0559, 0.1054] | 0.0646 [0.0561, 0.0731] | 0.00 | 140196 | 954.91 |
| concat[lr=0.016,concat_depth=3] | 0.0613 [0.0510, 0.0716] | 0.0345 [0.0315, 0.0376] | 0.0483 [0.0367, 0.0598] | 0.0422 [0.0369, 0.0476] | 0.0552 [0.0506, 0.0598] | 0.0363 [0.0330, 0.0396] | 0.0536 [0.0459, 0.0613] | 0.0571 [0.0492, 0.0650] | 0.0513 [0.0412, 0.0613] | 0.0532 [0.0438, 0.0627] | 0.0493 [0.0423, 0.0562] | 0.00 | 140515 | 1536.51 |
| lora[lr=0.008,rank=16] | **0.0492 [0.0431, 0.0554]** | **0.0212 [0.0207, 0.0217]** | **0.0311 [0.0247, 0.0375]** | **0.0309 [0.0267, 0.0350]** | **0.0374 [0.0323, 0.0426]** | **0.0227 [0.0217, 0.0238]** | **0.0377 [0.0310, 0.0443]** | **0.0371 [0.0334, 0.0408]** | **0.0366 [0.0345, 0.0388]** | **0.0308 [0.0180, 0.0437]** | **0.0334 [0.0308, 0.0359]** | 0.00 | 145673 | 1521.78 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0537 [0.0467, 0.0606] | 0.0546 [0.0482, 0.0611] | 0.0587 [0.0535, 0.0639] | 0.0597 [0.0531, 0.0662] | 0.0567 [0.0504, 0.0629] | 0.00 | 140196 | 954.91 |
| concat[lr=0.016,concat_depth=3] | 0.0466 [0.0395, 0.0537] | 0.0427 [0.0347, 0.0506] | 0.0515 [0.0435, 0.0595] | 0.0465 [0.0377, 0.0554] | 0.0468 [0.0389, 0.0547] | 0.00 | 140515 | 1536.51 |
| lora[lr=0.008,rank=16] | **0.0299 [0.0278, 0.0319]** | **0.0295 [0.0288, 0.0301]** | **0.0359 [0.0335, 0.0383]** | **0.0327 [0.0302, 0.0352]** | **0.0320 [0.0302, 0.0338]** | 0.00 | 145673 | 1521.78 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0719 [0.0601, 0.0837] | 0.0719 [0.0601, 0.0837] | 0.00 | 140196 | 954.91 |
| concat[lr=0.016,concat_depth=3] | 0.0782 [0.0657, 0.0908] | 0.0782 [0.0657, 0.0908] | 0.00 | 140515 | 1536.51 |
| lora[lr=0.008,rank=16] | **0.0491 [0.0462, 0.0519]** | **0.0491 [0.0462, 0.0519]** | 0.00 | 145673 | 1521.78 |

## rossler

phase 1: eps_enc_rel 0.0079, eps_dec 2.83, eps_pde_rms 3.11, eps_pde_ss_rms 0.692, l_dec_q95 136, l_dec_mean 43.6 · mean job wall 41.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.6% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.9% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.8% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.7768 [-1.9580, 3.5117] | 1.6658 [-4.9519, 8.2835] | 1.8666 [-5.5023, 9.2356] | 0.7447 [-1.9442, 3.4336] | 0.1538 [0.0601, 0.2476] | 0.1432 [0.0310, 0.2554] | 2.5544 [-7.8911, 13.00] | 1.9396 [-5.7967, 9.6759] | 1.6335 [-4.8280, 8.0950] | 0.1410 [0.0488, 0.2331] | 1.1455 [-3.2197, 5.5107] | 0.00 | 139627 | 615.98 |
| concat[lr=0.016,concat_depth=3] | 0.1751 [0.0952, 0.2549] | 0.1903 [0.0931, 0.2875] | 0.1899 [0.0978, 0.2820] | 0.1602 [0.0698, 0.2507] | 0.1939 [0.0618, 0.3260] | 0.1618 [0.0502, 0.2734] | 0.1778 [0.0780, 0.2776] | 0.2002 [0.0930, 0.3074] | 0.1828 [0.1034, 0.2622] | 0.1651 [0.0546, 0.2756] | 0.1799 [0.0806, 0.2792] | 0.14 | 139554 | 429.89 |
| lora[lr=0.008,rank=16] | **0.1043 [0.0981, 0.1106]** | **0.1112 [0.1064, 0.1159]** | **0.1094 [0.1026, 0.1162]** | **0.0858 [0.0787, 0.0928]** | **0.1004 [0.0961, 0.1047]** | **0.0824 [0.0711, 0.0937]** | **0.1004 [0.0960, 0.1048]** | **0.1069 [0.1028, 0.1109]** | **0.0998 [0.0853, 0.1143]** | **0.0873 [0.0841, 0.0905]** | **0.0988 [0.0942, 0.1035]** | 0.00 | 143481 | 497.06 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.2792 [-3.6378, 6.1961] | 0.9067 [-2.4241, 4.2374] | 1.2928 [-3.6831, 6.2687] | 1.7032 [-5.0801, 8.4865] | 1.2954 [-3.7063, 6.2972] | 0.00 | 139627 | 615.98 |
| concat[lr=0.016,concat_depth=3] | 0.1837 [0.0571, 0.3103] | 0.1823 [0.1073, 0.2573] | 0.1912 [0.0671, 0.3153] | 0.1749 [0.0723, 0.2775] | 0.1830 [0.0776, 0.2884] | 0.19 | 139554 | 429.89 |
| lora[lr=0.008,rank=16] | **0.0969 [0.0913, 0.1024]** | **0.1088 [0.0962, 0.1214]** | **0.1000 [0.0924, 0.1076]** | **0.0913 [0.0886, 0.0940]** | **0.0992 [0.0924, 0.1061]** | 0.00 | 143481 | 497.06 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 1.0909 [-2.9854, 5.1672] | 1.0909 [-2.9854, 5.1672] | 0.00 | 139627 | 615.98 |
| concat[lr=0.016,concat_depth=3] | 0.2054 [0.0179, 0.3929] | 0.2054 [0.0179, 0.3929] | 0.16 | 139554 | 429.89 |
| lora[lr=0.008,rank=16] | **0.0990 [0.0938, 0.1042]** | **0.0990 [0.0938, 0.1042]** | 0.00 | 143481 | 497.06 |

## two_link_arm

phase 1: eps_enc_rel 0.00653, eps_dec 4.64, eps_pde_rms 4.14, eps_pde_ss_rms 0.536, l_dec_q95 226, l_dec_mean 92.8 · mean job wall 33.9 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.4020 [0.2951, 0.5089] | 0.3947 [0.2942, 0.4953] | 0.3912 [0.2910, 0.4914] | 0.4012 [0.2968, 0.5056] | 0.4149 [0.3315, 0.4983] | 0.4006 [0.2978, 0.5034] | 0.3972 [0.2894, 0.5050] | 0.4244 [0.3278, 0.5210] | 0.4001 [0.2982, 0.5020] | 0.3848 [0.2817, 0.4880] | 0.4010 [0.3008, 0.5011] | 0.00 | 139657 | 816.27 |
| concat[lr=0.016,concat_depth=3] | 0.2862 [-0.0161, 0.5885] | 0.2870 [0.0086, 0.5653] | 0.3132 [0.1001, 0.5262] | 0.3108 [0.0684, 0.5531] | 0.2912 [0.0198, 0.5627] | 0.3026 [0.0426, 0.5626] | 0.3159 [0.0835, 0.5482] | 0.3216 [0.0637, 0.5795] | 0.2867 [-0.0049, 0.5783] | 0.2850 [0.0184, 0.5515] | 0.2992 [0.0378, 0.5606] | 0.01 | 139555 | 559.13 |
| lora[lr=0.008,rank=16] | **0.1200 [0.1164, 0.1235]** | **0.1132 [0.1110, 0.1155]** | **0.1144 [0.1090, 0.1199]** | **0.1144 [0.1114, 0.1174]** | **0.1468 [0.1157, 0.1779]** | **0.1161 [0.1113, 0.1208]** | **0.1189 [0.1168, 0.1210]** | **0.1321 [0.1141, 0.1502]** | **0.1173 [0.1113, 0.1232]** | **0.1093 [0.1083, 0.1104]** | **0.1203 [0.1157, 0.1250]** | 0.00 | 143465 | 654.58 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.3795 [0.2810, 0.4781] | 0.4079 [0.3051, 0.5107] | 0.3946 [0.3004, 0.4887] | 0.3907 [0.2915, 0.4900] | 0.3932 [0.2946, 0.4918] | 0.00 | 139657 | 816.27 |
| concat[lr=0.016,concat_depth=3] | 0.2685 [-0.0197, 0.5568] | 0.3245 [0.1029, 0.5462] | 0.2741 [-0.0217, 0.5698] | 0.3028 [0.0628, 0.5428] | 0.2925 [0.0315, 0.5535] | 0.01 | 139555 | 559.13 |
| lora[lr=0.008,rank=16] | **0.1134 [0.1096, 0.1172]** | **0.1202 [0.1153, 0.1251]** | **0.1315 [0.1138, 0.1492]** | **0.1117 [0.1090, 0.1143]** | **0.1192 [0.1124, 0.1261]** | 0.00 | 143465 | 654.58 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| latent[lr=0.008] | 0.4149 [0.3165, 0.5134] | 0.4149 [0.3165, 0.5134] | 0.00 | 139657 | 816.27 |
| concat[lr=0.016,concat_depth=3] | 0.3041 [0.0232, 0.5851] | 0.3041 [0.0232, 0.5851] | 0.00 | 139555 | 559.13 |
| lora[lr=0.008,rank=16] | **0.1318 [0.1123, 0.1512]** | **0.1318 [0.1123, 0.1512]** | 0.00 | 143465 | 654.58 |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### cartpole

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1540 | 1.1239 | 1.1548 | 1.2860 | 1.2510 | 0.00 (shape_ood/clean) | 0.645 | 891 |
| concat[lr=0.016,concat_depth=3] | 0.3823 | 0.3861 | 0.3989 | 0.4595 | 0.5093 | 0.00 (shape_ood/clean) | 0.659 | 736 |
| lora[lr=0.008,rank=16] | 0.0942 | 0.1069 | 0.1893 | 0.2248 | 0.3268 | 0.00 (shape_ood/clean) | 0.614 | 788 |

### fhn

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0224 | 0.0241 | 0.0338 | 0.0404 | 0.0420 | 0.00 (shape_ood/clean) | 0.853 | 789 |
| concat[lr=0.016,concat_depth=3] | 0.0253 | 0.0275 | 0.0485 | 0.0616 | 0.0603 | 0.00 (shape_ood/clean) | 0.822 | 510 |
| lora[lr=0.008,rank=16] | 0.0253 | 0.0277 | 0.0427 | 0.0545 | 0.0530 | 0.00 (shape_ood/clean) | 0.826 | 605 |

### highway_traffic

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.7649 | 0.7626 | 0.7575 | 1.1151 | 0.8205 | 0.00 (shape_ood/clean) | 0.543 | 796 |
| concat[lr=0.016,concat_depth=3] | 0.2700 | 0.2691 | 0.2681 | 0.4222 | 0.2913 | 0.01 (shape_ood/clean) | 0.810 | 482 |
| lora[lr=0.008,rank=16] | 0.0616 | 0.0617 | 0.0699 | 0.1071 | 0.1017 | 0.00 (shape_ood/clean) | 0.937 | 744 |

### network

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.0646 | 0.0567 | 0.0719 | 0.0800 | 0.0915 | 0.00 (shape_ood/clean) | 0.915 | 1.1e+03 |
| concat[lr=0.016,concat_depth=3] | 0.0493 | 0.0468 | 0.0782 | 0.0874 | 0.0997 | 0.00 (shape_ood/clean) | 0.949 | 1.06e+03 |
| lora[lr=0.008,rank=16] | 0.0334 | 0.0320 | 0.0491 | 0.0532 | 0.0639 | 0.00 (shape_ood/clean) | 0.918 | 1.02e+03 |

### rossler

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 1.1455 | 1.2954 | 1.0909 | 1.0463 | 3.8173 | 0.00 (shape_ood/clean) | 0.687 | 912 |
| concat[lr=0.016,concat_depth=3] | 0.1799 | 0.1830 | 0.2054 | 0.1969 | 0.3186 | 0.19 (shape_ood/clean) | 0.486 | 469 |
| lora[lr=0.008,rank=16] | 0.0988 | 0.0992 | 0.0990 | 0.0984 | 0.1592 | 0.00 (shape_ood/clean) | 0.499 | 790 |

### two_link_arm

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| latent[lr=0.008] | 0.4010 | 0.3932 | 0.4149 | 0.5724 | 0.4828 | 0.00 (shape_ood/clean) | 0.935 | 777 |
| concat[lr=0.016,concat_depth=3] | 0.2992 | 0.2925 | 0.3041 | 0.4203 | 0.4646 | 0.01 (id/clean) | 0.936 | 521 |
| lora[lr=0.008,rank=16] | 0.1203 | 0.1192 | 0.1318 | 0.1763 | 0.1586 | 0.00 (shape_ood/clean) | 0.914 | 632 |
