# C1-rank16 — LoRA rank 16 at the matched 140k budget (default 4)

*parent* `C1-where` · *contribution* C1 · *status* done · jobs 18/18

**Question.** Is the LoRA ~ hypernetwork tie an artefact of the low-rank choice?

**Hypothesis.** The weight-space advantage over latent/concat is stable across rank 1-16.

**Decision rule.** Stable if lora at this rank stays within an equivalence margin of +-10% of lora rank 4 (C1-where) or better, on >= 4/6 dev systems (sequence_ood).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (sequence_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0768 | 0.0694 | 0.0529 | 0.0255 | 0.0742 | 0.0195 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|

## Scoreboard (shape_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0555 | 0.0250 | 0.0477 | 0.0232 | 0.0558 | 0.0162 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|

## Scoreboard (id / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0465 | 0.0216 | 0.0453 | 0.0256 | 0.0576 | 0.0168 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|

## battery

phase 1: eps_enc_rel 0.00489, eps_dec 1.83, eps_pde_rms 0.302, eps_pde_ss_rms 0.0448, l_dec_q95 806, l_dec_mean 370 · mean job wall 1.1 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0768 [0.0561, 0.0975]** | **0.0295 [0.0272, 0.0318]** | **0.0485 [0.0419, 0.0551]** | **0.0682 [0.0429, 0.0935]** | **0.0553 [0.0257, 0.0848]** | **0.0299 [0.0083, 0.0516]** | **0.0364 [0.0234, 0.0494]** | **0.0510 [0.0363, 0.0657]** | **0.0510 [0.0135, 0.0885]** | **0.0226 [0.0186, 0.0266]** | **0.0465 [0.0328, 0.0601]** | 0.00 | 143833 | 918.08 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0509 [0.0278, 0.0739]** | **0.0626 [0.0123, 0.1129]** | **0.0697 [0.0300, 0.1093]** | **0.0389 [0.0204, 0.0573]** | **0.0555 [0.0242, 0.0868]** | 0.00 | 143833 | 918.08 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0768 [0.0048, 0.1488]** | **0.0768 [0.0048, 0.1488]** | 0.00 | 143833 | 918.08 |

## bioreactor

phase 1: eps_enc_rel 0.00825, eps_dec 0.0199, eps_pde_rms 0.00738, eps_pde_ss_rms 0.00456, l_dec_q95 44.1, l_dec_mean 18 · mean job wall 1.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0278 [0.0161, 0.0395]** | **0.0073 [0.0040, 0.0106]** | **0.0189 [0.0175, 0.0203]** | **0.0257 [0.0037, 0.0478]** | **0.0322 [0.0284, 0.0361]** | **0.0107 [0.0074, 0.0140]** | **0.0280 [-0.0111, 0.0671]** | **0.0299 [-0.0065, 0.0664]** | **0.0281 [0.0165, 0.0396]** | **0.0087 [0.0038, 0.0136]** | **0.0216 [0.0111, 0.0321]** | 0.00 | 142921 | 1862.03 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0282 [0.0254, 0.0309]** | **0.0146 [0.0110, 0.0183]** | **0.0270 [0.0169, 0.0370]** | **0.0302 [0.0105, 0.0498]** | **0.0250 [0.0160, 0.0339]** | 0.00 | 142921 | 1862.03 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0694 [-0.0119, 0.1506]** | **0.0694 [-0.0119, 0.1506]** | 0.00 | 142921 | 1862.03 |

## duffing

phase 1: eps_enc_rel 0.00444, eps_dec 0.00608, eps_pde_rms 0.00261, eps_pde_ss_rms 0.000269, l_dec_q95 295, l_dec_mean 137 · mean job wall 1.7 min

- state coverage id: 0.0% of eval states outside the training box, 1.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 2.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0675 [0.0609, 0.0741]** | **0.0376 [0.0325, 0.0427]** | **0.0444 [0.0395, 0.0493]** | **0.0496 [0.0450, 0.0543]** | **0.0522 [0.0483, 0.0561]** | **0.0360 [0.0298, 0.0423]** | **0.0440 [0.0394, 0.0486]** | **0.0460 [0.0400, 0.0521]** | **0.0397 [0.0350, 0.0443]** | **0.0384 [0.0317, 0.0451]** | **0.0453 [0.0403, 0.0503]** | 0.00 | 142921 | 1073.10 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0472 [0.0427, 0.0517]** | **0.0439 [0.0374, 0.0504]** | **0.0509 [0.0455, 0.0563]** | **0.0488 [0.0435, 0.0542]** | **0.0477 [0.0424, 0.0530]** | 0.00 | 142921 | 1073.10 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0529 [0.0487, 0.0571]** | **0.0529 [0.0487, 0.0571]** | 0.00 | 142921 | 1073.10 |

## lorenz

phase 1: eps_enc_rel 0.177, eps_dec 0.267, eps_pde_rms 77, eps_pde_ss_rms 8.57, l_dec_q95 10.1, l_dec_mean 3.84 · mean job wall 1.5 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0887 [0.0752, 0.1021]** | **0.0155 [0.0139, 0.0172]** | **0.0169 [0.0143, 0.0195]** | **0.0265 [0.0197, 0.0333]** | **0.0225 [0.0188, 0.0261]** | **0.0151 [0.0132, 0.0169]** | **0.0188 [0.0070, 0.0305]** | **0.0238 [0.0103, 0.0373]** | **0.0193 [0.0172, 0.0213]** | **0.0144 [0.0126, 0.0161]** | **0.0256 [0.0221, 0.0291]** | 0.00 | 143481 | 1050.57 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0243 [0.0201, 0.0285]** | **0.0190 [0.0161, 0.0219]** | **0.0222 [0.0134, 0.0311]** | **0.0271 [0.0190, 0.0352]** | **0.0232 [0.0175, 0.0289]** | 0.00 | 143481 | 1050.57 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0255 [0.0210, 0.0301]** | **0.0255 [0.0210, 0.0301]** | 0.00 | 143481 | 1050.57 |

## quad_tank

phase 1: eps_enc_rel 0.0128, eps_dec 0.139, eps_pde_rms 0.0188, eps_pde_ss_rms 0.0486, l_dec_q95 13.6, l_dec_mean 6.59 · mean job wall 1.9 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0659 [0.0510, 0.0807]** | **0.0418 [0.0390, 0.0446]** | **0.0606 [0.0490, 0.0722]** | **0.0487 [0.0436, 0.0537]** | **0.0706 [0.0493, 0.0919]** | **0.0444 [0.0404, 0.0485]** | **0.0522 [0.0433, 0.0611]** | **0.0690 [0.0550, 0.0829]** | **0.0623 [0.0537, 0.0709]** | **0.0597 [0.0553, 0.0641]** | **0.0576 [0.0507, 0.0644]** | 0.00 | 143465 | 926.47 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0567 [0.0536, 0.0598]** | **0.0529 [0.0491, 0.0568]** | **0.0573 [0.0503, 0.0644]** | **0.0562 [0.0500, 0.0625]** | **0.0558 [0.0515, 0.0601]** | 0.00 | 143465 | 926.47 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0742 [0.0665, 0.0818]** | **0.0742 [0.0665, 0.0818]** | 0.00 | 143465 | 926.47 |

## vdp

phase 1: eps_enc_rel 0.00385, eps_dec 0.0112, eps_pde_rms 0.275, eps_pde_ss_rms 0.0566, l_dec_q95 23.2, l_dec_mean 9.73 · mean job wall 1.4 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.4% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0229 [0.0223, 0.0235]** | **0.0152 [0.0138, 0.0165]** | **0.0167 [0.0157, 0.0177]** | **0.0159 [0.0158, 0.0160]** | **0.0187 [0.0181, 0.0194]** | **0.0166 [0.0154, 0.0177]** | **0.0143 [0.0140, 0.0146]** | **0.0171 [0.0152, 0.0190]** | **0.0172 [0.0164, 0.0181]** | **0.0142 [0.0136, 0.0147]** | **0.0168 [0.0162, 0.0175]** | 0.00 | 142921 | 1668.55 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0170 [0.0157, 0.0182]** | **0.0157 [0.0148, 0.0167]** | **0.0164 [0.0160, 0.0169]** | **0.0158 [0.0150, 0.0166]** | **0.0162 [0.0154, 0.0171]** | 0.00 | 142921 | 1668.55 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0195 [0.0187, 0.0202]** | **0.0195 [0.0187, 0.0202]** | 0.00 | 142921 | 1668.55 |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0465 | 0.0555 | 0.0768 | 0.0876 | 0.1542 | 0.00 (shape_ood/clean) | 0.900 | 922 |

### bioreactor

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0216 | 0.0250 | 0.0694 | 0.0787 | 0.2405 | 0.00 (shape_ood/clean) | 0.905 | 1.41e+03 |

### duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0453 | 0.0477 | 0.0529 | 0.0739 | 0.0597 | 0.00 (shape_ood/clean) | 0.602 | 1.12e+03 |

### lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0256 | 0.0232 | 0.0255 | 0.0231 | 0.0446 | 0.00 (shape_ood/clean) | 0.608 | 1.6e+03 |

### quad_tank

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0576 | 0.0558 | 0.0742 | 0.0988 | 0.1026 | 0.00 (shape_ood/clean) | 0.940 | 1.42e+03 |

### vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0168 | 0.0162 | 0.0195 | 0.0255 | 0.0226 | 0.00 (shape_ood/clean) | 0.616 | 1.27e+03 |
