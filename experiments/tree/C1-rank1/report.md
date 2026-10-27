# C1-rank1 — LoRA rank 1 at the matched 140k budget (default 4)

*parent* `C1-where` · *contribution* C1 · *status* done · jobs 18/18

**Question.** Is the LoRA ~ hypernetwork tie an artefact of the low-rank choice?

**Hypothesis.** The weight-space advantage over latent/concat is stable across rank 1-16.

**Decision rule.** Stable if lora at this rank stays within an equivalence margin of +-10% of lora rank 4 (C1-where) or better, on >= 4/6 dev systems (sequence_ood).

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (sequence_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0904 | 0.0631 | 0.0581 | 0.0421 | 0.0721 | 0.0235 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|

## Scoreboard (shape_ood / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0715 | 0.0257 | 0.0522 | 0.0406 | 0.0556 | 0.0198 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|

## Scoreboard (id / clean, all families)

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| lora | 0.0653 | 0.0201 | 0.0495 | 0.0412 | 0.0569 | 0.0196 |

Paired vs `lora` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|

## battery

phase 1: eps_enc_rel 0.00489, eps_dec 1.83, eps_pde_rms 0.302, eps_pde_ss_rms 0.0448, l_dec_q95 806, l_dec_mean 370 · mean job wall 1.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage shape_ood: 0.1% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0935 [0.0563, 0.1307]** | **0.0542 [0.0402, 0.0681]** | **0.0634 [0.0524, 0.0744]** | **0.0961 [0.0660, 0.1262]** | **0.0703 [0.0362, 0.1043]** | **0.0517 [0.0095, 0.0940]** | **0.0522 [0.0085, 0.0958]** | **0.0760 [0.0514, 0.1007]** | **0.0636 [0.0352, 0.0921]** | **0.0375 [0.0334, 0.0416]** | **0.0653 [0.0416, 0.0889]** | 0.00 | 140103 | 1587.82 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0722 [0.0321, 0.1123]** | **0.0701 [0.0355, 0.1047]** | **0.0886 [0.0390, 0.1382]** | **0.0549 [0.0183, 0.0914]** | **0.0715 [0.0314, 0.1115]** | 0.00 | 140103 | 1587.82 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0904 [0.0279, 0.1528]** | **0.0904 [0.0279, 0.1528]** | 0.00 | 140103 | 1587.82 |

## bioreactor

phase 1: eps_enc_rel 0.00825, eps_dec 0.0199, eps_pde_rms 0.00738, eps_pde_ss_rms 0.00456, l_dec_q95 44.1, l_dec_mean 18 · mean job wall 1.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0343 [0.0125, 0.0561]** | **0.0080 [0.0021, 0.0138]** | **0.0146 [0.0044, 0.0248]** | **0.0230 [0.0053, 0.0407]** | **0.0264 [0.0123, 0.0405]** | **0.0115 [0.0054, 0.0175]** | **0.0223 [0.0080, 0.0366]** | **0.0221 [0.0074, 0.0368]** | **0.0301 [0.0085, 0.0517]** | **0.0101 [-0.0015, 0.0217]** | **0.0201 [0.0080, 0.0323]** | 0.00 | 140449 | 911.15 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0283 [0.0152, 0.0413]** | **0.0159 [0.0040, 0.0277]** | **0.0281 [0.0124, 0.0437]** | **0.0306 [0.0027, 0.0585]** | **0.0257 [0.0091, 0.0423]** | 0.00 | 140449 | 911.15 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0631 [0.0248, 0.1015]** | **0.0631 [0.0248, 0.1015]** | 0.00 | 140449 | 911.15 |

## duffing

phase 1: eps_enc_rel 0.00444, eps_dec 0.00608, eps_pde_rms 0.00261, eps_pde_ss_rms 0.000269, l_dec_q95 295, l_dec_mean 137 · mean job wall 1.5 min

- state coverage id: 0.0% of eval states outside the training box, 1.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 2.7% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0710 [0.0692, 0.0729]** | **0.0428 [0.0297, 0.0560]** | **0.0482 [0.0357, 0.0608]** | **0.0541 [0.0448, 0.0634]** | **0.0563 [0.0425, 0.0701]** | **0.0396 [0.0281, 0.0512]** | **0.0484 [0.0373, 0.0595]** | **0.0489 [0.0447, 0.0531]** | **0.0433 [0.0362, 0.0505]** | **0.0440 [0.0308, 0.0572]** | **0.0495 [0.0398, 0.0591]** | 0.00 | 140449 | 1652.16 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0521 [0.0428, 0.0615]** | **0.0483 [0.0350, 0.0616]** | **0.0544 [0.0469, 0.0619]** | **0.0537 [0.0411, 0.0663]** | **0.0522 [0.0415, 0.0628]** | 0.00 | 140449 | 1652.16 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0581 [0.0480, 0.0682]** | **0.0581 [0.0480, 0.0682]** | 0.00 | 140449 | 1652.16 |

## lorenz

phase 1: eps_enc_rel 0.177, eps_dec 0.267, eps_pde_rms 77, eps_pde_ss_rms 8.57, l_dec_q95 10.1, l_dec_mean 3.84 · mean job wall 1.4 min

- state coverage id: 0.0% of eval states outside the training box, 0.3% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.1024 [0.0927, 0.1121]** | **0.0297 [0.0259, 0.0336]** | **0.0320 [0.0306, 0.0334]** | **0.0414 [0.0353, 0.0475]** | **0.0397 [0.0347, 0.0447]** | **0.0299 [0.0276, 0.0323]** | **0.0369 [0.0324, 0.0414]** | **0.0409 [0.0349, 0.0469]** | **0.0360 [0.0333, 0.0387]** | **0.0280 [0.0252, 0.0308]** | **0.0412 [0.0387, 0.0437]** | 0.00 | 139807 | 850.28 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0381 [0.0372, 0.0390]** | **0.0386 [0.0334, 0.0438]** | **0.0395 [0.0359, 0.0432]** | **0.0461 [0.0385, 0.0536]** | **0.0406 [0.0367, 0.0445]** | 0.00 | 139807 | 850.28 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0421 [0.0380, 0.0462]** | **0.0421 [0.0380, 0.0462]** | 0.00 | 139807 | 850.28 |

## quad_tank

phase 1: eps_enc_rel 0.0128, eps_dec 0.139, eps_pde_rms 0.0188, eps_pde_ss_rms 0.0486, l_dec_q95 13.6, l_dec_mean 6.59 · mean job wall 1.6 min

- state coverage id: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0701 [0.0536, 0.0866]** | **0.0449 [0.0349, 0.0549]** | **0.0548 [0.0518, 0.0578]** | **0.0479 [0.0395, 0.0562]** | **0.0658 [0.0534, 0.0782]** | **0.0498 [0.0430, 0.0566]** | **0.0512 [0.0429, 0.0594]** | **0.0625 [0.0547, 0.0702]** | **0.0653 [0.0599, 0.0706]** | **0.0563 [0.0453, 0.0672]** | **0.0569 [0.0494, 0.0645]** | 0.00 | 139806 | 1089.42 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0551 [0.0467, 0.0635]** | **0.0540 [0.0484, 0.0596]** | **0.0580 [0.0460, 0.0699]** | **0.0555 [0.0450, 0.0659]** | **0.0556 [0.0470, 0.0643]** | 0.00 | 139806 | 1089.42 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0721 [0.0653, 0.0788]** | **0.0721 [0.0653, 0.0788]** | 0.00 | 139806 | 1089.42 |

## vdp

phase 1: eps_enc_rel 0.00385, eps_dec 0.0112, eps_pde_rms 0.275, eps_pde_ss_rms 0.0566, l_dec_q95 23.2, l_dec_mean 9.73 · mean job wall 1.2 min

- state coverage id: 0.0% of eval states outside the training box, 0.4% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.5% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 1.4% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora | **0.0241 [0.0230, 0.0252]** | **0.0179 [0.0143, 0.0216]** | **0.0191 [0.0158, 0.0223]** | **0.0187 [0.0153, 0.0222]** | **0.0224 [0.0201, 0.0246]** | **0.0189 [0.0168, 0.0209]** | **0.0172 [0.0149, 0.0195]** | **0.0204 [0.0183, 0.0226]** | **0.0199 [0.0176, 0.0223]** | **0.0173 [0.0137, 0.0209]** | **0.0196 [0.0171, 0.0221]** | 0.00 | 140449 | 1532.92 |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | **0.0208 [0.0172, 0.0244]** | **0.0192 [0.0145, 0.0239]** | **0.0198 [0.0158, 0.0238]** | **0.0192 [0.0155, 0.0229]** | **0.0198 [0.0158, 0.0237]** | 0.00 | 140449 | 1532.92 |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| lora | **0.0235 [0.0221, 0.0249]** | **0.0235 [0.0221, 0.0249]** | 0.00 | 140449 | 1532.92 |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0653 | 0.0715 | 0.0904 | 0.1012 | 0.1505 | 0.00 (shape_ood/clean) | 0.899 | 1.46e+03 |

### bioreactor

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0201 | 0.0257 | 0.0631 | 0.0705 | 0.1342 | 0.00 (shape_ood/clean) | 0.884 | 1.34e+03 |

### duffing

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0495 | 0.0522 | 0.0581 | 0.0812 | 0.0672 | 0.00 (shape_ood/clean) | 0.602 | 1.5e+03 |

### lorenz

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0412 | 0.0406 | 0.0421 | 0.0391 | 0.0689 | 0.00 (shape_ood/clean) | 0.685 | 1.08e+03 |

### quad_tank

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0569 | 0.0556 | 0.0721 | 0.0941 | 0.0956 | 0.00 (shape_ood/clean) | 0.940 | 1.06e+03 |

### vdp

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| lora | 0.0196 | 0.0198 | 0.0235 | 0.0313 | 0.0281 | 0.00 (shape_ood/clean) | 0.613 | 1.07e+03 |
