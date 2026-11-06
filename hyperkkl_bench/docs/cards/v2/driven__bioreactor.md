# driven/bioreactor

system `bioreactor` · n_x 2 · measured [1] · unmeasured [0] · dt 0.0618 · N 1619 · T 100.1 · t_ref 1.0

Unmodelled term (partial / realistic levels): Monod saturation of the growth rate (model: first-order kinetics mu_max s / Ks)

## Input

* unit: dilution-rate change [1/h]; physical = (offset + u) x scale with (offset, scale, name) = [0.3, 1.0, 'D [1/h]']
* applied box [-0.15, 0.15]; actuator {'lo': -0.15, 'hi': 0.15, 'rate': 1.0, 'tau': 0.0, 'quant': 0.003, 'deadband': 0.0}
* training operators: op_chemostat_steps, op_shift_up_down
* OOD operators (test_op_ood): op_astat, op_feed_interruption
* input-record preset (input_realistic): {'noise': 0.01, 'noise_tau': 0.0, 'bias': 0.01, 'delay': 0.0, 'quant': 0.003, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}
* **uncited**: A-stat ramp duration (benchmark choice; no paper attached)

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/bioreactor/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | hyperkkl | 0.007545 |
| 2 | hk_base | 0.007546 |
| 3 | nlox_m | 0.007546 |
| 4 | ude32_ekf_gated | 0.007806 |
| 5 | node32_ekf_gated | 0.007909 |
| 6 | ude32_ukf_gated | 0.008024 |
| 7 | ude1_ekf_gated | 0.008034 |
| 8 | ude1_ukf_gated | 0.008253 |
| 9 | gru | 0.0127 |
| 10 | film | 0.02429 |
