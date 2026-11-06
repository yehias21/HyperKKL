# driven/vdp

system `vdp` · n_x 2 · measured [0] · unmeasured [1] · dt 0.0206 · N 2428 · T 50.02 · t_ref 1.0

Unmodelled term (partial / realistic levels): the nonlinear damping law mu (1 - x1^2) x2 (model: undamped oscillator)

## Input

* unit: forcing (dimensionless); physical = (offset + u) x scale with (offset, scale, name) = [0.0, 1.0, 'u']
* applied box [-1, 1]; actuator {'lo': -1, 'hi': 1, 'rate': 10.0, 'tau': 0.02, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_stepped_sine, op_random_band
* OOD operators (test_op_ood): op_swept_sine, op_burst_random
* input-record preset (input_realistic): {'noise': 0.005, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.00048828125, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}
* **uncited**: stepped / swept sine and band-random shaker tests (benchmark choice; no paper attached)

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/vdp/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | hk_base | 0.01602 |
| 2 | hyperkkl | 0.01645 |
| 3 | ude1_ekf_gated | 0.01739 |
| 4 | ude1_ukf_gated | 0.01741 |
| 5 | node32_ekf_gated | 0.01757 |
| 6 | ude32_ekf_gated | 0.01759 |
| 7 | ude32_ukf_gated | 0.0176 |
| 8 | dkkl_inj_m | 0.02174 |
| 9 | gru | 0.0223 |
| 10 | nlox_m | 0.02524 |
