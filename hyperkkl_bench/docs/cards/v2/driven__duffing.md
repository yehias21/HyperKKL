# driven/duffing

system `duffing` · n_x 2 · measured [0] · unmeasured [1] · dt 0.0103 · N 4855 · T 50.01 · t_ref 1.0

Unmodelled term (partial / realistic levels): the cubic stiffness -x1^3 (model: linear spring)

## Input

* unit: shaker force (dimensionless); physical = (offset + u) x scale with (offset, scale, name) = [0.0, 1.0, 'u']
* applied box [-1, 1]; actuator {'lo': -1, 'hi': 1, 'rate': 10.0, 'tau': 0.02, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_stepped_sine, op_random_band
* OOD operators (test_op_ood): op_swept_sine, op_burst_random
* input-record preset (input_realistic): {'noise': 0.005, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.00048828125, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}
* **uncited**: same as vdp (benchmark choice; no paper attached)

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/duffing/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ekf_gated | 0.01318 |
| 2 | hk_base | 0.01319 |
| 3 | ude1_ekf_gated | 0.01322 |
| 4 | ude32_ukf_gated | 0.01323 |
| 5 | hyperkkl | 0.01324 |
| 6 | ude1_ukf_gated | 0.01326 |
| 7 | node32_ekf_gated | 0.01355 |
| 8 | gru | 0.0172 |
| 9 | nlox_m | 0.04416 |
| 10 | latent | 0.1007 |
