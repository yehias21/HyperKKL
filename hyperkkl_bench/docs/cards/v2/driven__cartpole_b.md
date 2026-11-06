# driven/cartpole_b

system `cartpole_b` · n_x 4 · measured [0, 2] · unmeasured [1, 3] · dt 0.01725 · N 1739 · T 30.01 · t_ref 1.0

Unmodelled term (partial / realistic levels): large-angle pendulum terms (model: small-angle sin th = th, cos th = 1, no om^2)

## Input

* unit: cart command [x 5 N]; physical = (offset + u) x scale with (offset, scale, name) = [0.0, 5.0, 'F_cmd [N]']
* applied box [-1, 1]; actuator {'lo': -1, 'hi': 1, 'rate': 4.0, 'tau': 0.01, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_cart_reposition
* OOD operators (test_op_ood): op_cart_shuttle
* input-record preset (input_realistic): {'noise': 0.005, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.00048828125, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}
* **uncited**: minimum-jerk repositioning moves (benchmark choice; no paper attached)

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/cartpole_b/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ekf_gated | 0.01268 |
| 2 | ude32_ukf_gated | 0.01279 |
| 3 | hyperkkl | 0.01344 |
| 4 | node32_ekf_gated | 0.01442 |
| 5 | hk_base | 0.01573 |
| 6 | gru | 0.01768 |
| 7 | ude1_ekf_gated | 0.01806 |
| 8 | ude1_ukf_gated | 0.01829 |
| 9 | nlox_m | 0.02112 |
| 10 | dkkl_inj_m | 0.02327 |
