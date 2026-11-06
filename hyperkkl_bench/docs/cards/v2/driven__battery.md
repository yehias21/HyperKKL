# driven/battery

system `battery` · n_x 5 · measured [3, 4] · unmeasured [0, 1, 2] · dt 0.01483 · N 2023 · T 30.01 · t_ref 1.0

Unmodelled term (partial / realistic levels): the slow RC branch R2 || C2 (model: the textbook 1-RC equivalent circuit; V2 is not driven, not in the terminal voltage and dissipates no heat)

## Input

* unit: cell current [A], + discharge; physical = (offset + u) x scale with (offset, scale, name) = [0.0, 1.0, 'I [A]']
* applied box [-0.85, 0.85]; actuator {'lo': -0.85, 'hi': 0.85, 'rate': 102.0, 'tau': 0.0016666666666666668, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_drive_urban, op_drive_suburban, op_charge_rest
* OOD operators (test_op_ood): op_drive_highway_aggr, op_drive_congested
* input-record preset (input_realistic): {'noise': 0.005, 'noise_tau': 0.0, 'bias': 0.003, 'delay': 0.0, 'quant': 0.001, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/battery/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude1_ekf_gated | 0.008208 |
| 2 | ude32_ekf_gated | 0.008298 |
| 3 | hyperkkl | 0.008389 |
| 4 | hk_base | 0.008584 |
| 5 | ude1_ukf_gated | 0.01048 |
| 6 | ude32_ukf_gated | 0.0105 |
| 7 | node32_ekf_gated | 0.01265 |
| 8 | gru | 0.02582 |
| 9 | film | 0.04353 |
| 10 | nlox_m | 0.0846 |
