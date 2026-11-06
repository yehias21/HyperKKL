# driven/quad_tank

system `quad_tank` · n_x 4 · measured [0, 1] · unmeasured [2, 3] · dt 0.05 · N 6000 · T 300 · t_ref 15.0

Unmodelled term (partial / realistic levels): the square-root orifice law (model: flows linearised at the nominal equilibrium)

## Input

* unit: pump-voltage change [V]; physical = (offset + u) x scale with (offset, scale, name) = [3.0, 1.0, 'v1 [V]']
* applied box [-1, 1]; actuator {'lo': -1, 'hi': 1, 'rate': 0.5, 'tau': 1.0, 'quant': 0.00244140625, 'deadband': 0.0}
* training operators: op_tank_small_moves, op_tank_recipe
* OOD operators (test_op_ood): op_tank_grade_change, op_tank_pump_trip
* input-record preset (input_realistic): {'noise': 0.002, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.1, 'quant': 0.00244140625, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/quad_tank/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ekf_gated | 0.01695 |
| 2 | ude32_ukf_gated | 0.01695 |
| 3 | node32_ekf_gated | 0.01719 |
| 4 | ude1_ekf_gated | 0.01748 |
| 5 | ude1_ukf_gated | 0.01748 |
| 6 | hyperkkl | 0.01809 |
| 7 | hk_base | 0.01814 |
| 8 | film | 0.03138 |
| 9 | gru | 0.05166 |
| 10 | nlox_m | 0.1741 |
