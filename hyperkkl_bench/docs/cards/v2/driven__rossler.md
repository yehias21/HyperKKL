# driven/rossler

system `rossler` · n_x 3 · measured [1] · unmeasured [0, 2] · dt 0.02225 · N 7191 · T 160 · t_ref 8.0

Unmodelled term (partial / realistic levels): the folding nonlinearity x1 x3

## Input

* unit: forcing (dimensionless); physical = (offset + u) x scale with (offset, scale, name) = [0.0, 1.0, 'u']
* applied box [-1, 1]; actuator {'lo': -1, 'hi': 1, 'rate': inf, 'tau': 0.0, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_climatology, op_regime_shifts
* OOD operators (test_op_ood): op_scenario_trend, op_volcanic_pulses
* input-record preset (input_realistic): {'noise': 0.0, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.0, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 534 | 486 | 61 | 46 | 55 | 362 | 56 | 59 | 55 | 51 | 48 | 23 | 29 |
| realistic | 530 | 483 | 57 | 53 | 54 | 377 | 58 | 62 | 55 | 51 | 48 | 28 | 29 |

## v2 reference results (driven/rossler/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | hk_base | 0.1373 |
| 2 | hyperkkl | 0.1415 |
| 3 | node32_ekf_gated | 0.1618 |
| 4 | ude32_ukf_gated | 0.1642 |
| 5 | ude32_ekf_gated | 0.1647 |
| 6 | gru | 0.1797 |
| 7 | ude1_ukf_gated | 0.1945 |
| 8 | ude1_ekf_gated | 0.1959 |
| 9 | dkkl_m_trans | 0.1976 |
| 10 | dkkl_m | 0.1976 |
