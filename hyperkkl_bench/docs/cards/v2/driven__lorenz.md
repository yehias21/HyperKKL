# driven/lorenz

system `lorenz` · n_x 3 · measured [1] · unmeasured [0, 2] · dt 0.01113 · N 4494 · T 50 · t_ref 1.0

Unmodelled term (partial / realistic levels): the coupling -x1 x3 in the measured equation

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
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/lorenz/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | hk_base | 0.005154 |
| 2 | hyperkkl | 0.005156 |
| 3 | nlox_m | 0.006059 |
| 4 | node32_ekf_gated | 0.007099 |
| 5 | film | 0.009037 |
| 6 | dkkl_m | 0.009987 |
| 7 | dkkl_m_trans | 0.009987 |
| 8 | gru | 0.01024 |
| 9 | concat | 0.01098 |
| 10 | dkkl_m_asym | 0.01394 |
