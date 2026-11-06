# driven/lorenz96

system `lorenz96` · n_x 20 · measured [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] · unmeasured [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] · dt 0.00187 · N 12838 · T 24 · t_ref 1.0

Unmodelled term (partial / realistic levels): the spatial forcing anomaly dF cos(2 pi i / n) (model: uniform forcing F)

## Input

* unit: forcing anomaly on nodes 0/5/10/15; physical = (offset + u) x scale with (offset, scale, name) = [8.0, 1.0, 'F_i']
* applied box [-4, 4]; actuator {'lo': -4, 'hi': 4, 'rate': inf, 'tau': 0.0, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_climatology, op_regime_shifts
* OOD operators (test_op_ood): op_scenario_trend, op_volcanic_pulses
* input-record preset (input_realistic): {'noise': 0.0, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.0, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/lorenz96/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ekf_gated | 0.01222 |
| 2 | ude32_ukf_gated | 0.01222 |
| 3 | ude1_ukf_gated | 0.01483 |
| 4 | ude1_ekf_gated | 0.01483 |
| 5 | ukf_gated_tuned | 0.1499 |
| 6 | ekf_gated_tuned | 0.15 |
| 7 | ekf | 0.2445 |
| 8 | ukf | 0.2447 |
| 9 | gru | 0.3301 |
| 10 | node32_ekf_gated | 0.3475 |
