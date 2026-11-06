# driven/highway_v2

system `highway_traffic` · n_x 5 · measured [0, 2, 4] · unmeasured [1, 3] · dt 0.02611 · N 7662 · T 200 · t_ref 10.0

Unmodelled term (partial / realistic levels): congestion in the fundamental diagram (model: free-flow q = v_free rho)

## Input

* unit: on-ramp inflow change [veh/s]; physical = (offset + u) x scale with (offset, scale, name) = [0.3, 3600.0, 'ramp flow [veh/h]']
* applied box [-0.25, 0.2]; actuator {'lo': -0.25, 'hi': 0.2, 'rate': 0.01, 'tau': 10.0, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_peak_demand, op_metering_plan
* OOD operators (test_op_ood): op_signal_platoons, op_event_letout
* input-record preset (input_realistic): {'noise': 0.0, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.0, 'aggregate': 20.0, 'dropout': 0.0, 'use_command': False}

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/highway_v2/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | node32_ekf_gated | 0.02698 |
| 2 | ude32_ukf_gated | 0.03427 |
| 3 | ude1_ekf_gated | 0.03574 |
| 4 | ude1_ukf_gated | 0.03608 |
| 5 | ude32_ekf_gated | 0.06731 |
| 6 | hyperkkl | 0.06811 |
| 7 | hk_base | 0.06979 |
| 8 | nlox_m | 0.1236 |
| 9 | gru | 0.1335 |
| 10 | film | 0.1632 |
