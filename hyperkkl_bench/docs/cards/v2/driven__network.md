# driven/network

system `network` · n_x 10 · measured [0, 2, 6] · unmeasured [1, 3, 4, 5, 7, 8, 9] · dt 0.04351 · N 2299 · T 100 · t_ref 1.5

Unmodelled term (partial / realistic levels): the coupling edge 2-3 of the ring (model: chain 3-4-0-1-2, graph partly unknown)

## Input

* unit: injected current, node 0; physical = (offset + u) x scale with (offset, scale, name) = [0.0, 1.0, 'u']
* applied box [-1, 1]; actuator {'lo': -1, 'hi': 1, 'rate': 20.0, 'tau': 0.05, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_long_square, op_ramp_protocol
* OOD operators (test_op_ood): op_synaptic_noise, op_pulse_train
* input-record preset (input_realistic): {'noise': 0.005, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.00048828125, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/network/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude1_ekf_gated | 0.01364 |
| 2 | ude32_ekf_gated | 0.01368 |
| 3 | ude32_ukf_gated | 0.01391 |
| 4 | ude1_ukf_gated | 0.01393 |
| 5 | node32_ekf_gated | 0.01447 |
| 6 | hk_base | 0.01488 |
| 7 | hyperkkl | 0.01585 |
| 8 | gru | 0.02512 |
| 9 | dkkl_m | 0.03749 |
| 10 | dkkl_m_trans | 0.03749 |
