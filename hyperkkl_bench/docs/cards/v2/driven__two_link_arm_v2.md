# driven/two_link_arm_v2

system `two_link_arm_b` · n_x 4 · measured [0, 1] · unmeasured [2, 3] · dt 0.01294 · N 7728 · T 100 · t_ref 2.0

Unmodelled term (partial / realistic levels): Coriolis and centrifugal terms (model: M(q) qdd = tau - g(q) - kd qd)

## Input

* unit: shoulder torque [N m]; physical = (offset + u) x scale with (offset, scale, name) = [0.0, 1.0, 'tau [N m]']
* applied box [-12.0, 12.0]; actuator {'lo': -12.0, 'hi': 12.0, 'rate': 120.0, 'tau': 0.005, 'quant': 0.0, 'deadband': 0.0}
* training operators: op_fourier_excitation, op_pick_place
* OOD operators (test_op_ood): op_task_cycle, op_teleop
* input-record preset (input_realistic): {'noise': 0.005, 'noise_tau': 0.0, 'bias': 0.0, 'delay': 0.0, 'quant': 0.005859375, 'aggregate': 0.0, 'dropout': 0.0, 'use_command': False}
* **uncited**: minimum-jerk pick-and-place bumps (benchmark choice; no paper attached)

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_op_id | test_op_ood | test_legacy_id | test_legacy_shape_ood | test_legacy_sequence_ood | hidden_op_id | hidden_op_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 356 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 241 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 257 | 64 | 64 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (driven/two_link_arm_v2/realistic, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | gru | 0.1444 |
| 2 | dkkl_m_trans | 0.3658 |
| 3 | dkkl_m | 0.3665 |
| 4 | dkkl_m_asym | 0.3823 |
| 5 | ukf | 0.7103 |
| 6 | hk_base | 0.8187 |
| 7 | hyperkkl | 0.8717 |
| 8 | ekf_gated_tuned | 0.9053 |
| 9 | nlox_m | 1.102 |
| 10 | node32_ekf_gated | 1.192 |
