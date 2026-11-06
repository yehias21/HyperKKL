# autonomous/vdp

system `vdp` · n_x 2 · measured [0] · unmeasured [1] · dt 0.0206 · N 2428 · T 50.02 · t_ref 1.0

Unmodelled term (partial / realistic levels): the nonlinear damping law mu (1 - x1^2) x2 (model: undamped oscillator)

## Initial-condition regions

* train/val/test_id: core box, centre +- 0.6 half-width
* val_ood: shell 0.6-0.8 (normalised inf-norm)
* test_ic_ood: shell 0.8-1.0
* test_param_shift: parameters exp(+-0.15) further from the data plant

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_id | test_ic_ood | test_param_shift | hidden_id | hidden_ic_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 32 | 32 |
| mismatch30 | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 32 | 32 |
| realistic | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (autonomous/vdp/white, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | hyperkkl | 0.007155 |
| 2 | hk_base | 0.007819 |
| 3 | ude32_ekf_gated | 0.007938 |
| 4 | ude32_ukf_gated | 0.007948 |
| 5 | nlox_m | 0.009111 |
| 6 | dkkl_m | 0.01016 |
| 7 | dkkl_m_asym | 0.01022 |
| 8 | dkkl_m_trans | 0.01057 |
| 9 | ude1_ekf_gated | 0.01057 |
| 10 | ude1_ukf_gated | 0.01057 |
