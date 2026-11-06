# autonomous/lorenz

system `lorenz` · n_x 3 · measured [1] · unmeasured [0, 2] · dt 0.01113 · N 4494 · T 50 · t_ref 1.0

Unmodelled term (partial / realistic levels): the coupling -x1 x3 in the measured equation

## Initial-condition regions

* train/val/test_id: core box, centre +- 0.6 half-width
* val_ood: shell 0.6-0.8 (normalised inf-norm)
* test_ic_ood: shell 0.8-1.0
* test_param_shift: parameters exp(+-0.15) further from the data plant

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_id | test_ic_ood | test_param_shift | hidden_id | hidden_ic_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (autonomous/lorenz/white, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ukf_gated | 0.007953 |
| 2 | nlox_m | 0.009111 |
| 3 | hyperkkl | 0.009343 |
| 4 | hk_base | 0.00937 |
| 5 | dkkl_m_trans | 0.009374 |
| 6 | dkkl_m | 0.009458 |
| 7 | gru | 0.009812 |
| 8 | dkkl_m_asym | 0.01088 |
| 9 | ude1_ekf_gated | 0.01286 |
| 10 | ude1_ukf_gated | 0.02029 |
