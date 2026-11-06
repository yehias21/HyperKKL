# autonomous/rossler

system `rossler` · n_x 3 · measured [1] · unmeasured [0, 2] · dt 0.02225 · N 7191 · T 160 · t_ref 8.0

Unmodelled term (partial / realistic levels): the folding nonlinearity x1 x3

## Initial-condition regions

* train/val/test_id: core box, centre +- 0.6 half-width
* val_ood: shell 0.6-0.8 (normalised inf-norm)
* test_ic_ood: shell 0.8-1.0
* test_param_shift: parameters exp(+-0.15) further from the data plant

## Splits per data variant (episodes)

| variant | train | train_event | val | val_event | val_ood | phase1 | test_id | test_ic_ood | test_param_shift | hidden_id | hidden_ic_ood |
|---|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 600 | 64 | 64 | 64 | 400 | 64 | 64 | 64 | 32 | 32 |

## v2 reference results (autonomous/rossler/white, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | hyperkkl | 0.04523 |
| 2 | hk_base | 0.04795 |
| 3 | ude32_ukf_gated | 0.05132 |
| 4 | ude32_ekf_gated | 0.05148 |
| 5 | dkkl_m_asym | 0.05223 |
| 6 | dkkl_m | 0.05279 |
| 7 | gru | 0.05282 |
| 8 | dkkl_m_trans | 0.05319 |
| 9 | nlox_m | 0.05409 |
| 10 | ude1_ekf_gated | 0.128 |
