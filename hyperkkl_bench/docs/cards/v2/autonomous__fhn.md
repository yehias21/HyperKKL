# autonomous/fhn

system `fhn` · n_x 2 · measured [0] · unmeasured [1] · dt 0.0618 · N 1619 · T 100.1 · t_ref 2.0

Unmodelled term (partial / realistic levels): the recovery leak -eps b w

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

## v2 reference results (autonomous/fhn/white, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ekf_gated | 0.009422 |
| 2 | ude32_ukf_gated | 0.009423 |
| 3 | ude1_ekf_gated | 0.01105 |
| 4 | ude1_ukf_gated | 0.01105 |
| 5 | hyperkkl | 0.01279 |
| 6 | hk_base | 0.01381 |
| 7 | gru | 0.01604 |
| 8 | nlox_m | 0.01626 |
| 9 | dkkl_m | 0.01754 |
| 10 | dkkl_m_trans | 0.01754 |
