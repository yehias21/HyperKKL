# autonomous/lorenz96

system `lorenz96` · n_x 20 · measured [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] · unmeasured [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] · dt 0.00187 · N 12838 · T 24 · t_ref 1.0

Unmodelled term (partial / realistic levels): the spatial forcing anomaly dF cos(2 pi i / n) (model: uniform forcing F)

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

## v2 reference results (autonomous/lorenz96/white, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ukf_gated | 0.003949 |
| 2 | ude32_ekf_gated | 0.00395 |
| 3 | ude1_ukf_gated | 0.00416 |
| 4 | ude1_ekf_gated | 0.00416 |
| 5 | ukf_gated_tuned | 0.09353 |
| 6 | ekf_gated_tuned | 0.09363 |
| 7 | ekf | 0.1482 |
| 8 | ukf | 0.1485 |
| 9 | gru | 0.2068 |
| 10 | ekf_gated | 0.2298 |
