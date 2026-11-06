# autonomous/network

system `network` · n_x 10 · measured [0, 2, 6] · unmeasured [1, 3, 4, 5, 7, 8, 9] · dt 0.04351 · N 2299 · T 100 · t_ref 1.5

Unmodelled term (partial / realistic levels): the coupling edge 2-3 of the ring (model: chain 3-4-0-1-2, graph partly unknown)

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

## v2 reference results (autonomous/network/white, primary block, settled unmeasured NRMSE, seed mean)

| rank | method | mean |
|---|---|---|
| 1 | ude32_ekf_gated | 0.009002 |
| 2 | ude32_ukf_gated | 0.009007 |
| 3 | ude1_ekf_gated | 0.009204 |
| 4 | ude1_ukf_gated | 0.009253 |
| 5 | hyperkkl | 0.0102 |
| 6 | hk_base | 0.01268 |
| 7 | gru | 0.01333 |
| 8 | dkkl_m_trans | 0.01435 |
| 9 | dkkl_m | 0.01437 |
| 10 | dkkl_m_asym | 0.01505 |
