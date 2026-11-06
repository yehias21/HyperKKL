# Plant card: `network`

- role: **test**
- n_x = 10, n_y = 3, n_u = 1; dt = 0.0435087 s, N = 2299, T = 100 s, t_ref = 1.5 s
- measured: v0, v1, v3
- unmeasured: w0, w1, v2, w2, w3, v4, w4
- partial model misses: the coupling edge 2-3 of the ring (model: chain 3-4-0-1-2, graph partly unknown)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: rich

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.007898 / 0.007618 / 1.04 | `experiments/information/budgets.json:network.dists.eval_id.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.008712 / 0.00806 / 1.08 | `experiments/information/budgets_c3.json:network.dists.eval_id.c3.bayes` |
| observability Gramian cond (median) / full-rank fraction | 5.86e+03 / 1.00 | `experiments/cartpole_bounded/inclusion/network.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/network.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
