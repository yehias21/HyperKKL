# Plant card: `fhn`

- role: **test**
- n_x = 2, n_y = 1, n_u = 1; dt = 0.0618034 s, N = 1619, T = 100 s, t_ref = 2 s
- measured: v (membrane potential)
- unmeasured: w (recovery variable)
- partial model misses: the recovery leak -eps b w
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: rich

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.007765 / 0.00767 / 1.01 | `experiments/information/budgets.json:fhn.dists.eval_id.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.008091 / 0.007923 / 1.02 | `experiments/information/budgets_c3.json:fhn.dists.eval_id.c3.bayes` |
| observability Gramian cond (median) / full-rank fraction | 5.72 / 1.00 | `experiments/cartpole_bounded/inclusion/fhn.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/fhn.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
