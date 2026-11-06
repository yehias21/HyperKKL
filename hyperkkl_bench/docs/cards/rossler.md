# Plant card: `rossler`

- role: **test**
- n_x = 3, n_y = 1, n_u = 1; dt = 0.0222521 s, N = 7191, T = 160 s, t_ref = 8 s
- measured: x2
- unmeasured: x1, x3
- partial model misses: the folding nonlinearity x1 x3
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: rich

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.001427 / 0.001427 / 1.00 | `experiments/information/budgets.json:rossler.dists.eval_id.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.00134 / 0.00134 / 1.00 | `experiments/information/budgets_c3.json:rossler.dists.eval_id.c3.bayes` |
| largest Lyapunov exponent (1/s, id inputs) | 0.051 | `experiments/chaos_refs/results.json:lyapunov.rossler/nominal.tiers.id.lam_long.mean` |
| observability Gramian cond (median) / full-rank fraction | 2.9e+07 / 1.00 | `experiments/cartpole_bounded/inclusion/rossler.json:gramian` |
| inclusion: observable / weak / bounded | True / True / True | `experiments/cartpole_bounded/inclusion/rossler.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 399 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 491 | 57 | 372 | 55 | 51 | 48 | 54 | 26 | 24 | 21 |
| mismatch30 | 491 | 57 | 372 | 55 | 51 | 48 | 54 | 26 | 24 | 21 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
