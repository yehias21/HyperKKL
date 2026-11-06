# Plant card: `duffing`

- role: **dev**
- n_x = 2, n_y = 1, n_u = 1; dt = 0.0103006 s, N = 4855, T = 50 s, t_ref = 1 s
- measured: x1 (position)
- unmeasured: x2 (velocity)
- partial model misses: the cubic stiffness -x1^3 (model: linear spring)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: window-starved

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.01135 / 0.005311 / 2.14 | `experiments/information/budgets.json:duffing.dists.eval_id.bayes` |
| observability Gramian cond (median) / full-rank fraction | 82.9 / 1.00 | `experiments/cartpole_bounded/inclusion/duffing.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/duffing.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
