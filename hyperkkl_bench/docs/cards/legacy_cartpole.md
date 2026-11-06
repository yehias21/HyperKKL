# Plant card: `legacy_cartpole`

- role: **legacy**  (EXCLUDED: unbounded cart position (spread grows 9.4x from T to 5T); failed the boundedness rule (INCLUSION.md))
- n_x = 4, n_y = 2, n_u = 1; dt = 0.0172546 s, N = 1739, T = 30 s, t_ref = 1 s
- measured: x (cart position, m), th (pole angle, rad)
- unmeasured: v (cart velocity, m/s), om (pole rate, rad/s)
- partial model misses: large-angle pendulum terms (model: small-angle sin th = th, cos th = 1, no om^2)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: window-starved

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.04153 / 0.01066 / 3.89 | `experiments/information/budgets.json:cartpole.dists.eval_id.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.06476 / 0.01053 / 6.15 | `experiments/information/budgets_c3.json:cartpole.dists.eval_id.c3.bayes` |
| repairability rho (1 = grey-box residual removes all output model error) | 0.933 | `experiments/structural/scores.json:P-S3[2] (unbounded cartpole)` |
| observability Gramian cond (median) / full-rank fraction | 2.76e+04 / 1.00 | `experiments/cartpole_bounded/inclusion/cartpole.json:gramian` |
| inclusion: observable / weak / bounded | True / False / False | `experiments/cartpole_bounded/inclusion/cartpole.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood |
|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
