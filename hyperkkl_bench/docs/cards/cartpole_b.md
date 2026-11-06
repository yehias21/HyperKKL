# Plant card: `cartpole_b`

- role: **test**
- n_x = 4, n_y = 2, n_u = 1; dt = 0.0172546 s, N = 1739, T = 30 s, t_ref = 1 s
- measured: x (cart position, m), th (pole angle, rad)
- unmeasured: v (cart velocity, m/s), om (pole rate, rad/s)
- partial model misses: large-angle pendulum terms (model: small-angle sin th = th, cos th = 1, no om^2)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: rich

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.007303 / 0.007093 / 1.03 | `experiments/cartpole_bounded/budget_cartpole_b.json:dists.eval_id.white_gauss.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.008474 / 0.007529 / 1.13 | `experiments/cartpole_bounded/budget_cartpole_b.json:dists.eval_id.c3.bayes` |
| observability Gramian cond (median) / full-rank fraction | 8.58e+03 / 1.00 | `experiments/cartpole_bounded/inclusion/cartpole_b.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/cartpole_b.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
