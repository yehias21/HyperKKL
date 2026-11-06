# Plant card: `two_link_arm`

- role: **test**
- n_x = 4, n_y = 2, n_u = 1; dt = 0.012941 s, N = 7728, T = 100 s, t_ref = 2 s
- measured: q1 (shoulder angle, rad), q2 (elbow angle, rad)
- unmeasured: dq1 (rad/s), dq2 (rad/s)
- partial model misses: Coriolis and centrifugal terms (model: M(q) qdd = tau - g(q) - kd qd)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: absolutely starved

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.04561 / 0.04554 / 1.00 | `experiments/information/budgets.json:two_link_arm.dists.eval_id.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.02007 / 0.01988 / 1.01 | `experiments/information/budgets_c3.json:two_link_arm.dists.eval_id.c3.bayes` |
| observability Gramian cond (median) / full-rank fraction | 7e+03 / 1.00 | `experiments/cartpole_bounded/inclusion/two_link_arm.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/two_link_arm.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 335 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 305 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 305 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
