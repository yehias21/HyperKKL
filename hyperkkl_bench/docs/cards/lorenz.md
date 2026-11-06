# Plant card: `lorenz`

- role: **dev**
- n_x = 3, n_y = 1, n_u = 1; dt = 0.011126 s, N = 4494, T = 50 s, t_ref = 1 s
- measured: x2
- unmeasured: x1, x3
- partial model misses: the coupling -x1 x3 in the measured equation
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: rich

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.001398 / 0.001398 / 1.00 | `experiments/information/budgets.json:lorenz.dists.eval_id.bayes` |
| repairability rho (1 = grey-box residual removes all output model error) | 1.000 | `experiments/structural/scores.json:P-S3[6]` |
| largest Lyapunov exponent (1/s, id inputs) | 0.898 | `experiments/chaos_refs/results.json:lyapunov.lorenz/nominal.tiers.id.lam_long.mean` |
| observability Gramian cond (median) / full-rank fraction | 5.36e+03 / 1.00 | `experiments/cartpole_bounded/inclusion/lorenz.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/lorenz.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
