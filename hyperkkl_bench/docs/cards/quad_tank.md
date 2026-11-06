# Plant card: `quad_tank`

- role: **dev**
- n_x = 4, n_y = 2, n_u = 1; dt = 0.05 s, N = 6000, T = 300 s, t_ref = 15 s
- measured: h1 (lower tank 1, cm), h2 (lower tank 2, cm)
- unmeasured: h3 (upper tank 3, cm), h4 (upper tank 4, cm)
- partial model misses: the square-root orifice law (model: flows linearised at the nominal equilibrium)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: window-starved

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.0297 / 0.01981 / 1.50 | `experiments/information/budgets.json:quad_tank.dists.eval_id.bayes` |
| repairability rho (1 = grey-box residual removes all output model error) | 0.997 | `experiments/structural/scores.json:P-S3[0]` |
| observability Gramian cond (median) / full-rank fraction | 202 / 1.00 | `experiments/cartpole_bounded/inclusion/quad_tank.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/quad_tank.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
