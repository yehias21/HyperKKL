# Plant card: `vdp`

- role: **dev**
- n_x = 2, n_y = 1, n_u = 1; dt = 0.0206011 s, N = 2428, T = 50 s, t_ref = 1 s
- measured: x1 (position)
- unmeasured: x2 (velocity)
- partial model misses: the nonlinear damping law mu (1 - x1^2) x2 (model: undamped oscillator)
- input box: u in [-1.0, 1.0] (actuator: saturating)
- information group: rich

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.003786 / 0.003778 / 1.00 | `experiments/information/budgets.json:vdp.dists.eval_id.bayes` |
| repairability rho (1 = grey-box residual removes all output model error) | 0.359 | `experiments/structural/scores.json:P-S3[5]` |
| observability Gramian cond (median) / full-rank fraction | 7 / 1.00 | `experiments/cartpole_bounded/inclusion/vdp.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/vdp.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
