# Plant card: `highway_traffic`

- role: **test**
- n_x = 5, n_y = 3, n_u = 1; dt = 0.0261052 s, N = 7662, T = 200 s, t_ref = 10 s
- measured: rho1 (density cell 1, veh/m), rho3 (density cell 3, veh/m), rho5 (density cell 5, veh/m)
- unmeasured: rho2 (density cell 2, veh/m), rho4 (density cell 4, veh/m)
- partial model misses: congestion in the fundamental diagram (model: free-flow q = v_free rho)
- input box: u in [-0.15, 0.15] (actuator: saturating)
- information group: window-starved

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.05116 / 0.0186 / 2.75 | `experiments/information/budgets.json:highway_traffic.dists.eval_id.bayes` |
| info bound, unmeasured NRMSE (realistic): window 3/alpha / full past / ratio | 0.09648 / 0.02027 / 4.76 | `experiments/information/budgets_c3.json:highway_traffic.dists.eval_id.c3.bayes` |
| repairability rho (1 = grey-box residual removes all output model error) | 0.995 | `experiments/structural/scores.json:P-S3[3]` |
| observability Gramian cond (median) / full-rank fraction | 482 / 1.00 | `experiments/cartpole_bounded/inclusion/highway_traffic.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/highway_traffic.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
