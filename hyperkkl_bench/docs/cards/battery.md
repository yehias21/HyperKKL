# Plant card: `battery`

- role: **dev**
- n_x = 5, n_y = 2, n_u = 1; dt = 0.0148347 s, N = 2023, T = 30 s, t_ref = 1 s
- measured: dT (temperature rise, K), Vm (BMS voltage channel, V)
- unmeasured: SOC, V1 (fast RC voltage, V), V2 (slow RC voltage, V)
- partial model misses: the slow RC branch R2 || C2 (model: the textbook 1-RC equivalent circuit; V2 is not driven, not in the terminal voltage and dissipates no heat)
- input box: u in [-0.85, 0.85] (actuator: affine)
- information group: window-starved

## Difficulty axes

| axis | value | provenance |
|---|---|---|
| info bound, unmeasured NRMSE (white): window 3/alpha / full past / ratio | 0.008368 / 0.003615 / 2.31 | `experiments/information/budgets.json:battery.dists.eval_id.bayes` |
| repairability rho (1 = grey-box residual removes all output model error) | 0.884 | `experiments/structural/scores.json:P-S3[1]` |
| observability Gramian cond (median) / full-rank fraction | 3.03e+03 / 1.00 | `experiments/cartpole_bounded/inclusion/battery.json:gramian` |
| inclusion: observable / weak / bounded | True / False / True | `experiments/cartpole_bounded/inclusion/battery.json` |

## Episodes per split

| data variant | train | validation | phase1 | id | shape_ood | sequence_ood | val_ood | hidden_id | hidden_shape_ood | hidden_sequence_ood |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| realistic | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| mismatch30 | 600 | 64 | 400 | 64 | 64 | 64 | 64 | 32 | 32 | 32 |

Knowledge levels: `exact` = true dynamics at true parameters; white Gaussian sensor noise (3% y_std); `partial` = model misses one physically identified term (f_known); same data as exact; `mismatch` = true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3); `realistic` = f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise
