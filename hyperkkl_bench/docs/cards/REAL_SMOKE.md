# Real-data smoke runs (1 seed, through the benchmark API)

Each cell is the track's headline (battery: settled SOC RMSE [% SOC], mean of the 25 degC test families; emps: velocity NRMSE during encoder outages; silverbox: settled velocity NRMSE, mean of 3 test splits; neurobem: settled hidden NRMSE, mean of test_id and test_ood). Lower is better. One training seed, reduced budgets for UDE (1000 steps x 2 learning rates) and GRU (64 units, 1500 steps); HyperKKL = A3 recipe at full budget (phase 1 6000 + 6000 steps, phase 2 3000 steps), 1 seed, GPU. These show that every path runs end to end and gives sane numbers; they are NOT benchmark results (PROTOCOL: >= 3 seeds).

| dataset | prior_mean | coulomb | finite_diff | imu_static | ekf | ekf_gated | ude_ekf_gated | gru | hyperkkl |
|---|---|---|---|---|---|---|---|---|---|
| battery_pf | 19.07 | 20 | - | - | 3.996 | 4 | 1.895 | 1.176 | 1.157 |
| battery_lg | 22.72 | 20.01 | - | - | 1.845 | 1.845 | 1.891 | 0.9105 | 1.03 |
| emps | 0.9736 | - | 0.9791 | - | 0.01883 | 0.01883 | 0.01239 | 0.06856 | 0.8908 |
| silverbox | 1.08 | - | 0.343 | - | 0.2004 | 0.2005 | 0.1226 | 0.03217 | 0.06067 |
| neurobem | 1.117 | - | - | 1.22 | 0.3548 | 0.3552 | 0.2157 | 0.3662 | 0.4024 |
| calce | 19.1 | 20.03 | - | - | 7.925 | 7.925 | 9.29 | 5.075 | 15.42 |

## neurobem in physical units

| method | split | hidden NRMSE | v_x [m/s] | v_y [m/s] | v_z [m/s] | |v| [m/s] | roll [deg] | pitch [deg] | tilt [deg] | diverged |
|---|---|---|---|---|---|---|---|---|---|---|
| prior_mean | test_id | 0.999 | 3.56 | 3.89 | 2.07 | 5.91 | 32.5 | 28.6 | 42.0 | 0.000 |
| prior_mean | test_ood | 1.235 | 4.86 | 4.91 | 2.37 | 7.50 | 49.4 | 33.3 | 55.2 | 0.000 |
| imu_static | test_id | 1.084 | 2.49 | 3.41 | 2.09 | 4.87 | 49.7 | 33.3 | 54.1 | 0.000 |
| imu_static | test_ood | 1.356 | 3.04 | 3.92 | 2.51 | 5.70 | 66.1 | 42.2 | 68.0 | 0.000 |
| ekf | test_id | 0.327 | 1.10 | 0.94 | 1.42 | 2.13 | 6.3 | 4.5 | 5.9 | 0.000 |
| ekf | test_ood | 0.383 | 1.28 | 1.16 | 1.64 | 2.46 | 9.5 | 5.3 | 7.6 | 0.000 |
| ekf_gated | test_id | 0.327 | 1.11 | 0.94 | 1.42 | 2.13 | 6.3 | 4.5 | 5.9 | 0.000 |
| ekf_gated | test_ood | 0.383 | 1.28 | 1.16 | 1.64 | 2.47 | 9.6 | 5.3 | 7.6 | 0.000 |
| ude_ekf_gated | test_id | 0.193 | 0.61 | 0.64 | 0.78 | 1.22 | 4.9 | 2.7 | 3.9 | 0.000 |
| ude_ekf_gated | test_ood | 0.238 | 0.77 | 0.94 | 0.85 | 1.51 | 7.6 | 3.9 | 5.7 | 0.000 |
| gru | test_id | 0.199 | 0.60 | 0.66 | 0.60 | 1.10 | 7.8 | 4.8 | 6.8 | 0.000 |
| gru | test_ood | 0.533 | 1.11 | 1.33 | 1.25 | 2.18 | 27.5 | 15.2 | 23.3 | 0.000 |
| hyperkkl | test_id | 0.217 | 0.67 | 0.75 | 0.60 | 1.19 | 8.9 | 5.8 | 7.9 | 0.000 |
| hyperkkl | test_ood | 0.588 | 1.20 | 1.44 | 1.42 | 2.40 | 30.7 | 18.5 | 27.6 | 0.000 |
