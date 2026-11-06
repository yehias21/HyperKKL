### ekf on silverbox  (hyperkkl-bench 1.0.0rc4, scorer settled-velocity-nrmse-1)

settled velocity NRMSE (k >= N/2 of each 2048-sample window; velocity label = zero-phase derivative of V2, normalised by its training std); headline = mean of the three test splits

| split | nrmse_velocity | n |
|---|---|---|
| test_multisine | 0.1832 | 20 |
| test_arrow | 0.1294 | 30 |
| test_arrow_extrap | 0.2885 | 7 |
| **headline** | **0.2004** | |
