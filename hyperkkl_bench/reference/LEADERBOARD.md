# Reference leaderboard (package reports; regenerate with tools/build_leaderboard.py)

Headline per cell: mean over the three test tiers of the settled unmeasured NRMSE (clean).
Track score: geometric mean over cells of method / ekf_gated (< 1 is better); wins = cells below ekf_gated.
Learned baselines: one training seed (0); filters are deterministic. Seed variability is not included.

## Track `core` (48 cells with ekf_gated)

| baseline | cells | geo-mean ratio vs ekf_gated | wins |
|---|---|---|---|
| ekf_gated | 48 | 1.000 | 0 |
| ekf_gated_tuned | 48 | 0.815 | 27 |
| ekf_huber | 48 | 1.011 | 17 |
| ude_ekf_gated | 48 | 0.110 | 34 |

| cell | ekf_gated | ekf_gated_tuned | ekf_huber | ude_ekf_gated |
|---|---|---|---|---|
| battery/exact | 0.0056 | 0.0056 | 0.0057 | 0.0057 |
| bioreactor/exact | 0.0083 | 0.0083 | 0.0083 | 0.0083 |
| cartpole_b/exact | 0.0071 | 0.0073 | 0.0072 | 0.0072 |
| duffing/exact | 0.0059 | 0.0059 | 0.0060 | 0.0059 |
| fhn/exact | 0.0084 | 0.0084 | 0.0084 | 0.0084 |
| highway_traffic/exact | 0.0174 | 0.0174 | 0.0174 | 0.0173 |
| lorenz/exact | 0.0429 | 0.0429 | 0.0382 | 0.0593 |
| network/exact | 0.0086 | 0.0086 | 0.0086 | 0.0087 |
| quad_tank/exact | 0.0184 | 0.0184 | 0.0184 | 0.0184 |
| rossler/exact | 0.0454 | 0.0454 | 0.0457 | 0.0454 |
| two_link_arm/exact | 0.0837 | 0.0837 | 0.0846 | 0.0837 |
| vdp/exact | 0.0063 | 0.0065 | 0.0064 | 0.0064 |
| battery/mismatch | 0.1818 | 0.1818 | 0.1802 | 0.0056 |
| bioreactor/mismatch | 6.4378 | 6.5095 | 6.4354 | 0.0076 |
| cartpole_b/mismatch | 0.6057 | 0.5232 | 0.7079 | 0.0118 |
| duffing/mismatch | 0.0293 | 0.0303 | 0.0293 | 0.0061 |
| fhn/mismatch | 0.1725 | 0.1725 | 0.2234 | 0.0086 |
| highway_traffic/mismatch | 0.1000 | 0.1000 | 0.0994 | 0.2848 |
| lorenz/mismatch | 0.7090* | 0.4277 | 0.5951 | 0.0040 |
| network/mismatch | 0.3099 | 0.2325 | 0.3893 | 0.0087 |
| quad_tank/mismatch | 5.4783 | 5.4068 | 5.5282 | 0.0170 |
| rossler/mismatch | 1.9053 | 1.6672 | 1.8711 | 0.1173 |
| two_link_arm/mismatch | 8.2286* | 0.4029* | 0.4387 | 20.0253* |
| vdp/mismatch | 0.1627 | 0.1226 | 0.1621 | 0.0069 |
| battery/partial | 0.2104 | 0.2104 | 0.2233 | 0.0134 |
| bioreactor/partial | 1.6661 | 0.9939 | 1.6638 | 0.0084 |
| cartpole_b/partial | 0.0340 | 0.0368 | 0.0345 | 0.0075 |
| duffing/partial | 0.3221 | 0.3466 | 0.3332 | 0.0066 |
| fhn/partial | 0.3573 | 0.1472 | 0.3644 | 0.0084 |
| highway_traffic/partial | 0.1309 | 0.0980 | 0.1170 | 0.0179 |
| lorenz/partial | 3.9705 | 2.3757 | 6.8079* | 0.0785 |
| network/partial | 0.1193 | 0.1000 | 0.1210 | 0.0086 |
| quad_tank/partial | 0.0274 | 0.0272 | 0.0274 | 0.0191 |
| rossler/partial | 0.7282 | 0.6831 | 0.7664 | 0.3218 |
| two_link_arm/partial | 0.1119 | 0.1119 | 0.9356 | 2.5618* |
| vdp/partial | 0.5659 | 0.3658 | 0.6911 | 0.0071 |
| battery/realistic | 0.3218 | 0.3686 | 0.3203 | 0.0101 |
| bioreactor/realistic | 8.0173 | 7.0457 | 8.0155 | 0.0086 |
| cartpole_b/realistic | 0.5854 | 0.5072 | 0.6757 | 0.0159 |
| duffing/realistic | 0.3374 | 0.3626 | 0.3485 | 0.0154 |
| fhn/realistic | 0.4148 | 0.1356 | 0.4419 | 0.0133 |
| highway_traffic/realistic | 0.2038 | 0.2023 | 0.2005 | 0.0357 |
| lorenz/realistic | 3.3836 | 1.7509 | 2.8441* | 0.0074 |
| network/realistic | 0.3583 | 0.2478 | 0.4230 | 0.0134 |
| quad_tank/realistic | 5.4821 | 5.4215 | 5.5172 | 0.0173 |
| rossler/realistic | 1.3890 | 1.3224 | 1.4905 | 0.3161 |
| two_link_arm/realistic | 0.4013 | 0.3721 | 0.7512 | 3.3309* |
| vdp/realistic | 0.7159 | 0.4922 | 0.8661 | 0.0181 |

Per-cell values: test_sequence_ood, clean, unmeasured NRMSE; `*` = some episodes diverged.

## Track `deployment` (8 cells with ekf_gated)

| baseline | cells | geo-mean ratio vs ekf_gated | wins |
|---|---|---|---|
| ekf_gated | 8 | 1.000 | 0 |
| ekf_gated_tuned | 8 | 1.023 | 1 |
| ekf_huber | 8 | 0.982 | 4 |
| ude_ekf_gated | 8 | 0.491 | 6 |

| cell | ekf_gated | ekf_gated_tuned | ekf_huber | ude_ekf_gated |
|---|---|---|---|---|
| cstr_exo/exact | 0.0114 | 0.0125 | 0.0114 | 0.1545 |
| quad_planar/exact | 0.0230 | 0.0240 | 0.0232 | 0.0235 |
| cstr_exo/mismatch | 0.4421 | 0.4387 | 0.4060 | 0.3256 |
| quad_planar/mismatch | 0.2386 | 0.2386 | 0.2399 | 0.0752 |
| cstr_exo/partial | 0.2937 | 0.2937 | 0.3057 | 0.1603 |
| quad_planar/partial | 0.3229 | 0.3472 | 0.3030 | 0.0355 |
| cstr_exo/realistic | 0.7016 | 0.7070 | 0.7032 | 0.1534 |
| quad_planar/realistic | 0.3594 | 0.3594 | 0.3405 | 0.1329 |

Per-cell values: test_sequence_ood, clean, unmeasured NRMSE; `*` = some episodes diverged.

## Track `scaling` (20 cells with ekf_gated)

| baseline | cells | geo-mean ratio vs ekf_gated | wins |
|---|---|---|---|
| ekf_gated | 20 | 1.000 | 0 |
| ekf_huber | 20 | 1.030 | 6 |

| cell | ekf_gated | ekf_gated_tuned | ekf_huber | ude_ekf_gated |
|---|---|---|---|---|
| fhnring_n16/exact | 0.0101 |  | 0.0102 |  |
| fhnring_n32/exact | 0.0096 |  | 0.0097 |  |
| fhnring_n4/exact | 0.0097 |  | 0.0097 |  |
| fhnring_n64/exact | 0.0097 |  | 0.0098 |  |
| fhnring_n8/exact | 0.0091 |  | 0.0091 |  |
| fhnring_n16/mismatch | 0.2999 |  | 0.3082 |  |
| fhnring_n32/mismatch | 0.3339 |  | 0.3458 |  |
| fhnring_n4/mismatch | 0.2437 |  | 0.2550 |  |
| fhnring_n64/mismatch | 0.3421 |  | 0.3660 |  |
| fhnring_n8/mismatch | 0.2833 |  | 0.2879 |  |
| fhnring_n16/partial | 0.0871 |  | 0.0868 |  |
| fhnring_n32/partial | 0.0838 |  | 0.0828 |  |
| fhnring_n4/partial | 0.0966 |  | 0.0969 |  |
| fhnring_n64/partial | 0.0839 |  | 0.0823 |  |
| fhnring_n8/partial | 0.1029 |  | 0.1011 |  |
| fhnring_n16/realistic | 0.2720 |  | 0.3014 |  |
| fhnring_n32/realistic | 0.2986 |  | 0.3363 |  |
| fhnring_n4/realistic | 0.2776 |  | 0.2920 |  |
| fhnring_n64/realistic | 0.2995 |  | 0.3508 |  |
| fhnring_n8/realistic | 0.2936 |  | 0.3064 |  |

Per-cell values: test_sequence_ood, clean, unmeasured NRMSE; `*` = some episodes diverged.

