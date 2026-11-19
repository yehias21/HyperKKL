# ICLR-large — Row 11: a larger system (highway_traffic)

*parent* `ICLR` · *contribution* iclr · *status* done · jobs 3/3

**Question.** Do the methods scale to a higher-dimensional plant?

**Decision rule.** Paired relative NRMSE vs autonomous on id, 95% bootstrap CI.

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

## Scoreboard (id / clean, all families)

| method | highway_traffic |
|---|---|
| autonomous | 0.3085 |
| static | 0.2704 |
| dynamic | 0.0846 |
| curriculum | 0.0604 |
| curriculum[cur_levels=1] | 0.0621 |
| ekf | 0.0122 |
| kkl_mb | 0.2750 |

Paired vs `autonomous` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | highway_traffic |
|---|---|
| static | 10/0 (-12%*) |
| dynamic | 10/0 (-73%*) |
| curriculum | 10/0 (-80%*) |
| curriculum[cur_levels=1] | 10/0 (-80%*) |
| ekf | 10/0 (-96%*) |
| kkl_mb | 10/0 (-11%*) |

## Scoreboard (shape_ood / clean, all families)

| method | highway_traffic |
|---|---|
| autonomous | 0.2959 |
| static | 0.2672 |
| dynamic | 0.0852 |
| curriculum | 0.0542 |
| curriculum[cur_levels=1] | 0.0563 |
| ekf | 0.0123 |
| kkl_mb | 0.2707 |

Paired vs `autonomous` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | highway_traffic |
|---|---|
| static | 4/0 (-10%*) |
| dynamic | 4/0 (-71%*) |
| curriculum | 4/0 (-82%*) |
| curriculum[cur_levels=1] | 4/0 (-81%*) |
| ekf | 4/0 (-96%*) |
| kkl_mb | 4/0 (-9%*) |

## Scoreboard (sequence_ood / clean, all families)

| method | highway_traffic |
|---|---|
| autonomous | 0.3096 |
| static | 0.2681 |
| dynamic | 0.1086 |
| curriculum | 0.0584 |
| curriculum[cur_levels=1] | 0.0601 |
| ekf | 0.0119 |
| kkl_mb | 0.2682 |

Paired vs `autonomous` across systems (families where the 95% CI excludes 0: better / worse; all-column change):

| method | highway_traffic |
|---|---|
| static | 1/0 (-14%*) |
| dynamic | 1/0 (-65%*) |
| curriculum | 1/0 (-81%*) |
| curriculum[cur_levels=1] | 1/0 (-81%*) |
| ekf | 1/0 (-96%*) |
| kkl_mb | 1/0 (-13%*) |

## highway_traffic

phase 1: eps_enc_rel 0.0031, eps_dec 4.52, eps_pde_rms 0.000254, eps_pde_ss_rms 6.62e-05, l_dec_q95 2.25e+04, l_dec_mean 1.12e+04 · mean job wall 90.1 min

- state coverage id: 0.0% of eval states outside the training box, 0.1% beyond the training kNN q99
- state coverage shape_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage sequence_ood: 0.0% of eval states outside the training box, 0.0% beyond the training kNN q99
- state coverage fam_zero: 0.0% of eval states outside the training box, 0.2% beyond the training kNN q99

### id / clean

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autonomous | 0.2812 [0.2004, 0.3620] | 0.3273 [0.2405, 0.4141] | 0.2928 [0.2249, 0.3607] | 0.2763 [0.1948, 0.3579] | 0.3180 [0.2604, 0.3756] | 0.3060 [0.2327, 0.3794] | 0.2941 [0.2254, 0.3627] | 0.3391 [0.2564, 0.4217] | 0.3278 [0.2463, 0.4093] | 0.3135 [0.2519, 0.3751] | 0.3085 [0.2344, 0.3826] | 0.00 | – | 87.20 |
| static | 0.2708 [0.1873, 0.3542] | 0.2711 [0.1885, 0.3538] | 0.2612 [0.2144, 0.3080] | 0.2670 [0.1883, 0.3458] | 0.2714 [0.1999, 0.3428] | 0.2744 [0.2115, 0.3374] | 0.2701 [0.1985, 0.3416] | 0.2685 [0.1936, 0.3434] | 0.2747 [0.1995, 0.3499] | 0.2735 [0.2121, 0.3348] | 0.2704 [0.1994, 0.3414] | 0.00 | 140196 | 670.91 |
| dynamic | 0.1036 [0.0252, 0.1819] | 0.0691 [0.0382, 0.1000] | 0.0858 [0.0633, 0.1083] | 0.0815 [0.0396, 0.1233] | 0.1006 [0.0651, 0.1362] | 0.0728 [0.0436, 0.1020] | 0.0903 [0.0455, 0.1351] | 0.0870 [0.0523, 0.1217] | 0.0791 [0.0454, 0.1128] | 0.0781 [0.0362, 0.1200] | 0.0846 [0.0462, 0.1230] | 0.00 | 3658272 | 1799.23 |
| curriculum | 0.0389 [0.0345, 0.0433] | 0.0663 [0.0548, 0.0779] | 0.0710 [0.0664, 0.0756] | 0.0451 [0.0352, 0.0551] | 0.0690 [0.0550, 0.0830] | 0.0627 [0.0593, 0.0660] | 0.0546 [0.0477, 0.0615] | 0.0758 [0.0687, 0.0829] | 0.0614 [0.0550, 0.0678] | 0.0575 [0.0551, 0.0599] | 0.0604 [0.0586, 0.0623] | 0.00 | 35333 | 90.22 |
| curriculum[cur_levels=1] | 0.0419 [0.0375, 0.0464] | 0.0687 [0.0531, 0.0843] | 0.0713 [0.0636, 0.0790] | 0.0473 [0.0417, 0.0530] | 0.0697 [0.0579, 0.0815] | 0.0630 [0.0603, 0.0658] | 0.0564 [0.0518, 0.0610] | 0.0762 [0.0642, 0.0883] | 0.0625 [0.0499, 0.0751] | 0.0612 [0.0587, 0.0636] | 0.0621 [0.0580, 0.0661] | 0.00 | 35333 | 87.61 |
| ekf | **0.0109 [0.0109, 0.0109]** | **0.0114 [0.0114, 0.0114]** | **0.0113 [0.0113, 0.0113]** | **0.0154 [0.0154, 0.0154]** | **0.0129 [0.0129, 0.0129]** | **0.0120 [0.0120, 0.0120]** | **0.0124 [0.0124, 0.0124]** | **0.0130 [0.0130, 0.0130]** | **0.0109 [0.0109, 0.0109]** | **0.0119 [0.0119, 0.0119]** | **0.0122 [0.0122, 0.0122]** | 0.00 | – | 2097.29 |
| kkl_mb | 0.2707 [0.1893, 0.3520] | 0.2810 [0.2058, 0.3561] | 0.2652 [0.2173, 0.3132] | 0.2674 [0.1894, 0.3455] | 0.2738 [0.2116, 0.3360] | 0.2788 [0.2178, 0.3399] | 0.2727 [0.2059, 0.3395] | 0.2754 [0.2065, 0.3444] | 0.2819 [0.2114, 0.3523] | 0.2801 [0.2244, 0.3357] | 0.2750 [0.2084, 0.3416] | 0.00 | – | 4907.50 |

Paired vs `autonomous`:

| method | am | constant | drive | multisine | prbs | ramp | sinusoid | square | staircase | step | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| static | **-3.8% [-6, -2]** | **-17.3% [-21, -14]** | **-10.7% [-16, -5]** | **-3.4% [-4, -2]** | **-14.8% [-22, -8]** | **-10.3% [-12, -9]** | **-8.2% [-11, -5]** | **-20.9% [-24, -17]** | **-16.3% [-19, -13]** | **-12.8% [-16, -10]** | **-12.4% [-15, -10]** |
| dynamic | **-63.4% [-84, -43]** | **-79.0% [-84, -74]** | **-70.6% [-79, -62]** | **-70.6% [-80, -61]** | **-68.4% [-75, -62]** | **-76.2% [-83, -70]** | **-69.4% [-81, -58]** | **-74.4% [-80, -69]** | **-75.9% [-84, -68]** | **-75.2% [-84, -66]** | **-72.6% [-81, -64]** |
| curriculum | **-86.0% [-92, -80]** | **-79.7% [-82, -78]** | **-75.6% [-83, -68]** | **-83.4% [-92, -75]** | **-78.2% [-85, -71]** | **-79.4% [-85, -74]** | **-81.3% [-86, -77]** | **-77.6% [-81, -74]** | **-81.2% [-85, -78]** | **-81.6% [-86, -78]** | **-80.3% [-85, -75]** |
| curriculum[cur_levels=1] | **-84.9% [-90, -79]** | **-79.0% [-80, -78]** | **-75.4% [-84, -67]** | **-82.7% [-90, -76]** | **-78.0% [-83, -73]** | **-79.3% [-84, -75]** | **-80.7% [-85, -76]** | **-77.5% [-80, -75]** | **-80.9% [-83, -78]** | **-80.4% [-84, -77]** | **-79.8% [-84, -76]** |
| ekf | **-96.1% [-97, -95]** | **-96.5% [-97, -96]** | **-96.1% [-97, -95]** | **-94.4% [-96, -93]** | **-95.9% [-97, -95]** | **-96.1% [-97, -95]** | **-95.8% [-97, -95]** | **-96.1% [-97, -95]** | **-96.6% [-97, -96]** | **-96.2% [-97, -95]** | **-96.0% [-97, -95]** |
| kkl_mb | **-3.8% [-5, -2]** | **-14.2% [-16, -12]** | **-9.3% [-16, -3]** | **-3.2% [-4, -2]** | **-14.0% [-18, -10]** | **-8.8% [-12, -6]** | **-7.3% [-9, -6]** | **-18.8% [-22, -16]** | **-14.0% [-17, -11]** | **-10.7% [-14, -8]** | **-10.9% [-13, -9]** |

### shape_ood / clean

| method | chirp | filtered_noise | sawtooth | triangle | all | div | params | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.2860 [0.2105, 0.3614] | 0.2892 [0.2061, 0.3724] | 0.3142 [0.2360, 0.3925] | 0.2941 [0.2392, 0.3490] | 0.2959 [0.2230, 0.3687] | 0.00 | – | 87.20 |
| static | 0.2651 [0.1917, 0.3384] | 0.2655 [0.1831, 0.3479] | 0.2738 [0.1894, 0.3582] | 0.2643 [0.2008, 0.3278] | 0.2672 [0.1913, 0.3430] | 0.00 | 140196 | 670.91 |
| dynamic | 0.0869 [0.0422, 0.1317] | 0.0809 [0.0359, 0.1259] | 0.0863 [0.0381, 0.1344] | 0.0866 [0.0542, 0.1189] | 0.0852 [0.0429, 0.1274] | 0.00 | 3658272 | 1799.23 |
| curriculum | 0.0499 [0.0447, 0.0551] | 0.0497 [0.0491, 0.0503] | 0.0530 [0.0466, 0.0595] | 0.0643 [0.0571, 0.0715] | 0.0542 [0.0497, 0.0588] | 0.00 | 35333 | 90.22 |
| curriculum[cur_levels=1] | 0.0528 [0.0467, 0.0589] | 0.0523 [0.0484, 0.0561] | 0.0554 [0.0448, 0.0661] | 0.0646 [0.0574, 0.0718] | 0.0563 [0.0499, 0.0627] | 0.00 | 35333 | 87.61 |
| ekf | **0.0119 [0.0119, 0.0119]** | **0.0126 [0.0126, 0.0126]** | **0.0120 [0.0120, 0.0120]** | **0.0126 [0.0126, 0.0126]** | **0.0123 [0.0123, 0.0123]** | 0.00 | – | 2097.29 |
| kkl_mb | 0.2659 [0.1958, 0.3361] | 0.2695 [0.1905, 0.3486] | 0.2787 [0.2011, 0.3564] | 0.2685 [0.2125, 0.3244] | 0.2707 [0.2000, 0.3413] | 0.00 | – | 4907.50 |

Paired vs `autonomous`:

| method | chirp | filtered_noise | sawtooth | triangle | all |
|---|---|---|---|---|---|
| static | **-7.3% [-9, -5]** | **-8.3% [-11, -6]** | **-13.0% [-18, -8]** | **-10.2% [-15, -5]** | **-9.8% [-13, -6]** |
| dynamic | **-69.7% [-80, -60]** | **-72.1% [-83, -61]** | **-72.7% [-82, -63]** | **-70.6% [-80, -61]** | **-71.3% [-81, -61]** |
| curriculum | **-82.4% [-88, -77]** | **-82.7% [-88, -78]** | **-83.0% [-88, -78]** | **-78.0% [-84, -72]** | **-81.5% [-87, -76]** |
| curriculum[cur_levels=1] | **-81.4% [-86, -77]** | **-81.8% [-86, -78]** | **-82.3% [-85, -80]** | **-78.0% [-83, -73]** | **-80.9% [-85, -77]** |
| ekf | **-95.8% [-97, -95]** | **-95.6% [-97, -94]** | **-96.2% [-97, -95]** | **-95.7% [-97, -95]** | **-95.8% [-97, -95]** |
| kkl_mb | **-7.0% [-9, -5]** | **-6.8% [-9, -5]** | **-11.4% [-14, -9]** | **-8.8% [-12, -6]** | **-8.6% [-11, -7]** |

### sequence_ood / clean

| method | drive | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.3096 [0.2429, 0.3763] | 0.3096 [0.2429, 0.3763] | 0.00 | – | 87.20 |
| static | 0.2681 [0.1918, 0.3445] | 0.2681 [0.1918, 0.3445] | 0.00 | 140196 | 670.91 |
| dynamic | 0.1086 [0.0530, 0.1642] | 0.1086 [0.0530, 0.1642] | 0.00 | 3658272 | 1799.23 |
| curriculum | 0.0584 [0.0575, 0.0592] | 0.0584 [0.0575, 0.0592] | 0.00 | 35333 | 90.22 |
| curriculum[cur_levels=1] | 0.0601 [0.0575, 0.0628] | 0.0601 [0.0575, 0.0628] | 0.00 | 35333 | 87.61 |
| ekf | **0.0119 [0.0119, 0.0119]** | **0.0119 [0.0119, 0.0119]** | 0.00 | – | 2097.29 |
| kkl_mb | 0.2682 [0.1995, 0.3369] | 0.2682 [0.1995, 0.3369] | 0.00 | – | 4907.50 |

Paired vs `autonomous`:

| method | drive | all |
|---|---|---|
| static | **-13.5% [-20, -7]** | **-13.5% [-20, -7]** |
| dynamic | **-65.0% [-79, -51]** | **-65.0% [-79, -51]** |
| curriculum | **-81.1% [-85, -77]** | **-81.1% [-85, -77]** |
| curriculum[cur_levels=1] | **-80.5% [-84, -77]** | **-80.5% [-84, -77]** |
| ekf | **-96.1% [-97, -95]** | **-96.1% [-97, -95]** |
| kkl_mb | **-13.5% [-17, -10]** | **-13.5% [-17, -10]** |

### fam_zero / clean

| method | zero | all | div | params | µs/step |
|---|---|---|---|---|---|
| autonomous | 0.2632 [0.1892, 0.3372] | 0.2632 [0.1892, 0.3372] | 0.00 | – | 87.20 |
| static | 0.2632 [0.1892, 0.3372] | 0.2632 [0.1892, 0.3372] | 0.00 | 140196 | 670.91 |
| dynamic | 0.2632 [0.1892, 0.3372] | 0.2632 [0.1892, 0.3372] | 0.00 | 3658272 | 1799.23 |
| curriculum | 0.0467 [0.0437, 0.0497] | 0.0467 [0.0437, 0.0497] | 0.00 | 35333 | 90.22 |
| curriculum[cur_levels=1] | 0.0483 [0.0439, 0.0526] | 0.0483 [0.0439, 0.0526] | 0.00 | 35333 | 87.61 |
| ekf | **0.0126 [0.0126, 0.0126]** | **0.0126 [0.0126, 0.0126]** | 0.00 | – | 2097.29 |
| kkl_mb | 0.2632 [0.1892, 0.3372] | 0.2632 [0.1892, 0.3372] | 0.00 | – | 4907.50 |

Paired vs `autonomous`:

| method | zero | all |
|---|---|---|
| static | +0.0% [+0, +0] | +0.0% [+0, +0] |
| dynamic | +0.0% [+0, +0] | +0.0% [+0, +0] |
| curriculum | **-82.1% [-88, -76]** | **-82.1% [-88, -76]** |
| curriculum[cur_levels=1] | **-81.5% [-88, -75]** | **-81.5% [-88, -75]** |
| ekf | **-95.2% [-97, -94]** | **-95.2% [-97, -94]** |
| kkl_mb | +0.0% [+0, +0] | +0.0% [+0, +0] |

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### highway_traffic

| method | id | shape_ood | sequence_ood | unmeasured (seq) | p99 (seq) | div. rate (max) | alarm AUROC (seq) | µs/step |
|---|---|---|---|---|---|---|---|---|
| autonomous | 0.3085 | 0.2959 | 0.3096 | 0.4776 | 0.3676 | 0.00 (shape_ood/clean) | 0.834 | 87.8 |
| static | 0.2704 | 0.2672 | 0.2681 | 0.4190 | 0.3104 | 0.00 (shape_ood/clean) | 0.887 | 684 |
| dynamic | 0.0846 | 0.0852 | 0.1086 | 0.1697 | 0.1547 | 0.00 (shape_ood/clean) | 0.937 | 1.81e+03 |
| curriculum | 0.0604 | 0.0542 | 0.0584 | 0.0872 | 0.0945 | 0.00 (shape_ood/clean) | 0.956 | 90.2 |
| curriculum[cur_levels=1] | 0.0621 | 0.0563 | 0.0601 | 0.0905 | 0.0942 | 0.00 (shape_ood/clean) | 0.954 | 87.3 |
| ekf | 0.0122 | 0.0123 | 0.0119 | 0.0174 | 0.0184 | 0.00 (shape_ood/clean) | 0.928 | 2e+03 |
| kkl_mb | 0.2750 | 0.2707 | 0.2682 | 0.4189 | 0.3128 | 0.00 (shape_ood/clean) | 0.874 | 4.85e+03 |
