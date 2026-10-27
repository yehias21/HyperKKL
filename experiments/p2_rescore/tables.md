### Fold ratio HyperKKL / best grey-box, unmeasured-state NRMSE

| plant | tier | best grey-box | old lora | old fold [95% CI] | A2 | A2 fold [95% CI] | A2/old [95% CI] | A4 | A4 fold [95% CI] | A4/old [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | id | 0.0274 (ekf_oracle) | 0.0528 | 1.93x [1.29, 2.82] | 0.0154 | 0.56x [0.49, 0.64] | 0.29 [0.17, 0.49] | 0.0148 | 0.54x [0.40, 0.72] | 0.28 [0.14, 0.56] |
| battery | shape_ood | 0.0237 (ekf_oracle) | 0.0630 | 2.66x [1.44, 4.74] | 0.0178 | 0.75x [0.70, 0.80] | 0.28 [0.17, 0.49] | 0.0179 | 0.76x [0.60, 0.95] | 0.28 [0.13, 0.62] |
| battery | sequence_ood | 0.0192 (ekf_oracle) | 0.0841 | 4.39x [2.46, 7.79] | 0.0203 | 1.06x [0.68, 1.69] | 0.24 [0.17, 0.35] | 0.0278 | 1.45x [1.18, 1.81] | 0.33 [0.20, 0.54] |
| quad_tank | id | 0.0196 (ekf) | 0.0654 | 3.35x [3.13, 3.58] | 0.0367 | 1.87x [1.84, 1.90] | 0.56 [0.53, 0.59] | 0.0330 | 1.69x [0.82, 3.29] | 0.50 [0.23, 1.05] |
| quad_tank | shape_ood | 0.0199 (ekf_oracle) | 0.0646 | 3.24x [3.03, 3.47] | 0.0366 | 1.84x [1.73, 1.95] | 0.57 [0.51, 0.63] | 0.0337 | 1.69x [0.86, 3.18] | 0.52 [0.25, 1.05] |
| quad_tank | sequence_ood | 0.0185 (ekf) | 0.0950 | 5.13x [4.66, 5.64] | 0.0456 | 2.46x [2.25, 2.70] | 0.48 [0.45, 0.51] | 0.0422 | 2.28x [1.24, 4.05] | 0.44 [0.22, 0.86] |
| cartpole | id | 0.0478 (ukf) | 0.3164 | 6.62x [1.13, 28.29] | 0.5085 | 10.64x [7.17, 15.60] | 1.61 [0.29, 12.14] | 0.2726 | 5.71x [1.11, 23.38] | 0.86 [0.04, 19.65] |
| cartpole | shape_ood | 0.0455 (ukf) | 0.3059 | 6.72x [1.35, 25.70] | 0.5299 | 11.64x [7.72, 17.32] | 1.73 [0.33, 11.89] | 0.2458 | 5.40x [0.69, 30.01] | 0.80 [0.03, 21.31] |
| cartpole | sequence_ood | 0.0712 (ukf) | 0.6173 | 8.67x [1.04, 47.34] | 1.1237 | 15.77x [13.23, 18.77] | 1.82 [0.32, 15.61] | 0.5896 | 8.28x [1.64, 33.01] | 0.96 [0.04, 29.50] |
| fhn | id | 0.0140 (ekf_oracle) | 0.0520 | 3.70x [3.09, 4.44] | 0.0249 | 1.77x [1.62, 1.95] | 0.48 [0.43, 0.53] | - | - | - |
| fhn | shape_ood | 0.0131 (ekf_oracle) | 0.0509 | 3.89x [3.32, 4.57] | 0.0257 | 1.97x [1.82, 2.13] | 0.51 [0.47, 0.55] | - | - | - |
| fhn | sequence_ood | 0.0135 (ekf_oracle) | 0.0852 | 6.33x [5.19, 7.71] | 0.0481 | 3.58x [3.13, 4.09] | 0.57 [0.52, 0.61] | - | - | - |
| bioreactor | id | 0.2700 (ukf) | 0.0249 | 0.09x [0.06, 0.13] | 0.0090 | 0.03x [0.03, 0.03] | 0.36 [0.26, 0.50] | 0.0087 | 0.03x [0.03, 0.03] | 0.35 [0.26, 0.48] |
| bioreactor | shape_ood | 0.2623 (ukf) | 0.0270 | 0.10x [0.06, 0.18] | 0.0097 | 0.04x [0.04, 0.04] | 0.36 [0.22, 0.61] | 0.0092 | 0.04x [0.03, 0.04] | 0.34 [0.20, 0.59] |
| bioreactor | sequence_ood | 0.2696 (ukf) | 0.0776 | 0.29x [0.17, 0.48] | 0.0159 | 0.06x [0.04, 0.08] | 0.21 [0.11, 0.40] | 0.0147 | 0.05x [0.04, 0.08] | 0.19 [0.11, 0.35] |
| lorenz | id | 0.2390 (ekf_oracle) | 0.0269 | 0.11x [0.07, 0.18] | 0.0105 | 0.04x [0.03, 0.06] | 0.39 [0.21, 0.74] | 0.0075 | 0.03x [0.03, 0.03] | 0.28 [0.17, 0.48] |
| lorenz | shape_ood | 0.2390 (ekf_oracle) | 0.0261 | 0.11x [0.06, 0.19] | 0.0103 | 0.04x [0.03, 0.05] | 0.39 [0.20, 0.81] | 0.0074 | 0.03x [0.03, 0.03] | 0.28 [0.16, 0.51] |
| lorenz | sequence_ood | 0.2386 (ekf_oracle) | 0.0278 | 0.12x [0.06, 0.20] | 0.0104 | 0.04x [0.03, 0.06] | 0.38 [0.18, 0.80] | 0.0075 | 0.03x [0.03, 0.03] | 0.27 [0.15, 0.51] |
| vdp | id | 0.0248 (ekf) | 0.0201 | 0.81x [0.53, 1.28] | 0.0101 | 0.41x [0.25, 0.66] | 0.50 [0.48, 0.52] | 0.0065 | 0.26x [0.17, 0.42] | 0.32 [0.31, 0.34] |
| vdp | shape_ood | 0.0240 (ekf) | 0.0186 | 0.77x [0.60, 1.00] | 0.0097 | 0.41x [0.30, 0.56] | 0.52 [0.49, 0.56] | 0.0065 | 0.27x [0.20, 0.36] | 0.35 [0.33, 0.36] |
| vdp | sequence_ood | 0.0327 (ekf) | 0.0234 | 0.72x [0.59, 0.88] | 0.0112 | 0.34x [0.26, 0.45] | 0.48 [0.44, 0.51] | 0.0075 | 0.23x [0.17, 0.31] | 0.32 [0.28, 0.36] |

### Fold ratio HyperKKL / best grey-box, total NRMSE

| plant | tier | best grey-box | old lora | old fold [95% CI] | A2 | A2 fold [95% CI] | A2/old [95% CI] | A4 | A4 fold [95% CI] | A4/old [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | id | 0.0270 (ukf) | 0.0476 | 1.76x [1.13, 2.72] | 0.0141 | 0.52x [0.45, 0.61] | 0.30 [0.18, 0.50] | 0.0138 | 0.51x [0.37, 0.70] | 0.29 [0.14, 0.60] |
| battery | shape_ood | 0.0242 (ukf) | 0.0563 | 2.32x [1.32, 3.96] | 0.0161 | 0.67x [0.59, 0.74] | 0.29 [0.18, 0.47] | 0.0167 | 0.69x [0.56, 0.84] | 0.30 [0.14, 0.63] |
| battery | sequence_ood | 0.0191 (ekf) | 0.0744 | 3.89x [2.33, 6.52] | 0.0192 | 1.00x [0.64, 1.59] | 0.26 [0.19, 0.36] | 0.0265 | 1.38x [1.10, 1.76] | 0.36 [0.23, 0.56] |
| quad_tank | id | 0.0150 (ekf) | 0.0517 | 3.44x [3.29, 3.61] | 0.0296 | 1.97x [1.93, 2.01] | 0.57 [0.56, 0.59] | 0.0264 | 1.75x [0.80, 3.63] | 0.51 [0.22, 1.10] |
| quad_tank | shape_ood | 0.0152 (ukf) | 0.0508 | 3.34x [3.14, 3.54] | 0.0295 | 1.94x [1.82, 2.05] | 0.58 [0.53, 0.63] | 0.0268 | 1.76x [0.84, 3.49] | 0.53 [0.24, 1.11] |
| quad_tank | sequence_ood | 0.0142 (ekf) | 0.0717 | 5.06x [4.67, 5.47] | 0.0353 | 2.49x [2.25, 2.76] | 0.49 [0.47, 0.52] | 0.0324 | 2.29x [1.16, 4.31] | 0.45 [0.21, 0.92] |
| cartpole | id | 0.0406 (ukf) | 0.3258 | 8.02x [0.88, 45.79] | 0.5642 | 13.88x [10.47, 18.28] | 1.73 [0.25, 18.66] | 0.2760 | 6.79x [1.42, 26.32] | 0.85 [0.03, 28.46] |
| cartpole | shape_ood | 0.0390 (ukf) | 0.3033 | 7.78x [1.07, 38.30] | 0.5574 | 14.30x [10.45, 19.40] | 1.84 [0.30, 16.71] | 0.2415 | 6.20x [0.84, 32.96] | 0.80 [0.02, 29.06] |
| cartpole | sequence_ood | 0.0640 (ukf) | 0.5972 | 9.33x [0.91, 57.15] | 1.1398 | 17.80x [15.17, 20.85] | 1.91 [0.30, 20.03] | 0.5620 | 8.78x [1.55, 38.31] | 0.94 [0.03, 38.42] |
| fhn | id | 0.0150 (ekf) | 0.0421 | 2.80x [2.57, 3.04] | 0.0220 | 1.46x [1.44, 1.48] | 0.52 [0.48, 0.57] | - | - | - |
| fhn | shape_ood | 0.0150 (ekf) | 0.0418 | 2.78x [2.54, 3.04] | 0.0232 | 1.54x [1.49, 1.60] | 0.55 [0.52, 0.59] | - | - | - |
| fhn | sequence_ood | 0.0152 (ekf) | 0.0678 | 4.47x [3.88, 5.13] | 0.0394 | 2.60x [2.46, 2.75] | 0.58 [0.53, 0.64] | - | - | - |
| bioreactor | id | 0.2812 (ukf) | 0.0239 | 0.08x [0.06, 0.12] | 0.0069 | 0.02x [0.02, 0.03] | 0.29 [0.21, 0.41] | 0.0066 | 0.02x [0.02, 0.02] | 0.28 [0.20, 0.39] |
| bioreactor | shape_ood | 0.2739 (ukf) | 0.0261 | 0.10x [0.06, 0.15] | 0.0078 | 0.03x [0.03, 0.03] | 0.30 [0.19, 0.47] | 0.0072 | 0.03x [0.02, 0.03] | 0.28 [0.18, 0.45] |
| bioreactor | sequence_ood | 0.2813 (ukf) | 0.0753 | 0.27x [0.17, 0.42] | 0.0147 | 0.05x [0.04, 0.07] | 0.20 [0.11, 0.36] | 0.0135 | 0.05x [0.03, 0.07] | 0.18 [0.11, 0.31] |
| lorenz | id | 0.2274 (ekf_oracle) | 0.0298 | 0.13x [0.08, 0.21] | 0.0109 | 0.05x [0.04, 0.06] | 0.36 [0.20, 0.70] | 0.0078 | 0.03x [0.03, 0.04] | 0.26 [0.16, 0.45] |
| lorenz | shape_ood | 0.2273 (ekf_oracle) | 0.0287 | 0.13x [0.07, 0.21] | 0.0106 | 0.05x [0.04, 0.06] | 0.37 [0.19, 0.74] | 0.0077 | 0.03x [0.03, 0.04] | 0.27 [0.15, 0.47] |
| lorenz | sequence_ood | 0.2269 (ekf_oracle) | 0.0308 | 0.14x [0.08, 0.23] | 0.0108 | 0.05x [0.04, 0.06] | 0.35 [0.17, 0.73] | 0.0078 | 0.03x [0.03, 0.04] | 0.25 [0.14, 0.47] |
| vdp | id | 0.0196 (ekf) | 0.0157 | 0.80x [0.52, 1.25] | 0.0086 | 0.44x [0.28, 0.71] | 0.55 [0.53, 0.57] | 0.0059 | 0.30x [0.19, 0.48] | 0.38 [0.37, 0.39] |
| vdp | shape_ood | 0.0191 (ekf) | 0.0146 | 0.77x [0.60, 0.99] | 0.0084 | 0.44x [0.33, 0.60] | 0.58 [0.55, 0.61] | 0.0059 | 0.31x [0.23, 0.42] | 0.41 [0.39, 0.42] |
| vdp | sequence_ood | 0.0257 (ekf) | 0.0179 | 0.70x [0.57, 0.86] | 0.0094 | 0.37x [0.28, 0.48] | 0.52 [0.49, 0.56] | 0.0067 | 0.26x [0.19, 0.35] | 0.37 [0.34, 0.41] |

### P2 re-score (sequence_ood, unm fold, duffing excluded; SOAP = A2 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 4.39x | 1.06x | SOAP A2 |
| bioreactor | rich | 0.29x | 0.06x | SOAP A2 |
| lorenz | rich | 0.12x | 0.04x | SOAP A2 |
| quad_tank | window-starved | 5.13x | 2.46x | SOAP A2 |
| vdp | rich | 0.72x | 0.34x | SOAP A2 |
| cartpole | window-starved | 8.67x | 15.77x | SOAP A2 |
| fhn | rich | 6.33x | 3.58x | SOAP A2 |
| highway_traffic | window-starved | 5.64x | 5.64x | old (flagged) |
| rossler | rich | 0.34x | 0.34x | old (flagged) |
| two_link_arm | absolutely starved | 1.19x | 1.19x | old (flagged) |

**old runs only**: median fold window-starved 5.38x vs rich 0.34x (geometric means 5.76x vs 0.55x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A2 where available, both groups)**: median fold window-starved 4.05x vs rich 0.34x (geometric means 3.90x vs 0.25x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A2 on window-starved only, rich kept old**: median fold window-starved 4.05x vs rich 0.34x (geometric means 3.90x vs 0.55x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (id, unm fold, duffing excluded; SOAP = A2 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 1.93x | 0.56x | SOAP A2 |
| bioreactor | rich | 0.09x | 0.03x | SOAP A2 |
| lorenz | rich | 0.11x | 0.04x | SOAP A2 |
| quad_tank | window-starved | 3.35x | 1.87x | SOAP A2 |
| vdp | rich | 0.81x | 0.41x | SOAP A2 |
| cartpole | window-starved | 6.62x | 10.64x | SOAP A2 |
| fhn | rich | 3.70x | 1.77x | SOAP A2 |
| highway_traffic | window-starved | 4.21x | 4.21x | old (flagged) |
| rossler | rich | 0.37x | 0.37x | old (flagged) |
| two_link_arm | absolutely starved | 1.22x | 1.22x | old (flagged) |

**old runs only**: median fold window-starved 3.78x vs rich 0.37x (geometric means 3.66x vs 0.41x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A2 where available, both groups)**: median fold window-starved 3.04x vs rich 0.37x (geometric means 2.62x vs 0.21x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A2 on window-starved only, rich kept old**: median fold window-starved 3.04x vs rich 0.37x (geometric means 2.62x vs 0.41x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (sequence_ood, unm fold, duffing excluded; SOAP = A4 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 4.39x | 1.45x | SOAP A4 |
| bioreactor | rich | 0.29x | 0.05x | SOAP A4 |
| lorenz | rich | 0.12x | 0.03x | SOAP A4 |
| quad_tank | window-starved | 5.13x | 2.28x | SOAP A4 |
| vdp | rich | 0.72x | 0.23x | SOAP A4 |
| cartpole | window-starved | 8.67x | 8.28x | SOAP A4 |
| fhn | rich | 6.33x | 6.33x | old (flagged) |
| highway_traffic | window-starved | 5.64x | 5.64x | old (flagged) |
| rossler | rich | 0.34x | 0.34x | old (flagged) |
| two_link_arm | absolutely starved | 1.19x | 1.19x | old (flagged) |

**old runs only**: median fold window-starved 5.38x vs rich 0.34x (geometric means 5.76x vs 0.55x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A4 where available, both groups)**: median fold window-starved 3.96x vs rich 0.23x (geometric means 3.52x vs 0.24x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A4 on window-starved only, rich kept old**: median fold window-starved 3.96x vs rich 0.34x (geometric means 3.52x vs 0.55x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (id, unm fold, duffing excluded; SOAP = A4 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 1.93x | 0.54x | SOAP A4 |
| bioreactor | rich | 0.09x | 0.03x | SOAP A4 |
| lorenz | rich | 0.11x | 0.03x | SOAP A4 |
| quad_tank | window-starved | 3.35x | 1.69x | SOAP A4 |
| vdp | rich | 0.81x | 0.26x | SOAP A4 |
| cartpole | window-starved | 6.62x | 5.71x | SOAP A4 |
| fhn | rich | 3.70x | 3.70x | old (flagged) |
| highway_traffic | window-starved | 4.21x | 4.21x | old (flagged) |
| rossler | rich | 0.37x | 0.37x | old (flagged) |
| two_link_arm | absolutely starved | 1.22x | 1.22x | old (flagged) |

**old runs only**: median fold window-starved 3.78x vs rich 0.37x (geometric means 3.66x vs 0.41x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A4 where available, both groups)**: median fold window-starved 2.95x vs rich 0.26x (geometric means 2.16x vs 0.20x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A4 on window-starved only, rich kept old**: median fold window-starved 2.95x vs rich 0.37x (geometric means 2.16x vs 0.41x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (sequence_ood, tot fold, duffing excluded; SOAP = A2 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 3.89x | 1.00x | SOAP A2 |
| bioreactor | rich | 0.27x | 0.05x | SOAP A2 |
| lorenz | rich | 0.14x | 0.05x | SOAP A2 |
| quad_tank | window-starved | 5.06x | 2.49x | SOAP A2 |
| vdp | rich | 0.70x | 0.37x | SOAP A2 |
| cartpole | window-starved | 9.33x | 17.80x | SOAP A2 |
| fhn | rich | 4.47x | 2.60x | SOAP A2 |
| highway_traffic | window-starved | 4.76x | 4.76x | old (flagged) |
| rossler | rich | 0.32x | 0.32x | old (flagged) |
| two_link_arm | absolutely starved | 1.12x | 1.12x | old (flagged) |

**old runs only**: median fold window-starved 4.91x vs rich 0.32x (geometric means 5.44x vs 0.52x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A2 where available, both groups)**: median fold window-starved 3.63x vs rich 0.32x (geometric means 3.82x vs 0.24x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A2 on window-starved only, rich kept old**: median fold window-starved 3.63x vs rich 0.32x (geometric means 3.82x vs 0.52x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (id, tot fold, duffing excluded; SOAP = A2 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 1.76x | 0.52x | SOAP A2 |
| bioreactor | rich | 0.08x | 0.02x | SOAP A2 |
| lorenz | rich | 0.13x | 0.05x | SOAP A2 |
| quad_tank | window-starved | 3.44x | 1.97x | SOAP A2 |
| vdp | rich | 0.80x | 0.44x | SOAP A2 |
| cartpole | window-starved | 8.02x | 13.88x | SOAP A2 |
| fhn | rich | 2.80x | 1.46x | SOAP A2 |
| highway_traffic | window-starved | 3.54x | 3.54x | old (flagged) |
| rossler | rich | 0.35x | 0.35x | old (flagged) |
| two_link_arm | absolutely starved | 1.16x | 1.16x | old (flagged) |

**old runs only**: median fold window-starved 3.49x vs rich 0.35x (geometric means 3.62x vs 0.39x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A2 where available, both groups)**: median fold window-starved 2.75x vs rich 0.35x (geometric means 2.66x vs 0.19x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A2 on window-starved only, rich kept old**: median fold window-starved 2.75x vs rich 0.35x (geometric means 2.66x vs 0.39x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (sequence_ood, tot fold, duffing excluded; SOAP = A4 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 3.89x | 1.38x | SOAP A4 |
| bioreactor | rich | 0.27x | 0.05x | SOAP A4 |
| lorenz | rich | 0.14x | 0.03x | SOAP A4 |
| quad_tank | window-starved | 5.06x | 2.29x | SOAP A4 |
| vdp | rich | 0.70x | 0.26x | SOAP A4 |
| cartpole | window-starved | 9.33x | 8.78x | SOAP A4 |
| fhn | rich | 4.47x | 4.47x | old (flagged) |
| highway_traffic | window-starved | 4.76x | 4.76x | old (flagged) |
| rossler | rich | 0.32x | 0.32x | old (flagged) |
| two_link_arm | absolutely starved | 1.12x | 1.12x | old (flagged) |

**old runs only**: median fold window-starved 4.91x vs rich 0.32x (geometric means 5.44x vs 0.52x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A4 where available, both groups)**: median fold window-starved 3.53x vs rich 0.26x (geometric means 3.39x vs 0.23x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A4 on window-starved only, rich kept old**: median fold window-starved 3.53x vs rich 0.32x (geometric means 3.39x vs 0.52x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

### P2 re-score (id, tot fold, duffing excluded; SOAP = A4 where available)

| plant | group | old fold | fold used | source |
|---|---|---|---|---|
| battery | window-starved | 1.76x | 0.51x | SOAP A4 |
| bioreactor | rich | 0.08x | 0.02x | SOAP A4 |
| lorenz | rich | 0.13x | 0.03x | SOAP A4 |
| quad_tank | window-starved | 3.44x | 1.75x | SOAP A4 |
| vdp | rich | 0.80x | 0.30x | SOAP A4 |
| cartpole | window-starved | 8.02x | 6.79x | SOAP A4 |
| fhn | rich | 2.80x | 2.80x | old (flagged) |
| highway_traffic | window-starved | 3.54x | 3.54x | old (flagged) |
| rossler | rich | 0.35x | 0.35x | old (flagged) |
| two_link_arm | absolutely starved | 1.16x | 1.16x | old (flagged) |

**old runs only**: median fold window-starved 3.49x vs rich 0.35x (geometric means 3.62x vs 0.39x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**mixed (A4 where available, both groups)**: median fold window-starved 2.65x vs rich 0.30x (geometric means 2.16x vs 0.19x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**

**A4 on window-starved only, rich kept old**: median fold window-starved 2.65x vs rich 0.35x (geometric means 2.16x vs 0.39x); 4/4 window-starved above the rich median -> strict (4/4): **PASS**, >=3/4: **PASS**
