# Information budget vs achieved error: results

Generated 2026-10-26 11:27 CET by `score.py` from `budgets.json` and `achieved.json` (rules: `PREREG.md`). Nothing hand-edited.

## 1. Per-system information budget (eval id, Bayesian PCRB) and group

| system | n_x | n_y | bits full | bits 3/a | bits 1/a | unm lb full | unm lb 3/a (A) | unm lb 1/a | rho | total lb full | RD lb full | group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| battery | 5 | 2 | 39.3 | 37.5 | 33.4 | 0.0036 | 0.0084 | 0.0333 | 2.31 | 0.0037 | 0.0025 | window-starved |
| bioreactor | 2 | 1 | 8.6 | 8.6 | 8.6 | 0.0085 | 0.0085 | 0.0085 | 1.00 | 0.0063 | 0.0047 | rich |
| duffing | 2 | 1 | 16.1 | 14.5 | 11.7 | 0.0053 | 0.0113 | 0.0696 | 2.14 | 0.0045 | 0.0041 | window-starved |
| lorenz | 3 | 1 | 27.5 | 27.5 | 27.5 | 0.0014 | 0.0014 | 0.0014 | 1.00 | 0.0014 | 0.0014 | rich |
| quad_tank | 4 | 2 | 21.3 | 20.3 | 18.5 | 0.0198 | 0.0297 | 0.0517 | 1.50 | 0.0150 | 0.0104 | window-starved |
| vdp | 2 | 1 | 16.2 | 16.2 | 15.7 | 0.0038 | 0.0038 | 0.0051 | 1.00 | 0.0036 | 0.0036 | rich |
| cartpole | 4 | 2 | 29.4 | 27.0 | 25.1 | 0.0107 | 0.0415 | 0.1200 | 3.89 | 0.0084 | 0.0045 | window-starved |
| fhn | 2 | 1 | 14.9 | 14.8 | 14.0 | 0.0077 | 0.0078 | 0.0141 | 1.01 | 0.0063 | 0.0055 | rich |
| highway_traffic | 5 | 3 | 21.5 | 18.7 | 17.7 | 0.0186 | 0.0512 | 0.0721 | 2.75 | 0.0124 | 0.0083 | window-starved |
| network | 10 | 3 | 48.5 | 48.3 | 46.0 | 0.0076 | 0.0079 | 0.0110 | 1.04 | 0.0070 | 0.0051 | rich |
| rossler | 3 | 1 | 27.0 | 27.0 | 26.9 | 0.0014 | 0.0014 | 0.0014 | 1.00 | 0.0028 | 0.0020 | rich |
| two_link_arm | 4 | 2 | 21.0 | 21.0 | 20.0 | 0.0455 | 0.0456 | 0.0500 | 1.00 | 0.0352 | 0.0091 | absolutely starved |

Groups: **rich**: bioreactor, lorenz, vdp, fhn, network, rossler; **window-starved**: battery, duffing, quad_tank, cartpole, highway_traffic; **absolutely starved**: two_link_arm; **marginal**: none

## 2. P0 - every achieved error above the full-past bound (tolerance 5%)

**P0: FAIL** - 448 (system, knowledge, method, tier) cells, 3 violations of `rms >= 0.95 x bound` (total or any state).

| system | runs | method | tier | quantity | achieved rms | bound |
|---|---|---|---|---|---|---|
| quad_tank | full | ekf | sequence_ood | x3 | 0.0233 | 0.0245 |
| cartpole | full | ekf | id | total | 0.0079 | 0.0084 |
| cartpole | full | ekf | id | x2 | 0.0134 | 0.0145 |

Tightness (tier id, pooled RMS / bound; unmeasured = mean per-state ratio):

| system | EKF exact / bound (total) | EKF exact / bound (unmeasured) | best grey-box / bound (total) | HyperKKL / bound (total) | HyperKKL unm / A(3/a) | HyperKKL unm / bound full |
|---|---|---|---|---|---|---|
| battery | 1.75 | 1.99 | 9.09 (greybox_ekf_oracle) | 15.05 | 6.40 | 14.81 |
| bioreactor | 0.99 | 0.99 | 46.29 (greybox_ukf) | 3.94 | 2.46 | 2.46 |
| duffing | 1.04 | 1.04 | 1.71 (greybox_ukf) | 10.96 | 5.78 | 12.34 |
| lorenz | 4.63 | 4.61 | 157.05 (greybox_ekf_oracle) | 29.09 | 17.68 | 17.68 |
| quad_tank | 0.99 | 0.99 | 1.03 (greybox_ekf) | 3.62 | 2.24 | 3.37 |
| vdp | 1.58 | 1.67 | 5.78 (greybox_ekf) | 4.66 | 5.59 | 5.60 |
| cartpole | 0.95 | 0.95 | 5.31 (greybox_ukf) | 24.16 | 4.22 | 16.44 |
| fhn | 1.13 | 1.07 | 2.42 (greybox_ekf) | 4.78 | 4.65 | 4.71 |
| highway_traffic | 1.01 | 1.00 | 6.41 (greybox_ekf) | 5.89 | 2.07 | 5.68 |
| network | 1.14 | 1.17 | - (None) | 6.74 | 5.78 | 6.00 |
| rossler | 16.35 | 30.43 | 261.19 (greybox_ukf) | 38.11 | 71.74 | 71.74 |
| two_link_arm | 1.85 | 1.85 | 9.16 (greybox_ukf) | 3.66 | 3.76 | 3.77 |

## 3. P1 - rank correlation across the 12 plants

| test | x | y | n | Spearman rho_s | p (two-sided) | threshold | verdict |
|---|---|---|---|---|---|---|---|
| **P1a** | A | lora_unm_id | 12 | 0.650 | 0.022 | >= 0.5 | **PASS** |
| **P1b** | full_unm | ekf_unm_id | 12 | 0.580 | 0.0479 | >= 0.7 | **FAIL** |
| **P1c** | full_unm | gb_unm_id | 11 | -0.045 | 0.894 | >= 0.5 | **FAIL** |

Secondary (reported, no verdict):

| x | y | n | rho_s | p |
|---|---|---|---|---|
| A | lora_unm_id 11 plants | 11 | 0.609 | 0.0467 |
| bits3 | lora_unm_id  | 12 | 0.273 | 0.391 |
| full_tot | lora_tot_id  | 12 | 0.510 | 0.0899 |
| full_tot | ekf_tot_id  | 12 | 0.545 | 0.0666 |
| A | lora_unm_seq  | 12 | 0.755 | 0.00451 |
| full_unm | gb_unm_seq  | 11 | -0.036 | 0.915 |
| full_unm | lora_unm_id  | 12 | 0.483 | 0.112 |

Per-plant values used (tier id unless noted):

| system | group | A | unm lb full | HyperKKL unm | HyperKKL unm (seq) | EKF exact unm | best grey-box unm | best grey-box unm (seq) |
|---|---|---|---|---|---|---|---|---|
| battery | window-starved | 0.0084 | 0.0036 | 0.0535 | 0.0831 | 0.0062 | 0.0274 (greybox_ekf_oracle) | 0.0192 |
| bioreactor | rich | 0.0085 | 0.0085 | 0.0210 | 0.0751 | 0.0084 | 0.2700 (greybox_ukf) | 0.2696 |
| duffing | window-starved | 0.0113 | 0.0053 | 0.0655 | 0.0762 | 0.0055 | 0.0095 (greybox_ekf) | 0.0101 |
| lorenz | rich | 0.0014 | 0.0014 | 0.0247 | 0.0240 | 0.0064 | 0.2390 (greybox_ekf_oracle) | 0.2386 |
| quad_tank | window-starved | 0.0297 | 0.0198 | 0.0667 | 0.0933 | 0.0188 | 0.0196 (greybox_ekf) | 0.0185 |
| vdp | rich | 0.0038 | 0.0038 | 0.0212 | 0.0247 | 0.0062 | 0.0248 (greybox_ekf) | 0.0327 |
| cartpole | window-starved | 0.0415 | 0.0107 | 0.1753 | 0.2850 | 0.0095 | 0.0478 (greybox_ukf) | 0.0712 |
| fhn | rich | 0.0078 | 0.0077 | 0.0361 | 0.0671 | 0.0081 | 0.0140 (greybox_ekf_oracle) | 0.0135 |
| highway_traffic | window-starved | 0.0512 | 0.0186 | 0.1057 | 0.1188 | 0.0179 | 0.0434 (greybox_ekf) | 0.0334 |
| network | rich | 0.0079 | 0.0076 | 0.0457 | 0.0680 | 0.0087 | - (None) | - |
| rossler | rich | 0.0014 | 0.0014 | 0.1024 | 0.1050 | 0.0424 | 0.6577 (greybox_ukf) | 0.7091 |
| two_link_arm | absolutely starved | 0.0456 | 0.0455 | 0.1716 | 0.1874 | 0.0841 | 0.3148 (greybox_ukf) | 0.3165 |

## 4. P2 - the learned-vs-grey-box gap by group (partial-knowledge runs, unmeasured NRMSE)

**P2 (sequence_ood, primary): PASS** - median gap window-starved +464% vs rich -66%; 5/5 window-starved plants above the rich median (need >= 4).

| system | group | gap (lora - best grey-box) / best grey-box |
|---|---|---|
| battery | window-starved | +339% |
| bioreactor | rich | -71% |
| duffing | window-starved | +639% |
| lorenz | rich | -88% |
| quad_tank | window-starved | +413% |
| vdp | rich | -28% |
| cartpole | window-starved | +767% |
| fhn | rich | +533% |
| highway_traffic | window-starved | +464% |
| rossler | rich | -66% |
| two_link_arm | absolutely starved | +19% |

**P2 (id, secondary): PASS** - median gap window-starved +321% vs rich -63%; 5/5 window-starved plants above the rich median (need >= 4).

| system | group | gap (lora - best grey-box) / best grey-box |
|---|---|---|
| battery | window-starved | +93% |
| bioreactor | rich | -91% |
| duffing | window-starved | +624% |
| lorenz | rich | -89% |
| quad_tank | window-starved | +235% |
| vdp | rich | -19% |
| cartpole | window-starved | +562% |
| fhn | rich | +270% |
| highway_traffic | window-starved | +321% |
| rossler | rich | -63% |
| two_link_arm | absolutely starved | +22% |
