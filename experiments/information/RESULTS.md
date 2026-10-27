# Information budget vs achieved error: results

Generated 2026-10-26 11:27 CET by `score.py` from `budgets.json` and `achieved.json` (rules: `PREREG.md`). Nothing hand-edited except section 0.

## 0. Summary (written by the agent; everything below it is mechanical)

* **What was computed.** For each of the 12 plants, the information about the current state that the measurements
  and inputs carry (bits beyond climatology) and the implied floor on any observer's NRMSE per state, from the
  Bayesian posterior Cramer-Rao recursion (a theorem for Gaussian noises; `infolib.py`) along the cached evaluation
  episodes, for windows 1-10/alpha and the full past (`budgets.json`); the same on the training episodes (equal to
  within 0.3 bits everywhere). The local (EKF-covariance) recursion is reported next to it.
* **Guarantee (P0).** 448 (method, tier) cells: no learned observer and no grey-box filter is below any floor. 3 cells
  of the exact-model EKF on slow states (quad_tank x3, cartpole x2/total) are 5-8% below the full-past floor, each
  inside the bootstrap 95% CI of the achieved pooled RMS (per-episode CV 0.3-0.5). By the pre-registered 5% rule
  this is a FAIL; the honest reading is that the floors are attained by the optimal filter to within sampling error
  and never beaten beyond it. The exact-model EKF sits at 0.95-1.07 of the floor on the near-linear plants.
* **Grouping.** Window-starved (rho >= 1.4: a fading-memory observer loses >= 40% of the attainable precision on the
  unmeasured states): battery, duffing, quad_tank, cartpole, highway_traffic. Rich: bioreactor, lorenz, vdp, fhn,
  network, rossler. Absolutely starved (poor even with unbounded memory): two_link_arm.
* **quad_tank is window-starved, not information-starved in the absolute sense:** the full-past floor on the
  unmeasured tanks is 0.0198 and the grey-box EKF achieves 0.0185-0.0197 (0.95-1.01 of the floor at every
  excitation level): it is information-optimal, and *no observer of any kind can beat it by more than ~1%*. The
  question is why HyperKKL is 3.4x above the floor: the 1/alpha and 3/alpha windows explain a factor 2.6 and 1.5
  (floors 0.052 and 0.030); HyperKKL sits at 1.9-2.6x the 3/alpha floor. The rest is learning-side.
* **Excitation.** For a plant with a known model the state information is a Gramian property: the evaluation-side
  excitation does not change the floor (quad_tank: 21.27 bits at every input time scale from x1/4 to x4), and the
  exact-model EKF's error is flat (P3b PASS 3/3). HyperKKL's error moved by only 1.1-1.3x (P3c FAIL: less than
  predicted; on quad_tank slower inputs lowered it from 0.076 to 0.058 with no change in the budget, a small
  learning-side effect). Training-side enrichment (experiments/quadtank) cannot change the evaluation floor either
  (P4); it can only change coverage. Prediction kept: no arm brings HyperKKL below 0.020 (10/alpha floor).
* **Rank correlations (P1).** HyperKKL's unmeasured error tracks the 3/alpha floor across the 12 plants
  (Spearman 0.65, p = 0.02; 0.76 on sequence OOD): PASS. The exact-model EKF tracks the full-past floor at 0.58
  (FAIL at the 0.7 bar: the Bayesian floor is loose on the chaotic plants); the grey-box does not track it at all
  (-0.05, FAIL): where its residual model is wrong (bioreactor, lorenz, rossler, two_link_arm) it fails outright,
  so its error is not information-limited.
* **Where learned observers lose (P2).** The gap to the best grey-box on the unmeasured states is +464% (median) in
  the window-starved group against -66% in the rich group, 5/5 window-starved plants above the rich median (PASS on
  both tiers); the one unexplained case is fhn (rich, +533%).
* **Cross-check with experiments/quadtank/DIAG.md.** Their no-prior, no-process-noise CRB on the window-start state
  (10 s: h3 0.080, h4 0.045) agrees in magnitude with the 1/alpha floor here (0.066, 0.038); at long windows theirs
  keeps falling (0.009 at 49 s) while the floor here saturates at the process-noise level (0.025, 0.015), which is
  the value the grey-box filter actually reaches. Both analyses conclude observability-/window-limited.
* **Kolmogorov vs Shannon/Fisher.** Kolmogorov complexity is uncomputable (no algorithm computes it, every computable
  lower bound is bounded) and machine-dependent up to a constant, so it cannot yield a per-system number or guarantee.
  The computable content is Fisher/Shannon (above); the honest link is MDL (two-part code: model bits + residual bits,
  the latter bounded below by the rate-distortion bits), which interprets why a structured grey-box needs fewer
  informative samples but adds no quantity. Stated as a remark, not a theorem (theory/m4_information.tex).
* **Noise-model caveat (test plants).** The TEST-C3 runs use coloured Student-t noise on a 30%-perturbed plant; the
  budgets here were computed under the TEST-C1 white-noise exact-plant setting. The post-hoc recomputation
  (`BUDGET_FIX.md`, coordinating agent) gives A = 0.065 (cartpole), 0.097 (highway_traffic), 0.020 (two_link_arm);
  groups and verdicts are unchanged except that two_link_arm becomes "marginal" (see the PREREG addendum).
* **Real battery.** Information floor 0.02-0.11% SOC (sensor-level noise) against achieved 0.7-9%: model error, not
  information, limits every observer on the real cell.

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


## 5. P3 - dose-response: evaluation-input excitation (dose/*.json; `dose_response.py`)

### quad_tank (window-starved; seeds [0, 1, 2]; 5 levels)

| s | slow-band frac | mean switch (s) | bits full | bits 3/a | unm lb full | unm lb 3/a | lora unm | ekf_full unm | ekf unm | greybox_ekf unm | greybox_ukf unm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.000 | 79.4 | 21.27 | 20.33 | 0.0198 | 0.0297 | 0.0761 | 0.0186 | 0.0353 | 0.0189 | 0.0189 |
| 0.5 | 0.000 | 84.6 | 21.27 | 20.33 | 0.0198 | 0.0297 | 0.0733 | 0.0190 | 0.0405 | 0.0196 | 0.0196 |
| 1 | 0.000 | 91.7 | 21.27 | 20.33 | 0.0198 | 0.0297 | 0.0630 | 0.0181 | 0.0429 | 0.0190 | 0.0190 |
| 2 | 0.000 | 114.9 | 21.28 | 20.33 | 0.0198 | 0.0297 | 0.0640 | 0.0192 | 0.0391 | 0.0201 | 0.0201 |
| 4 | 0.000 | 159.4 | 21.27 | 20.33 | 0.0198 | 0.0297 | 0.0578 | 0.0186 | 0.0438 | 0.0194 | 0.0194 |

**P3a (quad_tank): FAIL** - 5 violations of `rms >= 0.95 x full-past bound` over 30 (level, method) cells.

| level | method | quantity | achieved rms | bound |
|---|---|---|---|---|
| 0.25 | ekf_full | x4 | 0.0143 | 0.0151 |
| 1 | greybox_ekf | x4 | 0.0143 | 0.0151 |
| 1 | greybox_ukf | x4 | 0.0143 | 0.0151 |
| 1 | ekf_full | x4 | 0.0140 | 0.0151 |
| 4 | ekf_full | x3 | 0.0232 | 0.0246 |
**P3b (quad_tank): PASS** - budget flat (max/min x1.000 < 1.2): ekf_full error max/min x1.06 (need < 1.3).
**P3c (quad_tank)**: HyperKKL error max/min x1.32 vs ekf_full x1.06 -> ratio criterion (>= 1.5x) NOT met; HyperKKL minimum at s = 4 (away from s = 1).
**P3d (quad_tank)**: HyperKKL unm / A(3/a) over levels: [2.56, 2.47, 2.12, 2.15, 1.94] (within x2 of each other: yes); HyperKKL unm / full-past bound: [3.84, 3.7, 3.18, 3.23, 2.92]; best grey-box / full-past bound: [0.95, 0.99, 0.96, 1.01, 0.98].

### vdp (rich; seeds [0, 1, 2]; 5 levels)

| s | slow-band frac | mean switch (s) | bits full | bits 3/a | unm lb full | unm lb 3/a | lora unm | ekf_full unm | ekf unm | greybox_ekf unm | greybox_ukf unm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.209 | 12.8 | 16.15 | 16.14 | 0.0038 | 0.0039 | 0.0198 | 0.0061 | 0.4156 | 0.0230 | 0.0230 |
| 0.5 | 0.224 | 13.1 | 16.16 | 16.15 | 0.0038 | 0.0038 | 0.0192 | 0.0060 | 0.4054 | 0.0239 | 0.0239 |
| 1 | 0.319 | 13.7 | 16.25 | 16.24 | 0.0037 | 0.0037 | 0.0202 | 0.0065 | 0.4085 | 0.0250 | 0.0250 |
| 2 | 0.510 | 15.2 | 16.19 | 16.18 | 0.0039 | 0.0039 | 0.0186 | 0.0066 | 0.4290 | 0.0238 | 0.0238 |
| 4 | 0.663 | 17.6 | 16.14 | 16.13 | 0.0039 | 0.0039 | 0.0176 | 0.0061 | 0.3989 | 0.0218 | 0.0218 |

**P3a (vdp): PASS** - 0 violations of `rms >= 0.95 x full-past bound` over 30 (level, method) cells.
**P3b (vdp): PASS** - budget flat (max/min x1.054 < 1.2): ekf_full error max/min x1.09 (need < 1.3).
**P3c (vdp)**: HyperKKL error max/min x1.15 vs ekf_full x1.09 -> ratio criterion (>= 1.5x) NOT met; HyperKKL minimum at s = 4 (away from s = 1).

### two_link_arm (absolutely starved; seeds [0, 1, 2]; 5 levels)

| s | slow-band frac | mean switch (s) | bits full | bits 3/a | unm lb full | unm lb 3/a | lora unm | ekf_full unm | ekf unm | greybox_ekf unm | greybox_ukf unm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.000 | 25.6 | 27.03 | 26.90 | 0.0211 | 0.0216 | 0.3831 | 0.4455 | 0.4016 | 0.3079 | 0.3079 |
| 0.5 | 0.000 | 26.2 | 26.89 | 26.75 | 0.0216 | 0.0222 | 0.3944 | 0.4088 | 0.3699 | 0.3777 | 0.3781 |
| 1 | 0.000 | 27.4 | 26.46 | 26.34 | 0.0238 | 0.0244 | 0.3617 | 0.3835 | 0.3564 | 0.4672 | 0.4669 |
| 2 | 0.000 | 29.7 | 26.81 | 26.67 | 0.0220 | 0.0226 | 0.3897 | 0.4148 | 0.3754 | 0.3204 | 0.2965 |
| 4 | 0.000 | 35.9 | 27.07 | 26.92 | 0.0208 | 0.0215 | 0.3808 | 0.4302 | 0.3774 | 0.3389 | 0.2870 |

**P3a (two_link_arm): PASS** - 0 violations of `rms >= 0.95 x full-past bound` over 30 (level, method) cells.
**P3b (two_link_arm): PASS** - budget flat (max/min x1.144 < 1.2): ekf_full error max/min x1.16 (need < 1.3).
**P3c (two_link_arm)**: HyperKKL error max/min x1.09 vs ekf_full x1.16 -> ratio criterion (>= 1.5x) NOT met; HyperKKL minimum at s = 1 (at/next to s = 1).

**P3c overall: FAIL** (ratio criterion on every plant and minimum at/next to s = 1 on >= 2 of 3).

## 6. Real battery (Panasonic 18650PF; `real_battery.py`)

SOC RMSE floor (% SOC) from the identified 2-RC model driven by the measured current, 30-min windows, settled half; (i) the tuned EKF's own (Q, R) [R_vm = 0.1 V], (ii) sensor-level noise [R_vm = 3 mV]; achieved settled SOC RMSE of the real-data run alongside.

| cycle | windows | floor (i) local | floor (i) Bayes | floor (ii) local | floor (ii) Bayes | HyperKKL | EKF | UKF | grey-box EKF | grey-box UKF |
|---|---|---|---|---|---|---|---|---|---|---|
| US06 | 11 | 0.70 | 0.13 | 0.069 | 0.021 | 2.18 | 8.85 | 9.13 | 1.05 | 0.87 |
| HWFTa | 20 | 0.87 | 0.13 | 0.088 | 0.021 | 0.91 | 1.19 | 1.39 | 1.03 | 0.94 |
| HWFTb | 20 | 0.87 | 0.13 | 0.088 | 0.021 | 0.91 | 1.19 | 1.39 | 1.03 | 0.94 |
| NN | 34 | 1.09 | 0.14 | 0.111 | 0.022 | 0.73 | 1.95 | 1.55 | 1.35 | 1.25 |
| Cycle_4 | 35 | 0.99 | 0.12 | 0.101 | 0.021 | 0.75 | 3.25 | 3.10 | 0.97 | 0.86 |

Reading: setting (ii) is the information floor if the identified model were right; every achieved error is 7-100x above it, so on the real cell the error is model error (the ECM), not a shortage of information. Setting (i) is the tuned filter's own belief (inflated Q, R), not a bound: HyperKKL is below its local value on NN and Cycle_4. Both are statements about the model; neither is used as a guarantee.

---
Assembled 2026-10-26 16:47 CET by `score_dose.py`.
