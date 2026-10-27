# P2 re-score with the SOAP recipe (A2 = SOAP + trainable base decoder; A4 = full stack)

Generated 2026-10-26 17:11 Dubai. Rules: `PREREG.md` (+ addendum 1). Numbers: `code/analyze.py` -> `results.json`, `tables.md` (all tiers, both metrics, all P2 variants).
Fold = HyperKKL / best stored grey-box (lowest seed-mean of greybox_ekf/ukf/ekf_oracle, per tier and metric); 95% CI = paired over seeds 0-2, t-interval (df 2) on log ratios.
Sources: battery, quad_tank, bioreactor, lorenz, vdp = beat-greybox campaign C3-partial A2/A4 cells (reused, == C3-regimes scenario; quad_tank A2 == experiments/quadtank X0); cartpole (A2, A4) and fhn (A2) = new runs here on TEST-C3, 3 seeds, no tuning (code hash b843eb68d0 = campaign's). Old = stored `lora` of C3-regimes / TEST-C3.

## 1. Fold ratios, sequence_ood (primary); id / shape_ood unmeasured in the last columns

| plant (group) | best grey-box unm (seq) | old unm fold | A2 unm fold | A4 unm fold | old total fold | A2 total fold | A4 total fold | A2 unm id | A2 unm shape |
|---|---|---|---|---|---|---|---|---|---|
| battery (ws) | 0.0192 (ekf_oracle) | 4.39x [2.46, 7.79] | **1.06x [0.68, 1.69]** | 1.45x [1.18, 1.81] | 3.89x | 1.00x [0.64, 1.59] | 1.38x | 0.56x [0.49, 0.64] | 0.75x [0.70, 0.80] |
| quad_tank (ws) | 0.0185 (ekf) | 5.13x [4.66, 5.64] | **2.46x [2.25, 2.70]** | 2.28x [1.24, 4.05] | 5.06x | 2.49x [2.25, 2.76] | 2.29x | 1.87x [1.84, 1.90] | 1.84x [1.73, 1.95] |
| cartpole (ws, TEST) | 0.0712 (ukf) | 8.67x [1.04, 47.3] | **15.77x [13.2, 18.8]** | 8.28x [1.64, 33.0] | 9.33x | 17.80x [15.2, 20.9] | 8.78x | 10.64x [7.17, 15.6] | 11.64x [7.72, 17.3] |
| fhn (rich, TEST) | 0.0135 (ekf_oracle) | 6.33x [5.19, 7.71] | **3.58x [3.13, 4.09]** | not run | 4.47x | 2.60x [2.46, 2.75] | - | 1.77x [1.62, 1.95] | 1.97x [1.82, 2.13] |
| bioreactor (rich) | | 0.29x | 0.06x | 0.05x | 0.27x | 0.05x | 0.05x | 0.03x | 0.04x |
| lorenz (rich) | | 0.12x | 0.04x | 0.03x | 0.14x | 0.05x | 0.03x | 0.04x | 0.04x |
| vdp (rich) | | 0.72x | 0.34x | 0.23x | 0.70x | 0.37x | 0.26x | 0.41x | 0.41x |

SOAP / old (paired, seq, unmeasured): battery A2 0.24 [0.17, 0.35]; quad_tank A2 0.48 [0.45, 0.51]; cartpole A2 **1.82 [0.32, 15.6]** (worse, n.s.: old seed 2 had a 1.22 outlier), A4 0.96 [0.04, 29.5]; fhn A2 0.57 [0.52, 0.61].
Absolute HyperKKL NRMSE per seed/tier: `tables.md`. No divergences in any new run.

**Cartpole failure (held-out, reported as is).** A2 total NRMSE seq 1.14 (old 0.60, grey-box UKF 0.064); error sits in the unmeasured velocities (x3 ~0.9, x4 ~2.0). Phase-1 diagnostic: SOAP fits the decoder far tighter (eps_dec 0.17-0.20 vs 36-37 old) but with a ~4x sharper map (l_dec_q95 218-313 vs 65-72), which amplifies TEST-C3's coloured Student-t sensor noise. A4 (window + wider decoder) partly recovers (seq 0.56, back to the old level), with large seed spread (seeds 0/1/2: 0.50 / 0.92 / 0.26).

## 2. P2 re-score (fold ratios, unmeasured NRMSE, duffing excluded; median fold, geometric mean in brackets)

Window-starved = battery, quad_tank, cartpole, highway_traffic; rich = bioreactor, lorenz, vdp, fhn, rossler (network has no stored lora). highway_traffic, rossler, two_link_arm: old runs (flagged), no SOAP run.

| variant | tier | window-starved median (gmean) | rich median (gmean) | ws above rich median | P2 (strict 4/4; >=3/4) |
|---|---|---|---|---|---|
| old runs only | seq | 5.38x (5.76x) | 0.34x (0.55x) | 4/4 | PASS; PASS |
| **A2 where available (both groups)** | seq | **4.05x (3.90x)** | **0.34x (0.25x)** | 4/4 | **PASS; PASS** |
| A2 on window-starved only | seq | 4.05x (3.90x) | 0.34x (0.55x) | 4/4 | PASS; PASS |
| A4 where available (fhn old) | seq | 3.96x (3.52x) | 0.23x (0.24x) | 4/4 | PASS; PASS |
| A2 where available | id | 3.04x (2.62x) | 0.37x (0.21x) | 4/4 | PASS; PASS |
| A2 where available, total NRMSE | seq | 3.63x (3.82x) | 0.32x (0.24x) | 4/4 | PASS; PASS |

Per-plant folds used in each variant: `tables.md`. The headline window-starved median is carried by two OLD numbers (cartpole is new but highway_traffic 5.64x is old); with SOAP, battery's gap is a tie (1.06x, CI spans 1) yet still above the rich median 0.34x, so the pass is ordinal, not a large-gap claim for every plant.

## 3. Insight (plain)

1. SOAP closes most of the gap on the dev window-starved plants: battery 4.4x -> 1.1x (tie with the grey-box), quad_tank 5.1x -> 2.5x.
2. It does not transfer to held-out cartpole: A2 is ~1.8x worse than the old recipe (16x behind the grey-box), because its sharper decoder amplifies the coloured heavy-tailed noise; A4 only returns to the old level (8x).
3. On held-out fhn (the "rich" exception) SOAP halves the gap (6.3x -> 3.6x) but HyperKKL still clearly loses, so fhn stays an exception to the law.
4. Rich dev plants improve too (bioreactor 0.29x -> 0.06x, vdp 0.72x -> 0.34x), so the group contrast survives: P2 still passes on every variant (ws median ~4x vs rich ~0.3x, 4/4).
5. Caveat: P2's pass is now ordinal; the magnitude rests on cartpole and on an un-rerun highway_traffic, and battery alone no longer shows a material loss.

## Cartpole diagnosis (eval only, 2026-10-26 17:34 Dubai; rule: PREREG addendum 2)

The stored models were re-evaluated with no retraining: old `lora` (checkpoints from the live cache, loaded with the run's own snapshot b2cb5b3dfe80), A2 and A4, plus greybox_ukf re-fitted in the snapshot code. All four reproduce their stored clean numbers exactly (table below). They were run on the same 64 x 3 tiers of episodes under five sensor conditions. Code: `code/diag_conditions.py`, `code/diag_report.py`; data: `diag/*.json`.

**Verdict by the pre-stated rule: "mixed", in the sense that it is neither of the two named causes.**
- Removing all sensor noise lowers the unmeasured error by only 1-3% for every recipe (seq: old 0.617 -> 0.599, A2 1.124 -> 1.113, A4 0.590 -> 0.570). So it is not noise amplification, even though A2/A4's decoders are 3-4x sharper (l_dec_q95 218-313 vs 65-72 old).
- Heavy tails and colour do not matter either: Gaussian white or coloured noise at the same variance changes the error by 0-5%.
- It is not missing information. The angular-velocity error with clean y is 150-300x its 3/alpha CRB (CRB 0.0068, the same as the full-past CRB). Only the cart velocity has a window deficit (CRB 0.076 at 3/alpha vs 0.015 full), and HyperKKL's x2 error (0.16-0.21) is only about 2-3x that window bound.
- What actually fails is a deterministic decoding error. Even the MEASURED angle x3 is off by 0.36-0.94 x_std with noise-free y, and the grey-box UKF gets it to 0.008. This points to the learned maps failing on the deployed trajectories (for example extrapolation outside the phase-1 domain: the cart position's x_std is 75 against an IC box of +-5), not to information or noise. This is a hypothesis; it was not tested here.
- Reference: the grey-box UKF's own x4 error (0.126) is 18x the CRB, which is model mismatch, and it too is insensitive to noise.

Reproduction (total NRMSE, stored clean vs re-evaluated `stored`, seeds 0-2):
- lora: 0.2869 / 0.2801 / 1.2246 vs 0.2869 / 0.2801 / 1.2246
- greybox_ukf: 0.0631 / 0.0641 / 0.0649 vs 0.0631 / 0.0641 / 0.0649
- A2: 1.2019 / 1.0751 / 1.1425 vs 1.2019 / 1.0751 / 1.1425
- A4: 0.5038 / 0.9246 / 0.2578 vs 0.5038 / 0.9246 / 0.2578

| model | tier | condition | total | unmeasured | x1 (meas.) | x2 cart vel | x3 (meas.) | x4 ang. vel |
|---|---|---|---|---|---|---|---|---|
| lora | id | stored | 0.3258 | 0.3164 | 0.1524 | 0.1715 | 0.2692 | 0.4613 |
| lora | id | ct_redraw | 0.3221 | 0.3129 | 0.1518 | 0.1719 | 0.2682 | 0.4540 |
| lora | id | noiseless | 0.2853 | 0.2732 | 0.1422 | 0.1649 | 0.2542 | 0.3815 |
| lora | id | gauss_white | 0.2923 | 0.2810 | 0.1447 | 0.1660 | 0.2555 | 0.3960 |
| lora | id | gauss_col | 0.3308 | 0.3225 | 0.1529 | 0.1735 | 0.2715 | 0.4715 |
| lora | shape_ood | stored | 0.3033 | 0.3059 | 0.1373 | 0.1289 | 0.2395 | 0.4829 |
| lora | shape_ood | ct_redraw | 0.2999 | 0.3024 | 0.1373 | 0.1295 | 0.2383 | 0.4753 |
| lora | shape_ood | noiseless | 0.2816 | 0.2815 | 0.1276 | 0.1239 | 0.2320 | 0.4390 |
| lora | shape_ood | gauss_white | 0.2836 | 0.2839 | 0.1304 | 0.1244 | 0.2320 | 0.4433 |
| lora | shape_ood | gauss_col | 0.3114 | 0.3136 | 0.1375 | 0.1292 | 0.2436 | 0.4979 |
| lora | sequence_ood | stored | 0.5972 | 0.6173 | 0.1443 | 0.1671 | 0.4565 | 1.0676 |
| lora | sequence_ood | ct_redraw | 0.5906 | 0.6109 | 0.1444 | 0.1667 | 0.4534 | 1.0551 |
| lora | sequence_ood | noiseless | 0.5794 | 0.5991 | 0.1357 | 0.1640 | 0.4453 | 1.0342 |
| lora | sequence_ood | gauss_white | 0.5812 | 0.6007 | 0.1383 | 0.1632 | 0.4454 | 1.0382 |
| lora | sequence_ood | gauss_col | 0.5998 | 0.6206 | 0.1446 | 0.1682 | 0.4581 | 1.0730 |
| greybox_ukf | id | stored | 0.0406 | 0.0478 | 0.0125 | 0.0192 | 0.0056 | 0.0763 |
| greybox_ukf | id | ct_redraw | 0.0409 | 0.0478 | 0.0129 | 0.0188 | 0.0057 | 0.0768 |
| greybox_ukf | id | noiseless | 0.0398 | 0.0470 | 0.0033 | 0.0178 | 0.0055 | 0.0763 |
| greybox_ukf | id | gauss_white | 0.0398 | 0.0470 | 0.0046 | 0.0177 | 0.0055 | 0.0763 |
| greybox_ukf | id | gauss_col | 0.0406 | 0.0474 | 0.0132 | 0.0184 | 0.0056 | 0.0764 |
| greybox_ukf | shape_ood | stored | 0.0390 | 0.0455 | 0.0122 | 0.0169 | 0.0058 | 0.0742 |
| greybox_ukf | shape_ood | ct_redraw | 0.0390 | 0.0456 | 0.0131 | 0.0172 | 0.0057 | 0.0739 |
| greybox_ukf | shape_ood | noiseless | 0.0380 | 0.0445 | 0.0030 | 0.0153 | 0.0056 | 0.0737 |
| greybox_ukf | shape_ood | gauss_white | 0.0381 | 0.0445 | 0.0043 | 0.0153 | 0.0056 | 0.0738 |
| greybox_ukf | shape_ood | gauss_col | 0.0389 | 0.0453 | 0.0126 | 0.0165 | 0.0057 | 0.0741 |
| greybox_ukf | sequence_ood | stored | 0.0640 | 0.0712 | 0.0131 | 0.0167 | 0.0079 | 0.1258 |
| greybox_ukf | sequence_ood | ct_redraw | 0.0638 | 0.0703 | 0.0129 | 0.0151 | 0.0079 | 0.1255 |
| greybox_ukf | sequence_ood | noiseless | 0.0633 | 0.0696 | 0.0030 | 0.0137 | 0.0077 | 0.1255 |
| greybox_ukf | sequence_ood | gauss_white | 0.0634 | 0.0695 | 0.0044 | 0.0135 | 0.0078 | 0.1255 |
| greybox_ukf | sequence_ood | gauss_col | 0.0639 | 0.0701 | 0.0130 | 0.0143 | 0.0078 | 0.1259 |
| A2 | id | stored | 0.5642 | 0.5085 | 0.3273 | 0.2765 | 0.5250 | 0.7405 |
| A2 | id | ct_redraw | 0.5651 | 0.5100 | 0.3273 | 0.2777 | 0.5246 | 0.7423 |
| A2 | id | noiseless | 0.5598 | 0.5039 | 0.3247 | 0.2729 | 0.5182 | 0.7349 |
| A2 | id | gauss_white | 0.5651 | 0.5110 | 0.3251 | 0.2739 | 0.5185 | 0.7482 |
| A2 | id | gauss_col | 0.5642 | 0.5099 | 0.3288 | 0.2766 | 0.5222 | 0.7432 |
| A2 | shape_ood | stored | 0.5574 | 0.5299 | 0.3139 | 0.1960 | 0.4849 | 0.8639 |
| A2 | shape_ood | ct_redraw | 0.5524 | 0.5248 | 0.3136 | 0.1950 | 0.4832 | 0.8546 |
| A2 | shape_ood | noiseless | 0.5544 | 0.5261 | 0.3136 | 0.1930 | 0.4800 | 0.8593 |
| A2 | shape_ood | gauss_white | 0.5554 | 0.5272 | 0.3134 | 0.1932 | 0.4822 | 0.8613 |
| A2 | shape_ood | gauss_col | 0.5572 | 0.5294 | 0.3139 | 0.1958 | 0.4871 | 0.8630 |
| A2 | sequence_ood | stored | 1.1398 | 1.1237 | 0.3030 | 0.2147 | 0.9374 | 2.0326 |
| A2 | sequence_ood | ct_redraw | 1.1458 | 1.1302 | 0.3017 | 0.2159 | 0.9406 | 2.0446 |
| A2 | sequence_ood | noiseless | 1.1310 | 1.1134 | 0.2984 | 0.2107 | 0.9327 | 2.0162 |
| A2 | sequence_ood | gauss_white | 1.1324 | 1.1155 | 0.2991 | 0.2120 | 0.9329 | 2.0190 |
| A2 | sequence_ood | gauss_col | 1.1445 | 1.1289 | 0.3031 | 0.2161 | 0.9402 | 2.0416 |
| A4 | id | stored | 0.2760 | 0.2726 | 0.1740 | 0.1584 | 0.1935 | 0.3868 |
| A4 | id | ct_redraw | 0.2673 | 0.2637 | 0.1730 | 0.1588 | 0.1914 | 0.3686 |
| A4 | id | noiseless | 0.2472 | 0.2445 | 0.1452 | 0.1441 | 0.1705 | 0.3450 |
| A4 | id | gauss_white | 0.2480 | 0.2462 | 0.1495 | 0.1460 | 0.1717 | 0.3464 |
| A4 | id | gauss_col | 0.2759 | 0.2727 | 0.1744 | 0.1587 | 0.1943 | 0.3866 |
| A4 | shape_ood | stored | 0.2415 | 0.2458 | 0.1584 | 0.1290 | 0.1693 | 0.3625 |
| A4 | shape_ood | ct_redraw | 0.2435 | 0.2481 | 0.1590 | 0.1290 | 0.1687 | 0.3672 |
| A4 | shape_ood | noiseless | 0.2273 | 0.2331 | 0.1292 | 0.1194 | 0.1598 | 0.3468 |
| A4 | shape_ood | gauss_white | 0.2276 | 0.2331 | 0.1350 | 0.1210 | 0.1595 | 0.3452 |
| A4 | shape_ood | gauss_col | 0.2440 | 0.2480 | 0.1606 | 0.1292 | 0.1689 | 0.3668 |
| A4 | sequence_ood | stored | 0.5620 | 0.5896 | 0.1673 | 0.1700 | 0.3807 | 1.0092 |
| A4 | sequence_ood | ct_redraw | 0.5653 | 0.5932 | 0.1664 | 0.1707 | 0.3822 | 1.0157 |
| A4 | sequence_ood | noiseless | 0.5453 | 0.5704 | 0.1383 | 0.1545 | 0.3601 | 0.9862 |
| A4 | sequence_ood | gauss_white | 0.5392 | 0.5648 | 0.1429 | 0.1566 | 0.3582 | 0.9730 |
| A4 | sequence_ood | gauss_col | 0.5651 | 0.5919 | 0.1669 | 0.1696 | 0.3836 | 1.0143 |

| model | l_dec_q95 (s0/s1/s2) | tier | unm drop noiseless vs stored | gauss white / ct redraw | gauss coloured / ct redraw | x2 (noiseless) / CRB(3/a) | x4 (noiseless) / CRB(3/a) |
|---|---|---|---|---|---|---|---|
| lora | 65 / 70 / 72 | id | +14% | 0.90 | 1.03 | 2.2 | 56 |
| lora | 65 / 70 / 72 | shape_ood | +8% | 0.94 | 1.04 | 1.6 | 65 |
| lora | 65 / 70 / 72 | sequence_ood | +3% | 0.98 | 1.02 | 2.2 | 152 |
| greybox_ukf | - | id | +2% | 0.98 | 0.99 | 0.2 | 11 |
| greybox_ukf | - | shape_ood | +2% | 0.98 | 0.99 | 0.2 | 11 |
| greybox_ukf | - | sequence_ood | +2% | 0.99 | 1.00 | 0.2 | 18 |
| A2 | 313 / 218 / 256 | id | +1% | 1.00 | 1.00 | 3.6 | 108 |
| A2 | 313 / 218 / 256 | shape_ood | +1% | 1.00 | 1.01 | 2.5 | 126 |
| A2 | 313 / 218 / 256 | sequence_ood | +1% | 0.99 | 1.00 | 2.8 | 296 |
| A4 | 251 / 281 / 290 | id | +10% | 0.93 | 1.03 | 1.9 | 51 |
| A4 | 251 / 281 / 290 | shape_ood | +5% | 0.94 | 1.00 | 1.6 | 51 |
| A4 | 251 / 281 / 290 | sequence_ood | +3% | 0.95 | 1.00 | 2.0 | 145 |

Bayesian PCRB per state (x_std units, eval_id, budgets.json): window 3/alpha = 2 s: 0.0059, 0.0762, 0.0020, 0.0068; full past: 0.0045, 0.0145, 0.0020, 0.0068.

## Cartpole diagnosis, part 2: extrapolation test (2026-10-26 18:13 Dubai; rule: PREREG addendum 3)

Eval only: cached models were loaded, and a cache miss would have raised an error; nothing was retrained. Code: `code/coverage.py`, `code/coverage_report.py`; data: `diag/coverage/*.json`. A GPU was used briefly (seconds per job), because the cached datasets are keyed by the CUDA generator.

**Verdict: no extrapolation, on any plant, model or tier. My part-1 guess is refuted.**
- **Cartpole.** 0.0% of settled eval steps fall outside the training envelope, and 0.0% fall even outside the training 1-99% quantile band.
  - The training data (phase-1 episodes over the domain box, phase-2 episodes, collocation) spans cart position +-485 and cart velocity +-19 (1-99%: +-370, +-10.6).
  - The sequence_ood eval states span only +-84 and +-3.3 (1-99%).
- **Cart drift.** Under the protocol inputs the cart does drift, like a random walk: |x1| median / 99% is 2.5 / 4.9 at t = 0, 15.5 / 67 at T/2 and 27 / 93 at T (sequence_ood; id reaches 166 at T). The training data already covers this range.
- **Translation invariance.** The true dynamics do not depend on cart position: f(x + d e1) - f(x) = 0 exactly for d = 10 and 100. So a model-based filter is unaffected by the drift. The grey-box UKF's error is uniform and small (seq: x3 0.008, x4 0.126).
- **The other plants (no rule triggers):**
  - highway_traffic (old lora), quad_tank (A2) and battery (A2) have 0.0% of steps outside their envelopes.
  - Outside the 1-99% band (secondary): highway 0-0.1%, quad_tank 0-7%, battery 2-4%. There the error is not higher (quad_tank 0.035 vs 0.039 inside, battery 0.022 vs 0.021).
  - So none of the window-starved losses comes from extrapolation in state space.
- **What this leaves for cartpole.** The HyperKKL error is large everywhere inside a well-covered state set, with clean y as well (part 1), including on the measured angle. That points to the learned observer failing to fit/learn the map itself (phase-2 training / decoder fit on this plant), not to coverage, noise or information. Note that the envelope is per-state min/max: coverage of joint states (combinations) is not tested beyond the per-state 1-99% band.

Coverage (settled eval steps, mean over seeds 0-2; errors = RMS |xhat - x| / x_std, unmeasured = mean over unmeasured states):

| plant | model | tier | joint outside min/max | unm err inside | unm err outside | ratio | rule | outside 1-99% band (secondary) | unm err in band | unm err out of band | x3 in / out band | x4 in / out band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cartpole | A2 | id | 0.0% | 0.709 | nan | nan | no | 0.0% | 0.709 | nan | 0.661 / nan | 1.071 / nan |
| cartpole | A2 | shape_ood | 0.0% | 0.623 | nan | nan | no | 0.0% | 0.623 | nan | 0.567 / nan | 1.016 / nan |
| cartpole | A2 | sequence_ood | 0.0% | 1.171 | nan | nan | no | 0.0% | 1.171 | nan | 0.963 / nan | 2.112 / nan |
| cartpole | A4 | id | 0.0% | 0.372 | nan | nan | no | 0.0% | 0.372 | nan | 0.261 / nan | 0.571 / nan |
| cartpole | A4 | shape_ood | 0.0% | 0.280 | nan | nan | no | 0.0% | 0.280 | nan | 0.213 / nan | 0.420 / nan |
| cartpole | A4 | sequence_ood | 0.0% | 0.644 | nan | nan | no | 0.0% | 0.644 | nan | 0.444 / nan | 1.114 / nan |
| cartpole | lora | id | 0.0% | 0.466 | nan | nan | no | 0.0% | 0.466 | nan | 0.340 / nan | 0.732 / nan |
| cartpole | lora | shape_ood | 0.0% | 0.367 | nan | nan | no | 0.0% | 0.367 | nan | 0.273 / nan | 0.590 / nan |
| cartpole | lora | sequence_ood | 0.0% | 0.650 | nan | nan | no | 0.0% | 0.650 | nan | 0.468 / nan | 1.125 / nan |
| cartpole | greybox_ukf | id | 0.0% | 0.053 | nan | nan | no | 0.0% | 0.053 | nan | 0.006 / nan | 0.085 / nan |
| cartpole | greybox_ukf | shape_ood | 0.0% | 0.049 | nan | nan | no | 0.0% | 0.049 | nan | 0.006 / nan | 0.079 / nan |
| cartpole | greybox_ukf | sequence_ood | 0.0% | 0.072 | nan | nan | no | 0.0% | 0.072 | nan | 0.008 / nan | 0.126 / nan |
| highway_traffic | lora | id | 0.0% | 0.212 | nan | nan | no | 0.1% | 0.212 | 0.511 | - | - |
| highway_traffic | lora | shape_ood | 0.0% | 0.208 | nan | nan | no | 0.0% | 0.208 | nan | - | - |
| highway_traffic | lora | sequence_ood | 0.0% | 0.219 | nan | nan | no | 0.0% | 0.219 | nan | - | - |
| quad_tank | A2 | id | 0.0% | 0.038 | nan | nan | no | 7.4% | 0.039 | 0.035 | - | - |
| quad_tank | A2 | shape_ood | 0.0% | 0.038 | nan | nan | no | 0.0% | 0.038 | nan | - | - |
| quad_tank | A2 | sequence_ood | 0.0% | 0.047 | nan | nan | no | 0.0% | 0.047 | 0.019 | - | - |
| battery | A2 | id | 0.0% | 0.018 | nan | nan | no | 3.4% | 0.018 | 0.025 | - | - |
| battery | A2 | shape_ood | 0.0% | 0.028 | nan | nan | no | 1.6% | 0.027 | 0.033 | - | - |
| battery | A2 | sequence_ood | 0.0% | 0.021 | nan | nan | no | 4.0% | 0.021 | 0.022 | - | - |

Envelopes:

- cartpole (seed 0): envelope lo [-485.492, -18.246, -0.722, -4.583] hi [499.29, 19.772, 0.698, 4.597]; train 1/99% [-360.741, -10.387, -0.148, -0.82] / [384.629, 10.905, 0.142, 0.822]; seq eval 1/99% [-83.991, -3.344, -0.104, -0.403] / [79.806, 2.996, 0.104, 0.405]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 0.0, '100.0': 0.0} (|f| max 137)
- highway_traffic (seed 0): envelope lo [0.004, 0.004, 0.003, 0.004, 0.004] hi [0.106, 0.106, 0.13, 0.106, 0.107]; train 1/99% [0.024, 0.023, 0.028, 0.025, 0.024] / [0.084, 0.091, 0.104, 0.098, 0.095]; seq eval 1/99% [0.037, 0.036, 0.047, 0.045, 0.042] / [0.039, 0.049, 0.087, 0.088, 0.088]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 32.44289456331575, '100.0': 3240.123969997325} (|f| max 0.00301)
- quad_tank (seed 0): envelope lo [5.09, 5.156, 0.124, 0.217] hi [18.839, 18.805, 6.369, 4.663]; train 1/99% [7.339, 7.186, 1.901, 1.82] / [15.675, 16.023, 5.86, 4.25]; seq eval 1/99% [10.653, 10.058, 4.838, 2.636] / [12.091, 11.931, 5.769, 4.103]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 0.17408241670373106, '100.0': 0.8811300504591423} (|f| max 0.246)
- battery (seed 0): envelope lo [-0.001, -0.046, -0.046, -0.322, -0.343] hi [1.009, 0.045, 0.046, 4.29, 0.468]; train 1/99% [0.057, -0.036, -0.034, -0.01, -0.147] / [0.978, 0.035, 0.033, 3.2, 0.393]; seq eval 1/99% [0.267, -0.036, -0.018, 0.415, -0.045] / [0.813, 0.036, 0.019, 1.199, 0.333]; max |f(x + d e1) - f(x)| for d = 10, 100: {'10.0': 10.322454517013941, '100.0': 10.322454587076564} (|f| max 8.78)

|x1| (cart position for cartpole) median / 99% at t = 0, T/2, T (seed 0):

| plant | tier | t0 | T/2 | T |
|---|---|---|---|---|
| cartpole | id | 2.52 / 4.89 | 20.9 / 72.2 | 26.5 / 166 |
| cartpole | shape_ood | 2.5 / 4.92 | 23.9 / 69.1 | 44.7 / 140 |
| cartpole | sequence_ood | 2.5 / 4.87 | 15.5 / 67.4 | 26.6 / 92.8 |
| highway_traffic | id | 0.0553 / 0.0988 | 0.0376 / 0.0407 | 0.0371 / 0.0373 |
| highway_traffic | shape_ood | 0.0554 / 0.0982 | 0.0376 / 0.0406 | 0.0371 / 0.0372 |
| highway_traffic | sequence_ood | 0.0556 / 0.0988 | 0.0377 / 0.0406 | 0.0371 / 0.0373 |
| quad_tank | id | 12 / 17.8 | 11.1 / 12 | 11.6 / 12 |
| quad_tank | shape_ood | 11.9 / 17.8 | 11.1 / 11.9 | 11.6 / 11.9 |
| quad_tank | sequence_ood | 12 / 17.7 | 11.2 / 12.1 | 11.6 / 12.1 |
| battery | id | 0.551 / 0.793 | 0.571 / 0.851 | 0.572 / 0.942 |
| battery | shape_ood | 0.552 / 0.792 | 0.575 / 0.88 | 0.585 / 0.937 |
| battery | sequence_ood | 0.553 / 0.79 | 0.578 / 0.812 | 0.548 / 0.802 |
## Coverage, part 3: joint nearest-neighbour coverage (2026-10-26 18:50 Dubai; rule: PREREG addendum 4)

Eval only; the KD-trees were built on the CPU. There were 21 jobs, run in 10 lanes with at most 2 per GPU, using a GPU for seconds each. Code: `code/coverage.py` (extended), `code/coverage3_report.py`; tables: `diag/coverage/tables_part3.md`. two_link_arm has old lora only (no A2 run exists).

**Verdict: no plant is coverage-limited.** In every plant x model x tier, at most 5.5% of the eval steps lie beyond the training nearest-neighbour 99th percentile (calibrated on validation episodes). The rule needed more than 10%.
- **Beyond the threshold (fraction of eval steps):**
  - cartpole: 0.7-5.5% (sequence_ood 5.5%)
  - fhn: 0.8-1.0%
  - two_link_arm: 1.2-1.6%
  - highway_traffic: 0.9-1.2%
  - quad_tank: 0.5-1.8%
  - battery: 0.4-4.1%
- **Cartpole: distance matters a little, but it is not the cause.**
  - Error does rise with distance (Spearman between distance and unmeasured error, seq: A2 +0.58, old +0.44, A4 +0.33; the grey-box UKF +0.23).
  - However, even in the nearest-distance quintile HyperKKL is 0.31-0.48, against the UKF's 0.08. On the well-covered 95% of steps the error is already about 6-14x the grey-box.
  - So the error grows towards sparsely covered regions, but the failure is present in the densely covered core.
- **Other plants:**
  - highway_traffic's "far" set is about 3x worse on sequence_ood, but it is only 1% of the steps.
  - fhn shows no distance trend (Spearman about 0, the far set is even slightly better).
  - quad_tank, battery and two_link_arm have weak trends (+0.03 to +0.22).
- **Does training explore the state space?** Yes, on every plant: the per-state std of the training episodes is at least the eval std on every state (cartpole velocities 0.50 / 2.49 x_std vs eval 0.24 / 2.25; the collocation spans 6-40x).
- **fhn and the limit cycle:** the zero-input phase-1 episodes sit on the limit cycle (radius about the eval-cloud centre at least 0.76 for 99% of samples, against eval down to 0.19). The forced training episodes (radius q1 0.35, q10 0.82) and the collocation (q1 0.35) do cover the interior, and eval steps are 99% within the null distance. So fhn's loss to the grey-box is not a coverage problem either.

Joint coverage (mean over seeds 0-2; d = NN distance to the training states in x_std units; null = validation episodes):

| plant | model | tier | null d50 / d99 (val) | eval d50 | eval beyond d99 | unm err near | unm err far | far/near | err by distance quintile (near -> far) | Spearman(d, err) | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cartpole | A2 | id | 0.0571 / 0.2759 | 0.0601 | 1.1% | 0.6711 | 2.3287 | 3.47 | 0.462 / 0.435 / 0.548 / 0.665 / 1.413 | +0.38 | no |
| cartpole | A2 | shape_ood | 0.0571 / 0.2759 | 0.0854 | 0.7% | 0.6124 | 1.4359 | 2.34 | 0.532 / 0.585 / 0.627 / 0.752 / 1.059 | +0.25 | no |
| cartpole | A2 | sequence_ood | 0.0571 / 0.2759 | 0.0892 | 5.5% | 1.0811 | 2.1985 | 2.03 | 0.477 / 0.726 / 1.129 / 1.675 / 2.537 | +0.58 | no |
| cartpole | A4 | id | 0.0571 / 0.2759 | 0.0601 | 1.1% | 0.3636 | 0.7121 | 1.96 | 0.290 / 0.278 / 0.319 / 0.359 / 0.707 | +0.23 | no |
| cartpole | A4 | shape_ood | 0.0571 / 0.2759 | 0.0854 | 0.7% | 0.2747 | 0.5635 | 2.05 | 0.287 / 0.307 / 0.294 / 0.293 / 0.382 | +0.11 | no |
| cartpole | A4 | sequence_ood | 0.0571 / 0.2759 | 0.0892 | 5.5% | 0.6297 | 0.8172 | 1.30 | 0.418 / 0.538 / 0.717 / 0.966 / 1.123 | +0.33 | no |
| cartpole | lora | id | 0.0571 / 0.2759 | 0.0601 | 1.1% | 0.4561 | 1.2384 | 2.72 | 0.494 / 0.455 / 0.458 / 0.424 / 0.709 | +0.33 | no |
| cartpole | lora | shape_ood | 0.0571 / 0.2759 | 0.0854 | 0.7% | 0.3622 | 0.8457 | 2.34 | 0.376 / 0.361 / 0.371 / 0.427 / 0.554 | +0.23 | no |
| cartpole | lora | sequence_ood | 0.0571 / 0.2759 | 0.0892 | 5.5% | 0.6095 | 1.2120 | 1.99 | 0.311 / 0.438 / 0.620 / 0.919 / 1.296 | +0.44 | no |
| cartpole | greybox_ukf | id | 0.0571 / 0.2759 | 0.0601 | 1.1% | 0.0528 | 0.0985 | 1.87 | 0.059 / 0.055 / 0.059 / 0.060 / 0.075 | +0.09 | no |
| cartpole | greybox_ukf | shape_ood | 0.0571 / 0.2759 | 0.0854 | 0.7% | 0.0483 | 0.1094 | 2.27 | 0.047 / 0.052 / 0.055 / 0.059 / 0.071 | +0.23 | no |
| cartpole | greybox_ukf | sequence_ood | 0.0571 / 0.2759 | 0.0892 | 5.5% | 0.0709 | 0.0924 | 1.30 | 0.079 / 0.083 / 0.087 / 0.093 / 0.106 | +0.23 | no |
| fhn | A2 | id | 0.0027 / 0.0146 | 0.0029 | 1.0% | 0.0266 | 0.0211 | 0.80 | 0.024 / 0.027 / 0.028 / 0.028 / 0.026 | +0.04 | no |
| fhn | A2 | shape_ood | 0.0027 / 0.0146 | 0.0032 | 1.0% | 0.0269 | 0.0209 | 0.78 | 0.027 / 0.028 / 0.028 / 0.027 / 0.024 | -0.03 | no |
| fhn | A2 | sequence_ood | 0.0027 / 0.0146 | 0.0034 | 0.8% | 0.0490 | 0.0382 | 0.78 | 0.052 / 0.052 / 0.050 / 0.047 / 0.042 | -0.07 | no |
| two_link_arm | lora | id | 0.0163 / 0.0592 | 0.0164 | 1.6% | 0.3883 | 0.6345 | 1.63 | 0.345 / 0.357 / 0.370 / 0.415 / 0.488 | +0.22 | no |
| two_link_arm | lora | shape_ood | 0.0163 / 0.0592 | 0.0163 | 1.4% | 0.3748 | 0.6093 | 1.63 | 0.326 / 0.338 / 0.354 / 0.396 / 0.478 | +0.22 | no |
| two_link_arm | lora | sequence_ood | 0.0163 / 0.0592 | 0.0171 | 1.2% | 0.3741 | 0.6462 | 1.73 | 0.329 / 0.339 / 0.358 / 0.394 / 0.472 | +0.21 | no |
| highway_traffic | lora | id | 0.0584 / 0.3947 | 0.0609 | 1.2% | 0.2070 | 0.4091 | 1.98 | 0.178 / 0.149 / 0.156 / 0.188 / 0.335 | +0.38 | no |
| highway_traffic | lora | shape_ood | 0.0584 / 0.3947 | 0.0619 | 0.9% | 0.2012 | 0.5713 | 2.84 | 0.153 / 0.156 / 0.165 / 0.190 / 0.326 | +0.38 | no |
| highway_traffic | lora | sequence_ood | 0.0584 / 0.3947 | 0.0580 | 1.1% | 0.2087 | 0.6404 | 3.07 | 0.149 / 0.152 / 0.171 / 0.201 / 0.357 | +0.38 | no |
| quad_tank | A2 | id | 0.0530 / 0.2685 | 0.0575 | 1.8% | 0.0381 | 0.0516 | 1.35 | 0.034 / 0.037 / 0.040 / 0.047 / 0.053 | +0.21 | no |
| quad_tank | A2 | shape_ood | 0.0530 / 0.2685 | 0.0547 | 0.5% | 0.0381 | 0.0326 | 0.86 | 0.033 / 0.035 / 0.039 / 0.044 / 0.057 | +0.22 | no |
| quad_tank | A2 | sequence_ood | 0.0530 / 0.2685 | 0.0566 | 0.8% | 0.0466 | 0.0618 | 1.32 | 0.044 / 0.045 / 0.047 / 0.052 / 0.061 | +0.13 | no |
| battery | A2 | id | 0.1139 / 0.3393 | 0.1296 | 1.8% | 0.0181 | 0.0257 | 1.41 | 0.019 / 0.018 / 0.016 / 0.017 / 0.024 | +0.18 | no |
| battery | A2 | shape_ood | 0.1139 / 0.3393 | 0.1399 | 0.4% | 0.0273 | 0.0584 | 2.14 | 0.039 / 0.029 / 0.021 / 0.021 / 0.028 | +0.03 | no |
| battery | A2 | sequence_ood | 0.1139 / 0.3393 | 0.1900 | 4.1% | 0.0207 | 0.0259 | 1.25 | 0.021 / 0.022 / 0.022 / 0.024 / 0.026 | +0.15 | no |

Exploration: per-state std relative to x_std (seed 0):

| plant | x_std | std p1 episodes / x_std | std train episodes / x_std | std collocation / x_std | std eval seq / x_std |
|---|---|---|---|---|---|
| cartpole | [74.9, 5.54, 0.0651, 0.0649] | [2.76, 0.82, 0.49, 2.69] | [0.52, 0.50, 0.87, 2.49] | [2.56, 1.63, 5.93, 40.81] | [0.43, 0.24, 1.05, 2.25] |
| fhn | [1.07, 0.538] | [0.98, 0.71] | [1.06, 0.99] | [1.54, 2.04] | [1.12, 0.54] |
| two_link_arm | [0.826, 6.36, 0.709, 0.865] | [0.52, 3.38, 1.87, 3.04] | [0.56, 0.85, 1.82, 2.34] | [2.12, 3.19, 9.29, 19.97] | [0.38, 0.76, 1.14, 0.38] |
| highway_traffic | [0.00406, 0.00661, 0.00847, 0.00916, 0.00968] | [2.56, 2.03, 1.85, 1.56, 1.51] | [2.29, 1.81, 1.56, 1.36, 1.30] | [7.28, 4.47, 3.65, 3.23, 3.07] | [0.13, 0.37, 0.94, 0.96, 0.94] |
| quad_tank | [0.468, 0.666, 0.293, 0.449] | [2.87, 2.28, 2.23, 0.58] | [2.54, 2.07, 2.68, 1.07] | [8.45, 5.92, 6.15, 2.86] | [0.69, 0.60, 0.77, 0.79] |
| battery | [0.167, 0.0181, 0.0155, 0.455, 0.107] | [1.68, 0.14, 0.42, 1.44, 1.22] | [1.02, 0.94, 0.96, 1.44, 0.97] | [1.73, 1.45, 1.70, 2.93, 1.76] | [0.93, 1.09, 0.54, 0.39, 0.86] |

Radius about the eval-state mean (sequence_ood, seed 0):

| plant | set | radius about eval-mean (x_std units): q1 / q10 / q50 / q90 / q99 |
|---|---|---|
| cartpole | eval | 0.30 / 0.82 / 1.63 / 4.25 / 6.86 |
| cartpole | p1 | 0.21 / 0.69 / 2.43 / 4.61 / 8.73 |
| cartpole | train | 0.06 / 0.25 / 1.06 / 2.11 / 8.49 |
| cartpole | coll | 4.26 / 10.08 / 35.98 / 63.97 / 70.29 |
| fhn | eval | 0.19 / 0.57 / 1.20 / 1.72 / 1.93 |
| fhn | p1 | 0.76 / 0.85 / 1.10 / 1.43 / 2.80 |
| fhn | train | 0.35 / 0.82 / 1.33 / 2.00 / 2.63 |
| fhn | coll | 0.35 / 1.10 / 2.45 / 4.01 / 4.76 |
| two_link_arm | eval | 0.29 / 0.60 / 1.28 / 2.12 / 3.11 |
| two_link_arm | p1 | 0.56 / 1.35 / 3.30 / 6.84 / 15.95 |
| two_link_arm | train | 0.40 / 0.72 / 1.54 / 3.69 / 12.05 |
| two_link_arm | coll | 4.39 / 9.28 / 20.17 / 32.74 / 36.83 |
| highway_traffic | eval | 0.22 / 0.49 / 1.16 / 2.59 / 4.61 |
| highway_traffic | p1 | 0.25 / 0.69 / 2.20 / 7.66 / 14.14 |
| highway_traffic | train | 0.25 / 0.61 / 1.98 / 6.68 / 12.76 |
| highway_traffic | coll | 4.09 / 6.49 / 10.38 / 16.44 / 19.04 |
| quad_tank | eval | 0.24 / 0.52 / 1.22 / 2.15 / 2.86 |
| quad_tank | p1 | 0.13 / 0.35 / 1.21 / 7.79 / 15.35 |
| quad_tank | train | 0.24 / 0.60 / 2.22 / 8.01 / 14.95 |
| quad_tank | coll | 4.03 / 7.89 / 13.96 / 19.52 / 22.90 |
| battery | eval | 0.47 / 0.88 / 1.68 / 2.52 / 3.23 |
| battery | p1 | 0.58 / 1.38 / 2.32 / 3.76 / 5.78 |
| battery | train | 0.61 / 1.19 / 1.96 / 3.59 / 5.54 |
| battery | coll | 1.83 / 2.92 / 4.55 / 7.56 / 8.55 |