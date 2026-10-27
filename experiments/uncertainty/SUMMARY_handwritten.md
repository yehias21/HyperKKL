## Summary and interpretation (hand-written, 2026-10-26 19:24 CET; everything below this section is generated)

**Scope.** 66 runs: 5 dev plants x {C1-where, C3-partial} and 6 held-out test plants x {TEST-C1, TEST-C3}, 3 seeds each. Each run has 3 tiers x 4 conditions, 64 test episodes per block, and 128 fresh calibration episodes. One exception: TEST-C3 rossler has no `plant_drift10` block, because every drifted-plant episode (30% + 10% parameter shift) diverged and was rejected by the data generator, so that condition was dropped automatically.

**1. Does HyperKKL get calibrated bounds? Yes, in the regime where the guarantee applies.**
- On id/clean, every state of every plant has coverage between 0.88 and 0.93 (target 0.90). This holds for U1, U2 and U2-mf, on dev and test plants, with an exact model, a partial model, and the realistic TEST-C3 setting. P1 passes.
- U1 is a constant per-state band. It fixes the scale but not the shape of the uncertainty: it covers only 0.81 in the 20% worst episodes.
- U2 adapts to the online signals. In high-error episodes it covers 0.88-0.90, and it has an 8-20% lower interval score at the same marginal coverage.
- Its Gramian feature uses the modeller's model. With the partial model (C3-partial) that feature misleads: on battery x0 the U2 width is 1.1 vs 0.17 for U1. P2 therefore fails, on C3-partial id: 3/5. The model-free U2-mf does not have this problem and is the variant to use when the model is incomplete.
- U4, the naive 3-seed ensemble band (1.645 x spread), is badly over-confident (coverage 0.23-0.70). Seed disagreement is not error. Conformalising the spread (U4c) restores coverage, but with wider bands than U2.

**2. Are the bounds observability-aware? Yes (P3 passes).**
- The bands of unmeasured states are 2.4-4.2x wider than those of measured states, matching their 2-4x larger error.
- quad_tank: the upper, unmeasured tanks get bands of 0.33 / 0.12 (x2 / x3), against 0.04 / 0.03 for the lower tanks, in both scenarios.
- Per-state U2 width tracks per-state RMSE across all dev (system, state) cells, Spearman 0.91.
- Within a trajectory, the width-error rank correlation is a modest 0.2-0.5. The per-state scale does most of the work; the time variation adds the conditional-coverage gain above.

**3. The bracket with the information study: floor <= achieved <= band.**
- No band is narrower than 1.645 x the information-floor std (P6: 0 of 176 cells; ratio min 1.8, median 7).
- The HyperKKL bands are honest but wide: 2-180x the floor and typically 3-10x the exact-model EKF's band, because the HyperKKL error is that much larger. The bands do not hide the accuracy gap; they quantify it.
- The floor used for C3-partial and TEST-C3 is the exact-plant, white-noise floor of the information study, so it is approximate there.

Unmeasured states, id/clean, seed mean (sigma-normalised). Floor = full-past Bayesian PCRB from experiments/information/budgets.json, RMS over unmeasured states:

| node | system | floor std | 1.645 x floor | HyperKKL RMSE | U2 90% half-width (coverage) | EKF RMSE | EKF 90% half-width (coverage) |
|---|---|---|---|---|---|---|---|
| C1-where | battery | 0.0041 | 0.0068 | 0.0647 | 0.0946 (0.914) | 0.0078 | 0.0085 (0.864) |
| C1-where | bioreactor | 0.0085 | 0.0140 | 0.0287 | 0.0333 (0.921) | 0.0085 | 0.0143 (0.908) |
| C1-where | lorenz | 0.0014 | 0.0023 | 0.0421 | 0.0304 (0.906) | 0.0066 | 0.0092 (0.896) |
| C1-where | quad_tank | 0.0204 | 0.0335 | 0.0750 | 0.1142 (0.908) | 0.0201 | 0.0326 (0.901) |
| C1-where | vdp | 0.0038 | 0.0062 | 0.0218 | 0.0347 (0.905) | 0.0063 | 0.0101 (0.906) |
| C3-partial | battery | 0.0041 | 0.0068 | 0.0661 | 0.2796 (0.919) | 0.6174 | 0.0100 (0.294) |
| C3-partial | bioreactor | 0.0085 | 0.0140 | 0.0400 | 0.0589 (0.925) | 1.7173 | 0.0135 (0.000) |
| C3-partial | lorenz | 0.0014 | 0.0023 | 0.0484 | 0.0319 (0.906) | 4.4086 | 0.0162 (0.041) |
| C3-partial | quad_tank | 0.0204 | 0.0335 | 0.0745 | 0.1155 (0.909) | 0.0469 | 0.0327 (0.609) |
| C3-partial | vdp | 0.0038 | 0.0062 | 0.0208 | 0.0320 (0.908) | 0.4246 | 0.0093 (0.032) |
| TEST-C1 | cartpole | 0.0113 | 0.0187 | 0.2780 | 0.2239 (0.908) | 0.0107 | 0.0176 (0.911) |
| TEST-C1 | fhn | 0.0077 | 0.0126 | 0.0388 | 0.0606 (0.911) | 0.0082 | 0.0137 (0.906) |
| TEST-C1 | highway_traffic | 0.0186 | 0.0306 | 0.1134 | 0.1766 (0.914) | 0.0187 | 0.0307 (0.900) |
| TEST-C1 | network | 0.0081 | 0.0134 | 0.0545 | 0.0726 (0.905) | 0.0092 | 0.0145 (0.904) |
| TEST-C1 | rossler | 0.0015 | 0.0024 | 0.1285 | 0.0691 (0.903) | 0.0552 | 0.0260 (0.898) |
| TEST-C1 | two_link_arm | 0.0496 | 0.0815 | 0.1807 | 0.2928 (0.914) | 0.0919 | 0.1383 (0.899) |
| TEST-C3 | cartpole | 0.0113 | 0.0187 | 0.5366 | 0.4589 (0.892) | 0.2375 | 0.0176 (0.149) |
| TEST-C3 | fhn | 0.0077 | 0.0126 | 0.0555 | 0.0880 (0.910) | 0.3850 | 0.0154 (0.022) |
| TEST-C3 | highway_traffic | 0.0186 | 0.0306 | 0.2125 | 0.4105 (0.921) | 0.2897 | 0.0223 (0.067) |
| TEST-C3 | network | 0.0081 | 0.0134 | 0.0638 | 0.0854 (0.908) | 0.2571 | 0.0160 (0.040) |
| TEST-C3 | rossler | 0.0015 | 0.0024 | 0.3085 | 0.1229 (0.908) | 1.6989 | 0.0064 (0.300) |
| TEST-C3 | two_link_arm | 0.0496 | 0.0815 | 0.3984 | 0.6120 (0.908) | 0.5309 | 0.1308 (0.272) |

**4. Shift: honest failure, and what online recalibration can and cannot do.**
- **Input shape shift (shape_ood): coverage holds** (0.906 dev, 0.908 test).
- **Event-sequence shift (sequence_ood): mild under-coverage.** U2 gives 0.86 dev / 0.88 test; U1 0.79 / 0.83.
- **Sensor faults and plant drift break exchangeability, and coverage fails.** U2 coverage averaged over all cells: outliers 0.45, noise10 0.79, plant_drift10 0.64. U2 fails (< 0.85) in 162 of 239 shift cells.
- **Deployable ACI (U3, output-residual feedback) does not restore coverage within the pre-registered width budget** (7/162; P4 fails), for two reasons:
  - (a) Where the shift is visible in the outputs (outliers, noise10), U3 does recover coverage (0.90 / 0.96). It does so by going to infinite bands 30-60% of the time, because the errors leave the calibration range. Its median width under outliers is 17 vs 0.39 for U2.
  - (b) Where the shift is in weakly observable directions (plant drift on lorenz, vdp, battery), the residual barely moves. U3 then neither widens nor recovers: width ratio ~1.0, coverage unchanged at 0.48-0.75.

  Point (b) is the fundamental limit predicted in THEORY.md section 2. Any feedback available online comes through the outputs, so a shift the outputs cannot see cannot be corrected online.
- **With true-error feedback, ACI (U3-oracle) holds about 0.90 in every condition** (0.89-0.91 averages), as its deterministic guarantee says.
  - P4-oracle nevertheless fails: 88% of cells are within 0.02 of the target, where 95% was required. 27 of the 30 misses are over-coverage (median 0.94): when the errors leave the calibration range, alpha_t goes below 0 and the infinite bands count as covered. Only 3 cells (TEST-C1 fhn outliers) sit at 0.879. The finite-episode slack (gamma T = 5) makes 0.02 a tight criterion.
- **Exploratory, not pre-registered (A3):** log-scale quantile tracking (U3qt) keeps the bands finite, with median widths 4.7 under outliers and 0.53 under noise10. It restores 36/162 failing cells within 3x the U2 width. That is better than U3, but it is not a fix.

**5. Comparison with EKF/UKF covariances (P5, Section 5).**
- **Exact model, clean (C1-where, TEST-C1):** the EKF covariance is well calibrated (5/5 dev plants count as calibrated: coverage 0.88-0.91, ANEES 0.96-1.35; battery is the loosest, with one state at 0.83) and 3-10x narrower than the HyperKKL band. Here the EKF wins on both accuracy and calibrated sharpness.
- **Partial model (C3-partial):** the EKF/UKF covariances are catastrophically over-confident: median coverage 0.04, ANEES about 1e3-1e4. Even the condition-oracle EKF, whose Q includes the model-error rate, reaches only 0.58 coverage with ANEES 90. HyperKKL-U2 keeps 0.906.
- **TEST-C3 (partial + 30% parameter error + coloured t-noise):** EKF coverage 0.02-0.30 against U2 0.89-0.92.
- **Faults:** the nominal filters are over-confident under outliers (coverage 0.22, ANEES ~1e4) and noise10 (0.64). The condition-oracle EKF, which is told the fault statistics, is calibrated under sensor faults. That is a level of knowledge neither the learned observer nor a deployed filter has.
- **Counts:** over all 261 (node, tier, condition, system) cells, HyperKKL-U2 is closer to 0.90 than the nominal EKF in 206 and than the UKF in 204, but only in 131 against the condition-oracle EKF.
- Conformalising the EKF (EKF-c) also restores id coverage under the partial model, but with bands of about 0.6 sigma (median), versus about 0.06 for HyperKKL. The EKF's error itself is large there.

**Bottom line.** HyperKKL comes with calibrated, observability-aware 90% bands wherever deployment matches calibration: id inputs, held-out input shapes, held-out plants, and an exact or partial model. This is something a model-based filter does not deliver once its model is incomplete. The bands are wide, honest about the observer's larger error, and never below the information floor. Under sensor faults and plant drift no calibrated band survives, neither HyperKKL's nor the EKF's. Online recalibration from output residuals can widen bands for shifts the outputs reveal, but not for shifts in weakly observable directions.

**Caveats.**
- The lorenz (and one rossler) re-runs reproduce the stored NRMSE only to 1-7% (Section 7): chaotic plant plus GPU nondeterminism, as already seen in experiments/diagnostics. Every other run reproduces to < 1%.
- Mean widths under outliers are dominated by the exponential scale model extrapolating far outside its training range. Use the median widths.
- 3 seeds. Coverage is pooled over 64 episodes per block, so its seed-to-seed sd is about 0.005-0.01 on id.
