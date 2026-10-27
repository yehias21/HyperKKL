# Uncertainty bands for HyperKKL (`lora`): results

Generated 2026-10-26 19:22 CET by `analysis/uncertainty/score.py` from the stage-2 outputs in `experiments/uncertainty/raw/` (rules: `PREREG.md`, theory: `THEORY.md`). The tables are generated and not hand-edited; only the section "Summary and interpretation" was written by hand (also in SUMMARY_handwritten.md; re-running score.py drops it). Seeds 0-2 per cell; `U4`/`U4c` are one ensemble of the 3 seeds. Widths are full widths `2w/sigma_j`; coverage target 0.90.

## Verdicts (pre-registered criteria)

| criterion | verdict | key numbers |
|---|---|---|
| P1 validity on id/clean (dev) | **PASS** | U1: mean 0.911, min 0.888 (n=32); U2: mean 0.911, min 0.901 (n=32); U2mf: mean 0.908, min 0.900 (n=32) |
| P2 adaptivity (U2 < U1 interval score; U2 > U1 high-error coverage, >= 4/5) | **FAIL** | C1-where id IS: 5/5; C1-where shape_ood IS: 5/5; C1-where sequence_ood IS: 5/5; C1-where id cov_hi: 5/5; C3-partial id IS: 3/5; C3-partial shape_ood IS: 4/5; C3-partial sequence_ood IS: 4/5; C3-partial id cov_hi: 5/5 |
| P3 observability awareness | **PASS** | (a) unmeasured > measured U2 width on 4/5; (b) quad_tank upper > lower: C1-where yes, C3-partial yes; (c) Spearman(width, RMSE) = 0.91 |
| P4 shift: ACI (deployable) restores failing cells | **FAIL** | U2 fails (< 0.85) in 162/239 shift cells; U3 restores 7 |
| P4 U3-oracle long-run coverage within 0.02 | **FAIL** | 88% of cells (need 95%) |
| P6 no band narrower than the information floor with coverage >= 0.9 | **PASS** | 0 violations in 176 (node, system, state, method) cells; floor: experiments/information/budgets.json (dists.eval_id.bayes.full.crb_state, read 19:22) |

P5 (filters) is a count, not pass/fail, and is in Section 5.

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


## 1. Validity on id/clean (conformal guarantee regime)

Coverage averaged over states (min over states in parentheses), seed mean.

**C1-where**

| system | U1 | U2 | U2mf | U3 | U3oracle | U4 | U4c | ekf | ukf | ekf_oracle | ekf-c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.915 (0.90) | 0.913 (0.91) | 0.909 (0.90) | 0.885 (0.88) | 0.911 (0.91) | 0.697 (0.62) | 0.915 (0.90) | 0.878 (0.83) | 0.949 (0.90) | 0.878 (0.83) | 0.902 (0.89) |
| bioreactor | 0.916 (0.92) | 0.919 (0.92) | 0.910 (0.91) | 0.911 (0.91) | 0.916 (0.92) | 0.603 (0.53) | 0.926 (0.93) | 0.907 (0.91) | 0.922 (0.92) | 0.907 (0.91) | 0.915 (0.91) |
| lorenz | 0.904 (0.90) | 0.905 (0.90) | 0.905 (0.90) | 0.897 (0.90) | 0.903 (0.90) | 0.604 (0.59) | 0.906 (0.90) | 0.895 (0.90) | 0.903 (0.90) | 0.895 (0.90) | 0.906 (0.90) |
| quad_tank | 0.908 (0.91) | 0.909 (0.91) | 0.908 (0.90) | 0.903 (0.90) | 0.909 (0.91) | 0.488 (0.24) | 0.911 (0.90) | 0.901 (0.90) | 0.902 (0.90) | 0.901 (0.90) | 0.903 (0.89) |
| vdp | 0.905 (0.90) | 0.905 (0.90) | 0.902 (0.90) | 0.896 (0.90) | 0.903 (0.90) | 0.230 (0.22) | 0.903 (0.90) | 0.906 (0.91) | 0.906 (0.91) | 0.906 (0.91) | 0.914 (0.91) |

**C3-partial**

| system | U1 | U2 | U2mf | U3 | U3oracle | U4 | U4c | ekf | ukf | ekf_oracle | ekf-c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.921 (0.91) | 0.919 (0.91) | 0.913 (0.91) | 0.891 (0.89) | 0.913 (0.91) | 0.687 (0.64) | 0.912 (0.89) | 0.306 (0.03) | 0.320 (0.07) | 0.578 (0.14) | 0.913 (0.90) |
| bioreactor | 0.922 (0.92) | 0.920 (0.91) | 0.916 (0.91) | 0.909 (0.90) | 0.916 (0.92) | 0.656 (0.58) | 0.918 (0.92) | 0.000 (0.00) | 0.000 (0.00) | 0.000 (0.00) | 0.890 (0.89) |
| lorenz | 0.903 (0.90) | 0.906 (0.91) | 0.906 (0.90) | 0.900 (0.90) | 0.903 (0.90) | 0.635 (0.62) | 0.905 (0.90) | 0.041 (0.00) | 0.041 (0.00) | 0.358 (0.00) | 0.908 (0.91) |
| quad_tank | 0.906 (0.89) | 0.906 (0.90) | 0.905 (0.90) | 0.900 (0.89) | 0.908 (0.91) | 0.494 (0.27) | 0.915 (0.91) | 0.755 (0.58) | 0.756 (0.58) | 0.844 (0.68) | 0.911 (0.90) |
| vdp | 0.906 (0.90) | 0.906 (0.90) | 0.905 (0.90) | 0.898 (0.90) | 0.903 (0.90) | 0.270 (0.22) | 0.915 (0.91) | 0.033 (0.03) | 0.033 (0.03) | 0.993 (0.99) | 0.903 (0.90) |

**TEST-C1**

| system | U1 | U2 | U2mf | U3 | U3oracle | U4 | U4c | ekf | ukf | ekf_oracle | ekf-c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cartpole | 0.922 (0.91) | 0.910 (0.91) | 0.911 (0.91) | 0.879 (0.87) | 0.916 (0.91) | 0.650 (0.37) | 0.912 (0.89) | 0.906 (0.90) | 0.932 (0.90) | 0.906 (0.90) | 0.905 (0.90) |
| fhn | 0.914 (0.91) | 0.910 (0.91) | 0.910 (0.91) | 0.904 (0.90) | 0.907 (0.91) | 0.516 (0.46) | 0.908 (0.90) | 0.906 (0.91) | 0.906 (0.91) | 0.906 (0.91) | 0.916 (0.91) |
| highway_traffic | 0.917 (0.91) | 0.911 (0.91) | 0.912 (0.91) | 0.900 (0.90) | 0.905 (0.90) | 0.609 (0.30) | 0.908 (0.89) | 0.901 (0.89) | 0.983 (0.97) | 0.901 (0.89) | 0.908 (0.89) |
| network | 0.904 (0.89) | 0.906 (0.90) | 0.904 (0.90) | 0.886 (0.88) | 0.907 (0.91) | 0.591 (0.42) | 0.904 (0.89) | 0.903 (0.90) | - | 0.903 (0.90) | 0.909 (0.89) |
| rossler | 0.909 (0.91) | 0.904 (0.90) | 0.904 (0.90) | 0.896 (0.89) | 0.902 (0.90) | 0.637 (0.51) | 0.908 (0.91) | 0.899 (0.90) | 0.900 (0.90) | 0.899 (0.90) | 0.906 (0.90) |
| two_link_arm | 0.909 (0.90) | 0.912 (0.91) | 0.913 (0.91) | 0.902 (0.90) | 0.904 (0.90) | 0.448 (0.20) | 0.911 (0.91) | 0.899 (0.90) | 0.899 (0.90) | 0.899 (0.90) | 0.909 (0.91) |

**TEST-C3**

| system | U1 | U2 | U2mf | U3 | U3oracle | U4 | U4c | ekf | ukf | ekf_oracle | ekf-c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cartpole | 0.906 (0.89) | 0.900 (0.88) | 0.902 (0.88) | 0.832 (0.81) | 0.924 (0.92) | 0.900 (0.82) | 0.922 (0.90) | 0.152 (0.02) | 0.152 (0.02) | 0.152 (0.02) | 0.934 (0.92) |
| fhn | 0.911 (0.91) | 0.913 (0.91) | 0.913 (0.91) | 0.892 (0.89) | 0.907 (0.91) | 0.378 (0.37) | 0.902 (0.90) | 0.026 (0.02) | 0.026 (0.02) | 0.860 (0.86) | 0.951 (0.95) |
| highway_traffic | 0.910 (0.91) | 0.916 (0.91) | 0.911 (0.90) | 0.873 (0.87) | 0.911 (0.90) | 0.629 (0.41) | 0.907 (0.90) | 0.092 (0.01) | 0.469 (0.10) | 0.092 (0.01) | 0.900 (0.88) |
| network | 0.908 (0.90) | 0.909 (0.90) | 0.908 (0.90) | 0.882 (0.88) | 0.907 (0.90) | 0.496 (0.33) | 0.905 (0.90) | 0.037 (0.00) | 0.035 (0.00) | 0.037 (0.00) | 0.899 (0.88) |
| rossler | 0.914 (0.91) | 0.910 (0.90) | 0.907 (0.90) | 0.893 (0.89) | 0.903 (0.90) | 0.664 (0.55) | 0.910 (0.91) | 0.257 (0.12) | 0.257 (0.12) | 0.849 (0.65) | 0.913 (0.91) |
| two_link_arm | 0.907 (0.90) | 0.909 (0.91) | 0.906 (0.90) | 0.869 (0.86) | 0.903 (0.90) | 0.513 (0.28) | 0.910 (0.91) | 0.238 (0.14) | 0.238 (0.14) | 0.265 (0.15) | 0.898 (0.88) |

Held-out test plants, id/clean: U1: mean 0.910, min 0.892 (n=56), worst TEST-C3/cartpole/x2=0.892, TEST-C3/cartpole/x1=0.893, TEST-C1/network/x3=0.895; U2: mean 0.909, min 0.881 (n=56), worst TEST-C3/cartpole/x1=0.881, TEST-C1/network/x7=0.899, TEST-C1/network/x3=0.900; U2mf: mean 0.908, min 0.881 (n=56), worst TEST-C3/cartpole/x1=0.881, TEST-C1/network/x7=0.898, TEST-C1/network/x1=0.898

Worst dev cells (P1): U1: C3-partial/quad_tank/x1=0.888, C1-where/battery/x3=0.899, C3-partial/lorenz/x0=0.902; U2: C3-partial/quad_tank/x1=0.901, C3-partial/quad_tank/x0=0.903, C3-partial/vdp/x0=0.904; U2mf: C3-partial/quad_tank/x1=0.900, C1-where/vdp/x1=0.901, C1-where/vdp/x0=0.903

## 2. Per-state bands on id/clean: observability awareness and the information floor

Full width `2w/sigma` (seed mean). `floor` = `2 x 1.645 x` the full-past Bayesian PCRB std from the information study (the narrowest 90% band any estimator could have with Gaussian posteriors; THEORY.md section 3). `rmse` = the lora settled RMSE of that state (normalised). m = measured.

**C1-where**

| system | state | m | rmse | floor | U1 width | U2 width (cov) | U2 width / floor | EKF width (cov) | U2 rho(width, abs err) |
|---|---|---|---|---|---|---|---|---|---|
| battery | x0 |  | 0.058 | 0.006 | 0.174 | 0.177 (0.92) | 29.2 | 0.019 (0.87) | 0.43 |
| battery | x1 |  | 0.079 | 0.009 | 0.204 | 0.207 (0.91) | 24.0 | 0.010 (0.89) | 0.45 |
| battery | x2 |  | 0.055 | 0.021 | 0.177 | 0.183 (0.91) | 8.7 | 0.022 (0.83) | 0.40 |
| battery | x3 | y | 0.023 | 0.013 | 0.069 | 0.068 (0.91) | 5.2 | 0.013 (0.89) | 0.44 |
| battery | x4 | y | 0.045 | 0.005 | 0.106 | 0.106 (0.91) | 19.9 | 0.012 (0.90) | 0.47 |
| bioreactor | x0 |  | 0.029 | 0.028 | 0.065 | 0.067 (0.92) | 2.4 | 0.029 (0.91) | 0.35 |
| bioreactor | x1 | y | 0.022 | 0.009 | 0.059 | 0.056 (0.92) | 6.4 | 0.009 (0.91) | 0.51 |
| lorenz | x0 |  | 0.015 | 0.004 | 0.039 | 0.040 (0.90) | 9.9 | 0.015 (0.90) | 0.22 |
| lorenz | x1 | y | 0.041 | 0.005 | 0.056 | 0.067 (0.90) | 13.4 | 0.020 (0.90) | 0.26 |
| lorenz | x2 |  | 0.058 | 0.005 | 0.073 | 0.081 (0.91) | 15.8 | 0.022 (0.90) | 0.36 |
| quad_tank | x0 | y | 0.013 | 0.019 | 0.041 | 0.041 (0.91) | 2.2 | 0.019 (0.90) | 0.25 |
| quad_tank | x1 | y | 0.010 | 0.019 | 0.033 | 0.034 (0.91) | 1.8 | 0.019 (0.90) | 0.24 |
| quad_tank | x2 |  | 0.098 | 0.081 | 0.323 | 0.334 (0.91) | 4.1 | 0.081 (0.90) | 0.22 |
| quad_tank | x3 |  | 0.040 | 0.050 | 0.118 | 0.122 (0.91) | 2.5 | 0.050 (0.90) | 0.29 |
| vdp | x0 | y | 0.009 | 0.011 | 0.031 | 0.032 (0.90) | 2.8 | 0.017 (0.91) | 0.04 |
| vdp | x1 |  | 0.022 | 0.012 | 0.071 | 0.069 (0.91) | 5.6 | 0.020 (0.91) | 0.27 |

**C3-partial**

| system | state | m | rmse | floor | U1 width | U2 width (cov) | U2 width / floor | EKF width (cov) | U2 rho(width, abs err) |
|---|---|---|---|---|---|---|---|---|---|
| battery | x0 |  | 0.055 | 0.006 | 0.172 | 1.113 (0.92) | 183.2 | 0.018 (0.09) | 0.41 |
| battery | x1 |  | 0.085 | 0.009 | 0.215 | 0.299 (0.92) | 34.6 | 0.011 (0.76) | 0.47 |
| battery | x2 |  | 0.053 | 0.021 | 0.170 | 0.266 (0.92) | 12.7 | 0.032 (0.03) | 0.41 |
| battery | x3 | y | 0.022 | 0.013 | 0.071 | 0.066 (0.92) | 5.0 | 0.013 (0.34) | 0.44 |
| battery | x4 | y | 0.043 | 0.005 | 0.106 | 0.101 (0.91) | 19.0 | 0.012 (0.32) | 0.47 |
| bioreactor | x0 |  | 0.040 | 0.028 | 0.072 | 0.118 (0.93) | 4.2 | 0.027 (0.00) | 0.39 |
| bioreactor | x1 | y | 0.036 | 0.009 | 0.069 | 0.090 (0.91) | 10.2 | 0.006 (0.00) | 0.58 |
| lorenz | x0 |  | 0.015 | 0.004 | 0.041 | 0.042 (0.91) | 10.5 | 0.025 (0.08) | 0.21 |
| lorenz | x1 | y | 0.047 | 0.005 | 0.061 | 0.072 (0.91) | 14.4 | 0.048 (0.04) | 0.27 |
| lorenz | x2 |  | 0.067 | 0.005 | 0.077 | 0.085 (0.91) | 16.5 | 0.040 (0.00) | 0.37 |
| quad_tank | x0 | y | 0.015 | 0.019 | 0.049 | 0.048 (0.90) | 2.5 | 0.019 (0.90) | 0.30 |
| quad_tank | x1 | y | 0.012 | 0.019 | 0.037 | 0.038 (0.90) | 2.0 | 0.019 (0.90) | 0.32 |
| quad_tank | x2 |  | 0.097 | 0.081 | 0.311 | 0.324 (0.90) | 4.0 | 0.081 (0.58) | 0.20 |
| quad_tank | x3 |  | 0.042 | 0.050 | 0.125 | 0.138 (0.91) | 2.8 | 0.050 (0.64) | 0.28 |
| vdp | x0 | y | 0.009 | 0.011 | 0.030 | 0.030 (0.90) | 2.7 | 0.016 (0.03) | 0.06 |
| vdp | x1 |  | 0.021 | 0.012 | 0.066 | 0.064 (0.91) | 5.2 | 0.019 (0.03) | 0.27 |

**TEST-C1**

| system | state | m | rmse | floor | U1 width | U2 width (cov) | U2 width / floor | EKF width (cov) | U2 rho(width, abs err) |
|---|---|---|---|---|---|---|---|---|---|
| cartpole | x0 | y | 0.014 | 0.015 | 0.045 | 0.041 (0.92) | 2.8 | 0.015 (0.90) | 0.30 |
| cartpole | x1 |  | 0.094 | 0.048 | 0.319 | 0.339 (0.91) | 7.1 | 0.048 (0.92) | 0.10 |
| cartpole | x2 | y | 0.096 | 0.006 | 0.194 | 0.162 (0.91) | 25.0 | 0.007 (0.90) | 0.62 |
| cartpole | x3 |  | 0.382 | 0.022 | 0.438 | 0.557 (0.91) | 24.9 | 0.023 (0.90) | 0.53 |
| fhn | x0 | y | 0.018 | 0.015 | 0.058 | 0.055 (0.91) | 3.7 | 0.019 (0.91) | 0.36 |
| fhn | x1 |  | 0.039 | 0.025 | 0.127 | 0.121 (0.91) | 4.8 | 0.027 (0.91) | 0.39 |
| highway_traffic | x0 | y | 0.016 | 0.016 | 0.053 | 0.050 (0.91) | 3.2 | 0.016 (0.91) | 0.29 |
| highway_traffic | x1 |  | 0.103 | 0.058 | 0.276 | 0.294 (0.91) | 5.1 | 0.058 (0.91) | 0.38 |
| highway_traffic | x2 | y | 0.014 | 0.016 | 0.048 | 0.046 (0.91) | 2.9 | 0.016 (0.90) | 0.25 |
| highway_traffic | x3 |  | 0.122 | 0.064 | 0.411 | 0.412 (0.91) | 6.4 | 0.065 (0.89) | 0.28 |
| highway_traffic | x4 | y | 0.016 | 0.016 | 0.052 | 0.050 (0.91) | 3.2 | 0.016 (0.90) | 0.28 |
| network | x0 | y | 0.024 | 0.010 | 0.075 | 0.071 (0.91) | 6.9 | 0.012 (0.90) | 0.43 |
| network | x1 |  | 0.077 | 0.037 | 0.229 | 0.240 (0.90) | 6.4 | 0.041 (0.90) | 0.28 |
| network | x2 | y | 0.021 | 0.010 | 0.065 | 0.062 (0.91) | 6.3 | 0.012 (0.90) | 0.42 |
| network | x3 |  | 0.058 | 0.034 | 0.172 | 0.181 (0.90) | 5.3 | 0.037 (0.90) | 0.31 |
| network | x4 |  | 0.031 | 0.011 | 0.084 | 0.084 (0.91) | 7.8 | 0.016 (0.91) | 0.40 |
| network | x5 |  | 0.057 | 0.031 | 0.140 | 0.150 (0.91) | 4.9 | 0.034 (0.92) | 0.36 |
| network | x6 | y | 0.023 | 0.011 | 0.064 | 0.064 (0.90) | 5.7 | 0.015 (0.90) | 0.37 |
| network | x7 |  | 0.064 | 0.026 | 0.136 | 0.144 (0.90) | 5.6 | 0.029 (0.90) | 0.42 |
| network | x8 |  | 0.034 | 0.013 | 0.093 | 0.091 (0.91) | 6.9 | 0.020 (0.90) | 0.41 |
| network | x9 |  | 0.046 | 0.023 | 0.123 | 0.125 (0.91) | 5.4 | 0.026 (0.91) | 0.39 |
| rossler | x0 |  | 0.031 | 0.006 | 0.094 | 0.092 (0.90) | 15.9 | 0.027 (0.90) | 0.28 |
| rossler | x1 | y | 0.020 | 0.014 | 0.061 | 0.058 (0.90) | 4.1 | 0.020 (0.90) | 0.31 |
| rossler | x2 |  | 0.179 | 0.004 | 0.195 | 0.185 (0.90) | 51.0 | 0.077 (0.90) | 0.58 |
| two_link_arm | x0 | y | 0.022 | 0.017 | 0.072 | 0.071 (0.91) | 4.2 | 0.024 (0.90) | 0.32 |
| two_link_arm | x1 | y | 0.018 | 0.005 | 0.057 | 0.054 (0.91) | 10.8 | 0.008 (0.90) | 0.50 |
| two_link_arm | x2 |  | 0.124 | 0.085 | 0.415 | 0.416 (0.91) | 4.9 | 0.156 (0.90) | 0.09 |
| two_link_arm | x3 |  | 0.223 | 0.214 | 0.755 | 0.755 (0.92) | 3.5 | 0.397 (0.90) | 0.12 |

**TEST-C3**

| system | state | m | rmse | floor | U1 width | U2 width (cov) | U2 width / floor | EKF width (cov) | U2 rho(width, abs err) |
|---|---|---|---|---|---|---|---|---|---|
| cartpole | x0 | y | 0.180 | 0.015 | 0.630 | 0.428 (0.91) | 29.0 | 0.015 (0.29) | 0.47 |
| cartpole | x1 |  | 0.200 | 0.048 | 0.590 | 0.872 (0.88) | 18.3 | 0.048 (0.14) | 0.22 |
| cartpole | x2 | y | 0.340 | 0.006 | 0.873 | 0.733 (0.90) | 113.5 | 0.007 (0.02) | 0.67 |
| cartpole | x3 |  | 0.732 | 0.022 | 1.105 | 0.963 (0.90) | 43.0 | 0.023 (0.15) | 0.57 |
| fhn | x0 | y | 0.030 | 0.015 | 0.098 | 0.096 (0.92) | 6.4 | 0.021 (0.03) | 0.32 |
| fhn | x1 |  | 0.056 | 0.025 | 0.181 | 0.176 (0.91) | 7.0 | 0.031 (0.02) | 0.39 |
| highway_traffic | x0 | y | 0.059 | 0.016 | 0.177 | 0.182 (0.92) | 11.5 | 0.016 (0.01) | 0.38 |
| highway_traffic | x1 |  | 0.197 | 0.058 | 0.529 | 0.623 (0.92) | 10.7 | 0.045 (0.02) | 0.42 |
| highway_traffic | x2 | y | 0.057 | 0.016 | 0.181 | 0.178 (0.91) | 11.2 | 0.016 (0.07) | 0.47 |
| highway_traffic | x3 |  | 0.227 | 0.064 | 0.756 | 1.019 (0.92) | 15.9 | 0.044 (0.11) | 0.34 |
| highway_traffic | x4 | y | 0.041 | 0.016 | 0.134 | 0.134 (0.91) | 8.4 | 0.016 (0.25) | 0.38 |
| network | x0 | y | 0.029 | 0.010 | 0.093 | 0.090 (0.91) | 8.8 | 0.013 (0.00) | 0.32 |
| network | x1 |  | 0.095 | 0.037 | 0.297 | 0.307 (0.91) | 8.2 | 0.041 (0.02) | 0.20 |
| network | x2 | y | 0.024 | 0.010 | 0.074 | 0.074 (0.91) | 7.6 | 0.015 (0.00) | 0.29 |
| network | x3 |  | 0.072 | 0.034 | 0.207 | 0.217 (0.92) | 6.4 | 0.041 (0.04) | 0.30 |
| network | x4 |  | 0.032 | 0.011 | 0.091 | 0.092 (0.91) | 8.6 | 0.029 (0.02) | 0.29 |
| network | x5 |  | 0.067 | 0.031 | 0.172 | 0.175 (0.91) | 5.7 | 0.036 (0.02) | 0.40 |
| network | x6 | y | 0.028 | 0.011 | 0.085 | 0.085 (0.91) | 7.6 | 0.018 (0.09) | 0.27 |
| network | x7 |  | 0.059 | 0.026 | 0.166 | 0.164 (0.91) | 6.3 | 0.028 (0.11) | 0.43 |
| network | x8 |  | 0.039 | 0.013 | 0.096 | 0.098 (0.90) | 7.4 | 0.024 (0.06) | 0.30 |
| network | x9 |  | 0.062 | 0.023 | 0.142 | 0.143 (0.90) | 6.1 | 0.026 (0.01) | 0.39 |
| rossler | x0 |  | 0.062 | 0.006 | 0.191 | 0.189 (0.91) | 32.7 | 0.022 (0.12) | 0.27 |
| rossler | x1 | y | 0.044 | 0.014 | 0.145 | 0.141 (0.91) | 9.9 | 0.018 (0.17) | 0.27 |
| rossler | x2 |  | 0.432 | 0.004 | 0.162 | 0.303 (0.90) | 83.5 | 0.004 (0.48) | 0.46 |
| two_link_arm | x0 | y | 0.059 | 0.017 | 0.189 | 0.183 (0.91) | 10.7 | 0.023 (0.27) | 0.34 |
| two_link_arm | x1 | y | 0.050 | 0.005 | 0.149 | 0.133 (0.91) | 26.5 | 0.008 (0.14) | 0.73 |
| two_link_arm | x2 |  | 0.331 | 0.085 | 1.081 | 1.090 (0.91) | 12.8 | 0.146 (0.25) | 0.14 |
| two_link_arm | x3 |  | 0.456 | 0.214 | 1.326 | 1.358 (0.91) | 6.3 | 0.377 (0.30) | 0.10 |

## 3. Coverage under shift and faults (state mean, averaged over systems; seed mean)

**dev (C1-where + C3-partial)** - coverage; width of U2 / U3 in brackets

| tier | cond | U1 | U2 | U3 | U3oracle | U3qt* | U3qt_oracle* | U4c | ekf | ukf | ekf_oracle | ekf-c | U2 width | U3 width (frac. infinite) | U3qt* width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| id | clean | 0.911 | 0.911 | 0.899 | 0.908 | 0.897 | 0.903 | 0.913 | 0.562 | 0.573 | 0.726 | 0.907 | 0.118 | 2.443 (0.00 inf) | 0.152 |
| id | outliers | 0.257 | 0.469 | 0.867 | 0.902 | 0.797 | 0.867 | 0.836 | 0.277 | 0.131 | 0.676 | 0.566 | 28.615 | 327390.504 (0.48 inf) | 392.467 |
| id | noise10 | 0.693 | 0.779 | 0.970 | 0.919 | 0.947 | 0.903 | 0.823 | 0.415 | 0.422 | 0.723 | 0.763 | 0.371 | 234.818 (0.35 inf) | 1.186 |
| id | plant_drift10 | 0.419 | 0.470 | 0.689 | 0.893 | 0.566 | 0.878 | 0.557 | 0.118 | 0.131 | 0.551 | 0.412 | 0.248 | 11.201 (0.23 inf) | 0.764 |
| shape_ood | clean | 0.905 | 0.906 | 0.897 | 0.906 | 0.896 | 0.902 | 0.917 | 0.566 | 0.576 | 0.733 | 0.919 | 0.105 | 0.164 (0.00 inf) | 0.118 |
| shape_ood | outliers | 0.262 | 0.476 | 0.867 | 0.902 | 0.802 | 0.868 | 0.846 | 0.274 | 0.134 | 0.685 | 0.570 | 19.803 | 168527.631 (0.47 inf) | 288.472 |
| shape_ood | noise10 | 0.691 | 0.785 | 0.972 | 0.918 | 0.949 | 0.902 | 0.832 | 0.417 | 0.428 | 0.730 | 0.775 | 0.299 | 38.281 (0.35 inf) | 0.880 |
| shape_ood | plant_drift10 | 0.415 | 0.469 | 0.701 | 0.891 | 0.574 | 0.878 | 0.561 | 0.108 | 0.121 | 0.556 | 0.408 | 0.240 | 8.630 (0.24 inf) | 0.734 |
| sequence_ood | clean | 0.792 | 0.863 | 0.887 | 0.898 | 0.886 | 0.898 | 0.933 | 0.571 | 0.579 | 0.743 | 0.924 | 0.141 | 0.324 (0.00 inf) | 0.186 |
| sequence_ood | outliers | 0.242 | 0.477 | 0.861 | 0.898 | 0.792 | 0.864 | 0.854 | 0.284 | 0.139 | 0.700 | 0.580 | 55.074 | 1640873.479 (0.47 inf) | 1698.603 |
| sequence_ood | noise10 | 0.597 | 0.749 | 0.966 | 0.908 | 0.931 | 0.895 | 0.860 | 0.421 | 0.429 | 0.740 | 0.778 | 0.318 | 231.818 (0.36 inf) | 0.987 |
| sequence_ood | plant_drift10 | 0.396 | 0.494 | 0.693 | 0.890 | 0.594 | 0.874 | 0.608 | 0.097 | 0.108 | 0.552 | 0.397 | 0.331 | 12.740 (0.21 inf) | 1.184 |

**test (TEST-C1 + TEST-C3)** - coverage; width of U2 / U3 in brackets

| tier | cond | U1 | U2 | U3 | U3oracle | U3qt* | U3qt_oracle* | U4c | ekf | ukf | ekf_oracle | ekf-c | U2 width | U3 width (frac. infinite) | U3qt* width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| id | clean | 0.911 | 0.909 | 0.884 | 0.908 | 0.870 | 0.903 | 0.909 | 0.518 | 0.527 | 0.639 | 0.912 | 0.287 | 0.321 (0.00 inf) | 0.286 |
| id | outliers | 0.216 | 0.487 | 0.919 | 0.895 | 0.855 | 0.862 | 0.886 | 0.267 | 0.238 | 0.618 | 0.513 | 19740.687 | 190997549.415 (0.60 inf) | 969036.416 |
| id | noise10 | 0.625 | 0.795 | 0.956 | 0.903 | 0.954 | 0.898 | 0.852 | 0.385 | 0.373 | 0.629 | 0.762 | 0.605 | 58.727 (0.29 inf) | 1.620 |
| id | plant_drift10 | 0.719 | 0.758 | 0.774 | 0.904 | 0.761 | 0.899 | 0.826 | 0.244 | 0.328 | 0.503 | 0.576 | 0.876 | 25.946 (0.01 inf) | 1.076 |
| shape_ood | clean | 0.911 | 0.908 | 0.883 | 0.906 | 0.869 | 0.902 | 0.907 | 0.517 | 0.527 | 0.641 | 0.919 | 0.281 | 0.276 (0.00 inf) | 0.262 |
| shape_ood | outliers | 0.213 | 0.491 | 0.922 | 0.895 | 0.859 | 0.863 | 0.884 | 0.264 | 0.244 | 0.608 | 0.517 | 30840.079 | 1464437068.339 (0.60 inf) | 1667105.435 |
| shape_ood | noise10 | 0.623 | 0.801 | 0.957 | 0.903 | 0.955 | 0.898 | 0.849 | 0.384 | 0.370 | 0.624 | 0.766 | 0.593 | 291.580 (0.29 inf) | 1.593 |
| shape_ood | plant_drift10 | 0.719 | 0.761 | 0.776 | 0.902 | 0.762 | 0.898 | 0.830 | 0.243 | 0.327 | 0.506 | 0.588 | 0.800 | 8.529 (0.00 inf) | 0.929 |
| sequence_ood | clean | 0.834 | 0.880 | 0.890 | 0.901 | 0.884 | 0.900 | 0.916 | 0.516 | 0.524 | 0.642 | 0.904 | 0.364 | 0.439 (0.00 inf) | 0.400 |
| sequence_ood | outliers | 0.197 | 0.491 | 0.924 | 0.894 | 0.860 | 0.862 | 0.885 | 0.260 | 0.252 | 0.618 | 0.499 | 20187.662 | 179437755.194 (0.61 inf) | 1008140.643 |
| sequence_ood | noise10 | 0.569 | 0.787 | 0.957 | 0.898 | 0.958 | 0.896 | 0.869 | 0.385 | 0.371 | 0.628 | 0.755 | 1.225 | 23543.192 (0.30 inf) | 3.425 |
| sequence_ood | plant_drift10 | 0.678 | 0.777 | 0.827 | 0.899 | 0.821 | 0.897 | 0.876 | 0.244 | 0.326 | 0.512 | 0.568 | 0.811 | 4.306 (0.01 inf) | 0.993 |

`*` = exploratory (PREREG amendment A3, not part of any verdict). Widths of U3 / U3-oracle include infinite steps capped at 3x the largest U2 band.

**Failing cells** (U2 coverage < 0.85; 162 of 239 shift cells), with the deployable ACI (U3) and, exploratory, U3qt:

| node | system | tier | cond | U2 cov | U3 cov | U3/U2 width | U3qt* cov | U3qt*/U2 width | U3-oracle cov | ekf_oracle cov |
|---|---|---|---|---|---|---|---|---|---|---|
| C1-where | quad_tank | shape_ood | plant_drift10 | 0.128 | 0.785 | 37.90 | 0.321 | 2.51 | 0.883 | 0.251 |
| TEST-C1 | two_link_arm | id | outliers | 0.139 | 0.896 | 2518.20 | 0.648 | 46.15 | 0.885 | 0.882 |
| TEST-C1 | two_link_arm | shape_ood | outliers | 0.141 | 0.897 | 3963.90 | 0.649 | 46.79 | 0.885 | 0.871 |
| TEST-C1 | two_link_arm | sequence_ood | outliers | 0.141 | 0.897 | 4239.28 | 0.650 | 46.03 | 0.885 | 0.872 |
| C1-where | vdp | id | outliers | 0.141 | 0.705 | 15.07 | 0.532 | 13.55 | 0.885 | 0.888 |
| C1-where | quad_tank | id | plant_drift10 | 0.142 | 0.777 | 49.21 | 0.328 | 2.38 | 0.885 | 0.245 |
| C1-where | vdp | shape_ood | outliers | 0.143 | 0.697 | 13.23 | 0.541 | 13.48 | 0.885 | 0.871 |
| C1-where | vdp | sequence_ood | outliers | 0.150 | 0.699 | 16.05 | 0.536 | 13.20 | 0.885 | 0.877 |
| TEST-C1 | highway_traffic | shape_ood | outliers | 0.155 | 0.912 | 47500.24 | 0.607 | 54.07 | 0.885 | 0.887 |
| TEST-C1 | highway_traffic | id | outliers | 0.156 | 0.912 | 9668.27 | 0.612 | 49.22 | 0.885 | 0.913 |
| TEST-C1 | highway_traffic | sequence_ood | outliers | 0.158 | 0.913 | 8891.00 | 0.612 | 49.95 | 0.885 | 0.896 |
| C1-where | quad_tank | sequence_ood | plant_drift10 | 0.170 | 0.673 | 35.44 | 0.303 | 2.23 | 0.883 | 0.253 |
| TEST-C1 | network | id | outliers | 0.179 | 0.859 | 43.60 | 0.629 | 14.22 | 0.884 | 0.854 |
| TEST-C1 | network | shape_ood | outliers | 0.185 | 0.856 | 43.68 | 0.631 | 14.03 | 0.884 | 0.862 |
| C3-partial | quad_tank | id | plant_drift10 | 0.186 | 0.874 | 90.67 | 0.476 | 3.49 | 0.887 | 0.170 |
| TEST-C1 | network | sequence_ood | outliers | 0.189 | 0.866 | 47.57 | 0.647 | 14.30 | 0.884 | 0.906 |
| C3-partial | quad_tank | shape_ood | plant_drift10 | 0.189 | 0.884 | 65.20 | 0.488 | 3.37 | 0.885 | 0.179 |
| C1-where | bioreactor | shape_ood | plant_drift10 | 0.198 | 0.830 | 30.89 | 0.505 | 3.85 | 0.883 | 0.951 |
| C1-where | bioreactor | id | plant_drift10 | 0.213 | 0.813 | 35.98 | 0.517 | 3.72 | 0.887 | 0.944 |
| C3-partial | vdp | id | outliers | 0.229 | 0.778 | 7.97 | 0.681 | 12.12 | 0.886 | 0.985 |
| C3-partial | quad_tank | id | outliers | 0.238 | 0.901 | 199.91 | 0.658 | 9.88 | 0.884 | 0.833 |
| C3-partial | quad_tank | shape_ood | outliers | 0.238 | 0.903 | 118.35 | 0.665 | 9.76 | 0.884 | 0.889 |
| C3-partial | vdp | sequence_ood | outliers | 0.242 | 0.771 | 8.51 | 0.681 | 11.96 | 0.886 | 0.982 |
| C3-partial | vdp | shape_ood | outliers | 0.244 | 0.781 | 6.85 | 0.703 | 12.03 | 0.886 | 0.988 |
| C1-where | quad_tank | sequence_ood | outliers | 0.247 | 0.902 | 15836.92 | 0.635 | 13.55 | 0.882 | 0.899 |
| C1-where | quad_tank | id | outliers | 0.248 | 0.903 | 12380.78 | 0.650 | 9.50 | 0.882 | 0.905 |
| C1-where | bioreactor | sequence_ood | plant_drift10 | 0.254 | 0.796 | 39.92 | 0.498 | 4.59 | 0.880 | 0.953 |
| C1-where | quad_tank | shape_ood | outliers | 0.256 | 0.905 | 7733.65 | 0.661 | 9.70 | 0.882 | 0.879 |
| C3-partial | quad_tank | sequence_ood | outliers | 0.265 | 0.910 | 342.14 | 0.687 | 10.23 | 0.883 | 0.909 |
| C3-partial | quad_tank | sequence_ood | plant_drift10 | 0.276 | 0.837 | 64.23 | 0.560 | 3.00 | 0.885 | 0.181 |
| C3-partial | bioreactor | id | plant_drift10 | 0.296 | 0.592 | 37.33 | 0.506 | 4.12 | 0.895 | 0.000 |
| TEST-C1 | rossler | shape_ood | outliers | 0.324 | 0.911 | 80.61 | 0.925 | 16.41 | 0.885 | 0.763 |
| TEST-C1 | rossler | sequence_ood | outliers | 0.325 | 0.910 | 113.68 | 0.925 | 16.50 | 0.885 | 0.788 |
| C3-partial | bioreactor | shape_ood | plant_drift10 | 0.327 | 0.653 | 50.55 | 0.569 | 3.89 | 0.897 | 0.000 |
| TEST-C1 | rossler | id | outliers | 0.334 | 0.909 | 96.78 | 0.925 | 16.17 | 0.885 | 0.788 |
| C3-partial | lorenz | sequence_ood | outliers | 0.374 | 0.734 | 204.43 | 0.757 | 15.58 | 0.884 | 0.639 |
| C1-where | vdp | sequence_ood | noise10 | 0.374 | 0.883 | 6.12 | 0.760 | 3.15 | 0.886 | 0.899 |
| C1-where | vdp | id | noise10 | 0.381 | 0.880 | 5.56 | 0.762 | 3.14 | 0.886 | 0.917 |
| C1-where | vdp | shape_ood | noise10 | 0.388 | 0.885 | 5.36 | 0.769 | 3.14 | 0.886 | 0.899 |
| C3-partial | lorenz | shape_ood | outliers | 0.392 | 0.750 | 206.93 | 0.775 | 15.93 | 0.885 | 0.640 |
| C3-partial | lorenz | id | outliers | 0.399 | 0.749 | 242.76 | 0.776 | 15.90 | 0.885 | 0.639 |
| C1-where | lorenz | shape_ood | outliers | 0.405 | 0.769 | 257.80 | 0.808 | 16.37 | 0.885 | 0.163 |
| C1-where | lorenz | id | outliers | 0.407 | 0.765 | 828.19 | 0.802 | 17.06 | 0.884 | 0.152 |
| C1-where | lorenz | sequence_ood | outliers | 0.410 | 0.771 | 610.92 | 0.805 | 16.32 | 0.885 | 0.175 |
| TEST-C1 | fhn | shape_ood | outliers | 0.423 | 0.843 | 17314.62 | 0.817 | 13.69 | 0.879 | 0.876 |
| TEST-C1 | fhn | id | outliers | 0.424 | 0.839 | 11177.82 | 0.802 | 14.22 | 0.880 | 0.896 |
| C1-where | lorenz | shape_ood | plant_drift10 | 0.447 | 0.480 | 1.23 | 0.483 | 1.21 | 0.890 | 0.666 |
| TEST-C1 | fhn | sequence_ood | outliers | 0.450 | 0.848 | 1108.69 | 0.820 | 10.15 | 0.879 | 0.887 |
| C3-partial | lorenz | shape_ood | plant_drift10 | 0.460 | 0.484 | 1.20 | 0.485 | 1.17 | 0.890 | 0.438 |
| C1-where | lorenz | id | plant_drift10 | 0.462 | 0.486 | 2.18 | 0.488 | 1.67 | 0.891 | 0.661 |
| C3-partial | bioreactor | sequence_ood | plant_drift10 | 0.463 | 0.726 | 44.07 | 0.666 | 4.42 | 0.899 | 0.000 |
| C1-where | lorenz | sequence_ood | plant_drift10 | 0.467 | 0.493 | 1.15 | 0.495 | 1.16 | 0.891 | 0.663 |
| C3-partial | lorenz | sequence_ood | plant_drift10 | 0.469 | 0.487 | 1.12 | 0.488 | 1.12 | 0.891 | 0.425 |
| C3-partial | lorenz | id | plant_drift10 | 0.475 | 0.491 | 1.95 | 0.492 | 2.16 | 0.891 | 0.419 |
| TEST-C3 | network | id | outliers | 0.500 | 0.896 | 34.63 | 0.917 | 11.60 | 0.889 | 0.059 |
| C1-where | battery | id | outliers | 0.505 | 0.956 | 67.15 | 0.959 | 27.90 | 0.893 | 0.890 |
| TEST-C3 | highway_traffic | shape_ood | outliers | 0.506 | 0.962 | 2983.21 | 0.888 | 15.22 | 0.890 | 0.022 |
| TEST-C3 | network | sequence_ood | outliers | 0.511 | 0.901 | 24.85 | 0.924 | 11.65 | 0.888 | 0.067 |
| TEST-C3 | highway_traffic | sequence_ood | outliers | 0.514 | 0.963 | 7716.41 | 0.891 | 15.46 | 0.890 | 0.017 |
| TEST-C3 | highway_traffic | id | outliers | 0.515 | 0.962 | 7204.74 | 0.888 | 15.87 | 0.890 | 0.017 |
| TEST-C3 | network | shape_ood | outliers | 0.516 | 0.899 | 22.28 | 0.921 | 11.65 | 0.889 | 0.066 |
| C1-where | battery | shape_ood | outliers | 0.527 | 0.960 | 52.46 | 0.964 | 27.31 | 0.894 | 0.901 |
| TEST-C3 | network | shape_ood | plant_drift10 | 0.541 | 0.631 | 1.27 | 0.643 | 1.28 | 0.901 | 0.012 |
| TEST-C3 | network | id | plant_drift10 | 0.551 | 0.621 | 1.33 | 0.630 | 1.30 | 0.901 | 0.018 |
| C1-where | quad_tank | sequence_ood | noise10 | 0.568 | 0.951 | 39.94 | 0.813 | 3.06 | 0.894 | 0.905 |
| C1-where | battery | sequence_ood | outliers | 0.575 | 0.959 | 62.93 | 0.966 | 27.36 | 0.893 | 0.923 |
| TEST-C1 | cartpole | sequence_ood | outliers | 0.576 | 0.922 | 1645.23 | 0.924 | 17.41 | 0.889 | 0.880 |
| TEST-C1 | cartpole | id | outliers | 0.582 | 0.915 | 13060.78 | 0.917 | 8.57 | 0.893 | 0.902 |
| TEST-C1 | fhn | id | plant_drift10 | 0.591 | 0.598 | 1.07 | 0.598 | 1.08 | 0.890 | 0.361 |
| TEST-C3 | network | sequence_ood | plant_drift10 | 0.595 | 0.703 | 1.49 | 0.718 | 1.41 | 0.894 | 0.017 |
| TEST-C3 | two_link_arm | id | outliers | 0.600 | 0.973 | 60.89 | 0.990 | 19.39 | 0.888 | 0.261 |
| TEST-C3 | two_link_arm | sequence_ood | outliers | 0.600 | 0.973 | 55.81 | 0.990 | 19.06 | 0.888 | 0.243 |
| TEST-C3 | two_link_arm | shape_ood | outliers | 0.602 | 0.973 | 40.97 | 0.990 | 19.20 | 0.888 | 0.255 |
| TEST-C1 | cartpole | shape_ood | outliers | 0.605 | 0.929 | 15536.29 | 0.938 | 34.54 | 0.893 | 0.918 |
| C3-partial | vdp | sequence_ood | noise10 | 0.617 | 0.955 | 7.17 | 0.938 | 3.16 | 0.890 | 0.986 |
| TEST-C1 | network | id | plant_drift10 | 0.622 | 0.664 | 1.58 | 0.660 | 1.28 | 0.894 | 0.405 |
| TEST-C1 | highway_traffic | shape_ood | noise10 | 0.626 | 0.970 | 569.95 | 0.881 | 3.39 | 0.892 | 0.901 |
| TEST-C1 | fhn | shape_ood | plant_drift10 | 0.626 | 0.644 | 1.10 | 0.644 | 1.09 | 0.890 | 0.379 |
| TEST-C1 | highway_traffic | id | noise10 | 0.630 | 0.970 | 515.03 | 0.883 | 3.39 | 0.892 | 0.900 |
| TEST-C1 | network | shape_ood | plant_drift10 | 0.632 | 0.678 | 1.47 | 0.674 | 1.25 | 0.894 | 0.388 |
| TEST-C1 | highway_traffic | sequence_ood | noise10 | 0.634 | 0.972 | 296.34 | 0.891 | 3.37 | 0.892 | 0.905 |
| C3-partial | battery | id | outliers | 0.641 | 0.972 | 10893.65 | 0.972 | 33.32 | 0.891 | 0.556 |
| TEST-C3 | fhn | shape_ood | plant_drift10 | 0.646 | 0.654 | 1.05 | 0.646 | 1.07 | 0.893 | 0.895 |
| C3-partial | vdp | id | noise10 | 0.647 | 0.961 | 6.05 | 0.949 | 3.14 | 0.894 | 0.990 |
| C1-where | quad_tank | id | noise10 | 0.658 | 0.964 | 57.84 | 0.877 | 3.04 | 0.902 | 0.904 |
| TEST-C3 | fhn | id | plant_drift10 | 0.660 | 0.679 | 1.11 | 0.673 | 1.14 | 0.894 | 0.859 |
| C1-where | quad_tank | shape_ood | noise10 | 0.664 | 0.966 | 54.26 | 0.885 | 3.04 | 0.900 | 0.891 |
| TEST-C1 | two_link_arm | sequence_ood | noise10 | 0.667 | 0.957 | 101.27 | 0.939 | 3.03 | 0.894 | 0.897 |
| C3-partial | vdp | shape_ood | noise10 | 0.669 | 0.966 | 5.08 | 0.957 | 3.14 | 0.896 | 0.990 |
| C3-partial | battery | shape_ood | outliers | 0.673 | 0.975 | 12275.62 | 0.976 | 25.15 | 0.892 | 0.615 |
| TEST-C1 | two_link_arm | id | noise10 | 0.674 | 0.957 | 75.67 | 0.939 | 3.00 | 0.895 | 0.903 |
| TEST-C1 | two_link_arm | shape_ood | noise10 | 0.677 | 0.958 | 71.31 | 0.942 | 2.99 | 0.894 | 0.901 |
| TEST-C1 | network | sequence_ood | plant_drift10 | 0.680 | 0.766 | 1.91 | 0.759 | 1.39 | 0.893 | 0.394 |
| TEST-C1 | fhn | sequence_ood | plant_drift10 | 0.683 | 0.749 | 1.76 | 0.746 | 1.33 | 0.887 | 0.380 |
| C3-partial | battery | sequence_ood | plant_drift10 | 0.690 | 0.730 | 1.58 | 0.737 | 1.32 | 0.892 | 0.556 |
| TEST-C3 | rossler | sequence_ood | outliers | 0.703 | 0.961 | 17.72 | 0.982 | 13.02 | 0.888 | 0.947 |
| C1-where | battery | sequence_ood | plant_drift10 | 0.706 | 0.749 | 1.49 | 0.755 | 1.43 | 0.893 | 0.526 |
| C3-partial | battery | sequence_ood | outliers | 0.707 | 0.978 | 35461.21 | 0.980 | 37.12 | 0.893 | 0.691 |
| C1-where | battery | id | plant_drift10 | 0.708 | 0.682 | 1.18 | 0.679 | 1.19 | 0.901 | 0.593 |
| C3-partial | battery | id | plant_drift10 | 0.711 | 0.684 | 70.52 | 0.679 | 1.23 | 0.902 | 0.518 |
| TEST-C3 | rossler | shape_ood | outliers | 0.711 | 0.961 | 14.22 | 0.983 | 13.30 | 0.890 | 0.948 |
| C3-partial | vdp | sequence_ood | plant_drift10 | 0.713 | 0.713 | 1.00 | 0.715 | 1.01 | 0.892 | 0.992 |
| TEST-C3 | fhn | sequence_ood | plant_drift10 | 0.719 | 0.786 | 1.36 | 0.797 | 1.42 | 0.890 | 0.927 |
| TEST-C3 | rossler | id | outliers | 0.719 | 0.962 | 21.80 | 0.983 | 12.88 | 0.889 | 0.947 |
| C1-where | battery | shape_ood | plant_drift10 | 0.727 | 0.708 | 1.27 | 0.708 | 1.24 | 0.898 | 0.593 |
| C1-where | vdp | sequence_ood | plant_drift10 | 0.728 | 0.727 | 1.00 | 0.728 | 1.01 | 0.892 | 0.966 |
| C3-partial | battery | shape_ood | plant_drift10 | 0.733 | 0.712 | 2.20 | 0.710 | 1.14 | 0.899 | 0.525 |
| C3-partial | vdp | shape_ood | plant_drift10 | 0.735 | 0.730 | 0.99 | 0.732 | 1.00 | 0.894 | 0.992 |
| TEST-C3 | network | id | noise10 | 0.738 | 0.936 | 9.36 | 0.967 | 2.64 | 0.894 | 0.023 |
| TEST-C3 | network | sequence_ood | noise10 | 0.739 | 0.937 | 9.11 | 0.968 | 2.70 | 0.893 | 0.024 |
| C1-where | lorenz | id | noise10 | 0.743 | 0.971 | 39.78 | 0.990 | 3.50 | 0.889 | 0.835 |
| C3-partial | vdp | id | plant_drift10 | 0.744 | 0.738 | 0.99 | 0.739 | 0.99 | 0.894 | 0.993 |
| C1-where | vdp | shape_ood | plant_drift10 | 0.748 | 0.743 | 0.99 | 0.745 | 1.00 | 0.894 | 0.965 |
| TEST-C3 | network | shape_ood | noise10 | 0.752 | 0.942 | 8.60 | 0.972 | 2.56 | 0.894 | 0.021 |
| C3-partial | quad_tank | sequence_ood | noise10 | 0.753 | 0.973 | 64.10 | 0.893 | 3.01 | 0.918 | 0.923 |
| C1-where | lorenz | shape_ood | noise10 | 0.755 | 0.974 | 23.35 | 0.991 | 2.90 | 0.889 | 0.837 |
| C3-partial | lorenz | sequence_ood | noise10 | 0.764 | 0.971 | 20.56 | 0.986 | 2.81 | 0.891 | 0.427 |
| C1-where | vdp | id | plant_drift10 | 0.764 | 0.756 | 0.98 | 0.757 | 0.99 | 0.895 | 0.967 |
| C1-where | lorenz | sequence_ood | noise10 | 0.765 | 0.978 | 22.18 | 0.993 | 2.88 | 0.889 | 0.836 |
| TEST-C3 | rossler | shape_ood | noise10 | 0.773 | 0.915 | 4.84 | 0.958 | 2.34 | 0.892 | 0.785 |
| TEST-C3 | rossler | id | noise10 | 0.774 | 0.915 | 4.94 | 0.957 | 2.34 | 0.892 | 0.785 |
| TEST-C3 | rossler | sequence_ood | noise10 | 0.775 | 0.920 | 4.96 | 0.961 | 2.35 | 0.891 | 0.786 |
| C3-partial | lorenz | id | noise10 | 0.775 | 0.975 | 29.78 | 0.989 | 3.54 | 0.892 | 0.425 |
| TEST-C1 | rossler | id | plant_drift10 | 0.781 | 0.827 | 1.73 | 0.829 | 1.29 | 0.891 | 0.891 |
| C3-partial | quad_tank | id | noise10 | 0.782 | 0.981 | 72.12 | 0.928 | 2.97 | 0.923 | 0.847 |
| TEST-C3 | cartpole | id | outliers | 0.786 | 0.920 | 862.18 | 0.959 | 6.69 | 0.917 | 0.339 |
| C3-partial | quad_tank | shape_ood | noise10 | 0.786 | 0.981 | 51.27 | 0.931 | 2.97 | 0.921 | 0.913 |
| C3-partial | lorenz | shape_ood | noise10 | 0.790 | 0.978 | 17.56 | 0.991 | 2.83 | 0.892 | 0.425 |
| TEST-C3 | two_link_arm | shape_ood | noise10 | 0.793 | 0.940 | 16.45 | 0.978 | 3.11 | 0.891 | 0.281 |
| TEST-C1 | rossler | shape_ood | plant_drift10 | 0.794 | 0.837 | 1.58 | 0.839 | 1.26 | 0.892 | 0.885 |
| TEST-C3 | two_link_arm | sequence_ood | noise10 | 0.795 | 0.942 | 11.90 | 0.979 | 3.14 | 0.892 | 0.275 |
| TEST-C3 | two_link_arm | id | noise10 | 0.796 | 0.942 | 21.25 | 0.979 | 3.10 | 0.891 | 0.280 |
| TEST-C1 | fhn | id | noise10 | 0.797 | 0.976 | 106.43 | 0.970 | 2.82 | 0.898 | 0.904 |
| C1-where | bioreactor | sequence_ood | outliers | 0.799 | 0.885 | 15.36 | 0.872 | 1.87 | 0.899 | 0.906 |
| TEST-C3 | cartpole | sequence_ood | noise10 | 0.799 | 0.887 | 2.68 | 0.898 | 1.57 | 0.897 | 0.163 |
| TEST-C3 | highway_traffic | sequence_ood | plant_drift10 | 0.799 | 0.884 | 18.10 | 0.853 | 1.64 | 0.913 | 0.080 |
| TEST-C1 | rossler | sequence_ood | plant_drift10 | 0.800 | 0.848 | 1.78 | 0.849 | 1.30 | 0.893 | 0.892 |
| TEST-C1 | fhn | sequence_ood | noise10 | 0.805 | 0.975 | 74.01 | 0.966 | 2.87 | 0.890 | 0.899 |
| TEST-C3 | highway_traffic | sequence_ood | noise10 | 0.810 | 0.976 | 39771.46 | 0.957 | 2.84 | 0.898 | 0.058 |
| TEST-C3 | highway_traffic | id | noise10 | 0.811 | 0.977 | 135.91 | 0.961 | 2.19 | 0.899 | 0.056 |
| TEST-C1 | fhn | sequence_ood | clean | 0.814 | 0.860 | 1.44 | 0.858 | 1.27 | 0.891 | 0.895 |
| C1-where | bioreactor | sequence_ood | clean | 0.816 | 0.902 | 2.79 | 0.892 | 1.53 | 0.889 | 0.906 |
| TEST-C3 | highway_traffic | shape_ood | noise10 | 0.817 | 0.980 | 4255.16 | 0.969 | 2.27 | 0.897 | 0.061 |
| TEST-C3 | highway_traffic | shape_ood | plant_drift10 | 0.817 | 0.887 | 22.47 | 0.861 | 1.62 | 0.917 | 0.079 |
| TEST-C3 | cartpole | shape_ood | outliers | 0.818 | 0.944 | 731.12 | 0.968 | 7.50 | 0.917 | 0.238 |
| TEST-C3 | highway_traffic | id | plant_drift10 | 0.819 | 0.881 | 5.95 | 0.858 | 1.44 | 0.920 | 0.077 |
| TEST-C1 | highway_traffic | id | plant_drift10 | 0.820 | 0.861 | 45.51 | 0.852 | 1.27 | 0.914 | 0.674 |
| TEST-C3 | cartpole | id | noise10 | 0.821 | 0.864 | 18.90 | 0.842 | 1.34 | 0.913 | 0.217 |
| TEST-C3 | cartpole | sequence_ood | outliers | 0.822 | 0.948 | 638.83 | 0.971 | 6.25 | 0.913 | 0.310 |
| TEST-C3 | cartpole | shape_ood | noise10 | 0.823 | 0.856 | 2.75 | 0.830 | 1.25 | 0.906 | 0.112 |
| C1-where | bioreactor | sequence_ood | noise10 | 0.823 | 0.974 | 31.33 | 0.946 | 2.99 | 0.902 | 0.907 |
| TEST-C1 | fhn | shape_ood | noise10 | 0.826 | 0.980 | 33.92 | 0.977 | 2.84 | 0.900 | 0.904 |
| C3-partial | battery | sequence_ood | clean | 0.829 | 0.861 | 1.90 | 0.865 | 1.41 | 0.897 | 0.659 |
| TEST-C1 | network | sequence_ood | noise10 | 0.831 | 0.984 | 19.34 | 0.979 | 2.83 | 0.904 | 0.904 |
| TEST-C1 | highway_traffic | shape_ood | plant_drift10 | 0.832 | 0.865 | 14.01 | 0.858 | 1.20 | 0.915 | 0.669 |
| TEST-C1 | cartpole | sequence_ood | plant_drift10 | 0.834 | 0.889 | 1.47 | 0.883 | 1.26 | 0.901 | 0.966 |
| TEST-C1 | highway_traffic | sequence_ood | plant_drift10 | 0.834 | 0.866 | 6.50 | 0.858 | 1.21 | 0.916 | 0.676 |
| C1-where | battery | sequence_ood | clean | 0.838 | 0.876 | 1.59 | 0.880 | 1.47 | 0.898 | 0.891 |
| TEST-C1 | network | id | noise10 | 0.839 | 0.984 | 20.49 | 0.979 | 2.78 | 0.906 | 0.903 |
| C3-partial | bioreactor | sequence_ood | clean | 0.839 | 0.926 | 5.40 | 0.930 | 1.69 | 0.893 | 0.000 |
| TEST-C1 | cartpole | shape_ood | plant_drift10 | 0.847 | 0.823 | 0.98 | 0.818 | 0.99 | 0.906 | 0.964 |
| TEST-C1 | cartpole | id | plant_drift10 | 0.847 | 0.835 | 1.83 | 0.829 | 1.15 | 0.913 | 0.967 |

Exploratory: U3qt brings 36/162 failing cells to >= 0.85 within 3x the U2 width (U3: 7).


## 4. Efficiency and conditional coverage (clean; state mean, mean over systems)

| node | tier | method | coverage | width | median width | interval score | coverage in top-20% error episodes | rho(width, abs err) |
|---|---|---|---|---|---|---|---|---|
| C1-where | id | U1 | 0.910 | 0.0889 | 0.0889 | 0.1412 | 0.811 | - |
| C1-where | id | U2 | 0.910 | 0.0912 | 0.0710 | 0.1167 | 0.884 | 0.31 |
| C1-where | id | U2mf | 0.907 | 0.0877 | 0.0717 | 0.1156 | 0.878 | 0.30 |
| C1-where | id | U3 | 0.899 | 0.1147 | 0.0681 | 0.1406 | 0.896 | 0.31 |
| C1-where | id | U3qt | 0.897 | 0.1112 | 0.0681 | 0.1369 | 0.895 | 0.31 |
| C1-where | id | U4 | 0.524 | 0.0351 | 0.0242 | 0.1789 | 0.525 | 0.21 |
| C1-where | id | U4c | 0.912 | 0.1372 | 0.0967 | 0.1582 | 0.899 | 0.21 |
| C1-where | id | ekf | 0.898 | 0.0227 | 0.0221 | 0.0296 | 0.842 | 0.16 |
| C1-where | id | ekf-c | 0.908 | 0.0233 | 0.0227 | 0.0296 | 0.854 | 0.16 |
| C1-where | shape_ood | U1 | 0.904 | 0.0889 | 0.0889 | 0.1454 | 0.805 | - |
| C1-where | shape_ood | U2 | 0.906 | 0.0934 | 0.0743 | 0.1217 | 0.883 | 0.28 |
| C1-where | shape_ood | U2mf | 0.909 | 0.0938 | 0.0762 | 0.1218 | 0.892 | 0.27 |
| C1-where | shape_ood | U3 | 0.898 | 0.1080 | 0.0719 | 0.1361 | 0.897 | 0.27 |
| C1-where | shape_ood | U3qt | 0.896 | 0.1042 | 0.0722 | 0.1323 | 0.897 | 0.27 |
| C1-where | shape_ood | U4 | 0.525 | 0.0375 | 0.0263 | 0.1805 | 0.522 | 0.21 |
| C1-where | shape_ood | U4c | 0.917 | 0.1443 | 0.1048 | 0.1643 | 0.909 | 0.20 |
| C1-where | shape_ood | ekf | 0.894 | 0.0226 | 0.0220 | 0.0301 | 0.837 | 0.16 |
| C1-where | shape_ood | ekf-c | 0.905 | 0.0232 | 0.0226 | 0.0300 | 0.850 | 0.15 |
| C1-where | sequence_ood | U1 | 0.794 | 0.0889 | 0.0889 | 0.2153 | 0.766 | - |
| C1-where | sequence_ood | U2 | 0.862 | 0.1289 | 0.1004 | 0.1933 | 0.856 | 0.20 |
| C1-where | sequence_ood | U2mf | 0.881 | 0.1301 | 0.1067 | 0.1841 | 0.874 | 0.21 |
| C1-where | sequence_ood | U3 | 0.886 | 0.2115 | 0.1152 | 0.2601 | 0.883 | 0.21 |
| C1-where | sequence_ood | U3qt | 0.884 | 0.1652 | 0.1147 | 0.2144 | 0.882 | 0.21 |
| C1-where | sequence_ood | U4 | 0.572 | 0.0591 | 0.0450 | 0.2252 | 0.568 | 0.16 |
| C1-where | sequence_ood | U4c | 0.933 | 0.2396 | 0.1859 | 0.2603 | 0.932 | 0.16 |
| C1-where | sequence_ood | ekf | 0.900 | 0.0226 | 0.0221 | 0.0286 | 0.848 | 0.16 |
| C1-where | sequence_ood | ekf-c | 0.910 | 0.0232 | 0.0227 | 0.0286 | 0.860 | 0.16 |
| C3-partial | id | U1 | 0.912 | 0.0911 | 0.0911 | 0.1496 | 0.807 | - |
| C3-partial | id | U2 | 0.911 | 0.1448 | 0.0715 | 0.1714 | 0.884 | 0.33 |
| C3-partial | id | U2mf | 0.909 | 0.0899 | 0.0713 | 0.1185 | 0.880 | 0.33 |
| C3-partial | id | U3 | 0.899 | 4.7717 | 0.0681 | 4.7981 | 0.900 | 0.33 |
| C3-partial | id | U3qt | 0.897 | 0.1929 | 0.0677 | 0.2193 | 0.900 | 0.33 |
| C3-partial | id | U4 | 0.548 | 0.0386 | 0.0256 | 0.1714 | 0.577 | 0.23 |
| C3-partial | id | U4c | 0.913 | 0.1423 | 0.0974 | 0.1627 | 0.923 | 0.23 |
| C3-partial | id | ekf | 0.227 | 0.0261 | 0.0253 | 16.9235 | 0.148 | 0.02 |
| C3-partial | id | ekf-c | 0.905 | 4.2945 | 3.9239 | 4.6382 | 0.774 | 0.02 |
| C3-partial | shape_ood | U1 | 0.906 | 0.0911 | 0.0911 | 0.1522 | 0.810 | - |
| C3-partial | shape_ood | U2 | 0.906 | 0.1162 | 0.0751 | 0.1461 | 0.888 | 0.29 |
| C3-partial | shape_ood | U2mf | 0.907 | 0.0975 | 0.0764 | 0.1292 | 0.883 | 0.29 |
| C3-partial | shape_ood | U3 | 0.897 | 0.2205 | 0.0723 | 0.2502 | 0.904 | 0.29 |
| C3-partial | shape_ood | U3qt | 0.895 | 0.1314 | 0.0725 | 0.1608 | 0.905 | 0.28 |
| C3-partial | shape_ood | U4 | 0.556 | 0.0416 | 0.0285 | 0.1706 | 0.577 | 0.22 |
| C3-partial | shape_ood | U4c | 0.917 | 0.1517 | 0.1072 | 0.1709 | 0.927 | 0.23 |
| C3-partial | shape_ood | ekf | 0.237 | 0.0261 | 0.0253 | 16.7270 | 0.178 | 0.05 |
| C3-partial | shape_ood | ekf-c | 0.934 | 4.2883 | 3.9170 | 4.5372 | 0.881 | 0.05 |
| C3-partial | sequence_ood | U1 | 0.791 | 0.0911 | 0.0911 | 0.2377 | 0.764 | - |
| C3-partial | sequence_ood | U2 | 0.864 | 0.1524 | 0.1037 | 0.2172 | 0.858 | 0.23 |
| C3-partial | sequence_ood | U2mf | 0.863 | 0.1416 | 0.1025 | 0.2101 | 0.855 | 0.23 |
| C3-partial | sequence_ood | U3 | 0.888 | 0.4360 | 0.1199 | 0.4843 | 0.885 | 0.23 |
| C3-partial | sequence_ood | U3qt | 0.888 | 0.2073 | 0.1219 | 0.2547 | 0.886 | 0.23 |
| C3-partial | sequence_ood | U4 | 0.606 | 0.0678 | 0.0508 | 0.2178 | 0.602 | 0.18 |
| C3-partial | sequence_ood | U4c | 0.934 | 0.2569 | 0.1960 | 0.2776 | 0.933 | 0.18 |
| C3-partial | sequence_ood | ekf | 0.243 | 0.0260 | 0.0253 | 16.5668 | 0.210 | 0.02 |
| C3-partial | sequence_ood | ekf-c | 0.937 | 4.2836 | 3.8990 | 4.5183 | 0.920 | 0.02 |
| TEST-C1 | id | U1 | 0.913 | 0.1782 | 0.1782 | 0.3009 | 0.834 | - |
| TEST-C1 | id | U2 | 0.909 | 0.1817 | 0.1452 | 0.2331 | 0.881 | 0.35 |
| TEST-C1 | id | U2mf | 0.909 | 0.1799 | 0.1483 | 0.2349 | 0.885 | 0.32 |
| TEST-C1 | id | U3 | 0.894 | 0.2099 | 0.1367 | 0.2635 | 0.893 | 0.35 |
| TEST-C1 | id | U3qt | 0.893 | 0.1876 | 0.1374 | 0.2406 | 0.891 | 0.34 |
| TEST-C1 | id | U4 | 0.575 | 0.0637 | 0.0469 | 0.4212 | 0.583 | 0.22 |
| TEST-C1 | id | U4c | 0.908 | 0.2650 | 0.2022 | 0.3298 | 0.911 | 0.21 |
| TEST-C1 | id | ekf | 0.902 | 0.0486 | 0.0439 | 0.0604 | 0.864 | 0.13 |
| TEST-C1 | id | ekf-c | 0.909 | 0.0498 | 0.0450 | 0.0605 | 0.870 | 0.13 |
| TEST-C1 | shape_ood | U1 | 0.911 | 0.1782 | 0.1782 | 0.2911 | 0.866 | - |
| TEST-C1 | shape_ood | U2 | 0.908 | 0.1832 | 0.1539 | 0.2387 | 0.889 | 0.30 |
| TEST-C1 | shape_ood | U2mf | 0.911 | 0.1844 | 0.1585 | 0.2397 | 0.895 | 0.27 |
| TEST-C1 | shape_ood | U3 | 0.894 | 0.1792 | 0.1453 | 0.2423 | 0.890 | 0.30 |
| TEST-C1 | shape_ood | U3qt | 0.893 | 0.1795 | 0.1459 | 0.2417 | 0.888 | 0.29 |
| TEST-C1 | shape_ood | U4 | 0.576 | 0.0625 | 0.0477 | 0.4362 | 0.561 | 0.18 |
| TEST-C1 | shape_ood | U4c | 0.907 | 0.2595 | 0.2049 | 0.3358 | 0.897 | 0.18 |
| TEST-C1 | shape_ood | ekf | 0.903 | 0.0487 | 0.0439 | 0.0610 | 0.861 | 0.12 |
| TEST-C1 | shape_ood | ekf-c | 0.909 | 0.0499 | 0.0450 | 0.0610 | 0.869 | 0.12 |
| TEST-C1 | sequence_ood | U1 | 0.821 | 0.1782 | 0.1782 | 0.4378 | 0.782 | - |
| TEST-C1 | sequence_ood | U2 | 0.873 | 0.2431 | 0.1919 | 0.3433 | 0.861 | 0.27 |
| TEST-C1 | sequence_ood | U2mf | 0.881 | 0.2325 | 0.1940 | 0.3437 | 0.869 | 0.25 |
| TEST-C1 | sequence_ood | U3 | 0.892 | 0.2976 | 0.2067 | 0.3759 | 0.894 | 0.27 |
| TEST-C1 | sequence_ood | U3qt | 0.890 | 0.2739 | 0.2048 | 0.3548 | 0.890 | 0.27 |
| TEST-C1 | sequence_ood | U4 | 0.590 | 0.0899 | 0.0680 | 0.5578 | 0.583 | 0.18 |
| TEST-C1 | sequence_ood | U4c | 0.911 | 0.3705 | 0.2910 | 0.4671 | 0.908 | 0.18 |
| TEST-C1 | sequence_ood | ekf | 0.901 | 0.0484 | 0.0438 | 0.0605 | 0.860 | 0.13 |
| TEST-C1 | sequence_ood | ekf-c | 0.907 | 0.0496 | 0.0449 | 0.0606 | 0.866 | 0.13 |
| TEST-C3 | id | U1 | 0.910 | 0.3815 | 0.3815 | 0.6619 | 0.829 | - |
| TEST-C3 | id | U2 | 0.910 | 0.3932 | 0.3032 | 0.5168 | 0.890 | 0.37 |
| TEST-C3 | id | U2mf | 0.908 | 0.3809 | 0.3028 | 0.5080 | 0.887 | 0.35 |
| TEST-C3 | id | U3 | 0.874 | 0.4321 | 0.2612 | 0.5947 | 0.893 | 0.36 |
| TEST-C3 | id | U3qt | 0.847 | 0.3846 | 0.2452 | 0.5896 | 0.881 | 0.35 |
| TEST-C3 | id | U4 | 0.597 | 0.1996 | 0.1463 | 0.7482 | 0.617 | 0.27 |
| TEST-C3 | id | U4c | 0.910 | 0.5117 | 0.3908 | 0.6136 | 0.912 | 0.28 |
| TEST-C3 | id | ekf | 0.134 | 0.0428 | 0.0429 | 3.0926 | 0.091 | 0.08 |
| TEST-C3 | id | ekf-c | 0.916 | 0.6700 | 0.6666 | 1.3520 | 0.805 | 0.08 |
| TEST-C3 | shape_ood | U1 | 0.911 | 0.3815 | 0.3815 | 0.6225 | 0.854 | - |
| TEST-C3 | shape_ood | U2 | 0.907 | 0.3781 | 0.3091 | 0.5023 | 0.881 | 0.34 |
| TEST-C3 | shape_ood | U2mf | 0.905 | 0.3731 | 0.3080 | 0.5013 | 0.881 | 0.32 |
| TEST-C3 | shape_ood | U3 | 0.871 | 0.3723 | 0.2658 | 0.5483 | 0.872 | 0.33 |
| TEST-C3 | shape_ood | U3qt | 0.844 | 0.3437 | 0.2497 | 0.5727 | 0.859 | 0.32 |
| TEST-C3 | shape_ood | U4 | 0.599 | 0.1893 | 0.1423 | 0.7541 | 0.603 | 0.24 |
| TEST-C3 | shape_ood | U4c | 0.908 | 0.4998 | 0.3915 | 0.6147 | 0.907 | 0.24 |
| TEST-C3 | shape_ood | ekf | 0.132 | 0.0428 | 0.0429 | 2.9404 | 0.100 | 0.11 |
| TEST-C3 | shape_ood | ekf-c | 0.929 | 0.6716 | 0.6684 | 1.2766 | 0.855 | 0.11 |
| TEST-C3 | sequence_ood | U1 | 0.847 | 0.3815 | 0.3815 | 0.9464 | 0.801 | - |
| TEST-C3 | sequence_ood | U2 | 0.887 | 0.4847 | 0.3867 | 0.7282 | 0.869 | 0.31 |
| TEST-C3 | sequence_ood | U2mf | 0.885 | 0.4555 | 0.3686 | 0.7219 | 0.867 | 0.30 |
| TEST-C3 | sequence_ood | U3 | 0.888 | 0.5795 | 0.3913 | 0.8090 | 0.887 | 0.30 |
| TEST-C3 | sequence_ood | U3qt | 0.879 | 0.5255 | 0.3825 | 0.7962 | 0.882 | 0.30 |
| TEST-C3 | sequence_ood | U4 | 0.627 | 0.2780 | 0.1976 | 0.8716 | 0.621 | 0.24 |
| TEST-C3 | sequence_ood | U4c | 0.921 | 0.6676 | 0.5027 | 0.7829 | 0.919 | 0.24 |
| TEST-C3 | sequence_ood | ekf | 0.131 | 0.0424 | 0.0424 | 2.9548 | 0.105 | 0.14 |
| TEST-C3 | sequence_ood | ekf-c | 0.901 | 0.6620 | 0.6531 | 1.3773 | 0.830 | 0.14 |

P2 detail (systems where U2 beats U1): C1-where id IS: battery, bioreactor, lorenz, quad_tank, vdp; C1-where shape_ood IS: battery, bioreactor, lorenz, quad_tank, vdp; C1-where sequence_ood IS: battery, bioreactor, lorenz, quad_tank, vdp; C1-where id cov_hi: battery, bioreactor, lorenz, quad_tank, vdp; C3-partial id IS: lorenz, quad_tank, vdp; C3-partial shape_ood IS: bioreactor, lorenz, quad_tank, vdp; C3-partial sequence_ood IS: battery, bioreactor, lorenz, vdp; C3-partial id cov_hi: battery, bioreactor, lorenz, quad_tank, vdp

## 5. EKF / UKF covariance calibration (P5)

Per cell: coverage of the Gaussian 90% band `1.645 sqrt(P_jj)` (state mean), ANEES = mean NEES / n_x (1 = calibrated). `calibrated` = coverage in [0.85, 0.95] and ANEES in [0.5, 2]. `U2 closer` = number of systems where HyperKKL-U2's coverage is closer to 0.90.

| node | tier | cond | filter | calibrated systems | U2 closer to 0.90 | median coverage | median ANEES | median U2 coverage |
|---|---|---|---|---|---|---|---|---|
| C1-where | id | clean | ekf | 5/5 | 2/5 | 0.901 | 0.99 | 0.909 |
| C1-where | id | clean | ukf | 5/5 | 3/5 | 0.906 | 0.95 | 0.909 |
| C1-where | id | clean | ekf_oracle | 5/5 | 2/5 | 0.901 | 0.99 | 0.909 |
| C1-where | id | outliers | ekf | 0/5 | 4/5 | 0.219 | 3.2e+04 | 0.407 |
| C1-where | id | outliers | ukf | 0/5 | 5/5 | 0.083 | 3e+03 | 0.407 |
| C1-where | id | outliers | ekf_oracle | 3/5 | 1/5 | 0.890 | 1.4 | 0.407 |
| C1-where | id | noise10 | ekf | 1/5 | 2/5 | 0.640 | 3.5 | 0.743 |
| C1-where | id | noise10 | ukf | 1/5 | 2/5 | 0.698 | 3.5 | 0.743 |
| C1-where | id | noise10 | ekf_oracle | 4/5 | 1/5 | 0.904 | 0.98 | 0.743 |
| C1-where | id | plant_drift10 | ekf | 0/5 | 4/5 | 0.164 | 4.4e+03 | 0.462 |
| C1-where | id | plant_drift10 | ukf | 0/5 | 4/5 | 0.164 | 4.3e+03 | 0.462 |
| C1-where | id | plant_drift10 | ekf_oracle | 1/5 | 1/5 | 0.661 | 4.8 | 0.462 |
| C1-where | shape_ood | clean | ekf | 5/5 | 2/5 | 0.897 | 1 | 0.907 |
| C1-where | shape_ood | clean | ukf | 5/5 | 3/5 | 0.904 | 0.99 | 0.907 |
| C1-where | shape_ood | clean | ekf_oracle | 5/5 | 2/5 | 0.897 | 1 | 0.907 |
| C1-where | shape_ood | outliers | ekf | 0/5 | 4/5 | 0.195 | 1.7e+04 | 0.405 |
| C1-where | shape_ood | outliers | ukf | 0/5 | 5/5 | 0.100 | 1.7e+04 | 0.405 |
| C1-where | shape_ood | outliers | ekf_oracle | 3/5 | 1/5 | 0.879 | 1.9 | 0.405 |
| C1-where | shape_ood | noise10 | ekf | 1/5 | 2/5 | 0.643 | 3.6 | 0.755 |
| C1-where | shape_ood | noise10 | ukf | 1/5 | 2/5 | 0.687 | 3.6 | 0.755 |
| C1-where | shape_ood | noise10 | ekf_oracle | 4/5 | 0/5 | 0.891 | 1 | 0.755 |
| C1-where | shape_ood | plant_drift10 | ekf | 0/5 | 4/5 | 0.170 | 4.4e+03 | 0.447 |
| C1-where | shape_ood | plant_drift10 | ukf | 0/5 | 4/5 | 0.171 | 4.4e+03 | 0.447 |
| C1-where | shape_ood | plant_drift10 | ekf_oracle | 0/5 | 1/5 | 0.666 | 4.9 | 0.447 |
| C1-where | sequence_ood | clean | ekf | 5/5 | 0/5 | 0.898 | 1 | 0.854 |
| C1-where | sequence_ood | clean | ukf | 5/5 | 0/5 | 0.909 | 0.96 | 0.854 |
| C1-where | sequence_ood | clean | ekf_oracle | 5/5 | 0/5 | 0.898 | 1 | 0.854 |
| C1-where | sequence_ood | outliers | ekf | 0/5 | 4/5 | 0.235 | 2e+03 | 0.410 |
| C1-where | sequence_ood | outliers | ukf | 0/5 | 5/5 | 0.090 | 2e+03 | 0.410 |
| C1-where | sequence_ood | outliers | ekf_oracle | 4/5 | 1/5 | 0.899 | 1.2 | 0.410 |
| C1-where | sequence_ood | noise10 | ekf | 1/5 | 2/5 | 0.654 | 4.5 | 0.765 |
| C1-where | sequence_ood | noise10 | ukf | 1/5 | 2/5 | 0.699 | 3.6 | 0.765 |
| C1-where | sequence_ood | noise10 | ekf_oracle | 4/5 | 1/5 | 0.899 | 1 | 0.765 |
| C1-where | sequence_ood | plant_drift10 | ekf | 0/5 | 4/5 | 0.154 | 4.4e+03 | 0.467 |
| C1-where | sequence_ood | plant_drift10 | ukf | 0/5 | 4/5 | 0.173 | 4.3e+03 | 0.467 |
| C1-where | sequence_ood | plant_drift10 | ekf_oracle | 0/5 | 1/5 | 0.663 | 4.8 | 0.467 |
| C3-partial | id | clean | ekf | 0/5 | 5/5 | 0.041 | 8.9e+03 | 0.906 |
| C3-partial | id | clean | ukf | 0/5 | 5/5 | 0.041 | 5.2e+03 | 0.906 |
| C3-partial | id | clean | ekf_oracle | 0/5 | 5/5 | 0.578 | 90 | 0.906 |
| C3-partial | id | outliers | ekf | 0/5 | 4/5 | 0.029 | 3.5e+05 | 0.399 |
| C3-partial | id | outliers | ukf | 0/5 | 4/5 | 0.029 | 3.4e+05 | 0.399 |
| C3-partial | id | outliers | ekf_oracle | 0/5 | 2/5 | 0.639 | 14 | 0.399 |
| C3-partial | id | noise10 | ekf | 0/5 | 5/5 | 0.039 | 2.4e+04 | 0.782 |
| C3-partial | id | noise10 | ukf | 0/5 | 5/5 | 0.039 | 5.2e+03 | 0.782 |
| C3-partial | id | noise10 | ekf_oracle | 0/5 | 3/5 | 0.544 | 19 | 0.782 |
| C3-partial | id | plant_drift10 | ekf | 0/5 | 5/5 | 0.029 | 7.6e+03 | 0.475 |
| C3-partial | id | plant_drift10 | ukf | 0/5 | 5/5 | 0.029 | 7.5e+03 | 0.475 |
| C3-partial | id | plant_drift10 | ekf_oracle | 0/5 | 4/5 | 0.419 | 56 | 0.475 |
| C3-partial | shape_ood | clean | ekf | 0/5 | 5/5 | 0.042 | 5.2e+03 | 0.907 |
| C3-partial | shape_ood | clean | ukf | 0/5 | 5/5 | 0.042 | 3.9e+03 | 0.907 |
| C3-partial | shape_ood | clean | ekf_oracle | 1/5 | 5/5 | 0.596 | 48 | 0.907 |
| C3-partial | shape_ood | outliers | ekf | 0/5 | 4/5 | 0.029 | 3.3e+05 | 0.392 |
| C3-partial | shape_ood | outliers | ukf | 0/5 | 4/5 | 0.029 | 3.2e+05 | 0.392 |
| C3-partial | shape_ood | outliers | ekf_oracle | 1/5 | 2/5 | 0.640 | 14 | 0.392 |
| C3-partial | shape_ood | noise10 | ekf | 0/5 | 5/5 | 0.040 | 7.5e+03 | 0.790 |
| C3-partial | shape_ood | noise10 | ukf | 0/5 | 5/5 | 0.040 | 3.9e+03 | 0.790 |
| C3-partial | shape_ood | noise10 | ekf_oracle | 1/5 | 3/5 | 0.569 | 19 | 0.790 |
| C3-partial | shape_ood | plant_drift10 | ekf | 0/5 | 5/5 | 0.025 | 7.3e+03 | 0.460 |
| C3-partial | shape_ood | plant_drift10 | ukf | 0/5 | 5/5 | 0.025 | 7.2e+03 | 0.460 |
| C3-partial | shape_ood | plant_drift10 | ekf_oracle | 0/5 | 4/5 | 0.438 | 44 | 0.460 |
| C3-partial | sequence_ood | clean | ekf | 0/5 | 5/5 | 0.043 | 4e+03 | 0.859 |
| C3-partial | sequence_ood | clean | ukf | 0/5 | 5/5 | 0.043 | 4e+03 | 0.859 |
| C3-partial | sequence_ood | clean | ekf_oracle | 1/5 | 4/5 | 0.659 | 17 | 0.859 |
| C3-partial | sequence_ood | outliers | ekf | 0/5 | 4/5 | 0.029 | 3.4e+05 | 0.374 |
| C3-partial | sequence_ood | outliers | ukf | 0/5 | 4/5 | 0.029 | 3.2e+05 | 0.374 |
| C3-partial | sequence_ood | outliers | ekf_oracle | 1/5 | 2/5 | 0.691 | 14 | 0.374 |
| C3-partial | sequence_ood | noise10 | ekf | 0/5 | 5/5 | 0.040 | 4e+03 | 0.764 |
| C3-partial | sequence_ood | noise10 | ukf | 0/5 | 5/5 | 0.040 | 4e+03 | 0.764 |
| C3-partial | sequence_ood | noise10 | ekf_oracle | 1/5 | 3/5 | 0.650 | 17 | 0.764 |
| C3-partial | sequence_ood | plant_drift10 | ekf | 0/5 | 5/5 | 0.021 | 7.4e+03 | 0.469 |
| C3-partial | sequence_ood | plant_drift10 | ukf | 0/5 | 5/5 | 0.021 | 7.4e+03 | 0.469 |
| C3-partial | sequence_ood | plant_drift10 | ekf_oracle | 0/5 | 4/5 | 0.425 | 46 | 0.469 |
| TEST-C1 | id | clean | ekf | 6/6 | 0/6 | 0.902 | 0.99 | 0.910 |
| TEST-C1 | id | clean | ukf | 4/5 | 2/5 | 0.906 | 0.96 | 0.910 |
| TEST-C1 | id | clean | ekf_oracle | 6/6 | 0/6 | 0.902 | 0.99 | 0.910 |
| TEST-C1 | id | outliers | ekf | 0/6 | 3/6 | 0.327 | 3.6e+05 | 0.256 |
| TEST-C1 | id | outliers | ukf | 0/5 | 2/5 | 0.312 | 1.3e+03 | 0.334 |
| TEST-C1 | id | outliers | ekf_oracle | 3/6 | 0/6 | 0.889 | 2 | 0.256 |
| TEST-C1 | id | noise10 | ekf | 0/6 | 5/6 | 0.642 | 3.8 | 0.818 |
| TEST-C1 | id | noise10 | ukf | 0/5 | 4/5 | 0.644 | 4.3 | 0.797 |
| TEST-C1 | id | noise10 | ekf_oracle | 6/6 | 0/6 | 0.903 | 0.99 | 0.818 |
| TEST-C1 | id | plant_drift10 | ekf | 0/6 | 6/6 | 0.376 | 62 | 0.801 |
| TEST-C1 | id | plant_drift10 | ukf | 0/5 | 4/5 | 0.447 | 52 | 0.820 |
| TEST-C1 | id | plant_drift10 | ekf_oracle | 0/6 | 5/6 | 0.782 | 6 | 0.801 |
| TEST-C1 | shape_ood | clean | ekf | 6/6 | 2/6 | 0.900 | 1 | 0.912 |
| TEST-C1 | shape_ood | clean | ukf | 4/5 | 3/5 | 0.905 | 0.95 | 0.910 |
| TEST-C1 | shape_ood | clean | ekf_oracle | 6/6 | 2/6 | 0.900 | 1 | 0.912 |
| TEST-C1 | shape_ood | outliers | ekf | 0/6 | 3/6 | 0.372 | 1.9e+04 | 0.255 |
| TEST-C1 | shape_ood | outliers | ukf | 0/5 | 2/5 | 0.319 | 1.4e+03 | 0.324 |
| TEST-C1 | shape_ood | outliers | ekf_oracle | 3/6 | 0/6 | 0.874 | 2.7 | 0.255 |
| TEST-C1 | shape_ood | noise10 | ekf | 0/6 | 5/6 | 0.639 | 4 | 0.839 |
| TEST-C1 | shape_ood | noise10 | ukf | 0/5 | 4/5 | 0.633 | 4.3 | 0.826 |
| TEST-C1 | shape_ood | noise10 | ekf_oracle | 6/6 | 0/6 | 0.902 | 1 | 0.839 |
| TEST-C1 | shape_ood | plant_drift10 | ekf | 0/6 | 6/6 | 0.369 | 60 | 0.813 |
| TEST-C1 | shape_ood | plant_drift10 | ukf | 0/5 | 5/5 | 0.438 | 48 | 0.832 |
| TEST-C1 | shape_ood | plant_drift10 | ekf_oracle | 0/6 | 5/6 | 0.777 | 5.6 | 0.813 |
| TEST-C1 | sequence_ood | clean | ekf | 6/6 | 2/6 | 0.902 | 0.99 | 0.886 |
| TEST-C1 | sequence_ood | clean | ukf | 4/5 | 2/5 | 0.901 | 0.99 | 0.896 |
| TEST-C1 | sequence_ood | clean | ekf_oracle | 6/6 | 2/6 | 0.902 | 0.99 | 0.886 |
| TEST-C1 | sequence_ood | outliers | ekf | 0/6 | 3/6 | 0.307 | 4e+04 | 0.257 |
| TEST-C1 | sequence_ood | outliers | ukf | 0/5 | 2/5 | 0.303 | 1.3e+03 | 0.325 |
| TEST-C1 | sequence_ood | outliers | ekf_oracle | 4/6 | 0/6 | 0.884 | 1.1 | 0.257 |
| TEST-C1 | sequence_ood | noise10 | ekf | 0/6 | 5/6 | 0.651 | 3.8 | 0.818 |
| TEST-C1 | sequence_ood | noise10 | ukf | 0/5 | 4/5 | 0.660 | 4.3 | 0.805 |
| TEST-C1 | sequence_ood | noise10 | ekf_oracle | 6/6 | 0/6 | 0.900 | 0.99 | 0.818 |
| TEST-C1 | sequence_ood | plant_drift10 | ekf | 0/6 | 6/6 | 0.398 | 61 | 0.817 |
| TEST-C1 | sequence_ood | plant_drift10 | ukf | 0/5 | 5/5 | 0.472 | 48 | 0.834 |
| TEST-C1 | sequence_ood | plant_drift10 | ekf_oracle | 0/6 | 5/6 | 0.784 | 5.5 | 0.817 |
| TEST-C3 | id | clean | ekf | 0/6 | 6/6 | 0.122 | 1.1e+03 | 0.910 |
| TEST-C3 | id | clean | ukf | 0/6 | 6/6 | 0.195 | 1.1e+03 | 0.910 |
| TEST-C3 | id | clean | ekf_oracle | 1/6 | 6/6 | 0.209 | 2.9e+02 | 0.910 |
| TEST-C3 | id | outliers | ekf | 0/6 | 6/6 | 0.061 | 1.2e+04 | 0.659 |
| TEST-C3 | id | outliers | ukf | 0/6 | 6/6 | 0.052 | 3.7e+03 | 0.659 |
| TEST-C3 | id | outliers | ekf_oracle | 1/6 | 5/6 | 0.300 | 3.6e+03 | 0.659 |
| TEST-C3 | id | noise10 | ekf | 0/6 | 6/6 | 0.095 | 9.6e+02 | 0.803 |
| TEST-C3 | id | noise10 | ukf | 0/6 | 6/6 | 0.130 | 9.8e+02 | 0.803 |
| TEST-C3 | id | noise10 | ekf_oracle | 0/6 | 5/6 | 0.249 | 1.2e+03 | 0.803 |
| TEST-C3 | id | plant_drift10 | ekf | 0/5 | 5/5 | 0.077 | 1.2e+03 | 0.819 |
| TEST-C3 | id | plant_drift10 | ukf | 0/5 | 5/5 | 0.135 | 8.3e+02 | 0.819 |
| TEST-C3 | id | plant_drift10 | ekf_oracle | 1/5 | 4/5 | 0.135 | 1.1e+03 | 0.819 |
| TEST-C3 | shape_ood | clean | ekf | 0/6 | 6/6 | 0.115 | 8.4e+02 | 0.911 |
| TEST-C3 | shape_ood | clean | ukf | 0/6 | 6/6 | 0.195 | 8.5e+02 | 0.911 |
| TEST-C3 | shape_ood | clean | ekf_oracle | 1/6 | 6/6 | 0.204 | 1.2e+02 | 0.911 |
| TEST-C3 | shape_ood | outliers | ekf | 0/6 | 6/6 | 0.054 | 1.2e+04 | 0.656 |
| TEST-C3 | shape_ood | outliers | ukf | 0/6 | 6/6 | 0.047 | 3.7e+03 | 0.656 |
| TEST-C3 | shape_ood | outliers | ekf_oracle | 1/6 | 5/6 | 0.247 | 3.1e+03 | 0.656 |
| TEST-C3 | shape_ood | noise10 | ekf | 0/6 | 6/6 | 0.085 | 8.7e+02 | 0.805 |
| TEST-C3 | shape_ood | noise10 | ukf | 0/6 | 6/6 | 0.122 | 8.8e+02 | 0.805 |
| TEST-C3 | shape_ood | noise10 | ekf_oracle | 0/6 | 5/6 | 0.197 | 1.3e+03 | 0.805 |
| TEST-C3 | shape_ood | plant_drift10 | ekf | 0/5 | 5/5 | 0.079 | 1.1e+03 | 0.817 |
| TEST-C3 | shape_ood | plant_drift10 | ukf | 0/5 | 5/5 | 0.127 | 6.8e+02 | 0.817 |
| TEST-C3 | shape_ood | plant_drift10 | ekf_oracle | 1/5 | 4/5 | 0.122 | 6.8e+02 | 0.817 |
| TEST-C3 | sequence_ood | clean | ekf | 0/6 | 6/6 | 0.122 | 7.5e+02 | 0.891 |
| TEST-C3 | sequence_ood | clean | ukf | 0/6 | 6/6 | 0.191 | 7.5e+02 | 0.891 |
| TEST-C3 | sequence_ood | clean | ekf_oracle | 1/6 | 5/6 | 0.202 | 1.3e+02 | 0.891 |
| TEST-C3 | sequence_ood | outliers | ekf | 0/6 | 6/6 | 0.055 | 3.3e+04 | 0.651 |
| TEST-C3 | sequence_ood | outliers | ukf | 0/6 | 6/6 | 0.047 | 7.4e+03 | 0.651 |
| TEST-C3 | sequence_ood | outliers | ekf_oracle | 1/6 | 5/6 | 0.277 | 6.5e+03 | 0.651 |
| TEST-C3 | sequence_ood | noise10 | ekf | 0/6 | 6/6 | 0.090 | 7.7e+02 | 0.797 |
| TEST-C3 | sequence_ood | noise10 | ukf | 0/6 | 6/6 | 0.123 | 7.8e+02 | 0.797 |
| TEST-C3 | sequence_ood | noise10 | ekf_oracle | 0/6 | 5/6 | 0.219 | 1.1e+03 | 0.797 |
| TEST-C3 | sequence_ood | plant_drift10 | ekf | 0/5 | 5/5 | 0.080 | 1.2e+03 | 0.799 |
| TEST-C3 | sequence_ood | plant_drift10 | ukf | 0/5 | 5/5 | 0.143 | 8.6e+02 | 0.799 |
| TEST-C3 | sequence_ood | plant_drift10 | ekf_oracle | 1/5 | 4/5 | 0.141 | 8.9e+02 | 0.799 |

Totals over all cells: ekf: calibrated 36/261, U2 closer to 0.90 in 206/261; ukf: calibrated 30/249, U2 closer to 0.90 in 204/249; ekf_oracle: calibrated 99/261, U2 closer to 0.90 in 131/261

## 6. Information-floor sanity check (P6)

Ratio half-width / (1.645 x floor std) over 176 cells: min 1.77, median 7.00, max 183.24. Cells below 1 (band narrower than the floor's 90% interval): 0, of which with coverage >= 0.9: 0.

## 7. Reproduction and data checks

Settled NRMSE of the re-run lora observer vs the run's stored results.json (clean, 3 tiers): 198 comparisons, 14 differ by > 1%. Differences: C1-where_lorenz/s0_nrmse_lora/id: 0.0266 vs 0.0274; C1-where_lorenz/s0_nrmse_lora/sequence_ood: 0.0254 vs 0.0272; C1-where_lorenz/s1_nrmse_lora/id: 0.0274 vs 0.0279; C1-where_lorenz/s1_nrmse_lora/shape_ood: 0.0252 vs 0.0255; C1-where_lorenz/s1_nrmse_lora/sequence_ood: 0.0267 vs 0.0272; C1-where_lorenz/s2_nrmse_lora/id: 0.0267 vs 0.0276; C1-where_lorenz/s2_nrmse_lora/shape_ood: 0.0229 vs 0.0240; C1-where_lorenz/s2_nrmse_lora/sequence_ood: 0.0250 vs 0.0254; C3-partial_lorenz/s0_nrmse_lora/id: 0.0359 vs 0.0367; C3-partial_lorenz/s0_nrmse_lora/shape_ood: 0.0369 vs 0.0360

Ensemble (U4) available (identical episodes across seeds): C1-where_battery: True, C1-where_bioreactor: True, C1-where_lorenz: True, C1-where_quad_tank: True, C1-where_vdp: True, C3-partial_battery: True, C3-partial_bioreactor: True, C3-partial_lorenz: True, C3-partial_quad_tank: True, C3-partial_vdp: True, TEST-C1_cartpole: True, TEST-C1_fhn: True, TEST-C1_highway_traffic: True, TEST-C1_network: True, TEST-C1_rossler: True, TEST-C1_two_link_arm: True, TEST-C3_cartpole: True, TEST-C3_fhn: True, TEST-C3_highway_traffic: True, TEST-C3_network: True, TEST-C3_rossler: True, TEST-C3_two_link_arm: True
