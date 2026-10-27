# Pre-registration: uncertainty bands for the learned KKL observer (HyperKKL, `lora`)

Written 2026-10-26 11:07 CET (10:07 UTC), **before any band was computed or any evaluation run**.
Any later change is appended in the "Amendments" section at the bottom, with a timestamp and reason.

## Object of study

The point estimate is the deployed HyperKKL observer of the rank-r hypernetwork decoder (method
`lora`, rank 4, SFM context), loaded read-only from the cached trained models of finished runs,
with the code snapshot each run was produced by (`experiments/code/<hash>.tar.gz`). No model is
trained or retrained. The score is the per-state normalised error
`e_j(t) = (x_j(t) - xhat_j(t)) / sigma_j` (sigma_j = the run's reference std `scales.x_std`) on
the settled window `t >= settle = floor(0.5 (N1 - 1))`, stored at a stride `s = max(1, floor(L_settled / 1000))`
(about 1000 samples per episode; coverage at a uniformly subsampled time is an unbiased estimate
of the full-window coverage). Nominal level alpha = 0.10 (target coverage 0.90), two-sided,
symmetric bands `[xhat_j - w_j, xhat_j + w_j]`.

## Data roles (per run)

- **fit set**: the run's 64 phase-2 validation episodes (`training_data(...)["val"]`, training input
  families; used in training only for early stopping). Used only to fit the U2 scale model.
- **calibration set**: 128 FRESH episodes from the id input distribution (`eval_data` of tier `id`
  with `eval.rep = 1`, `n_episodes = 128`), simulated in memory with the run's own code (never cached),
  sensor as in training. Exchangeable with the id test episodes (same generator, independent seeds).
  These are "validation episodes" in the sense of split conformal: never used for training or selection.
- **test blocks**: the run's evaluation episodes (64 per tier), tiers `id`, `shape_ood`,
  `sequence_ood`; conditions `clean`, `outliers` (2% outliers at 12 sd), `noise10` (10% sensor noise),
  `plant_drift10` (plant parameters drifted 10%). Conditions are defined in `configs/base.yaml`.

Scenarios: **C1-where** (exact model; dev plants battery, bioreactor, lorenz, quad_tank, vdp;
seeds 0-2) and **C3-partial** (partial model; same plants, seeds 0-2). duffing is excluded (author
decision). Held-out test plants (cartpole, fhn, highway_traffic, network, rossler, two_link_arm; seeds
0-2): **TEST-C1** (exact model, the test counterpart of C1-where) and **TEST-C3** (the frozen
realistic scenario: partial model + 30% parameter error + coloured Student-t noise; there is no
test counterpart of C3-partial alone). Test plants are evaluated once, with the recipe frozen here.

## Methods

- **U1 split conformal, per state.** Calibration scores `|e_j|` pooled over the settled samples of the
  n calibration episodes, every episode weighted equally. `w_j = Q_j`, the smallest q with
  `(1/(n+1)) sum_e F_e(q) >= 1 - alpha` (F_e = within-episode empirical CDF), i.e. the pooled quantile at
  level `(1 - alpha)(1 + 1/n)`. Guarantee (THEORY.md): `E[coverage of a new exchangeable episode] >= 1 - alpha`.
- **U2 locally adaptive conformal.** `w_j(t) = q_j * s_j(t)`, s = a scale model on online signals:
  1. `r_ema`: EMA (rate = the latent design abscissa alpha_KKL) of the squared output residual
     `(y - h(xhat)) / y_std`, square-rooted, per output;
  2. `ctx_dist`: mean distance of the context c_t (standardised) to its 10 nearest training contexts
     (20k settled training samples);
  3. `jac_j`: per-state row norm of the decoder Jacobian `d xhat_j / d z` (sigma units, z scaled by its
     training std) -- the KKL-native, model-free observability measure (a large row norm = state j
     is weakly determined by z, i.e. weakly observable through the learned map);
  4. `gram_j`: local observability from the MODELLER's model (exact in C1, partial in C3) linearised
     along xhat: information recursion `I_t = lam Phi^-T I_{t-1} Phi^-1 + H' R^-1 H`, lam = exp(-2 alpha_KKL dt),
     feature `sqrt([(I_t + eps I)^-1]_jj) / sigma_j`.
  Scale model: one MLP (inputs = log of all features, standardised; 2 hidden layers x 64, SiLU; outputs
  nx), fitted on the fit set by MSE on `log(|e_j| + 1e-3)`, Adam lr 1e-3, 3000 steps, weight decay 1e-4;
  `s_j = exp(output_j)`. Conformal score `|e_j| / s_j`, quantile as in U1 on the calibration set.
  **U2-mf** = the same without `gram` (model-free). U2 (with gram) is the primary U2.
- **U3 adaptive conformal inference (ACI, Gibbs & Candes 2021)**, online within each test episode,
  step size gamma = 0.005 per stored sample, alpha_0 = alpha, starting at the settle time; band
  `w_j(t) = Q^{U2}_j(1 - alpha_t) s_j(t)` (Q^{U2}_j = the calibration quantile function of the U2 scores).
  alpha_t <= 0 gives an infinite band, reported as covered, its width capped at 3 x the largest
  calibration score (the capped fraction is reported); alpha_t >= 1 gives a zero-width band.
  - **U3 (deployable, proxy feedback)**: the miscoverage signal is that of a U2-type conformal band
    on the measured output residual `|y - h(xhat)| / y_std` (calibrated on the same calibration set,
    with its own scale = the r_ema-only rule `s = r_ema`), err_t = 1 if any measured output is outside
    its band at level 1 - alpha_t. One alpha_t per episode drives every state.
  - **U3-oracle**: feedback = the true miss indicator of each state (not deployable; it shows the ACI
    long-run guarantee itself).
- **U4 deep-ensemble spread** over the 3 seeds' lora observers (centre = ensemble mean). NOTE (known
  before evaluation): the 3 seeds do NOT share training data (each seed draws its own training episodes),
  so this is a bagging-type ensemble. U4: `w_j = 1.645 * std_seeds` (naive Gaussian band). U4c: the
  ensemble std as the scale s of a conformal band (calibrated as U1), centred on the ensemble mean.
- **Reference filters**: EKF, UKF (nominal Q, R, P0 as in the runs) and `ekf_oracle` (condition-aware
  R and the model-error-rate Q), band `w_j = 1.645 sqrt(P_jj)` (Gaussian 90%), and NEES = e' P^-1 e
  (unnormalised error); ANEES = mean NEES / nx (1 = calibrated). Also **EKF-c**: conformalised EKF
  (s = sqrt(P_jj), calibrated as U1) to separate the EKF's shape of uncertainty from its scale.

## Metrics (per run x tier x condition x method x state, then seed means)

coverage (fraction of settled stored samples with |x_j - centre_j| <= w_j); width = mean `2 w_j / sigma_j`
(infinite ACI steps capped as above); interval score `IS = 2w + (2/alpha) dist(x, band)`, /sigma_j;
high-error coverage = coverage restricted to the test episodes whose settled lora NRMSE (all states) is in
the top 20% of the block; width-error rank correlation (Spearman, per state, over stored samples);
ANEES for filters. "Unmeasured" = states not in `system.sensors`.

## Pre-registered criteria (scored mechanically in RESULTS.md)

- **P1 validity on id (U1, U2, U2-mf)**: on `id/clean`, the seed-mean coverage of every state of every dev
  system (C1-where and C3-partial) is >= 0.87, and the mean over all such cells is in [0.88, 0.95].
  Expected: holds (guarantee). Also reported for the test plants.
- **P2 adaptivity**: U2 has a lower mean interval score than U1 (seed mean, averaged over states) on
  >= 4/5 dev systems in each tier of `clean` (C1-where and C3-partial separately), and higher high-error
  coverage than U1 on >= 4/5 dev systems on id/clean.
- **P3 observability awareness**: (a) on dev systems with unmeasured states, the U2 mean width of
  unmeasured states exceeds that of measured states on >= 4/5 systems (id/clean, C1-where); (b) on
  quad_tank, both upper tanks (states 2, 3) get wider U2 bands than both lower tanks (0, 1) in both
  scenarios; (c) the per-state U2 mean width is rank-correlated with the per-state lora RMSE across all
  (system, state) cells, Spearman rho >= 0.6.
- **P4 shift**: a cell (system x scenario x tier x condition, OOD tiers or a fault) "fails" when the U2
  seed-mean coverage averaged over states is < 0.85. Report the failure rate. **ACI restores** coverage if
  U3 (deployable) brings >= 50% of the failing cells to >= 0.85 with a mean width at most 3x U2's in them.
  U3-oracle: the per-episode long-run miscoverage is within 0.02 of alpha in >= 95% of cells (sanity of the
  guarantee; finite-episode deviations bounded by (max(alpha_1, 1-alpha_1) + gamma) / (gamma T)).
- **P5 filters**: EKF/UKF covariance bands are "calibrated" in a cell if the coverage averaged over states is
  in [0.85, 0.95] and ANEES in [0.5, 2]. Expected: calibrated on C1-where id/clean, over-confident on
  C3-partial and under faults. HyperKKL-U2 is "better calibrated than the EKF" in a cell if |cov - 0.9| is
  smaller; reported as a count over cells, plus widths (a calibrated wide band is not a win on accuracy).
- **P6 information floor sanity**: per (system, state), the U1/U2 half-width (seed mean, id/clean) must be
  >= 1.645 x the information-theoretic floor std (from experiments/information/ if available at scoring time,
  otherwise the exact-model EKF posterior std of C1-where as a proxy, labelled as such). Any violation together
  with coverage >= 0.9 is flagged as a contradiction to investigate (Gaussian-posterior argument, THEORY.md).

## Amendments

- **A1 (2026-10-26 11:20 CET, before any result was seen).** U3 deployable proxy: the output-residual band
  uses a CONSTANT scale (s = 1, i.e. the plain conformal quantile of |y_i - h_i(xhat)| / y_std,i on the
  calibration set) instead of `s = r_ema`. Reason: with s = r_ema the ratio |r_t| / r_ema,t is
  self-normalising and would not register a residual that grows under a shift, which is the very signal
  ACI needs. Everything else in U3 is unchanged.
- **A2 (2026-10-26 11:50 CET; the first stage-1 attempt crashed; only a pipeline smoke test on one vdp seed had been looked at: reproduction of the stored NRMSE and one U3-oracle coverage number).** The
  `gram_j` feature is computed in covariance (Riccati) form instead of information form, because the
  information recursion overflowed (Phi^-1 of strongly contracting modes) and produced singular matrices on
  battery, bioreactor and lorenz: `P_t = Phi P_{t-1} Phi' / lam + Q_p`, followed by the Kalman measurement
  update with R. Here `Q_p` = the run's process-noise covariance (noise_proc x sigma, floored at 1e-6 sigma)
  replaces the `(1 - lam) P0^-1` anchor, and each variance is capped at 100 x the prior variance.
  `gram_j = sqrt(P_jj) / sigma_j` is the same local-observability quantity: the posterior std of the
  linearised modeller's model along xhat, with forgetting. Nothing else changes.
- **A3 (2026-10-26 12:14 CET, EXPLORATORY, added AFTER seeing the stage-2 output of one system, C1-where
  bioreactor).** That output showed that ACI on the quantile level (U3, U3-oracle) has to go to infinite
  bands (alpha_t <= 0) whenever the errors exceed the largest calibration score. This happened under
  plant_drift10 and, for the proxy, under noise10. The reported width then depends on the arbitrary cap. Two
  variants are added and labelled exploratory everywhere; they are not part of any pre-registered verdict.
  **U3qt** (deployable) and **U3qt-oracle**: multiplicative quantile tracking (Angelopoulos, Candes &
  Tibshirani 2023), `w_j(t) = exp(theta_t) q_j s_j(t)`, `theta_{t+1} = theta_t + eta (err_t - alpha)`,
  eta = 0.05. For U3qt the feedback is the miss of the output-residual band scaled by the same `exp(theta_t)`,
  so theta tracks the inflation of the residual and transfers it to the states. For U3qt-oracle the feedback
  is the true per-state miss. A median width (`width_med`, insensitive to the cap) is also reported for every
  method.
- **A4 (2026-10-26 19:25 CET, engineering notes, after evaluation; no change to any method or criterion).**
  (i) From about 12:50 CET, newly started runs ran `ekf_oracle` as one batched pass with per-episode Q and R.
  On C1-where bioreactor s0 it was verified to be bit-identical to the per-condition filters.
  (ii) Simulated data that was not in the cache (calibration episodes, drifted-plant episodes) was shared
  across the seeds of a run through a scratchpad cache. It was never written to experiments/cache.
  (iii) TEST-C3 rossler: every `plant_drift10` episode (30% mismatch + 10% drift) diverged and was rejected by
  the generator, so that condition was dropped for this plant only. Its other blocks are complete.
  (iv) The lorenz re-runs reproduce the stored NRMSE only to 1-7% (chaotic plant, GPU nondeterminism); all
  other runs reproduce to < 1% (RESULTS.md section 7).
