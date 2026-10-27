# Uncertainty bands for a learned KKL observer: what is guaranteed and why

This note is meant to become an appendix remark. Notation follows PREREG.md. For a deployed observer
with estimate `xhat(t)` and state reference stds `sigma_j`, the per-state score is
`S_j(t) = |x_j(t) - xhat_j(t)| / (sigma_j s_j(t))`, with `s_j = 1` for U1 and `s_j` = a scale model for U2.
An *episode* is one deployment: an initial condition, an input signal, a sensor-noise realisation and a
plant. Its settled window has T stored times (the same for every episode).

## 1. Split conformal over episodes (U1, U2)

Samples within one episode are strongly dependent, so the exchangeable unit is the episode and not the
time sample. Let `F_e(q) = (1/T) sum_t 1{S_e(t) <= q}` be the within-episode empirical CDF of episode e,
calibration episodes `e = 1..n`, and

    qhat = inf { q : (1/(n+1)) sum_{e=1..n} F_e(q) >= 1 - alpha }

(the pooled quantile of all calibration scores at level `(1 - alpha)(1 + 1/n)`, every episode weighted
equally).

**Proposition 1.** If the score vectors of the calibration episodes and of a test episode `n+1` are
exchangeable (and the scale model was fitted on other data), then
`E[ F_{n+1}(qhat) ] >= 1 - alpha`: the expected fraction of settled time steps at which the band
`[xhat_j - qhat s_j, xhat_j + qhat s_j]` contains `x_j` is at least `1 - alpha`.

*Proof.* Let `q* = inf{ q : (1/(n+1)) sum_{e=1..n+1} F_e(q) >= 1 - alpha }`, a symmetric function of all
n+1 episodes. Since `F_{n+1} >= 0`, the defining set of qhat is contained in that of q*, so `qhat >= q*`
and `F_{n+1}(qhat) >= F_{n+1}(q*)`. By exchangeability `E[F_{n+1}(q*)] = E[(1/(n+1)) sum_{e<=n+1} F_e(q*)]`,
which is `>= 1 - alpha` by the definition of q* (right-continuity of the F_e). QED.

What the guarantee is and is not.
- It is a **marginal** guarantee: an average over episodes, time steps and calibration draws. It says nothing
  about one episode or one time instant (Section 4).
- **Exchangeability is the whole assumption.** Here it holds between the fresh calibration episodes and the
  `id` test tier: the same input generator, ICs, sensor and plant, with independent seeds. It is **not**
  implied for `shape_ood` / `sequence_ood` (other input laws), for sensor faults (outliers, 10% noise), or for
  plant drift. There the coverage is an empirical question, and RESULTS.md reports it.
- U2 has the same guarantee for any scale model `s`, however poor, provided `s` was fitted on episodes that
  are not used for calibration. A good `s` does not change the marginal coverage; it moves the width to the
  times and states where the error is large. That is the efficiency (interval score) and conditional-coverage
  gain.
- The phase-2 validation episodes were used in training only for early stopping. We use them to fit `s`, not
  to calibrate, so the selection effect cannot bias the coverage.

## 2. Adaptive conformal inference under shift (U3)

ACI (Gibbs & Candes, NeurIPS 2021) updates the working miscoverage level online,
`alpha_{t+1} = alpha_t + gamma (alpha - err_t)`, and issues the band at level `1 - alpha_t`, where
`err_t in {0, 1}` is the miss indicator of the band at time t. Their Proposition 4.1 is deterministic:
for **any** sequence of data, with no exchangeability or stationarity assumption,

    | (1/T) sum_{t<=T} err_t - alpha |  <=  (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T).

The reason is that `alpha_t` stays in `[-gamma, 1 + gamma]`. A level below 0 produces the infinite band,
which cannot be missed. A level above 1 produces the empty band, which is always missed. So the telescoping
sum `alpha_{T+1} - alpha_1 = gamma sum (alpha - err_t)` is bounded.

**The guarantee is about the feedback events err_t, not about the state.** This matters for an observer.
- U3-oracle uses the true state miss as feedback. It inherits the guarantee for every state, but it cannot
  be deployed, since x is unknown online.
- U3 (deployable) can only use what is measured: the output residual `y - h(xhat)`. ACI then guarantees
  long-run coverage of the **output-residual band**. The state bands, which share the same `alpha_t`, widen
  only to the extent that a shift shows in the outputs.
  - A shift that raises the residual without raising the state error makes U3 conservative. Sensor noise
    is the example: 10% noise inflates `|y - h(xhat)|` directly.
  - A shift that moves an unmeasured state with little effect on y, i.e. in a weakly observable direction,
    is invisible to any output-based feedback. No online method can restore coverage there without
    a model or ground truth.

  This is an observability statement about the feedback signal, not a weakness of ACI.
- With `gamma = 0.005` and about T = 1000 stored samples per episode, the bound above is 0.18 per episode.
  It is a long-run statement. In practice the deviation is much smaller (RESULTS.md).

## 3. The band against the information-theoretic floor

The information-theory study (experiments/information/, theory/m4_information.tex) bounds the achievable
error from below by Fisher information and Gramians. For any estimator the van Trees (Bayesian Cramer-Rao)
inequality gives `E[(x_j - xhat_j)^2] >= [J^{-1}]_{jj} =: v_j`, where J is the prior plus data information.

**Remark (no band can beat the floor with correct coverage, Gaussian case).** Suppose the posterior of
`x_j` given the data `y` is Gaussian (or symmetric unimodal) with variance `v(y) >= v_min`. Let B(y) be any
band of half-width w(y) around any centre with coverage `P(x_j in B) >= 1 - alpha`. By Anderson's lemma,
`P(x_j in B | y) <= 2 Phi(w(y)/sqrt(v(y))) - 1 <= 2 Phi(w(y)/sqrt(v_min)) - 1`. The right-hand side is
concave in `w >= 0`, so Jensen's inequality gives

    1 - alpha  <=  2 Phi( E[w] / sqrt(v_min) ) - 1,   i.e.   E[w]  >=  z_{1-alpha/2} sqrt(v_min)
                                                               (= 1.645 sqrt(v_min) at 90%).

So the mean half-width of a correctly covering 90% band must be at least 1.645 floor stds. A band that
covers at >= 0.9 while being narrower than that is impossible in the Gaussian-posterior case. For a
non-Gaussian posterior (strongly nonlinear plants, heavy-tailed noise) the inequality can fail only through
non-Gaussian error tails, so a violation is a red flag to investigate. RESULTS.md applies this check per
(system, state) as criterion P6.

Read together, the two studies bracket each state:

    1.645 sqrt(v_floor)  <=  1.645 x (HyperKKL RMSE, if near-Gaussian)  ~  calibrated band half-width.

The ratio band / floor is the price of using a learned, model-free observer instead of an ideal filter.

## 4. Why the bands are observability-aware

The learned observer is `xhat = T+(z, c)`, with `z` the latent filter state. Noise `delta z` (from sensor
noise, or from a latent filter that has not forgotten its initial condition) enters the estimate to first
order as `J delta z`, with `J = d T+ / d z`. Row j of J, the feature `jac_j`, measures how strongly state j
depends on the latent. For a linear plant `x' = F x`, `y = H x`, the KKL map solves `A T - T F = B H`, and
`T` is injective exactly when `(F, H)` is observable. Directions in which the observability Gramian is small
are the directions in which `T` is nearly singular. In those directions the left inverse `T+` has large
rows. So `||J_j||` is the learned, model-free counterpart of the inverse Gramian diagonal `gram_j`, which
the U2 scale model also receives when a modeller's model is available. A scale model on these features
widens the band of weakly observable states, such as the unmeasured upper tanks of quad_tank, and of times
when the trajectory passes through weakly observable regions. Marginal coverage stays exact on id while it
does so (Section 1).

The EKF's covariance carries the same information only if its model and noise statistics are right. Its
`P_jj` is the posterior variance of the linearised model with the assumed Q and R. With a partial model, a
drifted plant or unmodelled outliers, P stays small while the error grows: the over-confidence that the
NEES test detects.
