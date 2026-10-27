# Information budget vs achieved error: pre-registration

Written 2026-10-26 11:30 CET (14:30 Dubai), AFTER computing the per-system information budgets
(`budgets.json`, theory-side quantities only: plant, sensors, noise levels, input episodes) and BEFORE
any comparison of those budgets with achieved errors. What had been seen before writing this file,
stated for honesty: (i) the aggregate RESULTS.md of the campaign (total NRMSE of HyperKKL vs grey-box
per system; no per-state numbers except quad_tank), (ii) the per-state errors of the exact-model EKF
and HyperKKL on quad_tank and vdp, used to debug the bound's aggregation (the CRB bounds the pooled
RMS, not the mean of per-episode RMS; the 64-episode pooled EKF error on quad_tank x3 is 0.0240 against
the bound 0.0246), (iii) the quad_tank agent's PREREG.md. No test system's achieved per-state error and
no cross-system rank statistic had been looked at. Later changes are appended as timestamped addenda.

## 1. Quantities (fixed)

All from `budgets.json` (`budgets.py`, read-only on the cached evaluation episodes of C1-where / TEST-C1,
tier `id`, 64 episodes, 12 settled time points, prior = climatology covariance of the settled states):

* `B(W)`: Bayesian posterior CRB (Tichavsky-Muravchik-Nehorai 1998; information terms averaged over
  the episode distribution) from the last `W` samples of `(y, u)`, in sigma-normalised units;
  `W in {1, 2, 3, 5, 10}/alpha` and `full` (all data since t = 0).
* `bits(W) = 1/2 log2 det(Sigma_x) - 1/2 log2 det(B(W))`: information about the current state beyond
  climatology (exact mutual information in the linear-Gaussian case).
* `A := unmeasured_lb(3/alpha)`: mean over unmeasured states of the per-state CRB from a window of
  `3/alpha` seconds (the latent filter's weight on older data is below e^-3 = 5%). PRIMARY budget scalar.
* `rho := A / unmeasured_lb(full)`: window deficit (precision lost by a fading-memory observer).
* `bound_full_j := crb_state_j(full)`: per-state lower bound on the pooled RMS of ANY estimator with
  unbounded memory (the quantity every achieved error is checked against).
* Achieved errors (`achieved.json`): settled NRMSE from the stored runs, condition clean, mean over
  seeds. Two aggregations: `nrmse` (mean of per-episode NRMSE, the paper's metric) and `rms` (pooled
  RMS over episodes, the quantity a CRB bounds). `unmeasured` = mean of per-state NRMSE over unmeasured
  states (the paper's `nrmse_unmeasured`).
* Methods: HyperKKL = `lora` (r = 4). Full-knowledge runs (C1-where / TEST-C1): lora, latent, concat,
  film, hyper, ekf (exact model). Partial-knowledge runs (C3-regimes / TEST-C3): lora, hyper, ekf, ukf,
  greybox_ekf, greybox_ukf, greybox_ekf_oracle, ... "best grey-box" = min over greybox_ekf / greybox_ukf
  / greybox_ekf_oracle of the seed-mean (oracle pick favouring the baseline, as C3-regimes).

## 2. Grouping thresholds (fixed now)

Using `A` and `rho` of the `id` evaluation distribution:

* **rich**: `A < 0.02` and `rho < 1.25`;
* **window-starved**: `rho >= 1.4` (a fading-memory observer loses >= 40% of the attainable precision
  on the unmeasured states);
* **absolutely starved**: `A >= 0.03` and `rho < 1.4` (poor even with unbounded memory);
* **marginal**: everything else.

Values and resulting groups (from `budgets.json`, eval id; `A`, `rho`):

| system | A | rho | group |
|---|---|---|---|
| battery | 0.0084 | 2.31 | window-starved |
| bioreactor | 0.0085 | 1.00 | rich |
| duffing | 0.0113 | 2.14 | window-starved (excluded from the paper's headline counts; reported) |
| lorenz | 0.0014 | 1.00 | rich |
| quad_tank | 0.0297 | 1.50 | window-starved |
| vdp | 0.0038 | 1.00 | rich |
| cartpole | 0.0415 | 3.89 | window-starved |
| fhn | 0.0078 | 1.01 | rich |
| highway_traffic | 0.0512 | 2.75 | window-starved |
| network | 0.0079 | 1.04 | rich |
| rossler | 0.0014 | 1.00 | rich |
| two_link_arm | 0.0456 | 1.00 | absolutely starved |

quad_tank sits close to the window threshold (rho = 1.4995 before rounding); this is reported, not
hidden. There is no marginal plant under these thresholds; the dose-response "marginal" slot is
therefore taken by the absolutely-starved plant (two_link_arm).

## 3. Predictions and decision rules

**P0 (guarantee check).** For every system, every method with a stored result, both tiers (id,
sequence_ood) and every state: `rms_state_j >= 0.95 * bound_full_j`, and `rms >= 0.95 * nrmse_lb(full)`.
The 5% tolerance covers the Monte Carlo error of the bound (64 episodes, 12 time points) and the
linearisation in the nonlinear plants. PASS if zero violations; every violation is listed. Tightness
is reported as the ratio achieved / bound per system for the exact-model EKF and the best grey-box.

**P1 (rank correlation across plants).** Spearman rank correlation over the 12 plants between `A`
and the achieved `unmeasured` NRMSE (tier id):
* P1a HyperKKL (lora, full-knowledge runs): expected sign +, PASS if rho_s >= 0.5.
* P1b exact-model EKF (full-knowledge runs) against `unmeasured_lb(full)`: expected +, PASS if rho_s >= 0.7.
* P1c best grey-box (partial runs) against `unmeasured_lb(full)`: expected +, PASS if rho_s >= 0.5.
Secondary (reported, no verdict): the same with total NRMSE and with `bits(3/alpha)` (expected sign -),
and with tier sequence_ood; the 11-plant version without duffing.

**P2 (where learned observers lose to the grey-box).** Gap `g = (lora - best grey-box) / best grey-box`
on `unmeasured` NRMSE, partial-knowledge runs, tier sequence_ood (primary) and id. PASS if the median
`g` of the window-starved group (5 plants: battery, duffing, quad_tank, cartpole, highway_traffic)
exceeds the median `g` of the rich group (6 plants) AND at least 4 of the 5 window-starved plants have
`g` above the rich group's median. The absolutely-starved plant is reported separately (the mechanism
there is not the window).

**P3 (dose-response, evaluation-input excitation).** Plants quad_tank (window-starved), vdp (rich),
two_link_arm (absolutely starved); levels `s in {0.25, 0.5, 1, 2, 4}` multiplying the signal time unit
t_ref of the id families (same amplitude box, same IC box, same noise; new seeds; 32 episodes per
level); observers: the stored HyperKKL models (3 seeds, partial-knowledge runs), the node's EKF/UKF
(partial model), an exact-model EKF (`ekf_full`), grey-box EKF/UKF refitted on the run's own training
data (`baselines_data.fit_all`, as reeval). Budgets recomputed on each level's episodes with the
system's fixed climatology prior. Script `dose_response.py`; nothing is retrained.
* P3a (guarantee at every level): pooled `rms_state_j >= 0.95 * bound_full_j(level)` for every method,
  level and state. PASS if zero violations.
* P3b (information tracking by the information-optimal observer): over the 5 levels, if the full-past
  unmeasured bound varies by >= 20% (max/min) then Spearman(bound, `ekf_full` unmeasured error) >= 0.6;
  if it varies by < 20% ("flat budget"), `ekf_full`'s unmeasured error varies by < 30% (max/min).
  Theory: for a plant with a known model the state information is a Gramian property, so the
  evaluation-side excitation changes the bound only through the nonlinearity (operating points).
  Expected: near-flat on quad_tank (mild sqrt law), flat on two_link_arm, flat to mild on vdp.
* P3c (learned observers are excitation-sensitive beyond the information): on every plant the max/min
  ratio of HyperKKL's unmeasured error across levels exceeds that of `ekf_full` by at least a factor 1.5,
  and HyperKKL's error is minimal at or next to s = 1 (the training time scale) on >= 2 of 3 plants.
  Interpretation fixed now: a change of HyperKKL's error that is not matched by a change of the bound is
  a learning-side (distribution-shift) effect, not an information effect.
* P3d (window mechanism, quad_tank only, reported): the ratio HyperKKL unmeasured error / `A(level)`
  stays within a factor 2 across levels, whereas the ratio to the full-past bound is reported alongside.

**P4 (prediction for the quad_tank training-enrichment experiment, experiments/quadtank).** The
arms X1-X3 change the TRAINING inputs only; the evaluation episodes are the stored ones, so the
estimation-side bounds are unchanged: full past 0.0198 (x3 0.0245, x4 0.0151), 10/alpha 0.0201,
3/alpha 0.0297, 1/alpha 0.0517 (unmeasured NRMSE, pooled). The stored grey-box EKF (0.0185-0.0197
unmeasured, id) is within 1% of the full-past bound: it is information-optimal and no observer of any
kind can beat it by more than that. Predictions: (i) no arm brings HyperKKL's unmeasured error below
0.020 (the 10/alpha bound: the SFM context forgets at about 0.15 alpha); (ii) if the window is the
binding constraint, no arm brings it below 0.0297 (3/alpha); (iii) the gap to the grey-box that
training enrichment can close is at most from the current 0.065-0.095 down to the 0.03-0.05 band
(1/alpha to 3/alpha bounds); the remainder is the window deficit, which only a longer memory (lower
alpha, longer context) removes. Enriched-input bits: the budget of the slow-family inputs on the
evaluation episodes is unchanged by construction; what changes is the training coverage.

## 4. Scoring

`score.py` applies every rule above mechanically and writes `RESULTS.md` with PASS/FAIL per
prediction, the numbers, and every violation; nothing is dropped.

## Addendum (2026-10-26 16:20 CET / 19:20 Dubai, after scoring)

The budgets of `budgets.json` were computed under the setting of the full-knowledge runs (C1-where / TEST-C1:
white Gaussian sensor noise 0.03, exact plant). For the six DEV plants the partial-knowledge comparison
(C3-regimes, condition clean) uses the same setting, so P0-P2 are consistent there. For the six TEST plants the
partial-knowledge runs (TEST-C3) generate data with AR(1) Student-t noise (rho 0.9, nu 3) on a 30%-perturbed plant;
the budgets used for their P0 / P1c / P2 cells therefore belong to a different noise model. This was noticed after
scoring by the coordinating agent, who recomputed them post hoc (`BUDGET_FIX.md`, `budgets_c3.py`, whitened
recursion with the Student-t Fisher factor). Effect on this file's rules: A (3/alpha unmeasured floor) becomes
cartpole 0.065 (rho 6.2), highway_traffic 0.097 (rho 4.8), two_link_arm 0.020 (rho 1.0), fhn / network / rossler
unchanged. The window-starved and rich sets are unchanged; two_link_arm moves from "absolutely starved" to
"marginal" (A = 0.0201, just above 0.02). P0 on the TEST-C3 cells with the corrected bounds: 0 violations
(BUDGET_FIX.md). P2's verdicts are unchanged (same two groups). The dose-response budget for two_link_arm
(`dose_response.py`) linearised the nominal plant with white noise along the true-plant episodes (0.021-0.024
unmeasured, full past); the corrected full-past value is 0.0199, so its flat-budget verdict (P3b) stands. The
pre-registered scoring above is left as it was run.
