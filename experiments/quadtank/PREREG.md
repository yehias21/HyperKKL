# quad_tank: is HyperKKL limited by persistent excitation? (Stage 2 pre-registration)

Written 2026-10-26, finished 11:20 CET (14:20 Dubai), AFTER the Stage-1 diagnostics (`DIAG.md`,
which use only training data, stored C3 results and the campaign's A2 seed-0 result) and BEFORE
any Stage-2 training or evaluation run. Nothing below was chosen after looking at a Stage-2
result. Later changes are appended as timestamped addenda and never edit the text above them.

## 0. Question and isolation

Question: does HyperKKL lose to the grey-box filter on quad_tank because the training inputs do
not persistently excite the slow, unmeasured upper-tank directions (PE-limited), or because those
directions are only weakly observable through the measured lower tanks / the latent (observability-
limited)? Intervention: change ONLY the training data; everything else (model, recipe, steps,
evaluation episodes, validation episodes) is fixed.

* Code: scratch copy of the beat_greybox package (itself a copy of the live package + SOAP + the
  P option + `beat.py`), at
  `/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/quadtank/pkg/`.
  The package's data modules (`signals.py`, `data.py`, ...) are NOT modified, so the original
  episodes are read from the live cache under their own keys. The new input families live in
  `quadtank/qt_signals.py` (outside the package) and are routed into `data.generate` by replacing
  its module-level `_signals` in-process. The runner is `quadtank/run_qt.py`. The live package,
  `experiments/tree`, the queue and the caches are never written; new training sets are stored in
  `quadtank/data/` in the scratchpad.
* GPUs 5, 6, 7 only, at most 2 of these processes per GPU, < 6 GB each (shared with the running
  beat_greybox campaign and tuning worker).

## 1. Cells, seeds, metric, statistics (fixed)

* System quad_tank; scenarios **C3-partial** and **C3-mismatch30** (the stored C3 run configs).
* Evaluation: the C3 evaluation episodes, unchanged (live cache, 64 episodes per tier), tiers id,
  shape_ood, sequence_ood (**primary: sequence_ood**), condition clean, settled NRMSE of
  `metrics.episode_metrics` (settle = 0.5 N), per seed the mean over the 64 episodes.
* Seeds 0, 1, 2 for every arm.
* Comparison: `hyperkkl.metrics.paired_relative(arm, reference)` (per-seed relative change of the
  mean episode NRMSE on the same episodes, Student-t 95% CI over the 3 seeds, df = 2).

## 2. Model recipe (identical in every arm) = campaign arm A2

The beat_greybox campaign has no selected final arm yet (its verdict needs seeds 0-2 of all
cells; seeds 1-2 are running), so X0 uses the task's fallback recipe, which is exactly the
campaign's arm **A2**: HyperKKL lora r = 4, SFM context, SOAP (phase 1 and phase 2, lr 1e-2,
archive settings), base decoder trainable in phase 2 at lr x 0.1 (P1), every other setting from
the stored C3 config (phase-2 3000 steps, 32 episodes x 256 times per batch, budget 140k, nz 6,
alpha 0.1, dt 0.05). Phase 1 AND phase 2 are retrained on each arm's training set (phase 1 uses
the forced training episodes for its latent scale and as half of the decoder's trajectory
states). The number of optimiser steps is the same in every arm (more data = more diversity, not
more compute). Model selection: phase-2 validation loss on the ORIGINAL validation set
(`data.val`, 64 episodes, T = 300 s, original families) in every arm.

## 3. Arms (only the forced training set `data.train` changes)

Fixed in every arm: `data.p1` (400 zero-input phase-1 episodes), `data.coll` (collocation
states), `data.val`, the system scales (domain, x_std, washout t_burn = 86.7 s -> burn = 1735
samples), the latent design, the sensor noise (3%) and process noise (0.005 x_std / sqrt(s)),
k_cross = 3, the rejection envelope. Training seed s: base = 100000 (s + 1) as `data.training_data`.

| arm | training set | episodes | T | data seed |
|---|---|---|---|---|
| X0 | the C3 training set (live cache), 11 training families | 600 | 300 s | stored (base + 2) |
| X1 | the 11 training families, longer episodes | 600 | 900 s (T x 3) | base + 21 |
| X2 | slow-band re-weighting at the same size and length: episodes 0-299 of the X0 set (11 families) + 300 new slow-family episodes | 600 | 300 s | X0 / base + 22 |
| X3 | X1 + X2 + 2x episodes: X1's 600 episodes + 600 slow-family episodes at T = 900 s | 1200 | 900 s | base + 21 / base + 23 |

Slow families (`qt_signals.py`, TAU = 49 s = the config's median slow time constant; the
slowest linear mode is 84 s). Every family stays in the SAME amplitude box v(t) in [-1, 1]
(u = u_scale v, u_scale = 1) by construction, like the training families; families are drawn in
rotation with the first parameter stratified, as `data._signals`:

* `slow_staircase`: piecewise-constant, levels U[-1, 1], dwell U[1, 3] TAU (49-147 s).
* `slow_ramp`: piecewise-linear random walk, knot values U[-1, 1], knot spacing U[1, 3] TAU
  (long ramps between uniform levels; random phase of the first knot).
* `slow_multisine`: offset + sum of n in {2, 3, 4} sinusoids with frequencies U[0.2, 1] / TAU rad/s
  (all below the slow corner 1/TAU = 0.0204 rad/s), amplitudes Dirichlet x A, A U[0.1, 1],
  offset U[-1, 1] (1 - A); stored as 1-s knots (interpolation error < 1e-4).

The realised spectra / coverage of X1-X3 are reported with the Stage-1 measures (same code).
The episode set X1 is exactly the first half of X3 (nested design).

## 4. Grey-box comparators

* **Stored** (existing comparators): greybox_ekf, greybox_ukf, greybox_ekf_oracle per-episode NRMSE
  from the merged C3 `results.json`, seeds 0-2.
* **Retrained on each arm's data** (fair comparison): `baselines_data.fit_all` with
  `baselines = [greybox_ekf, greybox_ukf, greybox_ekf_oracle]` on the arm's `data` (its `train` +
  the fixed `p1` transitions, 1M-transition subsample cap and every other option as stored; the
  fixed `val` for early stopping), evaluated on the same episodes. GB-X0 (refit on the original
  data) doubles as the reproduction check against the stored C3 numbers.
* "Best grey-box" per (scenario, tier, data arm) = the one of the three with the lowest seed-mean
  NRMSE (oracle pick favouring the baseline, as C3-regimes and the campaign).

## 5. Predictions and decision rule (stated now)

Primary quantity, per scenario: r_k = paired_relative(HyperKKL X_k, HyperKKL X0) on sequence_ood,
k = 1, 2, 3. Gap closure g_k = (e_X0 - e_Xk) / (e_X0 - e_GB*) with e_GB* the best STORED grey-box
(also reported against the best grey-box retrained on X_k).

* **PE-limited** if, in BOTH scenarios, min(r_2, r_3) <= -30% with its 95% CI entirely below 0,
  and the corresponding g >= 0.5 (most of the gap to the grey-box closed).
* **Observability-limited** if, in BOTH scenarios, |r_2| < 10% and |r_3| < 10% (point estimates).
* Anything else is **mixed / partially PE-limited**, reported with the per-scenario numbers (e.g.
  10-30% gains, or a gain in one scenario only), and attributed with X1 (length alone) vs X2
  (slow content alone) vs X3.
* Sanity: an arm that makes HyperKKL significantly WORSE (CI above 0) is reported as such; it
  does not count as evidence for either hypothesis.

Author's prediction: if PE-limited, X2/X3 cut the error by >= 30% and close most of the gap; if
observability-limited, < 10% change.

Our prediction from Stage 1 (DIAG.md), stated before any run: **observability-limited** (|r| <
10% for X2 and X3), because (i) the evaluation states lie inside the training support (1-4% of
scored states beyond the 99th percentile of held-out validation distances), (ii) the training
inputs already carry MORE sub-corner energy (13.6% of the fluctuation energy below 1/49 s) and
more independent upper-tank variation than the primary tier sequence_ood (2.7%), (iii) the
per-episode error is uncorrelated with coverage (|Spearman| <= 0.13), whereas (iv) HyperKKL's
error is concentrated on x3 / x4 at about the level of the Cramer-Rao bound of a 10-s (= 1/alpha)
window of measurements, and the x3/x4 information sits ~40% in the three weakest latent
directions (variance 3e-6..6e-5 of the strongest). A possible exception we flag in advance: X1/X3
(longer episodes) could help through a different channel than PE (more labelled samples after the
washout, slower ramps), which would show as X1 ~ X3 >> X2.

## 6. Secondary (reported, no verdict)

id and shape_ood; per-state NRMSE (x1..x4) of HyperKKL and grey-box per arm; HyperKKL vs best
grey-box (stored and retrained) per arm (paired_relative); grey-box change with data
(GB-Xk vs GB-X0); phase-1 diagnostics (eps_dec, l_dec_q95); validation NRMSE; wall-clock; the
X0 reproduction check against the campaign's A2 (same code and data; seeds available at analysis
time).

## Addendum 1 (2026-10-26 11:40 CET, before the first real Stage-2 run; smoke tests only)

Implementation details fixed during the smoke tests (20-step / 24-episode runs, results discarded);
none changes the design above:

* New training episodes are generated on the CPU (`data.generate(..., device="cpu")`): the per-step
  RK4 loop is launch-bound and 24 episodes took 375 s on the saturated shared GPU vs 48 s for
  24 x 18000 steps on the CPU. The device only changes the noise draws.
* The retrained grey-box residual is fitted and its three filters are evaluated on the CPU (same
  code, `baselines_data.fit_all` / `evaluate.build_observers` / `evaluate.evaluate`), on the same
  evaluation episodes (read with their CUDA cache keys). HyperKKL is trained and evaluated on GPU.
  GB-X0 (CPU refit) vs the stored C3 grey-box (GPU) is the reproduction check of this path.
* `beat.phase2_bg` drew the validation time indices with the TRAINING episode length; with 900-s
  training episodes that indexes past the 300-s validation episodes. Fixed in the scratch copy to
  `bv.times(len(va), M, va.burn, va.x.shape[1])`; identical draws whenever train and validation
  have the same length (X0, X2), so X0 remains the campaign's A2.
* Dispatcher: 2 processes per GPU on GPUs 5, 6, 7; the arms of one (scenario, seed) never run
  concurrently (X1 and X3 share the generated 900-s episodes).
