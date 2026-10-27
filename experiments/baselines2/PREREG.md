# Learned, recipe-independent baselines (baselines2): pre-registration

Written 2026-10-27 00:50 Dubai (Asia/Dubai, UTC+4), BEFORE any training, evaluation or smoke run of this
study. Later changes go into timestamped addenda at the end; the text above them is never edited.

Purpose: close REVIEW_COVERAGE C3 (no generic learned sequence observer, no neural-ODE/UDE filter) and
C4 (NLOX under-sized) with comparators that do not depend on the HyperKKL recipe, so they can be dropped
unchanged into the consolidated re-run.

## 0. Isolation

* New code only, under `experiments/baselines2/code/` (package `b2`). The live package `hyperkkl/` is
  imported read-only and never edited (no cached-model hash changes). Nothing is written to
  `experiments/tree`, the queue, or `experiments/cache/models`.
* Data: training / validation / evaluation episodes come from `hyperkkl.data` with the run config stored
  in the tree run's own `results.json` (`experiments/tree/<node>/runs/<system>/seed<k>/results.json`;
  seeds 3-4 use the seed-0 config with `seed` changed). Files present in the live data cache are read
  read-only (so seeds 0-2 see exactly the episodes every stored observer was scored on); a missing file
  (seeds 3-4 training data) is built by the unchanged `hyperkkl.data` code into
  `experiments/baselines2/cache/` (never into the live cache).
* GPUs: 0-7 allowed. Before each launch `nvidia-smi` is checked; if a process whose command line does
  not contain `TMLR/hyperkkl` appears on GPUs 0-4, no new job goes to 0-4 (running ones finish). At most
  4 of our processes per GPU.

## 1. Cells, seeds, tiers, metric

* Dev: battery, bioreactor, duffing, lorenz, quad_tank, vdp in **C3-partial** and **C3-mismatch30** (12
  cells). duffing is reported in every table, and every dev count is given both with and without duffing
  (duffing is excluded from headlines by author decision).
* Held-out test: **TEST-C3** fhn, highway_traffic, network, rossler, two_link_arm and **TEST-C3-B**
  cartpole_b (cartpole_b replaces cartpole after the pre-registered inclusion check in
  `experiments/cartpole_bounded/`). 6 cells. Test plants: no per-plant choice of anything except the
  validation-split lr selection that every baseline gets on every cell.
* Seeds 0-4 for every baseline and for the re-fitted grey-box filters.
* Tiers id, shape_ood, sequence_ood; condition `clean` (the node's own sensor: white noise on dev,
  coloured Student-t on TEST-C3); 64 episodes per tier; **primary tier: sequence_ood**.
* Metric: settled NRMSE of `hyperkkl.metrics.episode_metrics` (settle = 0.5 N, divergence clamp as the
  tree), per seed the mean over the 64 episodes. Secondary: unmeasured-state NRMSE, divergence rate.

## 2. Baselines (all trained on the same labelled data: `data['train']` + `data['p1']` where a method
uses transitions, `data['train']` streams where it uses sequences; selection on `data['val']` only)

Every observer follows the package contract (`observers.Observer`): x̂_k uses y_0..y_k and u_0..u_k only.

| id | baseline | architecture | sizes |
|---|---|---|---|
| `gru` | GRU observer (direct sequence model) | 2-layer GRU on [u_k / u_scale, (y_k - ȳ) / y_std], linear read-out to (x - x̄) / x_std; h_0 = 0 (the prior x̂_0 is not used, as for NLOX whose z_0 = 0) | ≈140k, ≈500k (hidden width solved for the budget) |
| `ncde` | Neural CDE observer (Kidger, Morrill, Foster & Lyons, NeurIPS 2020, "Neural Controlled Differential Equations for Irregular Time Series") | dz = f_θ(z) dX, X = (t / t_ref, u / u_scale, normalised y) linearly interpolated between samples (so the step k → k+1 uses y_{k+1}, which x̂_{k+1} may use); z_0 = ζ(X_0) linear; f_θ = MLP d_z → W → W → d_z·d_X (SiLU, tanh output); one RK4 step per sample interval (torchcde is not installed offline; minimal implementation, documented); linear read-out; d_z = 32 | ≈140k (W solved) |
| `node_ekf`, `node_ukf` | Neural ODE + EKF / UKF | ẋ = f_θ(x, u), no physics; f_θ = the grey-box residual MLP class (standardised in/out, 2 hidden SiLU layers); h known (y = x[sensors]); trained by multi-step RK4 rollouts (below); used inside the unchanged `observers.EKF` / `UKF` with the grey-box filters' Q, R, P_0 | width 64 (≈5k, = grey-box residual) and width solved for ≈140k |
| `ude_ekf`, `ude_ukf` | Hybrid UDE + EKF / UKF | ẋ = f_known(x, u) + r_θ(x, u) (the modeller's model `data.model_plant`, as the grey-box), multi-step-trained; same filters / Q, R, P_0 | ≈5k and ≈140k |
| `nlox` (capacity-matched) | `train.nlox` unchanged, only `nlox.width` and `nlox.dec_width` scaled by a common factor | stored NLOX is 38.9k-41.8k params (width 64, dec 128, depth 3) | ≈140k and ≈500k |
| `greybox_*` (re-fit) | `baselines_data.fit_all` unchanged (greybox_ekf, greybox_ukf, greybox_ekf_oracle with `job.tune_oracle`) | re-fitted on every seed 0-4 so seeds 3-4 exist; seeds 0-2 must reproduce the stored tree numbers (reproduction check, reported) | ≈5k |

Training (fixed for all cells):
* Sequence models (gru, ncde): stateful truncated BPTT exactly as `train.nlox`: 32 streams, K = 256 steps
  per optimiser step, the hidden state carried (detached) across windows and restarted on a fresh episode;
  loss = mean over steps k ≥ burn (the episode set's washout index, as NLOX) of ||(x̂_k - x_k)/x_std||²;
  Adam, 3000 steps (= HyperKKL phase-2 / NLOX budget), warm-up 100 + cosine (`train._sched`), grad-norm
  clip 1.0; checkpoint = best validation loss (full validation episodes from h_0 = 0, steps ≥ burn),
  checked every 250 steps.
* Dynamics models (node, ude): windows of K = 32 consecutive training samples (train + p1 episodes);
  the model is rolled out from the true x at the window start with one RK4/ZOH step per sample (as
  `baselines_data.residual_targets`); loss = mean over the window of ||(x̂ - x)/x_std||²; first 300 steps
  with K = 1 (the grey-box one-step objective, as a warm start), then K = 32; batch 1024 windows; Adam,
  3000 steps, warm-up 100 + cosine, grad clip 1.0; checkpoint = best K = 32 validation rollout loss,
  every 100 steps. Output standardisation as the grey-box (target mean / std of the one-step targets).
* NLOX: `train.nlox` unchanged (RMSprop, 3000 steps, K = 256, its own validation checkpointing).
* **lr grid (4 values, every baseline): {3e-4, 1e-3, 3e-3, 1e-2}.** The lr is selected per (cell, seed)
  by the deployed observer's settled NRMSE on the 64 validation episodes (+ divergence rate, as
  `job.tune_oracle`); for node/ude the EKF and the UKF each pick their own lr. Evaluation episodes are
  never used for any choice.

## 3. What is reported (mechanically)

Per baseline × cell × tier (all three tiers; sequence_ood primary):
* (a) vs the **best grey-box filter**: the one of greybox_ekf / greybox_ukf / greybox_ekf_oracle (re-fit,
  seeds 0-4) with the lowest seed-mean NRMSE on that cell and tier (oracle pick, favours the grey-box).
  Seeds 0-4.
* (b) vs **HyperKKL**: stored A0 `lora` (tree runs, seeds 0-2), and, where stored, the campaign stack
  **A3** and **A4+D3** (`experiments/beat_greybox`: dev cells A3 and A4+D3; TEST-C3 and cartpole_b
  A4+D3). Common seeds 0-2.
* Statistic: per seed r_s = (mean_baseline - mean_ref) / mean_ref on the same episodes
  (`metrics.paired_relative`), mean and Student-t 95% CI over seeds; p-value = one-sample t-test of r_s
  vs 0; **Holm correction across the plants of one family** (family = one baseline × one comparator × one
  scenario × one tier; dev families have 6 plants, also reported as 5 without duffing; test family 6
  plants). "Beats" = Holm-adjusted p < 0.05 and mean r < 0; "worse" = Holm-adjusted p < 0.05 and r > 0;
  otherwise "tie".
* Parameter counts of every baseline (trained parameters; for the filters also the filter's own
  constants), wall time, chosen lr.
* No pass/fail rule for HyperKKL.

## 4. Predictions (written before any run; scored mechanically in RESULTS.md)

* P1. The **strongest learned baseline** (lowest median rank of sequence_ood seed-mean NRMSE over the 12
  dev cells) will be **UDE + EKF/UKF** (either size; the best of its four variants per cell counts as one
  entry only if it is the same size and filter for every cell -- otherwise each variant is ranked
  separately and P1 is scored on the best single variant).
* P2. Among the pure sequence models, the **GRU (140k) beats the Neural CDE (140k)** on more dev cells
  than the reverse (seed-mean sequence_ood NRMSE).
* P3. On the window-starved plants **quad_tank, battery (dev, both scenarios) and highway_traffic
  (test)**, **no learned baseline beats the best grey-box filter** (Holm, sequence_ood).
* P4. NODE + filter (no physics) is worse than the best grey-box filter (Holm) on at least 8/12 dev cells.
* P5. Capacity-matched NLOX-140k is better than the stored 40k NLOX (seed-mean, sequence_ood) on at most
  half of the dev cells, i.e. capacity is not what limits NLOX.
* P6. HyperKKL A4+D3 beats the GRU-140k (Holm, sequence_ood) on at least 6/12 dev cells.

## 5. Real battery (secondary, if feasible)

GRU and Neural CDE on the real-cell split of `experiments/real` (hyperkkl/realdata_run.py: same
episodes, masked loss, settled SOC RMSE), seeds 0-2, same lr grid selected on the validation cycle;
reported next to the stored grey-box UKF/EKF and HyperKKL numbers. No rule.

## Addenda

### Addendum 1 (2026-10-27 02:35 Dubai) -- implementation speed only, no change of protocol
The node/ude multi-step training loop ran launch-bound (~1.5 s per K = 32 step under the full load), i.e.
~5 h per job. It now replays each K's loss + backward as one CUDA graph (hyperkkl.train._Graphed, as
NLOX), with static index buffers. Checked before use: graph and eager training give bit-identical
validation losses (battery, UDE and NODE, 360 steps) and 5-6x speed; capture works on every plant except
network, which keeps the eager loop. The 23 ude5k jobs that had started (none finished) were killed and
re-run from scratch with the new loop. Nothing was looked at except training logs (no evaluation output
of any node/ude job existed).

### Addendum 1b (2026-10-27 06:44 Dubai) -- implementation speed only
The network plant was not capturable (its Laplacian is rebuilt on every f call); the job now caches that
constant matrix per dtype/device on its own model instance (identical values, checked), so network also
uses the graphed loop. Its 5 ude5k jobs (still training, no result) were restarted.


### Addendum 2 (2026-10-27 09:57 Dubai) -- post-hoc control, written after seeing interim ude5k results
Interim tables showed ude5k (multi-step) far below the one-step grey-box on many cells. The audit
(AUDIT_UDE.md) found no leak, but the comparison confounds two things: the objective (K = 32 rollouts vs
one-step) and tuning (ude5k gets the 4-value lr grid + deployed-validation selection and picked lr 1e-2
almost everywhere; the stored grey-box uses a fixed lr 1e-3 with early stopping). Control added,
labelled POST-HOC in every table: **ude5k1** = exactly ude5k (same MLP, data, lr grid, selection, step
budget, filters) but with the one-step objective (K = 1 throughout; checkpoint on the K = 1 validation
loss). ude5k1 vs grey-box isolates tuning/implementation; ude5k vs ude5k1 isolates the multi-step objective.
Seeds 0-4, all 18 cells. No prediction is attached to it.
