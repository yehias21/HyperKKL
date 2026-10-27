# Measurement memory for HyperKKL: pre-registration

Written 2026-10-26, finished 17:10 Dubai (14:10 CET), BEFORE any training or evaluation run of this
study. Inputs looked at so far: the quad_tank study (`experiments/quadtank/`), the beat_greybox
pre-registration, status and its seed-0..2 summary tables (A0-A4, D1-D3; wave 2 still running),
`experiments/information/` (budgets.json, RESULTS.md) and a feasibility probe of the latent designs
below (`code/probe_washout.py`: washout times only, no training). Nothing below was chosen after
looking at a result of this study. Later changes are appended as timestamped addenda and never edit
the text above them.

## 0. Hypothesis and isolation

On window-starved plants the KKL observer loses to the grey-box filter because its measurement memory
(~1-3/alpha) is shorter than the time the plant needs to reveal its slow states (quad_tank: h3/h4
errors at the CRB of a 10-s window, grey-box at the 30-s bound). Giving the observer longer memory,
at matched parameters and steps, should move its error toward the longer-window CRB and close the gap
to the grey-box; on rich plants it should not hurt materially.

* Code: `code/pkg/` = a copy of the beat_greybox scratch package (identical files; model-module hash
  `b843eb68d0`, data-module hash `2f0779bbcc`, sha1 list `code/campaign_pkg_snapshot.sha1`). No file
  of the copy is edited; the new pieces live in `code/pkg/hyperkkl/memory.py` (not part of the
  package's hashed module list; its own sha1 enters every new cache key) and the runner
  `code/run_md.py`. The live package, the tree, the caches and the campaign's folders are never
  written (the campaign's cached A2 phase-1 models are read, i.e. copied, by key).
* Data: the C3 runs' training / validation / evaluation episodes read READ-ONLY from the live cache
  (a miss is an error). The only new data are longer zero-input phase-1 episodes for the slow
  designs (section 2.2), generated on the CPU into `memory_design/data/`.
* GPUs: 0-4 preferred (cleared by the author), 5-7 if needed; at most 2 of our processes per GPU;
  if a non-HyperKKL process appears on 0-4 we yield them and use 5-7 only.

## 1. Cells, seeds, metric, statistics (fixed)

* Window-starved plants: **quad_tank, battery** (simulated); rich controls: **vdp, bioreactor**
  (group labels from information/RESULTS.md). Duffing excluded (author decision). Scenarios
  **C3-partial, C3-mismatch30** (stored configs). 8 cells.
* Tiers: sequence_ood (**primary**), id and shape_ood (secondary); condition clean; 64 episodes per
  tier; settled NRMSE of `metrics.episode_metrics` (settle 0.5 N); per seed, the mean over episodes.
* Seeds 0, 1, 2 for every arm.
* Paired comparison: `hyperkkl.metrics.paired_relative(arm, ref)` (per-seed relative change of the
  mean episode NRMSE on the same episodes, Student-t 95% CI over 3 seeds, df = 2).
* Grey-box comparator: the STORED best grey-box per (cell, tier) = the one of greybox_ekf /
  greybox_ukf / greybox_ekf_oracle with the lowest seed-mean NRMSE in the merged C3 results.json
  (oracle pick favouring the baseline, as in the campaign).

## 2. Arms

### 2.1 Base (M0) and matching

* **M0 = campaign arm A2** (HyperKKL lora r = 4, SFM context, SOAP lr 1e-2 in phases 1 and 2, base
  decoder trainable in phase 2 at lr x 0.1, every other setting from the stored C3 config: nz, alpha,
  phase-1 6000 + 6000 steps, phase-2 3000 steps, 32 episodes x 256 times per batch, conditioner budget
  140k). M0's numbers are the campaign's stored A2 results (same code, same hashes; the quad_tank
  study reproduced them to 4 digits). Reproduction check of this harness: the campaign's cached A2
  models of one cell/seed are re-evaluated here and must match the stored A2 NRMSE (< 0.5%).
* Every memory arm is built on M0's recipe and changes only the memory mechanism. **Same
  optimiser steps** in both phases. **Matched phase-2 trainable parameters**: conditioner + context +
  base decoder (+ history MLP or pole-scaling net) within 1% of M0's total (181,804 on quad_tank;
  per cell as stored): the conditioner budget passed to `match_width` is 140k minus the extra
  module's parameters. Phase-1 encoder / decoder sizes are unchanged. Reported per arm: every
  module's parameters, steps, wall-clock and GPU-hours.
* **Reference rows (no verdict)**: the campaign's stored A4 (S + P + H + R; the best stack) and,
  if its seeds 0-2 land before our analysis, A4+D3; A3 for comparison with M3.

### 2.2 Latent designs with longer memory (M1, M2) and how they are labelled

* **M1, slower poles**: the Butterworth design of the config with every pole divided by 3
  (`beat.custom_design(alpha_scale = 1/3)`), same n_z and B.
* **M2, log-spaced poles (HiPPO/S4-like)**: same n_z and B; the pole groups (conjugate pairs / real
  pole) of the Butterworth design, ordered slowest first, are scaled by 10^{-(m-1-k)/(m-1)}, i.e.
  the slowest group moves to alpha/10, the fastest stays where it is, the others are log-spaced in
  between (`custom_design(alpha_scale = 0.1, spread = 10)`). Real parts: quad_tank (0.01, 0.086,
  0.373), battery (0.15, 0.91, 2.82, 6.74), vdp (0.15, 1.24, 4.85), bioreactor (0.05, 0.41, 1.62).
  The SFM context's pole bank follows the design's eigenvalues (as in the package; part of the design).
* **How M2 differs from the campaign's D3** (multi-rate, helpful on 10/12 cells in wave 1): D3 keeps
  the default KKL latent and ADDS a second Butterworth bank (n_z more states, poles / 5) outside the
  KKL map, read only through an MLP into the hypernetwork context (extra parameters, no phase-1
  change). M2 has NO extra state: the slow modes are INSIDE the KKL latent, so the phase-1 encoder
  T0 / base decoder T0+ are re-learned for the log-spaced design and the decoder reads the slow
  modes directly; parameters are matched.
* **Labelling (needed because the washout of the slow designs exceeds the episodes).** The washout
  time `design.washout_time` (kappa 1, eps_label of the config) is 0.36-0.89 T for M1 and 1.3-3.05 T
  for M2 (probe), so the stored phase-1 episodes cannot be labelled. Therefore:
  1. Phase 1 (both M1 and M2): 400 new zero-input episodes per (cell, seed), generated exactly as
     `data.training_data` does for `p1` (zero family, k_cross 1, IC box = domain, data seed
     100000 (seed + 1) + 1, true plant of the scenario) but on the CPU and with length
     T_p1 = 1.05 t_burn(M2) + (T - t_burn(base)); labels z by the exact latent step of the arm's design
     from z = 0, burn = ceil(t_burn(arm) / dt). The labelled window per episode is then at least as
     long as the base's.
  2. Phase 2: the training / validation episodes are the stored ones. Their latents are recomputed
     with the arm's design from the **deployment initial condition** z_0 = T0(x_center) (x_center =
     the centre of the IC box = the evaluation prior "center"; T0 the arm's phase-1 encoder), i.e.
     exactly the latent the deployed observer computes. Phase-2 samples are drawn at k >= the base
     burn (same window as M0). (M0's latents start at 0 but are washed out at its burn, so this is
     the same rule in the limit of fast poles.)
  3. Model selection and the deployed observer are unchanged (`KKLObserver`, z_0 = T0(x0_hat)).

### 2.3 Plant-specific memory scale W* (a priori, from the information budget)

W* = the smallest window in {1, 2, 3, 5, 10}/alpha at which the Bayesian bound on the unmeasured
states (`information/budgets.json`, training distribution, `unmeasured_lb`) is within 25% of the
full-past bound (where the information stops improving):

| plant | alpha | full-past unm. bound | bound at windows 1, 2, 3, 5, 10 /alpha | W* |
|---|---|---|---|---|
| quad_tank | 0.1 /s | 0.0198 | 0.052, 0.037, 0.030, **0.024**, 0.020 | 5/alpha = 50 s |
| battery | 1.5 /min | 0.0037 | 0.037, 0.014, 0.0087, 0.0055, **0.0039** | 10/alpha = 6.67 min |
| vdp | 1.5 /s | 0.0038 | 0.0051, **0.0040**, 0.0038, ... | 2/alpha = 1.33 s |
| bioreactor | 0.5 /s | 0.0086 | **0.0086** (flat) | 1/alpha = 2 s |

### 2.4 Arms

| arm | memory mechanism | design / phase 1 | extra trainable params (taken from the conditioner budget) |
|---|---|---|---|
| M0 | none (A2) | default | - |
| M1 | all poles x 1/3 | M1 design, new phase 1 | 0 |
| M2 | log-spaced poles [alpha/10 .. fastest] | M2 design, new phase 1 | 0 |
| M3 | strided z-window spanning W* into the hypernetwork context | default (A2's phase 1) | history MLP |
| M4 | M2 + context-chosen pole scaling s(c) in [1/10, 1] | M2's phase 1 | pole-scaling net |
| M5 | M2 + M3 (window of the M2 latent spanning W*) | M2's phase 1 | history MLP |

* **M3 window**: the campaign's H channel (`beat.HistoryFeatures`, K = 20 samples
  [z_k, z_{k-s}, ..., z_{k-19 s}] / z_std, zeros before the start, bias-free tanh MLP 20 n_z -> 64 ->
  16, concatenated to the SFM context), with stride s = max(1, round(W* / (19 dt))): quad_tank 53
  (50 s), battery 24 (6.7 min), vdp 3 (1.2 s), bioreactor 2 (2.3 s). It differs from A3 (span 6/alpha
  for every plant, extra parameters) by the plant-specific span and the parameter matching. The
  window feeds the hypernetwork context, not the base decoder, for the reason given in the campaign
  PREREG section 7 (the base decoder is the autonomous phase-1 map). Causality is structural (lags
  >= 0) and unit-tested.
* **M4, context-chosen poles**: the latent of M2 is time-warped by a scalar s_k:
  z' = s(t) (A z + B y), discretised exactly for y held over a step,
  z_{k+1} = Ad(s_k) z_k + (Ad(s_k) - I) A^{-1} B y_k, Ad(s) = exp(s A dt).
  s_k = EMA_{rate alpha}( s_min + (1 - s_min) sigmoid(g(c_k)) ), s_min = 1/10, g a tanh MLP
  dc -> 16 -> 1 of the SFM context c_k (a causal function of u_0..u_k), output bias initialised to +3
  (s ~ 0.96, i.e. M4 starts at ~M2), EMA = first-order low-pass with rate alpha (rate limit). The
  latent is trained end to end through the scan (parallel associative scan over the episode) with the
  conditioner and base decoder; z_0 = T0(x_center) in training, T0(x0_hat) deployed.
  **Contraction**: A is in real Jordan form with kappa = 1, so its symmetric part is
  diag(Re lambda_i) <= -alpha_M2 I (alpha_M2 = alpha/10). For any measurable s(t) in [s_min, 1],
  two latents driven by the same y satisfy d/dt |dz|^2 / 2 = s(t) dz' A dz <= -s_min alpha_M2 |dz|^2,
  so the latent contracts uniformly at rate >= alpha/100 whatever the switching speed of s (the
  rate limit is for smoothness / identifiability, not needed for contraction); in discrete time
  ||Ad(s)||_2 = max_i exp(s Re(lambda_i) dt) < 1. Unit-tested. The price: M4's latent is no longer
  the KKL map of a fixed design, so the phase-1 theory applies only at s = 1.
* **M5**: M2's design and phase 1, plus the M3 window of the M2 latent (stride as M3).
* No hyperparameter is tuned inside any arm (K, strides, s_min, EMA rate, init are fixed above).

### 2.5 Selection of the "best memory arm" (validation only)

Per cell, the best memory arm = the one of M1-M5 with the lowest seed-mean deployed validation NRMSE
(`beat.validation_nrmse` on `data.val`, centre prior). The "best fixed-memory arm" = the same choice
among M1, M2, M3, M5. No evaluation episode enters any choice.

## 3. Pre-registered predictions and mechanical decision rules (primary tier sequence_ood)

Per state NRMSE n_j = per-seed mean over episodes of `nrmse_x{j}`; for M0 on quad_tank they are taken
from the quad_tank study's X0 runs (= A2, identical to 4 digits). Upper-tank error per seed
e_up = sqrt((n_3^2 + n_4^2) / 2). CRB_30 = the 30-s empirical-Gramian CRB of DIAG.md (c) (median
over training windows, x_std units): C3-partial h3 0.0172, h4 0.0135; C3-mismatch30 h3 0.0174, h4
0.0170 (`quadtank/s1.json`).

* **P-M1** (quad_tank, best memory arm per scenario): PASS iff in BOTH scenarios (a) the paired
  relative change of e_up vs M0 is <= -30% with its 95% CI entirely below 0, AND (b) the seed-mean n_3
  <= 1.5 x CRB_30(h3) and n_4 <= 1.5 x CRB_30(h4).
* **P-M2** (window-starved cells: quad_tank, battery x 2 scenarios): a cell succeeds iff the best
  memory arm is significantly better than M0 (paired CI entirely below 0) AND, if M0 loses to the
  grey-box on that cell (seed-mean e_M0 > e_GB), the gap closure g = (e_M0 - e_best) / (e_M0 - e_GB)
  >= 0.5; if M0 already beats the grey-box there, e_best < e_GB. PASS iff >= 3 of 4 cells succeed.
* **P-M3** (rich cells: vdp, bioreactor x 2 scenarios): PASS iff for every memory arm M1-M5 and every
  rich cell, the upper end of the 95% CI of paired_relative(arm, M0) is <= +10% (16 + 4 = 20
  comparisons; all must hold). Also reported: the count that hold and the point estimates.
* **P-M4**: PASS iff M4 is not significantly worse than the best fixed-memory arm (paired CI of
  M4 vs that arm not entirely above 0) AND its point estimate is <= +5%, on >= 3 of 4 window-starved
  cells AND on all 4 rich cells.

Authors' expectations (stated, not tested): P-M1 and P-M2 are the hypothesis; given that the campaign
found the multi-rate bank D3 helpful and the window H strongly helpful on quad_tank (A3 -67% vs A0),
we expect M3/M5 to carry most of the gain, M1 to trade transient for memory, and M2/M4 to be limited
by the slow initial-error decay (the slowest mode forgets the prior over 10/alpha, longer than the
quad_tank / battery settle windows).

## 4. Secondary (reported, no verdict)

id and shape_ood for all of the above; every arm vs M0 and vs the stored best grey-box on all 8 cells
x 3 tiers; per-state NRMSE on quad_tank; HyperKKL vs Bayesian windowed bounds of budgets.json;
M4's realised s(t) statistics; phase-1 diagnostics (eps_dec, l_dec_q95, noise gain); parameters,
wall-clock, GPU-hours. Optional, if time permits after the primary analysis and clearly labelled
post-primary: the validation-selected best memory arm applied on top of the A4 recipe on the
window-starved cells (the "best stack"), against stored A4 and, if available, A4+D3.

## 5. Tests before the full run

Unit tests (`code/test_memory.py`): (i) causality of the strided window (perturbing y/u after k
leaves xhat_k unchanged) and training gather == deployed ring buffer; (ii) M4: ||Ad(s)|| < 1 on an s
grid, two latents from different z_0 contract at >= the stated rate under a random fast-switching
s, and the parallel scan == the deployed step-by-step recursion; (iii) relabel with z_0 == the
deployed latent of `KKLObserver`; (iv) phase-2 trainable parameter counts of M1-M5 within 1% of M0
on every plant. Then a smoke run (20 steps) of every arm on quad_tank and vdp.

## Addendum 1 (2026-10-26 17:35 Dubai, before the first real run; unit tests + smoke runs only)

Implementation facts fixed during testing (smoke = 40/40 phase-1 and 20 phase-2 steps, 24 phase-1
episodes; its results are discarded and live under `runs/smoke/`). None changes the design above.

* Unit tests: 14/14 pass (`code/test_memory.py`): window causality and training gather == deployed
  ring buffer; M4 ||Ad(s)||_2 = exp(s max Re(lambda) dt) < 1 on an s grid and contraction at
  >= s_min alpha_M2 under random per-step switching of s (3000 steps, all four plants); the M4 modal
  step == matrix-exponential ZOH step; the M4 parallel scan == the deployed recursion (s varying,
  max abs error < 2e-4) and causal; gradients reach the pole-scaling net; relabel with z0 == the
  deployed `LatentStep` latent; parameter matching.
* Matched phase-2 trainable parameters (conditioner + context + base decoder + extra), vs M0:
  quad_tank M3/M5 -0.15%, M4 +0.16%; battery M3/M5 +0.54%, M4 +0.16%; vdp and bioreactor M3/M5
  -0.84%, M4 +0.16%; M1/M2 identical (the conditioner width is an integer; this is the closest).
* Long phase-1 episodes generated for all 24 (cell, seed) on the CPU (400 each; T_p1 = 1152 s
  quad_tank, 97 min battery, 112 s vdp, 284 s bioreactor; 0-17 of 400 episodes re-drawn by the
  data module's own rejection rule, as for the stored sets).
* M0 is re-evaluated in every job from the campaign's cached A2 models (copied by key); its
  numbers must equal the stored A2 numbers (the reproduction check of section 2.1, on all cells).
* Job groups per (cell, seed): {M0, M3}, {M1}, {M2, M5, M4}; dispatcher `code/dispatch_md.py`
  (2 processes per GPU on GPUs 0-4, which are shared with other agents' HyperKKL jobs; fallback 5-7
  if a non-project process appears on 0-4).
* Coordinator note received 16:5x Dubai (campaign wave 1: D3 helpful 10/12, wave 2 running
  A2+D3/A3+D3/A4+D3): M2's difference from D3 is stated in 2.2. The primary base stays A2 (the arms
  must be built on a base whose models exist for all cells now); A4 and A4+D3 are reference rows,
  and the optional post-primary pass of section 4 is where a memory arm meets the best stack.
