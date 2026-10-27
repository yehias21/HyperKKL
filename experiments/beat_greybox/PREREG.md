# Beat the grey-box filter: pre-registration

Written 2026-10-26, finished 10:00 Dubai (07:00 CET), BEFORE any training or evaluation run of this
campaign. Nothing below was chosen after looking at a result of this campaign. Later changes are
appended as timestamped addenda at the end and never edit the text above them.

## 0. Goal and isolation

Goal (author): make HyperKKL (the `lora` code path, rank r = 4: a hypernetwork that generates rank-r
updates of the decoder weights from the context) beat the grey-box filters (`greybox_ekf`,
`greybox_ukf`, `greybox_ekf_oracle`: the modeller's partial or nominal model + a learned residual
MLP inside an EKF/UKF, trained on the same labelled data, `hyperkkl/baselines_data.py`) on the
plants where it loses now, without losing where it wins now.

* Code: a scratch copy of the live package,
  `/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/beat_greybox/pkg/`,
  copied at 06:56 CET (sha1 of every file: `code/live_snapshot.sha1`). Into it: SOAP
  (`soap.py`, byte-identical to `experiments/archive_recipe/code/soap.py`), the P1 option of the
  pipeline study (base decoder trainable in phase 2 at lr x 0.1, re-implemented from
  `experiments/pipeline/code/hyperkkl/pipeline.py:phase2_variant`), and the new pieces below. The
  whole change ships as `code/beat_greybox.patch` + tests. The live package, `experiments/tree`, the
  queue and the caches are never written.
* Data: the C3 runs' training and evaluation episodes are read READ-ONLY from the live cache
  (`experiments/cache/data`, same keys; checked before this file was written: all 12 cells x
  {train, id, shape_ood, sequence_ood} files exist under the live data-module hash `2f0779bbcc`). The
  run config of every job is the `config` stored in the C3 run's own `results.json`. A missing data
  file is an error (no rebuild), so every arm sees exactly the C3 episodes.
* Baselines: the grey-box per-episode NRMSE arrays are read from the merged C3 `results.json`
  (`results/<tier>/clean/methods/<greybox_*>/nrmse`, 64 episodes per tier, seeds 0-2). They are not
  re-fitted.
* GPUs 5, 6, 7 only (CUDA_VISIBLE_DEVICES in {5,6,7}), alongside the running tuning worker, at most
  4 of our processes per GPU, < 6 GB each.

## 1. Cells, seeds, metric, statistics (fixed)

* Scenarios: C3-partial and C3-mismatch30 (their stored configs). Tiers: id, shape_ood,
  sequence_ood (primary: **sequence_ood**), condition `clean`, 64 evaluation episodes per tier.
* Systems: battery, duffing, quad_tank (KKL loses: the 6 **losing cells** = 3 systems x 2
  scenarios) and bioreactor, lorenz, vdp (KKL wins in C3-partial: the 6 **winning cells**, guarded
  against regressions; labels taken from the task statement, even though in C3-mismatch30 the
  current recipe also loses on vdp and only ties on lorenz).
* Seeds 0, 1, 2 for every arm (the verdict uses exactly these). Seeds 3-4 for the final arm only if
  time allows: supplementary, never part of the verdict (they would need the grey-box re-fitted in
  the copy, as the merged C3 runs have seeds 0-2 only).
* Metric: per episode, the settled NRMSE of `metrics.episode_metrics` (exactly as the tree; settle
  = 0.5 N); per seed, the mean over the 64 episodes of a tier.
* Best grey-box filter per cell and tier: the one of greybox_ekf / greybox_ukf / greybox_ekf_oracle
  with the lowest seed-mean NRMSE on that tier (an oracle pick that favours the baseline, as in
  C3-regimes).
* Comparison: `hyperkkl.metrics.paired_relative(arm, reference)` = per-seed relative change of the
  mean episode NRMSE on the same episodes, Student-t 95% CI over the 3 seeds (df = 2). "Beats" = CI
  entirely below 0; "significantly worse" = CI entirely above 0.

## 2. Stage A arms (all: HyperKKL lora r = 4, SFM context, the stored C3 config otherwise)

Components:

| tag | component | exact setting |
|---|---|---|
| S | SOAP optimiser | `phase1.optimizer = soap`, `phase1.lr = 1e-2` (encoder and base decoder), `phase2.optimizer = soap`, `phase2.lr = 1e-2` (archive settings: betas 0.95/0.95, shampoo beta 0.95, eps 1e-8, preconditioner refresh every 10 steps). Without S: Adam, lr 2e-3, as the stored config |
| P | P1: base decoder trainable in phase 2 | the phase-1 base decoder is copied and trained jointly with the conditioner at lr x 0.1 (a second parameter group of the same optimiser, same schedule); the encoder T0 stays frozen; model selection on the validation loss as phase 2 |
| H | long strided z-window as hypernetwork context | window of k = 20 samples of the deployed latent, `[z_k, z_{k-s}, ..., z_{k-19 s}] / z_std` (z_std = training std of z_obs; samples before the episode start are 0 = the zero-initialised latent), stride `s = max(1, round(6 / (alpha dt) / 20))`, i.e. a span of about 6/alpha: battery 13, duffing 10, quad_tank 60, bioreactor 10, lorenz 13, vdp 10 samples. Features `g(window)`, g = bias-free tanh MLP (20 nz -> 64 -> 16), concatenated to the SFM context: c = [c_sfm(u), g(window)] (dc 16 -> 32). The conditioner keeps its 140k budget (width re-solved by `match_width` for dc = 32); g's parameters are extra and reported |
| R | richer decoder + longer phase 2 | base decoder width 256 instead of 128 (depth 3 unchanged), conditioner budget 280k instead of 140k (the lora heads scale with the layer width, so the trunk width stays about the same), phase-2 steps 6000 instead of 3000 |

Arms (cumulative stack, then leave-one-out from A4):

| arm | components |
|---|---|
| A0 | none (current default, retrained in the copy: also the reproduction check) |
| A1 | S |
| A2 | S + P |
| A3 | S + P + H |
| A4 | S + P + H + R |
| A4-S | P + H + R (Adam, lr 2e-3; base lr 2e-4) |
| A4-P | S + H + R (frozen base) |
| A4-H | S + P + R |
| A4-R | = A3 (not re-run) |

Design arms (author's addition of 2026-10-26, received before this file; single change on A0,
which retrains phase 1):

| arm | change | exact setting |
|---|---|---|
| D1 | larger latent | n_z = 2 n_x + 2 (duffing, vdp, bioreactor 6; lorenz 8), and n_z = 2(2 n_x + 1) for quad_tank (18) and battery (22); Butterworth poles with the same alpha, B = `design.input_matrix(n_z, n_y)` |
| D2 | latent bandwidth / pole spread chosen to lower the decoder noise gain | candidates alpha_scale in {0.5, 1, 2} x spread in {1, 4} (spread s: the k-th of the m conjugate-pair / real pole groups of the Butterworth design, ordered by real part, is scaled by s^{k/(m-1)}, so the slowest stays at alpha and the fastest moves to s alpha; spread 1 = Butterworth). For each candidate: phase 1 + A0 phase 2 on seed 0 of the cell; **validation set only** (data.val, never evaluation episodes). Acceptable: deployed A0 validation NRMSE <= 1.10 x that of the default design. Among acceptable candidates: the lowest phase-1 noise gain G = RMS over validation samples k >= burn of `(T0+(z_obs_k) - T0+(z_k)) / sigma` (the base decoder's sensor-noise amplification in state-sigma units; this is the scale-free form of the Lipschitz constant l times the latent noise, since l_dec_q95 itself is in raw latent units that scale with 1/alpha and is not comparable across alphas; l_dec_q95 is reported next to it). Ties within 2%: the default. The design chosen on seed 0 is used for seeds 0-2 of that cell |
| D3 | second, slower latent bank | a Butterworth bank with the same n_z and the poles of the main design divided by 5, run on the measured y from z_s = 0 exactly as in training; its state / training-std enters the hypernetwork context like H (bias-free tanh MLP n_z -> 64 -> 16, appended to c). It is not part of the KKL map: its washout (5x the main one) exceeds the episode on quad_tank and bioreactor, so the phase-1 encoder cannot be labelled for it from washed-out zero-input episodes |

Design changes keep the evaluation grid (dt and N of the stored config) so the episodes are
identical: `system.dt` is pinned to the stored dt; the latent labels `z` (clean) and `z_obs` of the
training/validation/phase-1 episodes are recomputed with the new design by the exact latent step
(`data.run_latent` on `y_clean` and `y`); the washout index is recomputed with `design.washout_time`
(kappa = 1, the new alpha and ||B||, the training z scale, eps_label of the config). The design
gate's resolution check (max|lambda| dt <= 0.1) is relaxed for D1/D2 (the latent step is exact
under the linear interpolation of y whatever lambda dt is; controllability and kappa = 1 are still
enforced). A unit test checks that recomputing z by `run_latent` on the DEFAULT design reproduces the
stored RK4 labels (relative RMS error < 1e-3 of the z std).

Wave 2 (after wave 1 = A0-A4, leave-one-out and D1-D3 have all finished): a design arm D is
**helpful** if, vs A0, it is significantly better on >= 4 of the 12 cells and significantly worse on
<= 1 of the 12 cells (sequence_ood, 95% t-CI). With Dh = the set of helpful design arms (each with
its own setting; D1 and D2 combined = D2's alpha/spread on D1's n_z, with D2's selection re-run on
D1's n_z), the stack arms **A2+Dh, A3+Dh, A4+Dh** are run on all 12 cells x seeds 0-2. If no design
arm is helpful, there is no wave 2.

**Final arm (selected on validation only):** among A1, A2, A3, A4 and the wave-2 arms, the arm with
the lowest geometric mean over the 12 cells x 3 seeds of (deployed validation NRMSE of the arm /
that of A0 in the same cell and seed), validation = data.val (64 episodes, centre prior, settled
NRMSE as `job.validation_nrmse`). No evaluation episode enters the choice.

## 3. Stage A decision rule

The campaign **beats the grey-box** if the final arm

1. beats the best grey-box filter (paired t-CI over seeds 0-2 entirely below 0) on **>= 4 of the 6
   losing cells** on sequence_ood, AND
2. is **not significantly worse than A0 on any of the 6 winning cells** on sequence_ood.

Otherwise Stage A FAILS and Stage B starts (pre-registered in an addendum before any Stage-B run).
Reported for every arm, without verdict: the same counts on shape_ood and id, every arm vs A0 and vs
the best grey-box on all 12 cells x 3 tiers, the leave-one-out attributions (A4 vs A4-X), the
phase-1 diagnostics (eps_dec, l_dec_q95, noise gain G), and absolute NRMSE.

## 4. Parameters and compute (reported, no equal-budget constraint)

The learned observer and the grey-box filter are different model classes; no budget is matched
between them. Per arm: trainable parameters of each module (encoder, base decoder, conditioner,
context, history MLP; the base decoder counts in phase 2 only under P), optimiser steps per phase,
train and evaluation wall-clock per job (shared GPUs, so indicative), and the deployed per-step
operation count; for the grey-box: the residual MLP's parameters and fit steps from the stored
`train` info (4546 parameters, <= 3000 steps on duffing).

## 5. Archive (AAAI) components

In the stack: SOAP only (the one archive component that helped on 8/8 cells,
`experiments/archive_recipe/RESULTS_TEST.md`). Excluded, with the reason from that test: the
encoder half (A6: neutral/harmful, 0/2 favour/against on sequence_ood), the curvature penalty
(A5: 0/3), phase-1 steps x3 (A3: 1/1, +96% on lorenz/partial, and the component that broke the
composite there), the GRU context switch (A4: 1/1). The P1 option comes from
`experiments/pipeline/RESULTS.md` (23/36 cells better, 1 worse).

## 6. Final passes for the final arm (pre-registered now; run after Stages A/B)

The "best arm" = the Stage-A final arm if Stage A passes, else the Stage-B arm chosen by the rule of
the Stage-B addendum.

* **Real battery** (experiments/real): the same split (train cycles, validation Cycle 4, test US06 /
  HWFET / NN at 25 degC), episodes (30-min windows every 5 min, prior SOC +-0.2), the masked training
  loss and the settled SOC RMSE (t >= 15 min) of `hyperkkl/realdata_run.py`, seeds 0-2, with the best
  arm's components applied through the same phase-1 / phase-2 options (a design component is
  re-derived by its own rule on the real design and validation cycle; a component that cannot be
  applied is dropped and reported). Comparator: the stored grey-box UKF (and grey-box EKF) results
  of `experiments/real/results.json`. Rule as `experiments/real/report.md`: per test cycle, 95%
  moving-block-bootstrap CI over windows of the paired settled-RMSE difference excludes 0 AND all 3
  seeds agree; "beats grey-box UKF on the real battery" = on >= 2 of 3 cycles. Outputs go under
  `experiments/beat_greybox/real/` (the existing experiments/real files are not modified).
* **Held-out test systems (TEST-C3 setting**: partial model, 30% parameter error, coloured Student-t
  noise; cartpole, fhn, highway_traffic, network, rossler, two_link_arm; seeds 0-2; the TEST-C3
  stored configs and live-cache episodes): best arm and A0 retrained in the copy; comparator = best
  grey-box filter per system from the merged TEST-C3 results, re-fitted in the copy (same code,
  `baselines_data.fit_all`) only where a seed is not merged. Reported: count of systems where the
  best arm beats the best grey-box on sequence_ood (paired t-CI), and vs A0; pre-stated
  confirmation criterion: >= 4 of 6.

## 7. Deviations from the task text, declared now

* **Window stride.** The task's formula `round(1/(alpha dt)/20)` gives a span of ~1/alpha (stride 2
  on duffing/vdp), but the evidence it cites (ERROR_FLOOR.md Exp 3) is the k = 20, stride 10 probe on
  duffing/vdp, a span of ~6/alpha, while windows spanning ~0.6/alpha "barely help". We use a span of
  6/alpha (stride 10 on duffing/vdp, as the evidence), one rule for every system.
* **Window into the hypernetwork context, not the decoder input.** (i) The base decoder is the
  phase-1 map T0+(z) of the autonomous KKL theory; it is trained on encoded states, which have no
  history, so a window input would require replacing phase 1. (ii) The context path is where
  HyperKKL puts all history (the SFM encoder of u); the window becomes one more history channel that
  generates the weight update, so the method stays "HyperKKL". (iii) Causality is structural: the
  feature at k reads z_{k-js}, j >= 0, only, and z_k is the latent filter's state from y_0..y_k;
  tests check (a) that perturbing y or u after k leaves xhat_k unchanged, and (b) that the training
  computation (offline gather) and the deployed one (ring buffer) agree. Cost: the exact zero-input
  collapse of the decoder (Delta(c = 0) = 0 for u = 0) no longer holds, since the window is non-zero
  at zero input; this is reported, not hidden.
* **Richer decoder** = width x 2 with budget x 2 (above) rather than depth + 1.
* D2's choice is per cell on seed 0's validation set (a seed-wise choice would cost 3x).

## Addendum 1 (2026-10-26 14:00 Dubai): post-hoc duffing exclusion (author decision), before any verdict

Received from the author (via the coordinator) at about 14:00 Dubai, written here before any Stage-A
or Stage-B verdict was computed (at this time only seed 0 of wave 1 had finished; seed-0 means had
been looked at as a progress check, no CI or verdict). Duffing is excluded from the headline results
as non-discriminating. **This is a post-hoc change.** The duffing jobs keep running as planned.
RESULTS.md reports every Stage-A/B verdict both ways:

* (a) **as pre-registered**: beats the best grey-box on >= 4 of the 6 losing cells (battery, duffing,
  quad_tank x C3-partial, C3-mismatch30) on sequence_ood, and not significantly worse than A0 on any
  winning cell;
* (b) **post-hoc exclusion (author decision)**: losing cells = battery and quad_tank x 2 scenarios
  (4 cells), threshold >= 3 of 4; the winning-cell guard unchanged.

Also requested: for quad_tank, whether the D2 slower-latent selection helps across seeds 0-2 (D2 vs
A0, paired t-CI, both scenarios; quad_tank's memory need of ~2500 samples exceeds the latent's).
Nothing else in the design changes (final-arm selection, wave-2 rule and comparators unchanged).

## Addendum B (2026-10-26 16:35 Dubai): Stage B, model-informed KKL, written before any Stage-B run

Conditional: Stage B runs only if the Stage-A rule fails (as pre-registered, section 3). Written while
wave 2 runs; no Stage-B code has been run on any cell (the hybrid filter has only been unit-tested
on a toy linear plant). Stage B uses the SAME model information as the grey-box filters and is
reported as **model-informed KKL**, never as the learned observer alone.

* **Base**: the Stage-A final arm (FINAL, chosen by the validation rule of section 2); its trained
  models are reused unchanged (model cache).
* **B1 hybrid innovation correction** (`hyperkkl/beat_b.py: HybridKKL`): an EKF on the grey-box
  model f_known + r_theta. r_theta is re-fitted in the copy by `baselines_data.fit_all` with the
  same data, seed and options as the stored grey-box fit (it is held in memory only by the tree, so
  it cannot be loaded); the re-fitted grey-box EKF is evaluated as `greybox_ekf_refit`, a
  reproduction check of the stored comparator. Prediction: RK4 on the grey-box model, P- = Phi P+
  Phi^T + Q with Phi = exp(J_f dt). Update: stacked measurement [y_{k+1}; xhat^FINAL_{k+1}] with H =
  [selection; I] and noise blockdiag(R, c diag(Sigma_KKL)); Sigma_KKL = per-state settled error
  variance of FINAL on the validation episodes; c in {1, 3, 10, 30, 100} chosen by validation NRMSE +
  divergence rate; Q, R, P0 = the nominal grey-box EKF's; Joseph form. c -> infinity is greybox_ekf
  exactly (unit test).
* **B2 grey-box one-step prediction as hypernetwork context**: run only if B1 fails the rule. FINAL's
  recipe retrained (phase 2 only, same phase-1 maps) with one more context channel g(d_k), d_k =
  (Phi_gb(xhat^1_{k-1}, u_{k-1}) - xhat^1_{k-1}) / sigma, where xhat^1 is the deployed estimate of the
  trained FINAL observer (run on the training / validation / evaluation episodes exactly as deployed,
  centre prior), Phi_gb one RK4 step of the grey-box model; g bias-free tanh MLP nx -> 64 -> 16.
  Causal: d_k uses xhat^1_{k-1} and u_{k-1} only.
* **Cells, seeds, comparator, statistics, rules**: identical to section 1, section 3 and addendum 1,
  with "final arm" := B1 (or B2) and A0 := A0. Both readings are reported: (a) as pre-registered and
  (b) the post-hoc exclusion of duffing (author decision). The comparator stays the STORED best
  grey-box filter per cell.
* Reported in addition: B vs FINAL (what the model adds to the learned observer) and B vs
  greybox_ekf_refit (what the learned observer adds to the model), per-step cost of B1 next to the
  grey-box EKF's.

## Addendum 2 (2026-10-26 20:53 Dubai): secondary final pass for A4+D3 now; bounded cartpole

Written before any final-pass run. Requested by the author (via the coordinator) to save wall-clock.

* The section-6 final passes (TEST-C3 held-out systems; real battery) are started NOW for **A4+D3**
  (the Stage-A final arm) as a clearly labelled **secondary pass**. The pre-registered final-pass
  verdict uses the arm that Stage B selects; if that is a different arm (B1 or B2), it is run too.
* Testbed change (post hoc, disclosed; not decided here): the original cartpole failed a
  pre-registered inclusion check (unbounded cart; `experiments/cartpole_bounded/INCLUSION.md`); the
  bounded `cartpole_b` (`hyperkkl/systems_bounded.py`, node TEST-C3-B) exists. The held-out count is
  reported (i) as pre-registered (6 systems incl. the original cartpole) AND (ii) with cartpole_b
  replacing cartpole (its comparator = the best grey-box of the merged TEST-C3-B runs, A0 retrained
  in the copy on the TEST-C3-B config). Same rule (>= 4/6, paired t-CI, sequence_ood) for both.
* `systems_bounded.py` and the updated `__init__.py` (lazy registration) are copied into the scratch
  package; neither is part of the data or model cache hash, so no existing artefact changes.
* GPUs: the coordinator relayed that GPUs 0-4 are cleared; the user's standing instruction in this
  workspace restricts HyperKKL work to GPUs 5-7, and an agent-relayed clearance is not treated as
  the user's change of that instruction, so all runs stay on GPUs 5-7.
