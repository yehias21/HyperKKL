# Pre-registration: is the two-phase, frozen-base training pipeline optimal?

Written 2026-10-26 ~03:00 Dubai, **before any Stage-2 run**. Stage-1 evidence and the prediction
are in `EVIDENCE.md`. Code: a scratch copy of the package,
`/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/pipeline/pkg/`
(`hyperkkl/pipeline.py`, `run_pipeline.py`, `tests_pipeline/`). It is mirrored to `experiments/pipeline/code/`
when the runs start. The copy is identical to the live package plus one new module. MODEL_MODULES are
untouched, so P0 and the shared phase 1 are loaded from the live cache with the live keys. Each job
also checks that P0 reproduces the tree run episode by episode. Outputs go only to
`experiments/pipeline/` (`runs/`, `cache/`); the live `experiments/{tree,queue,cache}` are read only.

## Question

Does any alternative to "phase 1 (PINN encoder + labels, then decoder), then phase 2 (conditioner,
base frozen)" give a better deployed observer at equal compute and the same 140k conditioner? Does any
of them close the gap to the data-driven baselines (grey-box EKF/UKF, Koopman KF) on the systems where
KKL currently loses?

## Arms

All arms use the 140k conditioner budget (`match_width`), the same SFM context, the same data and the
same deployed observer `zhat' = A zhat + B y`, `xhat = T+(zhat; c)`, with `z0 = T0(x0_hat)` from the arm's
own encoder. Kinds: lora and hyper (latent, concat and film are not re-run).

| arm | what | trainable parameters (reported per run) |
|---|---|---|
| **P0** | current pipeline: `train.phase1` (6000 encoder + 6000 decoder steps), then `train.phase2` with frozen base (3000 steps). Cached tree models; `autonomous` (base only) evaluated too | conditioner ~140k + context |
| **P1** | P0's phase 1, then phase 2 with the **base decoder unfrozen**: a deep copy of T0+ in its own Adam param group at lr x 0.1 (2e-4), same cosine/warmup, 3000 steps, selection on validation | conditioner ~140k + base decoder (~35k) + context |
| **P2** | **one-shot joint training from scratch**: T0, T0+, conditioner and context in one Adam (lr 2e-3, cosine, warmup 100). Per step: label + lambda_pde * ramp(first 25%) * PDE (encoder) + recon of a fresh half-collocation/half-trajectory batch encoded by the *current* encoder, detached (decoder) + phase-2 decoder loss on z_obs (conditioner + decoder). Decoder input stats from the label latents (the encoder is untrained at init). Selection on the phase-2 validation loss, every 250 steps | conditioner ~140k + encoder + base decoder + context |
| **P3** | control: P0's phase 1, then an **unconditioned** fine-tune on forced data. Base decoder copy at lr x 0.1 plus an additive residual `R(z)` (MLP, 2 hidden, SiLU, zero-init output) matched to 140k at lr 2e-3, phase-2 loss, no input. Separates "repair the base" from "condition on the input" | R ~140k + base decoder |
| **P4** | **grey-box KKL**: `r_theta(x,u)` fitted by `baselines_data.fit_all` (same data, seed, options as the grey-box baselines). Phase 1 with `f0 = f_known(x,0) + r_theta(x,0)` in the PDE term, then the usual frozen phase 2 (lora, hyper); `P4-autonomous` also evaluated. **C3 scenarios only** (with the exact model the residual target is only process noise) | as P0 |

## Equal compute

The unit is one loss-batch gradient evaluation (one phase-1 encoder batch of 4096, one phase-1
decoder batch of 4096, or one phase-2 batch of 32 episodes x 256 times). In the two-phase arms one
optimiser step is one unit. Every arm totals **15000 units**:
P0 / P1 / P3 / P4 = 6000 + 6000 + 3000, and P2 = 5000 joint steps x 3 losses
(`pipeline.compute_plan`, unit-tested). The literal reading "same optimiser steps" (15000 joint steps)
would give P2 three times the gradient evaluations, so it is not used. P2 still gets 5000 phase-2
evaluations against 3000, and 5000 vs 6000 for each phase-1 loss. This is disclosed and favours P2.
**P4 exception (disclosed):** the residual fit (at most 3000 Adam steps, early stopping, of a ~5k-parameter
MLP) is the same fit the grey-box baselines use and is not deducted; its actual step count is recorded.
Wall-clock per stage is recorded for every arm, under shared-GPU contention.

## Systems, scenarios, seeds (rule fixed before running)

* **Systems rule: every dev system** (battery, bioreactor, duffing, lorenz, quad_tank, vdp). The rule
  selects nothing, so there is no selection effect. It includes the systems where KKL loses to the
  data-driven baselines (C3-partial: battery, bioreactor, duffing, quad_tank) and where it wins
  (lorenz, vdp).
* Scenarios: `C3-partial`, `C3-mismatch30` and `C1-where` (exact model, clean; for the C1 implication).
  Each job reuses the config, data and evaluation episodes of the corresponding tree run.
* Seeds 0, 1, 2. Evaluation: tiers id, shape_ood, sequence_ood; condition clean; 64 episodes per tier
  (the tree's episodes).
* GPUs: `CUDA_VISIBLE_DEVICES` in {5, 6, 7} only (the runner refuses anything else), <= 3 processes per
  GPU, < 10 GB each.

## Comparisons and decision rule

* Contrast: `hyperkkl.metrics.paired_relative` (relative change of mean episode NRMSE, Student-t 95% CI
  over the 3 seeds), same episodes.
  - P1-k, P2-k and P4-k against **P0-k** (same kind k).
  - P3 against **P0-best** (the better of P0-lora / P0-hyper by seed mean, chosen on these episodes;
    this choice favours P0) and against `autonomous`.
* **Primary tier: sequence_ood.** shape_ood and id are reported with no rule.
* **Rule (per scenario, per arm-kind):** the arm is **BETTER** if the CI excludes 0 in its favour on
  **>= 4/6 systems** and it is significantly worse on **none**. It is **WORSE** in the mirror case.
  Otherwise it is **NOT BETTER**. P4 is decided on the two C3 scenarios only.
* **Answer to the question:** "two-phase frozen is optimal among the tested pipelines" iff no arm-kind
  is BETTER in any scenario. If an arm is BETTER in a scenario, it is the recommended replacement
  there.
* **Gap closure (separate, no multiplicity correction):** a system is a *loss cell* if P0-best is
  significantly worse than the best data-driven baseline (paired, sequence_ood). An arm **closes** the
  gap on a loss cell if its best kind (by seed mean) is not significantly worse than that baseline. It
  **narrows** the gap if it is significantly better than P0-best there. Counts are reported per arm.
  Baselines: C3-partial from the tree's merged runs (greybox_ekf/ukf/oracle, koopman_kf, nlox).
  C3-mismatch30 from the tree (nlox), plus koopman_kf, greybox_ekf and greybox_ukf fitted and evaluated
  inside the pipeline job (same `fit_all` code and data), because the tree has not merged them there.
  C1-where (exact model): vs the tree's EKF, descriptive only.
* **C1 implication (exact scenario):** (i) whether the lora-vs-hyper ordering changes under P1/P2;
  (ii) P3 vs P0-best. If P3 is not significantly worse than P0-best on >= 4/6 systems, the C1 gain
  would be largely base repair, which changes how C1 reads. (iii) P3 vs the tree's latent and concat
  (paired, same episodes).
* **C3 implication:** whether any arm turns a C3-partial loss cell into a non-loss.

## Also reported (no rule)

Zero-input collapse. P0/P4: `cond(z,0) == T0+(z)` must be bitwise exact, checked on zero-input episodes.
P1/P2: structural exactness vs the arm's *own* base, plus the deviation of the zero-input map from
phase 1's T0+ (sigma units) and the zero-input NRMSE; the KKL autonomous guarantee is lost by that
amount. P3 has no collapse by construction.
Phase-1 diagnostics of P2 and P4 (eps_dec, l_dec_q95). Parameter counts. Wall-clock. Train/validation
curves.

## Known caveats (stated in advance)

3 seeds (t_0.975, df = 2, is 4.30), so "significant" needs large, consistent effects. Vote counts are
not corrected for multiplicity (5 arms x 2 kinds x 3 scenarios). The P1 learning-rate factor (0.1)
and P2's single learning rate are not tuned (one setting each, as specified). The best-of selection is
on the test episodes. Shared GPUs make wall-clock noisy.

## Pre-launch checks (appended 2026-10-26 02:41 Dubai, before the first real job)

* Unit tests (`tests_pipeline/test_pipeline.py`, CPU): 7 passed. They cover equal units (15000) across
  arms, bitwise zero-input collapse for the frozen-base decoder (P0/P4, lora and hyper, after
  perturbing the generators), P1/P2 structural collapse vs their own base with a non-zero deviation
  from the phase-1 base (reported), parameter counts within 2% of 140k with the base excluded from
  the conditioner count, the P3 budget and its zero-init start, and the collapse helper.
* GPU smoke run (`smoke/`, C3-mismatch30/vdp/seed0, 60/60/30 steps): every arm trains and evaluates.
  The collapse diagnostics behave as specified: P0/P4 exact, P1/P2 exact vs their own base and off
  phase 1's base. Trainable totals per run: P0 ~144-147k, P1 ~178-181k, P3 ~174k, P2 ~253-257k
  (encoder 75.7k + base decoder 34.1k + conditioner + context 6.2k).
