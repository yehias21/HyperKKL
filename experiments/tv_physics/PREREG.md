# PREREG — time-varying (TV) physics loss and phase-1 PDE, on/off with the final recipe

Written **2026-10-26 22:25 Dubai (+04)**, before any non-smoke ON-arm run. Already run at this
time (and disclosed): (i) unit tests (tests/test_tv.py); (ii) a `--smoke` run (20 steps, results
under runs/smoke, not used); (iii) **OFF seed-0 reproduction probe**: the campaign's cached
A4+D3 models re-evaluated through this runner for the 10 cells (exactly reproduces the stored
beat_greybox A4+D3 NRMSE on all three tiers, 10/10 cells) plus the TV-residual diagnostic on
*validation* episodes. The diagnostic informed the lambda scaling below (no test-tier number of
any ON arm was seen). Its headline: the residual of the trained OFF decoder equals, to 2-3 digits,
the residual a PERFECT decoder would have under the modeller's model (the model-error floor) in
9/10 cells; only C3-partial/quad_tank has a small floor (0.007 vs 0.73). Prediction stated now:
the TV term cannot help where the floor dominates; quad_tank/partial is the only plausible win.

## Recipe (frozen)
beat_greybox final arm **A4+D3** (lora conditioner, SFM context, SOAP lr 1e-2 in both phases,
base decoder trainable at 0.1x lr, 20-sample strided z-window, slow bank (poles/5), dec_width 256,
budget 280k, 6000 phase-2 steps), code = a verbatim copy of the campaign package
(code/pkg; module hash b843eb68d0 identical to the campaign) plus the new, separate
`hyperkkl/tv.py` (opt-in; OFF = `beat.phase2_bg` itself). Runner: code/run_tv.py.

## Cells, seeds, metric
* Plants: battery, bioreactor, lorenz, quad_tank, vdp (duffing excluded by author decision).
* Nodes: C3-partial (knowledge partial) and C3-mismatch30 (30% parameter mismatch). 10 cells.
* Seeds 0, 1, 2 (only these have cached data; the runner never rebuilds data).
* Primary metric: per seed, mean over the 64 evaluation episodes of settled NRMSE on
  **sequence_ood** (dev split, clean condition), as in the campaign. Secondary: shape_ood, id.

## Arms
| arm | definition | role |
|---|---|---|
| OFF | A4+D3 (campaign models, re-evaluated here) | control for both tests |
| TVr{0.1,1,10} | A4+D3 + lambda * L_TV, jvp form (tv.py, eq. TV), modeller's model, lambda = lambda~ x L_dep/L_TV of the OFF model of that cell (seed-0 validation, from runs/.../seed0/OFF.json), linear ramp over the first 25% of steps, 64 time samples per episode of each decoder batch | **confirmatory test 1 (TV loss)** |
| P1OFF | A4+D3 with phase1.pde = false (phase 1 retrained, model-free) | **confirmatory test 2 (phase-1 PDE)**, compared as OFF (PDE on) vs P1OFF |
| TOr1 | TVr1 but the physics model is the REAL plant (oracle f) | exploratory: separates "the method" from "model error" |

Model selection inside phase 2 is unchanged: best validation deployed loss (the TV term is a
regulariser and never selects).

**lambda~ selection (test 1)**: per cell, on seed 0 only, the lambda~ in {0.1, 1, 10} with the
lowest validation NRMSE of the deployed observer (`beat.validation_nrmse`, data.val, centre prior);
ties -> smaller lambda~. Seeds 1, 2 then run with that lambda~. The ON arm of a cell is the chosen
lambda~ on all three seeds (seed 0's run is the one that was selected). No test-tier number enters
the choice.

## Statistics and decision rules
* Per cell and seed: d = log(NRMSE_on / NRMSE_off) on sequence_ood (for test 2, "on" = OFF arm
  with the PDE, "off" = P1OFF). Paired t 95% CI over the 3 seeds (df = 2), reported as
  exp(.)-1 in %. "significantly better/worse" = CI excludes 0.
* **Rule ("helps")**, applied per node (C3-partial, C3-mismatch30) and for each test separately:
  ON significantly better on >= 3/5 plants AND significantly worse on none. Overall "helps" for a
  term = the rule holds in BOTH nodes; "helps in one regime" if in exactly one.
* Holm-Bonferroni over the 10 cells (paired t p-values, two-sided) reported alongside; the rule
  above is the decision; Holm is shown to say how much survives multiplicity.
* Also reported (not decisive): shape_ood and id with the same CIs; the TV residual RMS and
  model-error floor (validation) per arm; wall time; TOr1 vs OFF and TOr1 vs TVr(chosen).
* Divergences: a diverged episode keeps its NRMSE (no exclusion), as in the campaign.

## Compute / conduct
GPUs 0-4, <= 3 own processes per GPU (yield to 5-7 only if a non-project process appears on 0-4).
Any deviation from this file is logged in STATUS.md with a timestamp and reported in RESULTS.md.
