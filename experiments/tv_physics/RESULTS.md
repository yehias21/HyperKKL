# RESULTS — is the time-varying physics loss significant? (and the phase-1 PDE)

2026-10-26, protocol PREREG.md (written 22:25 Dubai, before any ON run). Audit: AUDIT.md.
Archive and current-pipeline evidence: evidence/EVIDENCE.md. Literature: evidence/LIT.md.
Full tables: tables.md; raw numbers: analysis.json; runs: runs/<node>/<plant>/seed*/<arm>.json.

## Insight (5 lines)
1. The archived "time-varying loss" never constrained the deployed observer. It acted on an encoder that inference discards, was off by default, and its dedicated ablations never actually switched it on (a string 'False' bug). The current pipeline has no TV loss at all.
2. The correct form for this architecture is the inverse TV PDE on the deployed decoder, with an exact ċ chain-rule term over every context channel. It is implemented (tv.py) and passes unit tests (an analytic linear KKL inverse gives zero residual; the residual is nonzero without the ċ term).
3. With the final recipe (A4+D3), the TV loss is **not significant**: 0/5 better and 0/5 worse in both C3-partial and C3-mismatch30, and every Holm-adjusted p is 1.00. Even with the TRUE plant as the physics model it does nothing (0/10).
4. The reason is measurable. The data-trained decoder's TV residual already equals the residual of a perfect decoder (the model-error floor) in 9/10 cells. Physics has nothing left to teach the decoder except the model's error.
5. The phase-1 autonomous PDE is also not significant with A4+D3 (0/5 better in both nodes; significantly worse on lorenz/partial, +7% [+5, +8], Holm p = 0.02). The bioreactor collapse without the PDE seen in the older pipeline does not occur once the base decoder is fine-tuned (P).

## Verdicts (pre-registered rule: ON significantly better on >= 3/5 plants on sequence_ood and worse on none, per node; "helps" overall = both nodes)

| test | C3-partial better/worse | C3-mismatch30 better/worse | Holm-significant cells (10) | verdict |
|---|---|---|---|---|
| 1. TV loss ON (lambda~ chosen on validation) vs OFF | 0 / 0 | 0 / 0 | 0 | **does not help** |
| 2. phase-1 PDE ON vs OFF (same recipe) | 0 / 1 (lorenz) | 0 / 0 | 1 (lorenz/partial, PDE worse) | **does not help** |
| expl. TV with oracle (true) f, lambda~ = 1, vs OFF | 0 / 0 | 0 / 0 | 0 | no effect |

## Test 1 — TV loss ON vs OFF, sequence_ood (primary), paired t 95% CI over 3 seeds

Effect = NRMSE_on/NRMSE_off − 1 (negative = TV better); **bold** = significantly better, *italic* = significantly worse.

| plant | C3-partial | C3-mismatch30 |
|---|---|---|
| battery | −7% [−30, +25] (λ~ 0.1) | −2% [−8, +3] (0.1) |
| bioreactor | +3% [−8, +16] (0.1) | +2% [−10, +15] (0.1) |
| lorenz | −0% [−2, +1] (1) | −1% [−7, +5] (0.1) |
| quad_tank | −15% [−57, +71] (0.1) | +8% [−39, +90] (1) |
| vdp | −0% [−2, +1] (0.1) | −0% [−2, +2] (0.1) |

Secondary tiers (tables.md): no cell is significantly better on shape_ood or id. One cell is
significantly worse: bioreactor/partial shape_ood, +2% [+0, +3]. quad_tank/partial: the TV
term rescued OFF's one bad seed (seed 0: 0.0406 → 0.0251). Seeds 1 and 2 are unchanged, so the CI
is very wide. This is the only cell with a small model-error floor (below), and it is the cell the
pre-registration named as the plausible win. It remains an anecdote, not a result.

λ~ selection (seed-0 validation): 0.1 in 8/10 cells, 1 in 2/10. λ~ = 10 was never chosen, and it
was clearly harmful on validation for battery (2x) and vdp (+13%/+40%). Larger physics weight
only hurts.

## Test 2 — phase-1 autonomous PDE ON vs OFF, same A4+D3 recipe, sequence_ood

| plant | C3-partial | C3-mismatch30 |
|---|---|---|
| battery | +15% [−1, +34] | −2% [−30, +35] |
| bioreactor | −4% [−10, +2] | +1% [−14, +19] |
| lorenz | *+7% [+5, +8]* (Holm p 0.02) | +1% [−4, +7] |
| quad_tank | −18% [−62, +77] | +5% [−30, +57] |
| vdp | +0% [−2, +2] | −1% [−2, +0] |

Secondary: the PDE is significantly better on battery/partial id (−13% [−19, −6]), battery/mm30
shape_ood (−12% [−21, −2]) and bioreactor/mm30 shape_ood (−1% [−2, −1]). It is significantly
worse on lorenz/partial in every tier. Taken together, the phase-1 PDE is neutral-to-mixed with
the final recipe. In the older pipeline (frozen base decoder), removing it made bioreactor 7-9x
worse (EVIDENCE.md §2). Base-decoder fine-tuning (component P) removes that dependence.

## Why: the model-error floor (validation, seed means, units σ_x·α)

| cell | floor = ‖f_true − f_model‖ at the true state | OFF decoder residual | TV-ON residual | oracle-TV residual (vs true f) |
|---|---|---|---|---|
| battery/partial | 1.35 | 1.36 | 1.34 | 0.165 |
| bioreactor/partial | 12.0 | 12.0 | 12.0 | 0.025 |
| lorenz/partial | 7.78 | 7.79 | 7.78 | 0.279 |
| quad_tank/partial | **0.008** | 0.361 | **0.135** | 0.136 |
| vdp/partial | 0.374 | 0.376 | 0.376 | 0.028 |
| battery/mm30 | 0.653 | 0.656 | 0.653 | 0.063 |
| bioreactor/mm30 | 4.77 | 4.77 | 4.77 | 0.028 |
| lorenz/mm30 | 2.07 | 2.07 | 2.07 | 0.046 |
| quad_tank/mm30 | 1.90 | 1.91 | 1.90 | 0.185 |
| vdp/mm30 | 0.150 | 0.155 | 0.153 | 0.020 |

Without any physics, the OFF decoder already moves exactly as the true state does: its residual
equals the floor of a perfect decoder. The TV term can lower the residual only by pulling x̂ toward
the wrong model, so λ selection drives it to the smallest value. Where the model is right
(quad_tank/partial), the term does cut the residual 0.36 → 0.135, but test NRMSE does not move
significantly. With the true f (oracle) the residual becomes small, yet accuracy is unchanged.
With 600 forced training episodes, the trajectory loss already pins down the decoder's dynamics.
The physics term is redundant here, not wrong in form.

## Conduct, cost, deviations
* OFF = the campaign's cached A4+D3 models, re-evaluated through this runner. They reproduce
  the stored A4+D3 NRMSE exactly on 30/30 (cell, seed) pairs, all tiers. ON arms reuse the same
  cached phase-1 models (the module hash is unchanged because tv.py sits outside MODEL_MODULES).
  P1OFF retrains phase 1.
* Cost: the TV term raises phase-2 wall time by about 1.7x (median 910 s vs 534 s for a no-TV
  phase 2 under the same load).
* Runs: 110 (30 TV grid seed 0, 20 TV seeds 1-2, 30 P1OFF, 30 oracle) plus 30 OFF re-evaluations.
  Only seeds 0-2 have cached data, so 5 seeds were not possible.
* Deviation (conduct only, STATUS.md): the dispatcher mis-flagged my own relative-path processes
  as foreign and placed some jobs on GPUs 5-7 (still ≤ 3 per GPU). A duplicate dispatcher
  briefly over-subscribed GPUs 0-4; its jobs were killed within 30 s and wrote nothing. The
  protocol was not changed.
* Supplementary tables in tables.md with a fixed arm (TVr1, oracle vs TVr1) are mostly n/a by
  design, because TVr1 has 3 seeds only where it was chosen.
* Unit tests: 7/7 pass (logs/pytest_tv_gpu_final.log). Note for interpreting λ: with SOAP/Adam and gradient clipping, λ sets only the RELATIVE weight of the two terms, and λ~ = 10 is already TV-dominated.
* Not run (exploratory, implemented and tested): the derivative-free step form (`tv_form=step`).
  Given the floor result, it has no headroom either.

## Recommendation
Do not claim a time-varying physics loss in the paper, and do not claim that the archive had a
working one. If physics is mentioned, the defensible statements are: (i) phase 1 uses the
autonomous KKL PDE as in Niazi et al. (it is neutral with the final recipe); (ii) a correct
non-autonomous residual (inverse TV PDE with exact ċ) was implemented and tested, and it gave no
significant gain. The reason is that, under the realistic model knowledge of C3, its residual is
dominated by model error, which the data-trained decoder already sits at. If you keep the phase-1
PDE, consider fixing its 1/‖f0‖² weighting (AUDIT §1): 1% of the collocation points carry most
of the weight.
