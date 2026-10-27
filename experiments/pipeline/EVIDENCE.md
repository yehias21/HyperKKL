# Stage 1: eval-only evidence on the two-phase, frozen-base pipeline

Written 2026-10-26 ~02:40 Dubai. Read-only on `experiments/tree` (no training). Script and raw output:
`experiments/pipeline/stage1/stage1.py` and `stage1.json`. Data: `C3-partial`, `C3-mismatch30` and `C1-where`
runs, seeds 0-2, all at code `a1e5d75cce66`. MODEL_MODULES are identical to the live package, so cached
base maps are the ones these runs used. Clean condition. "KKL" means the best conditioned learned
observer (latent, concat, film, lora or hyper), chosen by seed mean. "Data-driven" is the best of nlox,
koopman_kf*, greybox_* among the methods present on all seeds.

**Coverage caveat.** C3-partial has every data-driven baseline merged (18/18 runs). C3-mismatch30 has
**only NLOX** (the greybox/koopman re-evaluation has not reached it). C1-where has no data-driven
baselines at all. So "gap" means gap vs greybox/koopman/nlox on partial and gap vs NLOX on mismatch30.

## (a) Does base-map quality predict KKL's losses?

Sequence OOD, seed means. (Shape OOD and id show the same pattern: `stage1.json`.)

| scenario | system | eps_dec | eps_enc_rel | eps_pde_rms | l_dec_q95 | KKL NRMSE | best data-driven | KKL vs it (paired, t-CI over seeds) |
|---|---|---|---|---|---|---|---|---|
| partial | battery | 1.83 | 0.005 | 0.29 | 756 | 0.072 hyper | greybox_ekf 0.019 | +276% [+257, +295]* |
| partial | bioreactor | 0.096 | 0.009 | 0.024 | 84 | 0.070 hyper | koopman_kf 0.027 | +158% [+49, +266]* |
| partial | duffing | **0.006** | 0.004 | 0.001 | 298 | 0.054 lora | greybox_ukf 0.0078 | **+593%** [+403, +784]* |
| partial | lorenz | 0.357 | 0.178 | 105 | 11 | 0.019 hyper | nlox 0.029 | -33% [-52, -14]* |
| partial | quad_tank | 0.170 | 0.013 | 0.011 | 15 | 0.063 hyper | greybox_ekf 0.014 | +341% [+280, +402]* |
| partial | vdp | 0.011 | 0.003 | 0.061 | 23 | 0.018 lora | greybox_ekf 0.026 | -30% [-44, -15]* |
| mismatch30 | battery | 1.83 | 0.005 | 0.33 | 691 | 0.052 lora | nlox 0.284 | -81%* |
| mismatch30 | bioreactor | 0.185 | 0.007 | 0.010 | 57 | 0.098 lora | nlox 0.035 | +194% [-123, +510] |
| mismatch30 | duffing | 0.006 | 0.004 | 0.002 | 295 | 0.056 lora | nlox 0.074 | -24%* |
| mismatch30 | lorenz | 0.225 | 0.021 | 33 | 12 | 0.007 lora | nlox 0.007 | +3% (n.s.) |
| mismatch30 | quad_tank | 0.270 | 0.061 | 0.018 | 12 | 0.042 lora | nlox 0.130 | -68%* |
| mismatch30 | vdp | 0.014 | 0.004 | 0.27 | 26 | 0.021 lora | nlox 0.032 | -35%* |
| exact (C1-where) | battery / bioreactor / duffing / lorenz / quad_tank / vdp | 1.83 / 0.020 / 0.006 / 0.267 / 0.139 / 0.011 | | | 806 / 44 / 295 / 10 / 14 / 23 | 0.072 / 0.044 / 0.057 / 0.018 / 0.064 / 0.019 | (none merged) | |

Spearman correlation of the log diagnostic with log KKL NRMSE (abs) and with the log ratio KKL / best
data-driven (gap), on sequence OOD:

| diagnostic | abs, pooled n=18 | gap, pooled n=12 | gap, partial only n=6 | within system, across seeds (n=54, centred) |
|---|---|---|---|---|
| eps_dec | +0.13 (p 0.60) | -0.31 (p 0.32) | -0.31 | +0.13 (p 0.36) |
| eps_enc_rel | -0.02 | +0.01 | -0.26 | +0.04 |
| eps_pde_rms | -0.39 (p 0.11) | -0.49 (p 0.11) | **-0.83 (p 0.04)** | +0.13 |
| l_dec_q95 | **+0.64 (p 0.004)** | +0.25 (p 0.43) | +0.54 | +0.16 (p 0.25) |

**Finding (a): the quality of the phase-1 base maps does NOT predict where KKL loses.**
- The largest loss is on **duffing, which has the best base maps of all systems** (eps_dec 0.006,
  PDE residual 0.001). Battery's base decoder fails outright (eps_dec 1.83 sigma, worse than
  climatology), and yet battery KKL *beats* NLOX by 81% under mismatch30. eps_dec and eps_enc have no
  rank correlation with either the absolute error or the gap. Within a system, the seed with the worse
  base map is not the seed with the worse observer (all |rho| <= 0.16).
- The PDE residual correlates *negatively* with the gap: lorenz, whose base is worst by the PDE metric,
  is where KKL wins. That is a system-identity confound, not a mechanism.
- Only the decoder Lipschitz proxy `l_dec_q95` tracks the **absolute** error (rho +0.64 pooled; battery
  806, duffing 295, both with an error floor near 0.05). It does not track the gap. It is a property of
  the plant's immersion (it is nearly identical across the three scenarios for each system), not of how
  well phase 1 trained.
- The KKL error is close to **scenario-invariant** on the losing systems: duffing 0.057 / 0.054 / 0.056
  (exact / partial / mismatch30), battery 0.072 / 0.072 / 0.052, quad_tank 0.064 / 0.063 / 0.042. The
  model error enters only the phase-1 PDE term, and it barely moves the deployed error (consistent with
  C3-partial-modelfree: dropping the PDE term entirely changed nothing on 0/6). The losses to
  greybox/koopman therefore come from a **KKL error floor of ~0.05**. The data-driven filters go below
  it (0.008-0.027) once they get a residual-corrected model or a lifted linear model.
- That floor is a fit floor, not a generalisation gap. At the end of phase 2, train and validation loss
  are equal: sqrt(loss) for duffing is 0.053 / 0.053, quad_tank 0.077 / 0.080 and vdp 0.018 / 0.019
  (train / val, partial). Id, shape and sequence NRMSE are within ~10% of each other (duffing 0.049 /
  0.051 / 0.054). On duffing, the validation loss is still falling when the cosine schedule ends.

## (b) What phase 2 can and cannot fix

The autonomous observer (base maps only) vs the best conditioned variant, exact model (C1-where, seeds
0-2; the C3 nodes did not train `autonomous`, and Stage 2's P0 arm adds it there):

| system | autonomous seq / shape / id | best conditioned seq | reduction seq (paired t-CI) | shape | id |
|---|---|---|---|---|---|
| battery | 1.50 / 0.74 / 0.75 | 0.072 | -95% [-98, -92] | -93% | -94% |
| bioreactor | 4.70 / 2.98 / 3.26 | 0.044 | -99% [-99, -99] | -99% | -99% |
| duffing | 0.30 / 0.20 / 0.25 | 0.057 | -81% [-83, -80] | -74% | -81% |
| lorenz | 0.52 / 0.51 / 0.51 | 0.018 | -96% [-99, -94] | -97% | -96% |
| quad_tank | 1.62 / 1.07 / 1.15 | 0.064 | -96% [-96, -96] | -95% | -95% |
| vdp | 0.11 / 0.07 / 0.08 | 0.019 | -83% [-85, -81] | -78% | -80% |

- **Phase 2 does almost all of the work.** Under forcing the autonomous observer is near or above
  climatology on 4/6 systems. Conditioning removes 74-99% of its error on every system and tier. The
  "base map" is mainly a frozen feature extractor for a conditioner that re-learns most of the map
  `(z_obs, c) -> x`. Duffing and vdp, the systems with the best base maps, are where phase 2 removes the
  *least* (about 80%).
- **What phase 2 cannot fix:** the floor in (a). With the base frozen, the conditioner can only add a
  140k low-rank or subspace correction to a decoder whose features were fitted to zero-input states.
  On the losing systems the remaining error is (i) the same across scenarios, (ii) equal on train and
  validation, and (iii) still falling at the end of training. Three explanations fit that pattern and
  the eval data cannot separate them: the frozen features are the wrong basis (P1, P2, P3 test this),
  phase 2 is too short, or the 140k conditioner is too small. Capacity already looks unlikely: C1-budget-l
  at 560k is within ~10% of 140k.
- **What a better phase 1 would not fix (from (a)):** the base-map metrics do not track the losses, so a
  phase-1 recipe change aimed at eps_dec or eps_pde (longer phase 1, PDE reweighting R5, grey-box PDE P4)
  should not be expected to close the greybox gap. Stage 2 tests P4 directly.

## (c) What the archive says about freezing and joint training

Sources: `experiments/archive_recipe/STUDY.md` (B7, T1-T5, section 4) and `../archive/` (read-only):
`HyperKKL/QUEUE.md`, `final paper/AAAI_PLAN.md`, `final paper/spine_supplement.tex` ~L376,
`HyperKKL/README.md`, `DECK_BACKGROUND.md`, `Previous_submission_versions_comments.md`.

1. **The base weights were frozen in every archived run.** The supplement says so: "The base weights
   are frozen for the whole of Phase 2 in every rung", and so do the README and every config. **No
   archived run unfroze the base or trained end to end.** The archive has no direct evidence on the
   question.
2. The only freeze lever it tested was **`freeze_encoder_deltas`**, i.e. whether the *encoder* also
   gets context-dependent LoRA deltas plus a characteristic-residual and label loss in phase 2 (base
   still frozen). Re-freezing them (B7, FD_loo_frz vs FA_wt, 3 systems x 3 seeds, in-sample) improved the
   graded tier by **-22.5%** median (bioreactor -30 ±18, vdp -23 ±10, both significant; duffing_mult
   -4) and was neutral on the waveform tier (-1%). Adding trainable pathways and losses in phase 2 hurt.
   Legacy `QUEUE.md` notes that only 4 legacy runs ever had the lever off, and all were confounded.
3. **Phase-1 length mattered** (T1: 60 -> 200 epochs gave -25% on 3/3 systems, single seed). The
   composed recipe (T5) gained -42% on the waveform tier, and the leave-one-outs that did land do not
   explain it. Its likely carriers are phase-1 length (about 3.3x) and lr 1e-2, but that is inferred,
   not measured. Mechanism (`diag.py`): eps_dec went down on 6/8 systems and `l_dec_q95` went up on 5/8.
   Phase 2 at 120 vs 40 epochs also helped (T2, -16%, 3/3, single seed), and the two were not additive
   (T3, -6%).
4. The closest archive relative of P3 is the old **`curriculum`** baseline, which fine-tuned the base
   maps on staged forced data with no conditioning. It was much worse than autonomous (SMAPE 33-103 vs
   19-65). Its reverse-order control worked as well as forward (DECK: "a curriculum whose reverse works
   equally well is not a curriculum"). These numbers predate the L (label-order) fix and use SMAPE,
   so they are invalid as magnitudes. They are not evidence about P3 under protocol v2.
5. An AAAI reviewer asked "Is there a performance benefit from the joint fine-tuning performed in
   Phase 2?". It was never answered with an experiment.

**Net.** The archive has no test of an unfrozen base or of joint training. Its one related LOO (B7)
says *more trainable pathways in phase 2 hurt*, and its schedule results say *more phase-1 steps
help*. Those two point in opposite directions for P1 and P2, so Stage 2 is needed.

## Stage 1 verdict (before Stage 2 launches)

- Base-map quality (eps_dec, eps_enc, eps_pde) does **not** predict KKL's losses to the data-driven
  baselines. Only the decoder Lipschitz proxy tracks absolute error, and it is a plant property.
- The losses come from a scenario-invariant fit floor of ~0.05 NRMSE on battery, duffing and quad_tank
  (bioreactor ~0.07-0.10), which the frozen-base phase 2 does not get below, while it removes 74-99% of
  the autonomous error everywhere.
- Prediction for Stage 2, written down before the runs: P4 (grey-box PDE) will *not* close the gap,
  because the PDE term does not move the deployed error. If anything helps, it will be an arm that lets
  the decoder features change on forced data (P1, P2 or P3). P3 will show how much of that is base repair.
