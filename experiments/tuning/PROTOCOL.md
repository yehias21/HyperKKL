# C1 per-variant tuning study: pre-registration

Written 2026-10-25 22:45 CET, **before** any tuning code was run on real data and before
any tuning job was queued. Sections 1-6 are frozen from this point; anything changed later
goes into section 8 (deviations) with a date and a reason, and never edits sections 1-6.

Motivation: Park review (`theory/reviews/editors/park.md`), section 4 (a) one learning rate
(2e-3) for every variant, (b) hypernetworks are step-size sensitive, R3 (per-variant lr chosen
on validation), R8 (generator depth), MUST-FIX 1 (hyper is a k = 4 linear-subspace generator at
140k, `paper/generated/tab6_capacity.tex`). Question: does the C1 conclusion (weight-space
conditioning beats latent injection and the additive "concat" residual) survive when every
variant gets its own, equally sized, tuning budget?

## 1. Fairness definition

Every variant v in {latent, concat, film, lora, hyper} gets:

**(a) The same trainable-parameter budget.** `phase2.budget = 140000` conditioner parameters,
solved per configuration by `conditioning.match_width` (one width knob per variant; for hyper
the trunk output width k, i.e. the dimension of the weight-update subspace). The context
encoder (SFM, dim 16, shared) is not counted (`budget_includes_context: false`, as in
C1-where). Actual counts and widths are reported per configuration and system in
`selection.json` (match_width's rounding: C1-where counts lie in 137.5k-145.2k).

**(b) The same training compute, in optimiser steps and samples.** 3000 Adam steps, warm-up
100, cosine decay, gradient clipping as now (`monitor.TrainMonitor`), batch = 32 episodes x
256 time points (decoder variants: 8192 (z_obs, c, x) samples per step; latent: 32 rollouts of
K = 256 steps from T0(x_k0), unchanged), validation every 250 steps on the fixed validation
draw, best-validation checkpoint. Same frozen base maps T0, T0+ (phase 1) and the same context
encoder architecture as C1-where. Reported per variant, not equalised: measured phase-2
wall-clock (`train.<method>.wall_s`; shared GPUs, 4 jobs per GPU, so indicative) and matmul
FLOPs per optimiser step, forward + backward (`torch.utils.flop_counter`, CPU, real shapes;
script `analysis/tuning_cost.py`). Latent's FLOPs per step differ by construction (K-step
BPTT); that is the existing protocol and is reported, not changed.

**(c) The same tuning budget.** N = 8 configurations per variant = 4 learning rates x 2 values
of one architecture knob from that variant's own design space (section 2). Every
configuration is trained once (seed 0) on each of the 6 dev systems.

## 2. Design spaces (8 configurations each)

Learning rate grid, identical for all variants: `phase2.lr` in {5e-4, 1e-3, 2e-3, 4e-3}.

| variant | knob (config key) | values | default |
|---|---|---|---|
| hyper | `phase2.gen_layers`: the hypernetwork generates weight/bias deltas only for the LAST L layers of the base decoder; at the same 140k budget match_width solves the trunk width k | all layers (key absent) / 2 | all |
| lora | `phase2.rank` (existing key) | 4 / 16 (rank 1 already shown worse, C1-rank1) | 4 |
| film | `phase2.film_depth`: hidden layers of the FiLM generator MLP | 1 / 2 | 1 |
| concat | `phase2.concat_depth`: hidden layers of the residual MLP R | 2 / 3 | 2 |
| latent | `phase2.latent_depth`: hidden layers of the injection MLP M | 2 / 3 | 2 |

Default configuration of every variant = lr 2e-3 and the default knob = exactly the C1-where
recipe (the new keys are absent there, and absent keys reproduce the old code bit for bit;
tested).

Per-method values are passed with a new method-string syntax `kind[key=value,...]`
(phase-2 keys only), e.g. `hyper[lr=0.001,gen_layers=2]`, composable with the existing
`kind@context:ctxkey=value` syntax.

## 3. Selection (node group C1-tune-select)

* Data: seed 0, the 6 dev systems (battery, bioreactor, duffing, lorenz, quad_tank, vdp).
  The selection nodes run with `eval.tiers: []`: no evaluation-tier episode (id, shape_ood,
  sequence_ood) is generated or scored. Test systems are not touched.
* Score of a configuration on a system: mean settled NRMSE (`metrics.episode_metrics`,
  settle_frac 0.5, divergence clamped as in evaluation) of the deployed observer (exact latent
  filter from the centre-of-box prior, decoded through the trained conditioner, as in the
  `clean` evaluation condition) on the 64 phase-2 validation episodes (`data.val`, the set
  phase 2 already uses for checkpoint selection). Stored by the job as
  `train.<method>.val_nrmse`.
* Score of a configuration: arithmetic mean of its 6 per-system scores.
* Selected configuration per variant: the argmin. A non-finite score counts as +inf. If any
  of the 48 (configuration, system) results of a variant is missing, the script refuses to
  select (no partial selection).
* Ties go to the default: if the argmin is not the default and its score is not at least 0.5%
  lower than the default's score (score_best > 0.995 x score_default), the default is selected.
* Computed only by `analysis/select_tuned.py` from the runs' results.json, written to
  `experiments/tuning/selection.json`. The C1-tuned and TEST-C1-tuned node.yaml files are
  generated from selection.json by `experiments/make_tree_tuning.py tuned`; no manual choice.
  Selection is never revisited after C1-tuned or TEST-C1-tuned results exist.
* Reported, not used: geometric mean and mean rank across systems, and the phase-2
  validation loss (`train.<method>.val`), with a flag where they would choose differently.

## 4. Confirmation nodes

* **C1-tuned** (parent C1-where): the 5 selected configurations, seeds 0, 1, 2, the 6 dev
  systems, all tiers (id, shape_ood, sequence_ood), clean condition. Same episodes as C1-where
  (`tree.EPISODE_KEYS` unchanged), so it pairs with C1-where seeds 0-2.
* **TEST-C1-tuned** (parent TEST-C1): the same 5 configurations, the 6 test systems, seeds 0,
  1, 2, run once.

## 5. Decision rule (C1-tuned, applied by `analysis/evidence.py`)

Weight-space conditioning (the better of tuned lora / tuned hyper by mean NRMSE in that block,
as in C1-where) beats tuned latent AND tuned concat with a paired 95% CI excluding 0 (I1:
Student-t over seeds) on >= 4/6 dev systems (C1-tuned) and on >= 4/6 test systems
(TEST-C1-tuned), on shape_ood and on sequence_ood, each separately. PASS iff all four
(dev x tier, test x tier) votes pass; FAIL if all four fail; otherwise MIXED/INCONCLUSIVE as
`analysis/rules.combine` decides. Reported next to it, no rule: id tier; tuned film; tuned vs
default per variant (C1-tuned vs C1-where seeds 0-2, TEST-C1-tuned vs TEST-C1, paired on the
same episodes through `evidence.cross_pair`).

## 6. Compute plan and reduction clause

Cost unit measurements from finished C1-where / TEST-C1 jobs (4 jobs per GPU): phase 1 about
125-140 s per (system, seed), cached; phase 2 latent 960-1475 s, concat/film/lora 32-68 s,
hyper 55-205 s; evaluation 60-540 s per job. Default plan: N = 8 per variant (above). The
reduced design (lr {5e-4, 1e-3, 2e-3} x 2 knob values = 6 per variant, equal across variants)
is used only if the estimate made before queueing exceeds ~6 h on GPUs 5-7 with 4 slots per
GPU; that decision and the estimate are recorded in section 7 before queueing.

## 7. Cost estimate and queueing record

Estimate written 2026-10-25 23:45 CET, before queueing (unit costs: section 6, measured on the
finished C1-where / TEST-C1 jobs at 4 jobs per GPU; data/scales caches are reused because no
DATA_MODULES file changes; phase 1 is rebuilt once per (system, seed)).

Selection: 10 nodes x 6 dev systems = 60 jobs (240 phase-2 trainings, seed 0, no evaluation).

| job type | jobs | per job | slot-seconds |
|---|---|---|---|
| latent (4 trainings x 960-1475 s) | 12 | ~5000 s | ~60k |
| hyper (4 x 55-205 s, + val/cost) | 12 | ~900 s | ~11k |
| concat, film, lora (4 x 32-68 s) | 36 | ~450 s | ~16k |
| phase-1 rebuild (6 x ~130 s, first job per system) | | | ~1k |

~88k slot-s; on GPUs 5-7 x 4 slots = 12 slots: latent nodes first (priority 32, the others 31), so
the 12 latent jobs fill the slots for ~1.4 h and the rest takes ~0.7 h: **~2.2 h** (less if
the worker still has GPUs 0-7 when it starts).

Confirmation: C1-tuned 18 jobs (seed 0's 5 trainings are cache hits of the selection) and
TEST-C1-tuned 18 jobs; per job 5 trainings (latent ~1200 s, others 60-200 s) + phase 1 +
evaluation of 5 methods on 3 tiers (100-550 s): ~2100 s, ~500 s for dev seed 0. ~66k slot-s,
36 jobs on 12 slots: **~1.8 h**. FLOPs per optimiser step (vdp, `analysis/tuning_cost.py`, scratch
run): latent 31.3 G, concat 17.0 G, film 10.9 G, lora 11.0 G (r16 11.3 G), hyper 12.4 G (gen2
10.7 G); SFM context 2.9 G of each.

Total ~4 h of GPU time after the main queue is empty, within the ~6 h target. **Decision: N = 8
per variant (no reduction).**

Queueing record: package change deployed 2026-10-26 02:51 CET (after the main queue's pending
list was empty; running jobs had already imported the old code). Live tests before queueing:
test_core + test_tuning (fast part, incl. bit-identity against the C1-where code snapshot
a1e5d75cce66) + test_verify_pipeline: 97 passed; in the scratch copy beforehand: test_tuning 46
passed, test_core/test_verify_models/test_verify_pipeline 186 passed (5 skipped), test_graphs +
test_verify_train on GPU 37 passed (1 known quad_tank xfail). Queued 2026-10-26 02:55 CET:
C1-tune-{latent-d2,latent-d3} (priority 32), C1-tune-{concat-d2,concat-d3,film-d1,film-d2,
lora-r4,lora-r16} (31), C1-tune-{hyper-all,hyper-gen2} (30, ablation): 60 jobs.

## 8. Deviations

* 2026-10-25 23:45 CET, before any job: clarification of section 5. The joint label is
  `analysis/rules.combine` of the four votes, which is an AND rule: PASS iff all four pass, FAIL
  if ANY of the four fails, otherwise INCONCLUSIVE. ("FAIL if all four fail" in section 5 was a
  wording slip; the implementation in `analysis/evidence.py` follows combine.)
* 2026-10-25 23:45 CET, before any job: selection nodes are 10 nodes `C1-tune-<variant>-<knob>`
  (4 learning rates each, children of the umbrella node C1-tune-select), not one node, so the
  latent trainings run in parallel. Same 40 configurations x 6 systems.

## Known caveats (stated in advance)

* The new knobs require editing `conditioning.py` and `train.py`, which are part of every cache
  key (`cache.MODEL_MODULES`), so phase-1 base maps are rebuilt once per (system, seed) for this
  study (same config and seed as C1-where) and shared by all its configurations. The
  cross-node comparison with C1-where therefore also re-draws the base maps; whether they are
  identical is checked by comparing the phase-1 diagnostics in results.json.
* The test systems are touched a second time (TEST-C1 ran the default recipe). The tuned
  recipe is fixed on dev data only, before TEST-C1-tuned is queued.
* Compute is equalised in optimiser steps and samples, not in FLOPs or wall-clock.
## 8 (continued). Deviations and records, in time order

* 2026-10-26 00:52 CET, before any tuning job was queued (author decision, relayed by the
  coordinator): the paper has ONE weight-space method, **HyperKKL** = the hypernetwork that
  generates rank-r low-rank decoder updates (the `lora` code path); the dense `hyper` generator
  is an appendix ablation only.
  - HyperKKL's 8 configurations = lr {5e-4, 1e-3, 2e-3, 4e-3} x rank {4, 16} (the `lora` design
    space of section 2, unchanged). Rank 16 is kept, not replaced by 32: C1-rank16 trained rank 16
    only at lr 2e-3, with no validation score and under the previous code hash, so it covers no
    part of this selection.
  - latent, concat, film: unchanged (8 configurations each).
  - hyper (dense): kept with the same 8-configuration budget as an ablation; selected by the same
    rule, reported in the appendix, never in the headline comparison.
  - Headline rule of section 5, replaced: **tuned HyperKKL (lora) beats tuned latent AND tuned
    concat** with a paired 95% CI excluding 0 on >= 4/6 dev systems (C1-tuned) and >= 4/6 test
    systems (TEST-C1-tuned), on shape_ood and on sequence_ood separately; label =
    `rules.combine` of the four votes. No best-of-two pick, so the rule is stricter than the
    original "better of tuned lora / hyper". The original version and tuned hyper vs every other
    variant are reported as secondary lines, not as the outcome.
* 2026-10-26 05:53 CET, record (not a deviation): all 60 selection jobs done 05:52; `analysis/select_tuned.py`
  wrote selection.json; geometric mean, mean rank and val loss agree with every choice. Selected:
  latent[lr=0.004,latent_depth=3], concat[lr=0.004,concat_depth=3], film[lr=0.004,film_depth=2],
  lora[lr=0.004] (HyperKKL), hyper[lr=0.004] (ablation). **Every variant selects the top of the lr
  grid (4e-3)**: the grid edge binds for all of them, so "tuned" means best within the
  pre-registered grid, not an interior optimum. Per protocol the grid is not extended. C1-tuned and
  TEST-C1-tuned generated from selection.json and queued 05:53 (36 jobs, priorities 25 / 24).
  Phase-1 base maps rebuilt under the new code hash are bit-identical to C1-where's (battery seed 0:
  T0, T0+ state dicts equal), so the C1-tuned vs C1-where pairing has identical base maps.

## 9. Addendum: equal learning-rate extension (written 2026-10-26 07:15 CET)

Reason (coordinator, after the selection): every variant selected lr = 4e-3, the top of the grid,
so the pre-registered selection is censored and every variant may be under-tuned. This addendum
does NOT replace sections 1-8. The pre-registered verdict (C1-tuned / TEST-C1-tuned, section 5 as
amended in section 8) is reported as it is. The extension below is reported next to it, labelled
"extended".

* ext1: for ALL five variants, lr in {8e-3, 1.6e-2} x the same two knob values, seed 0, the 6
  dev systems, eval.tiers = [] (nodes `C1-tune-<variant>-<knob>-ext1`, 2 methods each). That is
  4 new configurations per variant, equal for all.
* Selection: the same rule as section 3, over the union (12 configurations per variant). The
  default and the 0.5% tie rule are unchanged. Script: `analysis/select_tuned.py --stages pre,ext1`
  -> `experiments/tuning/selection_ext1.json` (selection.json is never rewritten).
* ext2: if ANY variant's ext1 choice is at lr 1.6e-2 (the new edge), lr 3.2e-2 x both knob
  values is added for ALL variants (`-ext2` nodes), selected over the union of 14 ->
  selection_ext2.json. No further extension.
* Confirmation: only variants whose extended selection differs from selection.json are re-run,
  with 3 seeds on dev (`C1-tuned-ext`) and test (`TEST-C1-tuned-ext`). Unchanged variants reuse
  their C1-tuned / TEST-C1-tuned runs. Same episodes, same base maps, paired per seed.
* Extended verdict: the section-8 headline rule (tuned HyperKKL beats tuned latent AND tuned
  concat, >= 4/6 dev and >= 4/6 test, shape_ood and sequence_ood) applied to the extended
  selection.
* SOAP is added as a separate, equally applied factor only if the beat-the-grey-box campaign
  (experiments/beat_greybox/RESULTS.md) confirms it; this is decided by reading that file, and
  none of its runs are duplicated here.
* Queued before TEST-C1-tuned finished, at a lower priority (22 vs 24), so it cannot delay that node.
  The design above was fixed before any TEST-C1-tuned result was read.

## 10. Addendum: shared context-scale extension (written 2026-10-26 09:26 CET, before any ext1 selection was computed)

Reason (coordinator, from experiments/c1_conditioning/AUDIT_HYPERKKL.md): the context RMS at
the conditioner input differs by system (0.07-0.23 at init) and by variant once trained. concat
benefits from per-system unit-RMS context normalisation; HyperKKL is hurt by it. For an equal
comparison, every variant's design space is extended by the same knob.

* Naming: the conditional 3.2e-2 step of section 9 is renamed **ext1b** (its rule is unchanged:
  it runs for ALL variants if any ext1 choice sits at 1.6e-2). **ext2** is the extension below.
* Knob `phase2.ctx_rms` in {absent (= no scaling, all earlier runs), 0.3, 1.0}. Implemented once,
  shared by all variants (`conditioning.ContextScale`, set up in `train.phase2`). The
  conditioner receives s * c with a FIXED scalar s = ctx_rms / RMS(c), where RMS(c) is taken
  over the post-washout training samples of the freshly initialised context encoder (per
  system, seed and variant, from training data only). No parameters are added (s is a buffer).
  c = 0 still maps to 0. Absent key = bit-identical to before (tested).
* ext2 design, equal for every variant: the knob value and lr* selected after ext1 (or ext1b);
  lr in {lr*/2, lr*, 2 lr*} x ctx_rms in {0.3, 1.0} = 6 new configurations per variant. The
  ctx_rms-absent row is the configurations already run. Seed 0, the 6 dev systems,
  eval.tiers = []. The design is written mechanically from the ext1(b) selection file by
  `analysis/select_tuned.py --write-ext2-design` -> experiments/tuning/ext2_design.json.
* Selection: the section-3 rule over the union of every configuration a variant has run (pre +
  ext1 [+ ext1b] + ext2; equal counts across variants) -> selection_ext2.json.
* Confirmation as in section 9 (3 seeds, dev + test, only variants whose selection changed vs the
  previous stage), nodes `C1-tuned-ext2` / `TEST-C1-tuned-ext2`.
* Verdicts reported separately and labelled: pre-registered (selection.json), ext1
  (selection_ext1[b].json), ext2 (selection_ext2.json), each with the section-8 headline rule.
* Added reporting (no rule): per system, tuned HyperKKL vs tuned concat and vs tuned latent
  (paired change and CI), with the plant dimensions (n_x, n_y) next to each system.
* 2026-10-26 10:16 CET, record: the ext1 choice for concat is concat[lr=0.016,concat_depth=3], at the
  new grid edge, so ext1b (lr 3.2e-2 x both knob values, ALL variants, section 9) was triggered and
  queued: 10 nodes `C1-tune-<variant>-<knob>-ext1b`, 60 jobs. The latent ext1 runs were still
  running; they cannot undo the trigger. Code deployed 09:42 (ctx_rms knob, absent = bit-identical).
* 2026-10-26 11:46 CET, record / correction: a post-hoc author decision (analysis/store.py,
  EXCLUDED_SYSTEMS = {duffing}) changed `store.DEV` to 5 systems. `analysis/select_tuned.py` used
  store.DEV, so the first selection_ext1b.json (11:44) silently averaged over 5 systems. That file
  was deleted unused. The script now uses the frozen pre-registered list TUNE_SYSTEMS (all 6 dev
  systems, section 3), and selection_ext1b.json was recomputed at 11:46. selection.json and
  selection_ext1.json were computed on all 6 systems and are unaffected.
  Result: ext1b changes no choice (lr 3.2e-2 never selected), so concat's lr 1.6e-2 is interior.
  Changed vs selection.json: latent -> latent[lr=0.008], concat -> concat[lr=0.016,concat_depth=3],
  HyperKKL -> lora[lr=0.008,rank=16]; film and hyper unchanged. Verdicts: dev systems are
  reported in both readings (all 6 = as pre-registered; 5 without duffing = the post-hoc headline
  reading of the results layer).
* 2026-10-26 16:22 CET, record: ext2 grid done (60 jobs); selection_ext2.json over 20 configurations per
  variant (8 pre + 4 ext1 + 2 ext1b + 6 ext2; equal for all). Selected: latent[lr=0.008,ctx_rms=0.3],
  concat[lr=0.016,concat_depth=3,ctx_rms=0.3] (0.6% better than the unscaled choice; the gmean,
  rank and val-loss checks prefer the unscaled one), film[lr=0.004,film_depth=2,ctx_rms=1],
  lora[lr=0.008,rank=16,ctx_rms=1] (HyperKKL), hyper[lr=0.002,ctx_rms=1]. All five changed, so all
  five are confirmed: C1-tuned-ext2 / TEST-C1-tuned-ext2 (36 jobs) queued 16:22.

## 11. Outcomes (mechanical, analysis/evidence.py; recorded 2026-10-26 17:42 CET)

Headline rule (section 8): tuned HyperKKL (lora) beats tuned latent AND tuned concat, paired 95% CI,
>= 4/6 dev and >= 4/6 test, shape_ood and sequence_ood. Dev counts are given for all 6 systems
(as pre-registered) and, in brackets, for the post-hoc 5-system reading without duffing (need 4/5).

| verdict | HyperKKL | dev shape | dev seq | test shape | test seq | label |
|---|---|---|---|---|---|---|
| pre-registered (selection.json) | lora[lr=0.004] | 3/6 (3/5) | 2/6 (2/5) | 4/6 | 3/6 | FAIL |
| ext1 (selection_ext1b.json) | lora[lr=0.008,rank=16] | 4/6 (4/5) | 3/6 (3/5) | 4/6 | 4/6 | FAIL |
| ext2 (selection_ext2.json) | lora[lr=0.008,rank=16,ctx_rms=1] | 3/6 (3/5) | 1/6 (1/5) | 5/6 | 5/6 | FAIL |

At every stage, on shape_ood HyperKKL beats both tuned baselines on every plant with n_x >= 4
(quad_tank, battery; cartpole, two_link_arm, highway_traffic, network). The misses are on the
2-3-state plants.
