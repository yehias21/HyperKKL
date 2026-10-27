# P2 re-score with the SOAP recipe: pre-registration

Written 2026-10-26 17:05 Dubai, BEFORE any training or evaluation run of this study (no smoke run
either). Later changes go into timestamped addenda below; the text above them is never edited.

## Goal

Re-measure the HyperKKL-vs-best-grey-box gap of the information-theory result P2
(`experiments/information/RESULTS.md` section 4, `score.py`) with the SOAP recipe instead of the old
runs (C3-regimes / TEST-C3: Adam, frozen base decoder), on battery, quad_tank, cartpole (+ fhn,
optional), and recompute P2 with fold ratios.

## Recipe (fixed, copied, not tuned)

* Code: the beat-greybox campaign's scratch package and runner, copied byte-identical at 17:00
  Dubai into `code/pkg` (sha1 list `code/pkg.sha1`); `code/run_p2.py` = the campaign's `run_bg.py`
  with only the output/model directories changed. The campaign's files and the live package are not
  touched.
* Arms, exactly as `experiments/beat_greybox/PREREG.md` section 2:
  **A2** = SOAP (phase 1 and 2, lr 1e-2) + base decoder trainable in phase 2 at lr x 0.1;
  **A4** = A2 + strided z-window context (k = 20, stride rule `max(1, round(6/(alpha dt)/20))`) +
  richer decoder (width 256, budget 280k, 6000 phase-2 steps). No hyperparameter is changed for any
  plant.
* Config of every job = the `config` stored in the old run's own `results.json`
  (TEST-C3 for cartpole/fhn), evaluation condition `clean`, 64 episodes per tier, tiers id,
  shape_ood, sequence_ood; training and evaluation episodes read read-only from the live cache (no
  rebuild), so they are the episodes the stored grey-box filters were scored on. Seeds 0, 1, 2.

## Cartpole and fhn are held-out TEST plants

No tuning, no selection, no re-run "to fix" a result: A2 and A4 are run once per seed with the
recipe above and every result is reported, good or bad. A failed/diverged job is reported as such
(re-launch only for an infrastructure failure, e.g. CUDA OOM, with the same settings). fhn: A2 only
(optional, lower priority).

## Reused results (not re-run)

* battery, quad_tank: the campaign's C3-partial A2 and A4 cells, seeds 0-2
  (`.../scratchpad/beat_greybox/runs/C3-partial/<system>/seed*/A{2,4}.json`). C3-partial and
  C3-regimes have identical system/data/phase configs and identical clean results for every stored
  method (checked: lora, ekf and all grey-box filters equal per seed and tier), so these are the
  P2 scenario. quad_tank X0 (`experiments/quadtank/runs`) equals A2 to 4 digits (its own check).

## Quantities and statistics

* Per seed and tier: HyperKKL total NRMSE = mean over the 64 episodes of the settled NRMSE; and the
  unmeasured-state NRMSE = scorecard `nrmse_unmeasured` (as `information/score.py`/
  `collect_achieved.py`).
* Best grey-box per plant x tier x metric: among greybox_ekf, greybox_ukf, greybox_ekf_oracle of the
  stored partial-knowledge run, the one with the lowest seed-mean (oracle pick favouring the
  baseline, as score.py).
* Fold ratio = HyperKKL / best grey-box. Point estimate = ratio of seed-means (as score.py).
  95% CI = paired over seeds: d_s = log(HyperKKL_s / greybox_s), Student-t with df = 2,
  exponentiated.
* Old recipe = the `lora` method of the stored run (C3-regimes / TEST-C3), same seeds/episodes.
  Reported: old fold, new fold, and new/old (paired log-ratio t-CI).

## P2 re-score

Same groups as `information/RESULTS.md` table 1; duffing excluded (author decision, as the
campaign's addendum 1). Gap measure = fold ratio of unmeasured NRMSE (sequence_ood primary, id
secondary). Per plant the SOAP fold where available (best of A2/A4 is NOT used for the headline:
headline = A2, the recipe requested; A4 reported alongside), the old fold elsewhere (flagged).
Statistic: median fold of window-starved (battery, quad_tank, cartpole, highway_traffic) vs median
fold of rich (bioreactor, lorenz, vdp, fhn, rossler; network if available); geometric means
reported too. P2 holds if median_ws > median_rich and the number of window-starved plants above the
rich median is >= 4 of 4 (strict, the proportion of the original 4/5) — also reported at >= 3 of 4.

## Addendum 1 (2026-10-26 16:57 Dubai), before any cartpole/fhn result exists

* The campaign also has C3-partial A2/A4 seeds 0-2 for the rich dev plants bioreactor, lorenz, vdp
  (C3-partial == C3-regimes config; clean results of lora and the best grey-box identical per seed;
  on lorenz only non-best grey-box filters differ slightly between the two stored runs). The P2
  re-score uses them too ("SOAP where available" applies to both groups; otherwise the rich side
  would be scored with the old recipe while the window-starved side gets the new one). Reported both
  ways: (i) SOAP on all plants where available, (ii) SOAP on the window-starved side only.
* The task's quoted old gaps (battery 3.9x, quad_tank 5.1x, cartpole 9.3x, fhn 4.5x) are folds of
  TOTAL NRMSE on sequence_ood; score.py's P2 uses UNMEASURED NRMSE. Both are reported; P2 verdicts
  for both metrics, unmeasured stays the primary (as score.py).

## Addendum 2 (2026-10-26 17:21 Dubai): cartpole diagnosis (eval only, author approved), written before any diagnostic evaluation

* Models (no retraining): old-recipe stored `lora` of TEST-C3 cartpole seeds 0-2 (phase-1 + phase-2
  checkpoints from the live model cache, loaded read-only with the run's own code snapshot
  `experiments/code/b2cb5b3dfe80.tar.gz`; if they do not load, reported and dropped), and A2, A4
  seeds 0-2 (this study's cached models). Reference: greybox_ukf, re-fitted in the snapshot code
  (`baselines_data.fit_all`, same data/seed/options; the stored fit was never saved) and checked
  against the stored clean numbers.
* Same 64 x {id, shape_ood, sequence_ood} evaluation episodes; conditions on the same simulated
  trajectories: `stored` (the original coloured Student-t draw), `ct_redraw` (same law, new draw),
  `noiseless` (y = clean y), `gauss_white` (Gaussian, same variance), `gauss_col` (Gaussian AR(1),
  rho 0.9, same variance: colour without heavy tails).
* Reported: per state x1..x4 NRMSE (x2 cart velocity, x4 angular velocity unmeasured; x1, x3
  measured), total and unmeasured NRMSE, l_dec_q95 per model; Bayesian PCRB per state at window
  3/alpha (= 2 s) vs full past from `information/budgets.json` (already computed, eval_id, Gaussian
  sensor model, TEST-C1 plant; indicative).
* Decision rule, per recipe (old, A2, A4), on the seed-mean unmeasured NRMSE, sequence_ood primary:
  drop = 1 - noiseless / stored. drop >= 50% -> **noise amplification** (fix: decoder smoothness);
  drop < 20% AND HyperKKL velocity error within 2x the 3/alpha CRB -> **memory/observability**;
  otherwise **mixed**. "Heavy tails/colour vs level" = gauss_white and gauss_col vs ct_redraw.

## Addendum 3 (2026-10-26 17:35 Dubai): extrapolation test (eval only, no retraining), written before any coverage number was computed

* Training envelope per plant and seed = per-state [min, max] over every state the learned maps saw:
  phase-1 episodes, phase-2 training episodes (all time steps) and phase-1 collocation points
  (`data.training_data`: p1, train, coll). Eval states = all settled time steps (k >= 0.5 N, the
  NRMSE window) of the 64 episodes per tier. A step is "outside" for a state if that state is
  outside its [min, max]; "jointly outside" if any state is.
* Per tier: per-state quantiles (1, 50, 99%) of train vs eval, the fraction of steps outside (per
  state, joint), and the HyperKKL error RMS (|xhat - x| / x_std) for steps inside vs outside (jointly):
  unmeasured mean, and x3 and x4 for cartpole. Models: cartpole old lora + A2 + A4 (+ greybox_ukf as a
  reference), highway_traffic old lora (TEST-C3), quad_tank and battery A2 (campaign C3-partial cached
  models, read-only). Seeds 0-2.
* Cart drift: the distribution of x1 at t = 0, T/2 and T per tier, and a numeric check of whether
  the true dynamics f(x, u) are invariant to a shift of x1.
* Rule, per plant x model x tier (sequence_ood primary): **extrapolation** if > 30% of the settled eval
  steps are jointly outside the envelope AND the unmeasured error outside >= 3x the error inside.

## Addendum 4 (2026-10-26 18:19 Dubai): joint coverage (eval only), written before any joint-coverage number was computed

* Plants/models: cartpole (old lora, A2, A4), fhn (A2), two_link_arm (old lora; no A2 exists), highway_traffic
  (old lora) on TEST-C3; quad_tank, battery (A2) on C3-partial; seeds 0-2, all tiers.
* State space normalised by x_std. Reference set R = training states (phase-1 episodes, phase-2 training
  episodes, every 5th time step; plus all collocation points). NN distance d(x) = Euclidean distance to the
  nearest point of R (KD-tree, CPU).
* Null distribution: d over the settled steps of the VALIDATION episodes (data.val: same law as the training
  episodes but not in R, so near-duplicate neighbours from the same trajectory cannot shrink it).
  Threshold d99 = its 99th percentile. Eval: fraction of settled eval steps with d > d99, and HyperKKL's
  unmeasured error (RMS of |xhat - x| / x_std over unmeasured states) for d > d99 vs d <= d99; error by
  distance quintile; Spearman rho between d and the per-step unmeasured error.
* Rule, per plant x model x tier (sequence_ood primary): **coverage-limited** if > 10% of settled eval steps
  have d > d99 AND the unmeasured error there is >= 2x the rest.
* Exploration (reported): per-state std of training episodes, collocation and eval; fhn: the radius
  distribution of training vs eval states about the equilibrium-centre of the eval cloud, to see whether
  training only visits the limit cycle.
