# bench_v2: pre-registration of the all-baselines comparison

Written 2026-11-03 14:30 Dubai (UTC+4), BEFORE any non-smoke training or evaluation on v2 data. Only smoke runs
(20 steps, 8 episodes; numbers unused) existed. Later changes go into dated addenda at the end; the text above them is
never edited. Design: DESIGN.md. Code: code/ (v2run.py one job, build_data.py data, conditions_v2.py, dispatch.py).

## 1. Question
On the v2 benchmark (two families: autonomous A, non-autonomous B with realistic operating-profile tiers), how do all
existing baselines rank against HyperKKL default under equal data, a reported tuning budget, validation-only
selection and one scorer? Also: is HyperKKL in family A equivalent to an autonomous KKL observer (no conditioning
benefit)?

## 2. Cells
* Family B, level **realistic** (partial model, 30 % mismatch, AR(1) Student-t(3) noise): battery, bioreactor,
  duffing, lorenz, quad_tank, vdp, cartpole_b, fhn, highway_v2, network, rossler, two_link_arm_v2, lorenz96 (13 cells).
* Family A, level **partial** (white data, nominal plant): vdp, fhn, rossler, lorenz, lorenz96, network,
  lorenz96_n10 (7 cells).
* Real tracks (B by nature): battery_pf, battery_lg, emps, silverbox, with the subset of methods that the real-data
  adapters support (section 3), 3 seeds.
* Deployment (quad_planar incl. outliers, cstr_exo) and scaling (fhnring): lowest priority (section 6).

## 3. Methods and tuning budget (every method trained on the cell's MIXED data: train + train_event, 1200 episodes;
val + val_event, 128, for checkpoints; selection score = mean settled all-state NRMSE + divergence rate on val and
val_ood, never a test block)
| id | method | configurations tried (budget) |
|---|---|---|
| ekf, ukf | plain filters, nominal Q / R (no tuning) | 1 |
| ekf_gated, ukf_gated | + published chi-square gate (p 0.999, forced update after 5 rejections) | 1 |
| ekf_gated_tuned, ukf_gated_tuned | gated, Q std x {1, 3, 10, 30}, R std x {1, 3}, chosen on 32 val + 32 val_ood episodes | 8 |
| ude1_{ekf,ukf}_gated | corrected grey-box, ude5k1 style (f_known + 5k MLP, K = 1 through RK4, + phase1 episodes as transitions), gated | lr in {3e-4, 1e-3, 3e-3, 1e-2} x 3000 Adam steps (4) |
| ude32_{ekf,ukf}_gated | multi-step UDE (K = 32 RK4 rollouts after 300 K = 1 steps), gated | 4 x 3000 |
| node32_ekf_gated | neural ODE + gated EKF (no physics) | 4 x 3000 |
| gru | 2-layer GRU, ~140k, stateful TBPTT K = 256 | 4 x 3000 |
| ncde | neural CDE, ~140k (cost: dev subset, section 6) | 2 x 3000 (1e-3, 3e-3) |
| auto | autonomous KKL observer from HyperKKL's phase-1 maps | 0 (phase 1 only) |
| latent, concat, film | conditioning baselines, research harness phase 2 (Adam, 3000 steps, 140k conditioner) | lr in {2e-3, 1e-3} (2) |
| nlox_m | NLOX (Woelk et al. 2026) at the fairness rules: SOAP, params = hk_budget, 15000 steps | lr in {1e-2, 1e-3} (2) |
| dkkl_m | Peralez & Nadri 2024 switching DKKL, input-blind | 2 x 15000 |
| dkkl_inj_m | DKKL + model-based injection Psi (eq. 17) (cost: subset, section 6) | 2 x 15000 |
| hk_base | HyperKKL A3 + mixed (phase 1 A3 then phase 2 3000 SOAP steps) | phase-2 lr in {1e-2, 1e-3} (2) |
| **hyperkkl** | HyperKKL default = hk_base + C5 (jump rho 3, slope rho 0.3, spectral cap 0.5; per-plant rule) | 2 (same lr as its base) |
Every comparator gets at least as many configurations as HyperKKL (2). Within a method, the lr is selected per
(cell, seed) by the selection score. In family A, latent / concat / film / dkkl_inj_m are not run: with u = 0 they
reduce to the autonomous decoder (injection and context are zero or constant), so they would duplicate `auto` /
`dkkl_m`. Stated, not hidden.

## 4. Seeds, metric, statistics
* Seeds 0, 1, 2 for every learned method (minimum 3; 5 only if section 6 leaves capacity, reported per method).
  Filters are deterministic (seed 0 only), and their value enters every paired comparison as a constant.
* Primary metric: settled UNMEASURED NRMSE (bench scorer `settled-nrmse-1`), per seed the mean over the 64 episodes.
  Primary block: B `test_op_ood/clean`; A `test_id/clean`. Secondary blocks: every other block (tables).
* Paired comparison of method M against hyperkkl in a cell: per-seed log ratio log(M / hyperkkl), mean and 95 %
  t-interval over seeds; Holm correction over the cells of one family (per method pair). "M better" if the
  Holm-adjusted interval of the log ratio is entirely below 0, "worse" if entirely above, else "tie".
* Ranking: per cell, the methods sorted by the seed-mean primary metric. Reported per family: the best method per cell,
  hyperkkl's rank, and the median rank over cells.

## 5. Predictions (made now)
* B (realistic, op_ood): the corrected grey-box (ude1 / ude32 with the gated UKF or EKF) is the best method in the
  largest number of cells (>= 5 of 13). hyperkkl ranks in the top 3 in >= 6 of 13 cells and is best in 3-5, likely
  bioreactor, lorenz, vdp, lorenz96, battery. Plain ungated filters rank last or near last. Peers: nlox_m beats
  dkkl_m in >= 10 of 13 cells, and hyperkkl beats nlox_m in >= 10 of 13. auto is worse than every trained KKL variant.
* B op_ood / op_id ratio is close to 1 (<= 1.5) for model-based methods and larger for learned ones.
* Lorenz-96 (B, equal data): ude32 (multi-step UDE + EKF/UKF) is >= 10x worse than hyperkkl on op_ood (stopped
  earlier runs suggested 42-46x).
* network: C5 costs roughly one extra phase 2 of training time (c5_extra_wall_s ~ base_wall_s); C5 vs hk_base
  accuracy change on network within +-10 % on clean blocks, better under outliers.
* A (partial, id): **autonomous HyperKKL ~ base KKL**. The hypernetwork context is constant (u = 0), so any
  difference between hyperkkl and an autonomous KKL trained with the same budget comes from training, not
  conditioning. Prediction: hyperkkl / nlox_m (end-to-end autonomous KKL at the same budget; its injection is
  identically 0 at u = 0) lies within [0.67, 1.5] in >= 5 of 7 cells. hyperkkl < auto (phase-1 maps alone). No
  conditioning benefit will be claimed in family A, whatever the numbers.
* A: model-based filters (gated, tuned) and ude rank high on vdp / fhn (smooth, low-dimensional); chaotic lorenz96 is
  hard for every filter on the partial model.

## 6. Grid, priorities and cuts (no silent caps)
Order of the queue (dispatch.py, GPUs 2-7, <= 3 processes per GPU):
1. B core: all methods of section 3 except ncde / dkkl_inj_m, seeds 0-2.
2. Real tracks (filters, ude1/ude32, gru, hyperkkl A3, nlox_m where the masked-label adapter works).
3. A: all applicable methods except node32 and ncde, seeds 0-2.
4. ncde on B vdp, battery, cartpole_b, lorenz96 (seeds 0-2).
5. dkkl_inj_m on B vdp, battery, cartpole_b (seeds 0-2, both lr).
6. Deployment (quad_planar with outliers, cstr_exo), scaling, extra seeds 3-4.
Whatever of this order is not finished by the deadline is reported as CUT, item by item, in RESULTS.md. Exact
timings from the pilot go into Addendum 1 (before any main-run result is read).

## Deviations / addenda

### Addendum 1 (2026-11-03 15:35 Dubai; timings only, no main-run accuracy read)
Measured on GPUs 2-7 with 3 processes per GPU (wall per job, first non-smoke runs): phase 1 1.7-2.8 min; NLOX-M
24-28 min and DKKL 22-33 min per 15000-step lr configuration; real tracks on CPU: gated filters 0.1-2.7 min, tuned
0.3-4.2 min, ude1 1.4-6.7 min, GRU ~5.6 min, node32 ~30 min, ude32 ~36 min. Estimate of the whole grid of
section 6 (items 1-5): ~330 GPU-process-hours, i.e. ~18-19 h wall on the 18 slots. That is inside the window, so
no item of section 6 is cut in advance; items 1-5 run in the stated order. If capacity remains, item 6 adds seeds
3-4 for hyperkkl, hk_base, ude1, ude32 and nlox_m on family B (reported per method).

Fixes before any main result (code only, no protocol change):
(a) the real-track tuned filters crashed in hyperkkl_bench's TunedKF, which needs context["prior_mean"]; the
    bench_v2 subclass selects with the same grid and score from each validation episode's own track prior
    (PITFALLS P23);
(b) C5 weight guard: a penalty term that is exactly 0 gets weight 0 instead of fit/0. The guard never triggered in
    the smokes; in family A the context-slope term is small but not 0, so the per-plant rule applies unchanged;
(c) autonomous rossler at +0.3 (realistic / mismatch30 data, not run here): the further +0.15 shifted plant
    escapes the envelope from every IC, so test_param_shift is UNAVAILABLE there (MANIFEST). Family A runs at the
    white level, where the tier exists.

### Addendum 2 (2026-11-03, coordinator / author instruction, no result read for this decision)
The optional item 6 (seeds 3-4) was removed from the queue before any of its jobs started. Every learned method
reports seeds 0, 1, 2; filters seed 0 (deterministic).

### Addendum 3 (2026-11-04 08:15 Dubai; robustness of the harness, not of any method)
(a) A filter episode that fails numerically (torch LinAlgError: singular innovation covariance after divergence, seen
    for the gated UKF on driven/network under outliers) crashed the whole filters job. It is now rerun episode by
    episode, and an episode that still fails is scored as DIVERGED (NaN -> clipped error 100), the protocol's rule
    for diverged episodes. Affects only the filters jobs that crashed (network); they are rerun from scratch.
(b) Transient cuFFT CUFFT_INTERNAL_ERROR in CUDA-graph capture (latent method's input filter, as in the research
    runs): failed jobs get a final retry pass after the main queue. Up to 3 more attempts per job (as
    experiments/retry_cufft.sh); a job that still fails is reported as missing.
(c) Coordinator request (2026-11-04): the filter copy (code/pkg/hyperkkl/observers.py; the live package is untouched)
    solves with S through safe_solve / safe_inv: plain solve, else S + 1e-9 mean(diag S) I, else pseudo-inverse. Only
    reached when the plain solve raises, so no completed result changes. When it is reached, the regular episodes of the
    same batch see a 1e-9 relative jitter. The network filters job is rerun with it. Backup of the file before the
    change: code/observers_pre_jitter_fix.py.bak.

### Addendum 4 (2026-11-05, after all results; review round REVIEW_7). Deviations found in review, declared here
None of these changed any job or number; they correct what the text above says the study did.
(a) **phase1 is extra data for some methods (equal-data deviation).** Section 3 says "every method trained on the
    cell's MIXED data". In fact the HyperKKL family (hyperkkl, hk_base, auto, latent, concat, film; phase 1) and the
    UDE / NODE filters (one-step / rollout transitions; the research ude5k1 recipe) also used the 400 zero-input
    phase1 episodes (HyperKKL's phase 1 also uses model collocation states, which PROTOCOL counts as part of the
    method). GRU, NCDE, NLOX-M and DKKL used train + train_event only. This is an equal-data deviation (PITFALLS P29),
    stated in RESULTS.md next to every comparison.
(b) **phase1 overlaps the family-A IC shell.** Its ICs cover the whole envelope, and 56-100 % of them lie in the
    test_ic_ood shell. The test_ic_ood numbers of the methods in (a) are therefore not out-of-distribution
    (PITFALLS P30). The primary family-A block (test_id) is unaffected (core ICs, in-distribution by design).
(c) **Selection rule.** For ude1 / ude32 / node32 / gru / ncde, the lr was chosen INSIDE the job by the deployed
    validation NRMSE on val (+ val_event) only, as in the baselines2 recipe, not val + val_ood. The tuned filters used
    32 val + 32 val_ood episodes; HyperKKL, hk_base, peers and conditioning variants used the val + val_ood score
    (analyze.py). hyperkkl and hk_base are selected independently over the two lr runs, and C5 always uses the lr
    of the base it was trained from (within one run).
(d) **Seeds.** 3 seeds (Addendum 2). By PROTOCOL 5 these are reference results, not headline claims (5 seeds).
(e) Addendum 1 (a) refers to "PITFALLS P23"; the item is P28 in the package.
