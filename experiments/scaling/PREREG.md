# Scaling study (many states): pre-registration

Written 2026-10-27 (Dubai), finished before ANY full observer run of this study (timestamp at the end of
section 9). Before this file: plant sanity simulations, the inclusion / information pre-checks
(section 1), unit tests, and three smoke runs at n_x = 8 (20 optimiser steps, 8 evaluation episodes:
plumbing only, numbers meaningless, on plant revision 1, outputs moved to `runs_smoke_rev1/`). Later
changes go into timestamped addenda at the end; the text above them is never edited.

Question (author): with many states, which observer is best as the state dimension grows, when every
learned observer's capacity and training data grow by the same fixed rule?

## 0. Isolation and code

* Code: `code/` (new). `code/pkg/` = byte-identical copy of the beat_greybox A3 scratch package
  (`code/PKG_SOURCE.txt`); `code/b2s/` = byte-identical copies of `experiments/baselines2/code/b2/{models,
  train}.py`. The live package, its caches, `experiments/tree`, the paper, CLAIMS.md and other agents'
  folders are never written. New plants are registered from `code/scaleplants.py` (no existing plant
  class or cache-hashed module is touched). Caches: `experiments/scaling/cache_rev2/`.
* Data (scales, training, evaluation episodes) are generated on the CPU (one thread) and shared by all
  GPU jobs; every method at a given (n_x, seed, dataset) sees exactly the same episodes.
* GPUs: 0-7, <= 3 of our processes per GPU, least-loaded first; no new job on 0-4 while
  `experiments/GPU_0_4_BLOCKED` exists or a non-hyperkkl process is visible there; jobs killed by the guard
  are re-run on 5-7. CPU latency on one core with taskset, torch threads = 1.

## 1. Plant family and inclusion

`fhnring{N}` (code/scaleplants.py, docstring has the equations): a ring of N heterogeneous
FitzHugh-Nagumo nodes (the `network` plant generalised), n_x = 2N in {4, 8, 16, 32, 64} (+128 for the
pre-checks only, section 7). Node parameters of `network`; gap-junction coupling kappa = 0.5; recovery
rates eps_i = 0.08 exp(spread (2i/(N-1) - 1)), spread = 1.0; an excitatory chemical synapse between ring
neighbours g_syn (E_syn - v_i) sum_j a_ij sig((v_j - th)/s), g_syn 0.15, E_syn 2, th 1, s 0.25; input on
node 0 through the `network` saturating actuator. Every node has two neighbour slots at every N (N = 2:
a_01 = 2). T = 100, dt = 0.05 (N_steps = 2000) at every n_x, t_ref 1.5, IC box v in [-2.5, 2.5],
w in [-1.5, 1.5] per node.

* **Sensors**: v of every even node (n_y = N/2 = n_x/4, 25 % of the states, 50 % of the nodes); every
  unmeasured node has two measured neighbours (n_x = 4: v_0 only).
* **Missing physics (partial model)**: the chemical synapse (the coupling nonlinearity) is absent from
  f_known; the gap junctions are known. Identical per node at every N. Relative one-step model error
  ||f - f_known|| / ||f|| ~ 0.26 at every n_x (sanity runs).
* **Knowledge levels** (the author's spectrum): `exact` (true f, white Gaussian sensor 0.03 y_std),
  `off30` (nominal f, real plant with every parameter off by exp(+-0.3), alternating over sorted names,
  I_ext = an operating condition, not perturbed), `partial` (f_known, real plant nominal), `realistic`
  (f_known + 30 % parameter error + AR(1) rho 0.9 Student-t nu 3 sensor noise, as TEST-C3).
  Datasets: D0 (exact and partial share the episodes), D30 (off30), DR (realistic).
* **Revision history (before any observer run)**: rev 1 used `network`'s spread 0.6 and failed the
  boundedness criterion (2a) at n_x = 16 and 32 (slow phase dispersion: S(5T)/S(T) up to 2.3 while
  |v| <= 2.5 always); rev 2 (this) uses spread 1.0 (diagnostic at n_x = 16: 1.16). STATUS.md has the log.
* **Inclusion (pre-registered rule, experiments/cartpole_bounded/INCLUSION.md, applied unchanged with
  its functions; code/precheck.py; outputs inclusion/ring{n_x}.json)** - table in section 8 (filled
  before any observer run). Extra, not part of the rule: boundedness of the 30 %-perturbed real plant.

## 2. Latent design

n_y = 1 (n_x = 4): the package convention, Butterworth, n_z = 2 n_x + 1 = 9, alpha = 0.3. n_y >= 2: the
package convention n_z = n_x + 2 on one Butterworth circle does not scale (pole radius alpha / sin(pi / 2
n_z): dt and the episode sample count grow ~ n_z), so each measured channel gets its own 5-pole
Butterworth bank (alpha 0.3): A = blockdiag(A_5, ...), B = blockdiag(1_5, ...), n_z = 5 n_y = 1.25 n_x
(= n_x + 2 at n_x = 8, the reference size). The package design gate holds per bank (unit-tested). One
grid (dt = 0.05) for every n_x.

## 3. Capacity and data rule (fixed)

Reference n_x = 8 = the package defaults. With r = n_x / 8:
* every MLP hidden width x sqrt(r) (so its parameter count ~ x r): phase-1 encoder 192 sqrt(r), base
  decoder 128 sqrt(r), NLOX widths (64, 128) sqrt(r), grey-box / UDE / NODE residual MLP 64 sqrt(r);
* every solved-for budget x r: HyperKKL / concat / latent conditioner 140k r, GRU 140k r;
  context encoder (SFM, input u only, scalar) and the A3 history channel keep their package sizes;
* training data x r: n_train = 600 r forced episodes, n_p1 = 400 r zero-input episodes, collocation
  32768 r states; validation 64 episodes (selection only) at every n_x. Transitions = 2000 (n_train +
  n_p1) = 2e6 r, so (training transitions) / (trainable parameters) is constant over n_x for each method
  up to width rounding; all methods at one (n_x, seed, dataset) get the same episodes. Grey-box residual
  fit: its least-squares/Adam pool = all transitions capped at 4e6 (drawn from every episode; the cap
  binds at n_x >= 32 and is reported). Trainable parameters and transitions per parameter are reported
  per method and n_x.
* optimiser steps and batch sizes are the package values at every n_x (not scaled).
* **Fixed-data variant**: at n_x = 64 only, every learned method trained on the n_x = 4 amount of data
  (n_train 300, n_p1 200, collocation 16384) with the n_x = 64 capacity; levels partial and realistic.

## 4. Observers (one fixed configuration each; no per-cell tuning)

| name | what | knowledge |
|---|---|---|
| hkkl_A3 | HyperKKL lora r = 4, recipe A3 = SOAP (phase 1 and 2, lr 1e-2) + base decoder trainable in phase 2 (lr x 0.1) + strided z-window (20 samples, stride round(6/(alpha dt)/20)) into the hypernetwork context (beat_greybox) | model in phase 1 |
| hkkl_A0 | HyperKKL lora r = 4, package default (Adam) | model in phase 1 |
| concat, latent | the C1 alternatives on the SAME A0 base maps, SAME conditioner budget, SFM context | model in phase 1 |
| ekf, ukf | package EKF / UKF on the modeller's model, nominal Q, R, P0 | f_known |
| greybox_ekf, greybox_ukf | modeller's model + one-step residual MLP (baselines_data.fit_all; gb_lr 1e-3) | f_known |
| ude_ekf, ude_ukf | modeller's model + residual trained on K = 32 rollouts (baselines2 train_dyn), lr 1e-2 | f_known |
| node_ekf | model-free neural ODE (same trainer, no f_known) + EKF, lr 1e-2 | none |
| gru | 2-layer GRU observer, TBPTT (baselines2), lr 1e-2 | none |
| nlox | NLOX (train.nlox), lr 3e-3 | none |

Fixed learning rates for the baselines2 families = their modal selection in baselines2 (no grid here:
budget). Model-based observers are trained per knowledge level; model-free ones per dataset (their
exact and partial entries are the same run). Seeds 0, 1, 2 (3, 4 at n_x = 4, 8 only if time remains).

## 5. Metric and statistics

* Primary: settled NRMSE of the **unmeasured** states (`metrics.episode_metrics`, settle 0.5 N,
  divergence clamp as the package), per seed the mean over the 64 episodes of a tier; tiers id,
  shape_ood, sequence_ood (clean = the level's own sensor). Secondary: all-state NRMSE, divergence rate.
* "average over levels" = mean over the 4 levels of the seed-mean; "worst case" = max over levels.
* Paired comparison A vs B on one (n_x, level, tier): per-seed relative change of the mean episode
  error, Student-t 95 % CI over seeds (paired_relative); "better" = CI below 0. Families of comparisons
  are Holm-corrected (alpha 0.05) where a prediction aggregates several tests.
* Training cost: wall-clock process-hours per unit (shared GPUs: an upper bound on GPU time) and data
  used; inference: section 6.

## 6. Inference cost

code/latency.py: per-step latency of advance + estimate at batch 1, (a) one CPU core (taskset, threads =
1), (b) GPU, median of 7 x 200 steps; FLOPs per step by torch FlopCounterMode (matmul-type ops only:
matrix_exp / inv / cholesky are not counted, so filter FLOPs are lower bounds). Scaling exponent b in
cost ~ n_x^b: OLS of log cost on log n_x over n_x in {4, ..., 64} (+128 if it ran), slope with a 95 % t-CI.

## 7. Grid, budget and declared cuts

Grid: 5 n_x x 3 seeds x {kkl units x 4 levels, gb units x 4 levels, model-free units x 3 datasets} +
fixed-data variant (n_x = 64, 2 levels / 2 datasets, 3 seeds). Projected cost (smoke timings x full
steps): ~7 process-hours per (n_x, seed) => ~110 process-hours on shared GPUs, above the ~60 GPU-h guide,
so these cuts are declared now (not silent): **n_x = 128 gets the pre-checks and the latency benchmark
only (no observer training); NODE + EKF is not trained at n_x = 64 (latency only); the fixed-data variant
only at partial and realistic; seeds 3-4 at n_x = 4, 8 only after everything else.** If the projected
wall time after the seed-0 wave exceeds 30 h, seed 2 of the n_x = 64 `off30` cells is dropped next
(announced in STATUS.md before it is applied).

## 8. Predictions (stated before any observer result)

**P1 (inference exponent).** The CPU batch-1 latency exponent of each of ekf, ukf, ude_ekf, ude_ukf is
larger than hkkl_A3's (and hkkl_A0's). Test: slope difference with 95 % CI (combined OLS standard errors),
Holm over the 8 pairs; supported if all 8 are significant. Expected: EKF ~ n^3 (Jacobian + matrix_exp
cubic, 2 < b <= 3 over this range where launch overhead flattens small n), UKF ~ n x model cost (b ~ 2),
HyperKKL ~ linear in the latent + decoder width (b < 1 at these sizes).

**P2 (C1 dimension sweep).** HyperKKL's advantage over concat and over latent grows with n_x:
adv_B(n_x) = mean over seeds and the 4 levels of (err_B - err_A0) / err_B (unmeasured, A0 = the
budget- and base-matched lora). Spearman rho over the 5 n_x >= 0.8 on shape_ood AND on sequence_ood, for
B = concat AND B = latent (4 checks; supported if all 4 hold; per-level rho reported too). Also reported
for hkkl_A3.

**P3 (most accurate at n_x = 64, sequence_ood, unmeasured; my guess):** exact -> ukf; off30 -> ude_ukf;
partial -> ude_ukf; realistic -> hkkl_A3. Scored as hit / miss per level (best seed-mean).

**P4 (bits rule, experiments/information).** A = Bayesian PCRB of the unmeasured states for the window
3/alpha under each dataset's real plant and sensor (precheck, inclusion/ring*.json `budget`). Rule:
HyperKKL cannot beat the best grey-box filter on a cell where A > eps (eps = the best grey-box / UDE
unmeasured error on that cell); where A < eps the rule is silent ("can win"). Per n_x the A values are
in the table below; prediction: A stays ~0.007-0.009 at every n_x (it does not grow with n_x because the
sensing is local), so the rule never forbids a win, i.e. every cell is "can win"; whether HyperKKL then
wins is P3's question. Reported mechanically: fraction of cells where the rule forbids, and the outcome
there.

Reporting is mechanical: every table for every method, tier and level in RESULTS.md / results.json,
including the ones that contradict these predictions.

## 9. Inclusion and budget table (filled from inclusion/ring*.json before any observer run)

(see addendum below)

### Section 9 table (rev 2; generated by code/incl_table.py at 2026-10-27 14:20 Dubai)

| n_x | n_y | n_z | observable? | bounded? (rule) | max unmeasured full-past CRB; Gramian median lmin/lmax (frac full rank) | worst S(5T)/S(T); secular | perturbed-30% plant bounded? (extra; worst ratio) | A = window PCRB unmeasured D0 / D30 / DR |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 9 | yes | yes | 0.0110; 2.9e-03 (100%) | 1.35; 1.17 | yes (1.27) | 0.0073 / 0.0072 / 0.0074 |
| 8 | 2 | 10 | yes | yes | 0.0119; 1.5e-05 (100%) | 1.11; 1.20 | yes (1.14) | 0.0075 / 0.0074 / 0.0076 |
| 16 | 4 | 20 | yes | yes | 0.0123; 5.7e-06 (100%) | 1.14; 1.14 | no (2.14) | 0.0078 / 0.0083 / 0.0087 |
| 32 | 8 | 40 | yes | yes | 0.0140; 3.7e-06 (100%) | 1.64; 1.24 | yes (1.74) | 0.0086 / 0.0090 / 0.0095 |
| 64 | | | pending | | | | | |
| 128 | | | pending | | | | | |

n_x = 4-32 pass both pre-registered criteria. The extra check on the 30 %-perturbed real plant (off30 /
realistic levels; not part of the rule) is marginal at n_x = 16 (2.14 > 2 on the training-input set,
slow phase dispersion, states bounded); disclosed, cells kept. n_x = 64 and 128: the check is still
running; **the launcher gives no observer job at an n_x before inclusion/ring{n_x}.json exists and
passes** (code/plan.py, function included), and the row is appended here when it lands.

**PREREG fixed: 2026-10-27 14:20 Dubai (before any full observer run).**

## Addendum 1 (2026-10-27 15:27 Dubai, before any rev-3 observer run): plant revision 3

The n_x = 64 inclusion check of rev 2 FAILED boundedness (2a) (worst S(5T)/S(T) = 4.2 on shape_ood inputs;
all states stayed within |v| <= 2.5, secular ratio <= 1.64; mechanism: with a monotone eps ramp along a
32-node ring, neighbours are nearly identical, phase-lock in clusters and disperse slowly, so the
across-episode spread at T is far below its stationary value). Per the rule no observer may run at that
n_x. To keep ONE plant family over all n_x, rev 3 changes only the placement of the recovery rates:
eps_i = 0.08 exp(spread (2 frac(i g) - 1)), g = golden ratio conjugate, spread 1.0 (same range; ring
neighbours always differ). Diagnostic before adoption (n_x 64 / 16, train inputs): 1.10 / 1.24
(perturbed plant 1.49 / 1.32). Everything else in this file is unchanged. The rev-2 observer runs
(started 14:20, stopped now) are void, moved to runs_rev2_void/ and never analysed (only six gb and six
mf 'done' summary lines at n_x 4 / 8 were displayed; they did not inform this change, which the
inclusion rule dictates). The inclusion pre-checks are re-run for every n_x on rev 3 and the launcher
again gates every n_x on its own pass; the table is appended below when it lands. If an n_x fails on rev
3, it is excluded from the observer grid (no further plant change).
