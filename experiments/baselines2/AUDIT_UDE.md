# Sanity / leakage audit of the UDE and NODE pipeline

Written 2026-10-27 ~10:00 Dubai, at the coordinator's request, after the interim tables showed ude5k
beating the best grey-box filter and HyperKKL A3/A4+D3 on most cells. Script: `code/audit_ude.py`.
Raw output: `audit/<cell>.json`. It ran on CPU for 8 cells, seed 0: C3-partial bioreactor, battery,
lorenz and quad_tank; C3-mismatch30 battery and vdp; TEST-C3 highway_traffic; TEST-C3-B cartpole_b.
The code-path checks below hold for every cell, because the same code runs everywhere.

**Verdict: no leak or bug found.** The UDE filter gets no information that the grey-box filter does not
also get. Its advantage comes from the model it learns. Once both models are run open-loop, the UDE
model is 10-1000x more accurate than the one-step residual model, and on several cells it is almost as
accurate as the true plant. That advantage is partly confounded with tuning: the UDE gets an lr grid
and the grey-box does not. A post-hoc control, `ude5k1`, separates the two effects; it is described in
PREREG addendum 2 and its results are in section 6.

## (1) Which model the filter holds

* `job.py` builds `self.model = hyperkkl.data.model_plant(cfg["system"])`, the function that every
  grey-box and model-based filter uses. It calls `build_plant(scfg, knowledge=...)` with **no
  `perturb` argument**, so the model has nominal parameters. Under `knowledge: partial` it is also
  wrapped in `KnownPart`, so the model's `f` is `f_known`.
* The audit checked this numerically. The model's parameters equal the nominal parameters on all 8
  cells. They differ from the data-generating plant's parameters on every 30%-mismatch cell
  (C3-mismatch30 and TEST-C3: `params==real False`).
* `f_known` is not the true `f`. The median relative error |f_real - f_model| / |f_real| on training
  states is: bioreactor/partial 643, lorenz/partial 1.5, battery/partial 1.37, battery/mm30 0.38,
  vdp/mm30 0.17, quad_tank/partial 0.025, cartpole_b 0.89, highway 2.6.
* The real plant, `build_plant(sc, perturb=mismatch)`, appears in only two places:
  * the data generator (unchanged package code);
  * the audit's "TRUE plant (reference only)" row below.

  `b2/` never references it.
* Q, R and P_0 are computed exactly as in `evaluate.build_observers` for the nominal EKF and the
  grey-box:
  * Q = noise_proc · x_std;
  * R = nominal sensor noise · y_std (on TEST-C3 this is not the coloured-noise oracle);
  * P_0 = (IC-box width) / √12.

  The initial state is the same centre-of-box prior (`evaluate.initial_guess`, "center").

## (2) Training data

* The UDE and NODE train on `data['train']` (600 forced episodes) plus `data['p1']` (400 zero-input
  episodes). These are the same sets the grey-box residual fits on (`baselines_data._sets(data, "train")`).
  HyperKKL phase 1 uses `p1` and phase 2 uses `train`.
* The episode length N+1 is 1620-7663 samples, so there are 1.6M-7.7M transitions. The grey-box
  sub-samples at most 1M of them (`gb_n_fit`); the UDE samples windows from all of them.
* Training families: zero, constant, step, staircase, prbs, square, sinusoid, multisine, ramp, am and
  drive (`signals.TRAIN_FAMILIES`). These are the same for every method.
* Evaluation episodes: on every audited cell, **0 of 64 id / shape_ood / sequence_ood episodes and 0 of
  64 validation episodes match a training episode** (hash of the first 4 states). Evaluation seeds come
  from `split_seed`, which is independent of the training seed.

## (3) Selection

The lr is chosen by `train.deployed_val` on `data['val']` (64 episodes), centre prior, settled NRMSE plus
divergence rate. The checkpoint is chosen on the validation rollout loss over windows of `data['val']`.
No evaluation episode is used for either choice.

## (4) Evaluation

The UDE, NODE and the re-fitted grey-box all go through the same `hyperkkl.evaluate.evaluate` call:
* the same cached evaluation files, read from the live cache;
* `episode_metrics` with settle = 0.5 N and the same divergence clamp;
* the same prior.

The re-fitted grey-box reproduces the stored tree numbers **exactly** (relative difference 0.0 on all 17
cells that have a stored grey-box, seeds 0-2). This confirms that the episodes, metric and harness are
identical.

## (5) Filters

* `observers.EKF` linearises with `vmap(jacrev(self.m.f))`, where `self.m` is the deployed
  `GreyBoxModel(f_model, r_theta)`. The Jacobian is therefore the learned model's, never the true
  plant's.
* The UKF propagates sigma points through the same `self.m`.
* `h` is the known selection y = x[sensors] (`_selection(self.m.sensors)`).
* The NODE uses `NoPhysics` (f = 0) plus `r_theta`, with the same known `h`.
* A unit test checks that the training RK4 step equals the filter's RK4 step (`test_dynamics_models_in_filters`).

## (6) Why the one-step grey-box is so much worse

Open-loop rollouts from the true state on 2048 validation windows give the normalised MSE after k
steps, in units of x_std² (seed 0). Here "grey-box" is the one-step residual, re-fitted on CPU with the
job's own code.

| cell | f_known k=8 / k=32 | grey-box k=8 / k=32 | ude5k k=8 / k=32 | true plant k=8 / k=32 | eval seq-OOD NRMSE: grey-box / ude5k (EKF) |
|---|---|---|---|---|---|
| C3-partial/bioreactor | 2.1 / 2.8 | 4.3e-2 / 1.1e-1 | 1.5e-5 / 3.4e-5 | 9.7e-6 / 3.1e-5 | 0.29 / 0.0063 |
| C3-partial/lorenz | 5.7 / 3.9e5 | 1.6e-2 / 1.3e-1 | 1.2e-5 / 7.4e-5 | 2.0e-6 / 2.9e-5 | 1.22 (oracle 0.23) / 0.0082 |
| C3-partial/battery | 4.1e-3 / 8.3e-3 | 1.5e-4 / 1.7e-4 | 2.8e-6 / 1.1e-5 | 2.5e-6 / 9.1e-6 | 0.020 / 0.010 |
| C3-mismatch30/battery | 1.6e-3 / 8.9e-3 | 3.5e-5 / 5.2e-5 | 2.9e-6 / 1.1e-5 | 2.6e-6 / 9.2e-6 | 0.0155 / 0.0051 |
| C3-mismatch30/vdp | 1.3e-3 / 1.1e-2 | 8.4e-6 / 5.3e-5 | 4.8e-6 / 2.1e-5 | 4.1e-6 / 1.9e-5 | 0.0095 / 0.0064 |
| TEST-C3-B/cartpole_b | 3.1e-2 / 2.8e-2 | 1.2e-3 / 8.1e-4 | 2.2e-5 / 4.0e-5 | 1.2e-5 / 3.1e-5 | 0.057 / 0.0126 |
| C3-partial/quad_tank | 2.1e-5 / 1.9e-4 | 9.7e-6 / 3.9e-5 | 9.7e-6 / 3.8e-5 | 9.7e-6 / 3.8e-5 | 0.0141 / 0.0142 (tie) |
| TEST-C3/highway | 2.2e-4 / 3.2e-3 | 5.4e-6 / 2.2e-5 | 5.3e-6 / 2.1e-5 | 5.3e-6 / 2.0e-5 | 0.0265 / 0.0417 (UDE worse in mean, equal in median) |

Reading:

* The ranking of the filters follows the accuracy of their models. Where the grey-box model is already
  close to the true plant (quad_tank, highway), the UDE ties it or is worse. Where it is not (bioreactor,
  lorenz, cartpole_b, battery), the UDE model is 10-1000x more accurate over 8-32 steps, and its filter
  wins by the same pattern.
* On bioreactor/partial the grey-box residual has a validation R² of 0.9997 and still fails. The
  modeller's model is wrong by a factor of about 600 (median), so a residual with 0.03% relative error
  is still large in absolute terms: its one-step error is 1.5e-3 against 2.8e-6 for the UDE. The grey-box
  EKF then runs at NRMSE 0.29 with no divergence.
* The stored grey-box fit is weak relative to the precision it needs:
  * fixed lr 1e-3, 3000 steps; it hit the step cap and did not early-stop;
  * no lr grid;
  * one-step targets (x_{k+1} - Φ(x_k))/dt, where process noise is divided by dt. The validation R² is
    near 0 on several states (battery/partial states 1 and 2, all quad_tank states, vdp/mm30 state 0).
    On those states the one-step target is mostly noise.

  The UDE picked lr = 1e-2, the top of its grid, on 7 of the 8 audited cells.
* Part of the gap may therefore be tuning rather than the multi-step objective. The post-hoc control
  `ude5k1` measures this: the same MLP, lr grid, selection and budget, with the one-step objective. Its
  results are in section 6 below and in RESULTS.md.

## (7) Seeds

* Baselines and the re-fitted grey-box: seeds 0-4.
* HyperKKL A0, A3 and A4+D3: seeds 0-2.
* `analyze.compare` pairs on the intersection of seeds, so every HyperKKL comparison uses seeds 0-2 and
  every grey-box comparison uses seeds 0-4.

## Section 6: what makes the one-step grey-box weak (results)

### 6a. Control `ude5k1`: one-step objective, same lr grid and pipeline as ude5k (seeds 0-4, seq OOD, EKF)

| cell | best stored grey-box | ude5k1 (one-step) | ude5k (K = 32) |
|---|---|---|---|
| C3-partial/battery | 0.0197 | 0.0142 (-28%, beats) | 0.0090 (-54%) |
| C3-partial/bioreactor | 0.2804 | 0.0077 (-97%) | 0.0063 (-98%) |
| C3-partial/duffing | 0.0076 | 0.0053 (-30%) | 0.0051 (-33%) |
| C3-partial/lorenz | 0.2275 | 0.0099 (-96%) | 0.0078 (-97%) |
| C3-partial/quad_tank | 0.0143 | 0.0142 (tie) | 0.0141 (tie) |
| C3-partial/vdp | 0.0257 | 0.0069 (-73%) | 0.0067 (-74%) |
| C3-mismatch30/battery | 0.0157 | 0.0062 (-60%) | 0.0053 (-66%) |
| C3-mismatch30/bioreactor | 0.0435 | 0.0056 (-87%) | 0.0056 (-87%) |
| C3-mismatch30/duffing | 0.0059 | 0.0055 (tie) | 0.0049 (-16%) |
| C3-mismatch30/lorenz | 0.0202 | 0.0039 (-81%) | 0.0039 (-81%) |
| C3-mismatch30/quad_tank | 0.0131 | 0.0134 (tie) | 0.0124 (-5%) |
| C3-mismatch30/vdp | 0.0098 | 0.0063 (-36%) | 0.0065 (-34%) |
| TEST-C3/fhn | 0.0152 | 0.0148 (-2%) | 0.0148 (-2%) |
| TEST-C3/highway_traffic | 0.0263 | 0.0265 (tie) | 0.0334 (tie) |
| TEST-C3/network | 0.0138 | 0.0133 (-3%) | 0.0133 (-4%) |
| TEST-C3/rossler | 0.8263 | 0.2426 (-70%) | 0.1785 (-78%) |
| TEST-C3/two_link_arm | 0.2337 | 0.4386 (tie, +90%) | 0.2447 (tie) |
| TEST-C3-B/cartpole_b | 0.0589 | 0.0192 (-67%) | 0.0132 (-77%) |

The same MLP, trained with the same one-step objective inside this study's pipeline, already removes
most of the gap. The multi-step objective adds a smaller, real improvement on battery, rossler,
cartpole_b, duffing/mm30 and quad_tank/mm30.

### 6b. It is not the learning rate

The stored grey-box was re-fitted on CPU with lr 1e-2 instead of 1e-3, 3000 steps. The open-loop
validation errors were essentially unchanged:
* bioreactor k = 1: 1.48e-3 → 1.48e-3;
* lorenz k = 32: 0.136 → 0.129;
* battery and cartpole_b: within 5%.

Raw output: `audit/gb_lr_*.log`.

### 6c. Cause found: the grey-box residual is fitted in one form and deployed in another (`hyperkkl/baselines_data.py`)

The residual r is fitted to the discrete-map targets (x_{k+1} - Φ_model(x_k, u_k)) / dt, so that
Φ_model(x) + dt·r(x) ≈ x_{k+1}. The filter, however, integrates RK4(f_model + r). In that form r is
re-evaluated at the RK4 stage points, where f_model is wrong. When f_model is poor, the two maps differ
by far more than the fitting error. One-step validation nMSE (x_std² units, 20k transitions, seed 0):

| cell | f_known alone | Φ_model + dt·r (fitted form) | RK4(f_model + r) (what the filter uses) | ratio |
|---|---|---|---|---|
| C3-partial/bioreactor | 1.3e-1 | 3.3e-6 | 1.5e-3 | 470x |
| C3-partial/lorenz | 3.1e-2 | 3.9e-6 | 3.5e-4 | 90x |
| TEST-C3-B/cartpole_b | 4.8e-3 | 1.5e-5 | 1.3e-4 | 8x |
| C3-partial/battery | 5.3e-4 | 4.2e-7 | 2.0e-5 | 48x |

The fitted residual is accurate. The deployed one-step model is 8-470x less accurate, and that is
exactly where the grey-box filters fail. `ude5k1` and `ude5k` are trained through the same RK4 that the
filter integrates, so they do not have this mismatch.

**This is a flaw in the grey-box baseline as implemented in the live package, not a leak in the UDE.**
The live package was not edited. Any of the following removes the mismatch:
* fit r through the deployed RK4 step (as `ude5k1` does);
* or deploy the discrete map Φ_model + dt·r in the filter;
* or use a step size at which the two forms agree.

All earlier "vs grey-box" verdicts in this repository (the C3 nodes, beat_greybox, cartpole_bounded,
real battery) compare against this weakened grey-box. Where f_known is poor (bioreactor, lorenz,
cartpole_b, rossler, battery) the true grey-box bound is much lower than reported.

### 6d. Consequence for this study

The pre-registered comparison (a) "vs best grey-box" is reported as registered, against the stored
grey-box implementation re-fitted on seeds 0-4. A second, POST-HOC reference is also reported in
RESULTS.md: the best of {stored grey-box filters, ude5k1_ekf, ude5k1_ukf}, which is the
consistency-corrected grey-box.
