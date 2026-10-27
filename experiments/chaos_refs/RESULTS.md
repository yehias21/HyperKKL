# Chaos references: Pecora-Carroll observer and lambda_max (lorenz, rossler)

Written 2026-10-26, CPU only. This answers `theory/reviews/editors/claims_v1/reviewer_d.md` E2/E3 and `SYNTHESIS.md` §4 rank 5.

**Files** (all in `experiments/chaos_refs/`):
- `chaos_refs.py`: the script.
- `results.json`: per-episode metrics, paired comparisons, Lyapunov numbers per family.
- `lyap_mismatch_levels.json`: lambda_max of the perturbed plants at 10/20/30/50%.
- `run.log`.

Nothing in `hyperkkl/`, `experiments/tree`, the queue or `experiments/cache` was written.
- The cache was opened through a read-only loader, so a cache miss raises.
- The device string `"cuda"` is used only to reproduce the runs' cache key; no GPU was used.

## Protocol

**Pecora-Carroll (PC) observer.**
- y = x2 replaces x2 in the model, and the unmeasured equations are integrated, driven by y. There is no learning, gain or tuning.
- x2_hat is the raw y_k.
- Integration is RK4 at the run's dt, with y linear between y_k and y_{k+1} and the input held (ZOH). This gives the same causal information as one EKF `advance` step.
- Initial state: the run's prior x0_hat (centre of the IC box, or the corner for poor_init).

The two response subsystems:
- lorenz: x1' = sigma (y - x1), x3' = x1 y - beta x3.
- rossler: x1' = -(y + x3), x3' = b + x3 (x1 - c).

The input enters only the measured x2 equation in both plants, actuator included, so the PC response never uses u. The input faults (noise, bias, delay) therefore leave PC unchanged.

Two PC variants:
- **`pc`** uses the node's modeller model (`data.model_plant`: nominal parameters, `system.knowledge`).
- **`pc_true`** uses the block's real plant. It is a reference only.

**Episodes.**
- These are exactly the runs' evaluation episodes: `evaluate._blocks` on each run's own `config` and `scales`, loaded from cache.
- They match the saved `traj.npz` true trajectories exactly (max abs difference 0.0 on every node, seed and OOD tier).
- Sensor and input conditions are rebuilt by the same code with the same seeds.

**Metric.**
- `metrics.episode_metrics`: settled NRMSE over the second half, with the same sigma_j.
- Also reported: `nrmse_unmeasured` (x1, x3 only), because PC's x2 is the raw measurement and its 3% noise alone gives NRMSE ≈ 0.017.

**Paired comparisons.**
- `metrics.paired_relative` against each method in the runs' `results.json`, on the same episodes.
- PC is deterministic, so its array is paired with every training seed. The Student-t CI over seeds therefore carries only the other method's training variance.
- `*` marks a CI that excludes 0.
- HyperKKL = `lora`; "dense" = the `hyper` ablation.

**Lyapunov exponents.**
- Method: two-trajectory Benettin. RK4 in float64, d0 = 1e-7, renormalised every step.
- Plant: the real plant with its saturating actuator, without process noise.
- Inputs: the exact evaluation inputs of every episode, all families of every tier (protocol v2, bounded box), repeated cyclically K = 6 episode lengths.
- lambda_max is taken over cycles 2-6:
  - lorenz: 250 s, about 225 Lyapunov times;
  - rossler: 800 s, about 40 Lyapunov times.
- The finite-time exponent over the episode itself (t from 0.2 T to T) is also reported.
- A u = 0 reference uses 16 id initial states.
- The conditional Lyapunov exponent (CLE) of the PC response comes from two response copies driven by the same reference x2, computed in the same augmented ODE.

## (b) lambda_max under forcing

| plant (params) | id | shape_ood | sequence_ood | u = 0 | episodes with lambda > 0 | min over episodes | T_L = 1/lambda |
|---|---|---|---|---|---|---|---|
| lorenz nominal (10, 28, 8/3) | 0.898 | 0.899 | 0.900 | 0.900 | 100% (192/192) | 0.859 (chirp) | 1.11 s; T = 50 s ≈ 45 T_L |
| rossler nominal (0.1, 0.1, 14; actuator k = 0.02) | 0.051 | 0.051 | 0.045 | 0.049 | 100% | 0.0014 (one `constant` id episode); OOD min 0.024 | ≈ 20-22 s; T = 160 s ≈ 7-8 T_L |
| rossler, TEST-C3 real plant (+30%) | 0.081 | 0.080 | 0.078 | 0.075 | 100% | 0.0075 (`constant`) | ≈ 12.5 s |
| **lorenz, C3-realistic real plant (+30%)** | **-0.374** | **-0.374** | **-0.374** | -0.375 | **0%** | | not chaotic |

**Per family.** lambda_max is flat across families and tiers on lorenz, at 0.891-0.907 per family mean. On rossler it is 0.044-0.059 nominal and 0.072-0.092 perturbed. The finite-time exponent over the evaluation window agrees: lorenz 0.87-0.89, rossler 0.042-0.050.

**Conclusion: both nominal plants are chaotic under every input family and tier.** Rossler is weakly chaotic, at only about 7-8 Lyapunov times per episode. A few `constant`-input id episodes are close to marginal.

**New caveat for C3 (lorenz mismatch nodes).**
- The ±30% parameter perturbation gives sigma = 13.5, rho = 20.7, beta = 3.6. This is below the Hopf threshold, rho_H = sigma (sigma + beta + 3) / (sigma - beta - 1) ≈ 30.5. The "real" lorenz plant of C3-realistic, and of C3-mismatch30, therefore converges to a stable equilibrium: lambda = -0.37 under every input.
- From `lyap_mismatch_levels.json` (8 sequence_ood episodes each):

| mismatch | sigma, rho, beta | lambda_max | behaviour |
|---|---|---|---|
| 10% | 11.05, 25.3, 2.95 | +0.74 to +0.89 on 5/8, -0.04 on 3/8 | bistable: chaotic attractor coexists with stable equilibria |
| 20% | | -0.20 | not chaotic |
| 30% | | -0.37 | not chaotic |
| 50% | | -0.82 | not chaotic |

- This explains why lorenz errors fall about 3x from mismatch 20% on (HyperKKL id 0.022 at 10% → 0.007 at 20%), and why the climatology is 0.61 there (1.0 for the nominal plant).
- **Every lorenz cell in C3-mismatch20/30/50 and C3-realistic is a non-chaotic plant.** Any "chaotic plant" reading of those cells must be dropped.
- Rossler stays chaotic at all levels (0.04-0.10).

## (a) Pecora-Carroll results

### Which subsystem synchronises

**lorenz.**
- The (x1, x3) response driven by x2 is triangular, with CLEs exactly -sigma and -beta. The largest CLE is **-8/3** (measured -2.6667 on every episode).
- It synchronises globally, whatever the input.
- The partial model removes -x1 x3 from the measured x2 equation only, so the response is unchanged: `pc` with the partial model equals `pc_true`.

**rossler.**
- With y = x2, the only PC response is (x1, x3).
- Its largest CLE is **positive**: +0.045 / +0.040 / +0.027 (id / shape / seq; max 0.067) with the true model, against lambda_max ≈ 0.05.
- **The x2-driven (x1, x3) subsystem does not synchronise at these parameters (a = 0.1, c = 14).** This differs from the classic y-drive result at a = b = 0.2, c = 9.
- There is no other single-output drive available, so no PC observer exists for this sensor.

**rossler, partial model.**
- The model drops x1 x3 from the x3 equation, which lies in the response itself. The CLE is then 0 (x1 is neutral: x1' = -(y + x3)) and x3 relaxes to b/c, so the observer drifts.
- **PC with the modeller's model is invalid on rossler.**

### NRMSE, clean condition

Each cell gives PC's change relative to the method in the row (negative = PC better).

| node (plant model) | tier | PC NRMSE (unmeasured) | vs HyperKKL | vs dense abl. | vs NLOX | vs best grey-box | vs best filter | climatology |
|---|---|---|---|---|---|---|---|---|
| C3-faults lorenz (exact) | shape | 0.0215 (0.0141) | 0.0251: -12% [-34, +10]; unm -38%* | 0.0168: +31%* | 0.0285: -24%* | n/a | ukf 0.0065: PC +230%* | 1.01 |
| | seq | 0.0214 (0.0141) | 0.0259: -14% [-37, +9]; unm -40%* | 0.0173: +27%* | 0.0285: -25%* | n/a | ukf 0.0065: PC +232%* | 1.00 |
| C3-partial lorenz (partial) | shape | 0.0215 (0.0141) | 0.0288: -23% [-62, +17]; unm -44%* | 0.0184: +19% [-33, +72] | 0.0283: -24%* | gb_ekf_oracle 0.227: **-91%*** | ukf_oracle 0.532: -96%* | 1.01 |
| | seq | 0.0214 (0.0141) | 0.0308: -28% [-67, +11]; unm -47%* | 0.0192: +13% [-24, +50] | 0.0287: -25%* | gb_ekf_oracle 0.227: **-91%*** | ukf_oracle 0.530: -96%* | 1.00 |
| C3-realistic lorenz (partial, 30% off, coloured-t) | seq | 0.478 (pc_true 0.048) | 0.0104: +4512%* | 0.0150 | 0.0129 | gb_ukf 0.102: +374%* | kkl_mb 0.0257: +1816%* | 0.61 |
| TEST-C1 rossler (exact) | shape | 0.633 (0.667) | 0.104: +509%* | 0.106: +501%* | n/a | n/a | ekf 0.0475: +1233%* | 1.04 |
| | seq | 0.629 (0.663) | 0.105: +501%* | 0.108: +484%* | n/a | n/a | ekf 0.0474: +1227%* | 1.07 |
| TEST-C3 rossler (realistic) | seq | 6.21 (pc_true 1.29) | 0.247: +2434%* | 0.303 | 0.376: +1551%* | gb_ukf 0.764: +729%* | kkl_mb 1.14: +455%* | 1.62 |

The id tier gives the same numbers: lorenz PC 0.0215, rossler 0.670. Every node has 3-5 seeds; PC diverged on 0 episodes.

**Lorenz partial.**
- The no-learning PC observer beats every grey-box filter by 10x (0.021 vs 0.227-1.27), and every model-based filter.
- It ties HyperKKL on the full NRMSE, with a point estimate 23-28% better and CIs that include 0.
- It is 44-47% better on the unmeasured states. It is also better than NLOX.
- Only the dense ablation is numerically better on the full NRMSE (n.s.).
- **So the lorenz "HyperKKL beats the grey-box" cell (C3.4) is matched by a tuning-free synchronisation observer that uses only the known part of the model.** It is a property of the plant/sensor (x2-drive synchronisation), not evidence for learning.

**Lorenz exact.**
- EKF/UKF are 3.3x better than PC: they also filter the x2 noise and use the x2 equation.
- PC ties HyperKKL and beats NLOX.

**Lorenz realistic.**
- PC with the nominal sigma and beta is biased (0.48). The real plant is at a stable equilibrium (see above), so this is not a chaos result.

**Rossler.**
- PC is near climatology (0.63 vs 1.04-1.07) on the exact model, because there is no synchronisation (CLE > 0).
- With the partial model it diverges to 6.2, above the climatology of 1.62.
- The held-out rossler result (HyperKKL 0.25 vs NLOX 0.37) therefore cannot be challenged by a PC baseline: none exists for y = x2.

### Lorenz under stress conditions

Paired NRMSE, sequence_ood. The other tiers agree.

| condition | PC | vs HyperKKL | vs grey-box (C3-regimes, partial) | vs best filter (C3-faults, exact) |
|---|---|---|---|---|
| noise10 | 0.071 | +60%* (0.045) | -68%* vs gb_ekf_oracle 0.221 | +309%* (ukf_oracle 0.017) |
| outliers | 1.27 | +112%* (0.60) | +92%* vs gb_ekf_oracle 0.672 | ukf_huber 0.0069 |
| dropout | 0.033 | +11% n.s. | | +376%* |
| colored / colored_t | 0.059 / 0.049 | +23%* / +15%* | | +175%* / +173%* |
| quantized | 0.055 | +34%* | | +314%* |
| sensor_drift | 0.43 | +67%* | | +83%* |
| poor_init, input_noise, input_bias, input_delay | 0.0214 (identical to clean) | -14% n.s. | | |
| plant_drift10 | 0.138 (pc_true 0.021) | -18%* (0.169) | +23%* vs gb_ekf_oracle 0.112 | -17%* (ukf_oracle 0.166) |
| plant_drift30 | 0.472 | -3%* | | +1%* |
| burst_outage | 0.567 | +57%* (0.36) | +52%* | +2483%* (ekf_oracle 0.022) |

PC has no outlier rejection and holds no measurement noise model, so it loses under outliers, sensor drift, larger noise and frozen sensors. It does not depend on the prior, the input or the input faults. Under plant drift it degrades like every model-based method.

### Bursts and tails, per Lyapunov time

- **Source.** Saved trajectories (4 clean episodes per seed) on sequence_ood, settled window.
- **Measures.**
  - Rate = up-crossings of ne(t) per Lyapunov time, at thresholds 0.1 and 0.5.
  - p99.9 of ne(t).
- **T_L.** lorenz 1.11 s; rossler 22 s (TEST-C1) and 12.8 s (TEST-C3).

| node | method | crossings of 0.1 per T_L | crossings of 0.5 per T_L | ne p99.9 |
|---|---|---|---|---|
| C3-faults lorenz | HyperKKL | 0.31 | 0.024 | 0.26 |
| | dense / NLOX | 0.05 / 0.10 | 0.011 / 0 | 0.10 / 0.19 |
| | EKF / UKF | 0 | 0 | 0.026 |
| | PC | 0 | 0 | 0.061 |
| C3-partial lorenz | HyperKKL | 0.34 (0.4% of the time) | 0.041 | 0.29 |
| | PC | 0 | 0 | 0.061 |
| | grey-box EKF/UKF | above 0.1 for 99% of the time | | 2.7-2.8 |
| TEST-C1 rossler | HyperKKL | 6.2 | 0.85 | 1.09 |
| | EKF | 2.8 | 0.14 | 0.71 |
| | PC | 3.0 (above 0.1 for 60% of the time) | 2.8 | 6.4 |
| TEST-C3 rossler | HyperKKL | 3.8 | 1.35 | 2.5 |
| | NLOX | 3.2 | 1.42 | 4.9 |
| | grey-box UKF | 1.2 | 1.08 (above 0.1 for 14% of the time) | 8.5 |

On lorenz, HyperKKL's heavy tail consists of about 0.3 desynchronisation events (ne > 0.1) per Lyapunov time, about 1 every 3 T_L. PC and the exact-model filters have none. On rossler every observer has excursions above 0.5 roughly once per 1-2 Lyapunov times.

## What this changes in the claims

1. **"Chaotic" may be used for nominal lorenz and nominal/perturbed rossler under every tier and family** (lambda_max > 0 on 100% of episodes). Do not use it for lorenz in C3-mismatch20/30/50 or C3-realistic: that real plant is not chaotic (lambda = -0.2 to -0.8). C3-mismatch10 lorenz is bistable.
2. **The lorenz win of C3.4 is not evidence for learning.** A tuning-free Pecora-Carroll observer, which uses only the known part of the model, ties HyperKKL (-23% to -28%, n.s.), is better on the unmeasured states (-44% to -47%*), and beats the grey-box filters by 10x.
3. **The rossler held-out win stands against PC.** For y = x2 no PC observer synchronises (CLE +0.03 to +0.05), and with the partial model it diverges. HyperKKL beats PC by 5-24x there.
4. **Side note.** The "lorenz grey-box 1.22-1.25 above climatology 0.61" in SYNTHESIS mixes nodes. The climatology is 1.00 in C3-partial, where the grey-box sits at 0.23-1.27. The 0.61 belongs to C3-realistic (the non-chaotic perturbed plant).
