# HyperKKL (TMLR) — where the input should enter a learned KKL observer

> KKL theory says a forced system needs an input-dependent observer map, but not how a
> learned observer should depend on the input. We show **where the input should enter**,
> **why a short input window suffices**, and **when a learned KKL observer beats
> classical filters**.

This directory is a clean rebuild of the archived AAAI/CDC code (`../archive/`).
`TASKS.md` lists what was wrong or redundant there and every design decision taken here.

## Contributions and where their evidence comes from

| | contribution | code | tree nodes |
|---|---|---|---|
| C1 | controlled study of five conditioning locations at matched parameters and matched input history | `conditioning.py`, `train.phase2` | `C1-*` |
| C2 | fading-memory lemma, window rule, error bound in the conditioning network's Lipschitz constant | `THEORY.md`, `nets.SFMEncoder`, `nets.WindowEncoder` | `C2-*` |
| C3 | when learned KKL beats EKF / UKF / PF / model-based KKL / NLOX | `observers.py`, `configs/base.yaml: conditions` | `C3-*` |
| C4 | open testbed: systems, input families, held-out splits, metrics, checkpoints | `systems.py`, `signals.py`, `metrics.py`, `export.py` | — |

## Install

```bash
# the venv in .venv inherits torch/numpy/scipy from an existing conda env
/path/to/python -m venv --system-site-packages .venv && .venv/bin/pip install pytest
# or: pip install -r requirements.txt
.venv/bin/python -m pytest tests -q          # full suite: ~30+ min on CPU
```

## Quick start

```bash
P=.venv/bin/python
$P -m hyperkkl tree                          # the experiment tree
$P -m hyperkkl show C1-where                 # one node: hypothesis, rule, config diff
$P -m hyperkkl queue C1-where                # create its jobs
$P -m hyperkkl worker --gpus 0-7 --per-gpu 4 # run everything queued
$P -m hyperkkl report C1-where               # -> experiments/tree/C1-where/report.md
$P -m hyperkkl.job --node R0 --system vdp --seed 0   # one job, in the foreground
```

## How a run works

1. **Design** (`design.py`, `data.make_design`): Butterworth latent spectrum with abscissa
   `alpha`, `dt` with `max|lambda| dt = 0.1`, design gate.
2. **Scales** (`data.system_scales`, cached): the operating envelope reached by the inputs
   of all three evaluation tiers (all in-bounds), reference stds under ID inputs (the NRMSE
   yardstick), washout and memory horizons.
3. **Data** (`data.training_data`, cached): zero-input episodes for phase 1, forced
   training/validation episodes (crossed IC × signal design), Sobol collocation set.
4. **Phase 1** (`train.phase1`): `T0` from labels + PDE residual, `T0+` on fresh states.
5. **Phase 2** (`train.phase2`): each conditioning variant at a matched parameter budget.
6. **Evaluation** (`evaluate.py`): every observer × tier (`id`, `shape_ood`,
   `sequence_ood`) × condition (clean, sensor faults, input faults, poor init, plant
   drift), paired episodes; plus the state-coverage diagnostic per tier
   (`metrics.state_coverage`: fraction of evaluation states outside the training-state
   box and beyond the training kNN q99).

## Input protocol (v2)

Out-of-distribution means what it means for a deployed estimator: the input is bounded by
fixed actuator limits (a pedal between released and fully pressed), and what changes is
its **shape** or its **sequence** (another driver, another road), never its amplitude.
Every family in every tier realises `v(t)` in `[-1, 1]` by construction (applied input
`u_scale * v`), and training covers that whole box. Time-like signal parameters are in
units of a per-plant `t_ref` (`configs/systems/*.yaml`, rule `max(1, tau_slow / 3)`).

| tier | families | what differs from training |
|---|---|---|
| `id` | zero (train only), constant, step, staircase, prbs, square, sinusoid, multisine, ramp, am, drive | nothing (independent seeds) |
| `shape_ood` | chirp, filtered_noise (OU), triangle, sawtooth | waveform shape; same box, same frequency band |
| `sequence_ood` | drive ("stop-and-go driver") | event order and timing: large level jumps, short dwells, abrupt ramps; same level grid and range |

Parameter ranges: `signals.RANGES` (reviewable as `signals.regime_table()`).

## Deployment scorecard

What makes a good estimator in real life is more than its clean NRMSE. Every evaluation
block (method x tier x condition) stores, next to the per-episode metrics, a `scorecard`
(`metrics.block_scorecard`; definitions in the `metrics.py` docstring): per-state NRMSE with
the measured and the **unmeasured** states summarised separately, p95 / p99 of the episode
NRMSE and the divergence rate, convergence time (median / p90, s and t_ref) and the paired
"forget the prior" time (poor_init estimate vs the same observer's nominal-prior estimate on
the same episode; robust to input-driven error swings), recovery time
after a burst sensor outage, the long-horizon drift ratio, and the AUROC of an output-residual
alarm (EMA over t_ref of ||(y - h(xhat)) / y_std||) for detecting a state error above 3x the
method's median. Every node report has a **Deployment scorecard** section (one row per
method: id / shape_ood / sequence_ood NRMSE, unmeasured NRMSE, worst sensor fault, p99,
divergence rate, convergence from `poor_init`, recovery after `burst_outage`, drift on
`long`, alarm AUROC, µs/step; columns appear when the node has the condition), and
`python -m hyperkkl scorecard ID [ID ...]` writes a cross-node table (`.md` + `.tex`) to
`experiments/scorecard/`.

Opt-in conditions (`eval.conditions`) for it, in `configs/base.yaml`:

| condition | what | cost |
|---|---|---|
| `poor_init` | prior in a far corner of the IC box (convergence time) | none extra (batched) |
| `burst_outage` | sensor frozen (sample-and-hold) for 2 t_ref from 0.6 T (recovery time) | none extra (batched) |
| `long` | 16 id episodes of 5 T (drift ratio) | one extra sequential pass of 5 N steps per observer (smoke, vdp: +730 s over 9 observers vs 320 s for the batched 3-tier x 4-condition pass; filters ~110-160 s each, learned ~35 s) |
| `colored_t` | AR(1) (rho 0.9) Student-t (nu 3) sensor noise, unit stationary variance | none extra |

**Fair filters** (`baselines`): `ekf_huber`, `ukf_huber` (Huber M-estimation update, nominal
Q, R) and the condition-oracle filters `ekf_oracle`, `ukf_oracle`, `ekf_oracle_huber`
(Huber + chi-square gate), rebuilt per condition with `observers.oracle_noise` for that
condition's sensor and a Q that includes the model-error rate between the filter's model and
the real plant of the block (its weight q_inflate and an r_inflate chosen per job on validation
episodes, `job.tune_oracle`, recorded as `oracle_tuning`); they skip held samples (dropout, burst) unless the sensor is
quantised. One extra sequential pass per distinct (sensor, plant shift) of a node.

**Incomplete knowledge** (`system.knowledge: full | partial`): every model-based component
(the phase-1 PDE term, EKF/UKF/PF/kkl_mb, the oracle model-error rate) holds the modeller's
model `data.model_plant` (nominal parameters, `Plant.f_known` when partial, see
`Plant.UNMODELLED`); the real plant that generates the data keeps its full dynamics. NLOX
and the learned observers' phase 2 are model-free.

**Data-driven baselines** (`baselines_data.py`, reviewer E1/E2), trained on exactly the
learned observers' data (`data.train` + `data.p1`, real-plant state labels; selection on
`data.val`) and fitted once per job in memory: `koopman_kf` / `koopman_kf_bilinear` (EDMD with
inputs, dictionary x + thin-plate RBFs, ridge LS, Kalman filter on the lifted state with
residual Q, nominal R, lifted IC-box prior) and `greybox_ekf` / `greybox_ukf` /
`greybox_ekf_oracle` (`data.model_plant` + an MLP residual fitted to
(x_{k+1} - RK4_model(x_k, u_k)) / dt, inside the unchanged EKF / UKF). They can be added
to finished runs without the learned models:
`python -m hyperkkl reeval NODE --baselines koopman_kf greybox_ekf ... [--device cuda:0] [--shard i/n]`
merges their metrics, scorecard, cost and train info into each run's `results.json`
(original kept as `results.json.bak`, provenance under `reeval`), checking that the data
modules, device type, scales and episode families match the original run.

**C2 memory probe** (`eval.memory_probe: true`, `evaluate.memory_probe`): for every
decoder-side method with an sfm / gru / window context, the input of the last L samples is
kept and everything older is replaced by another episode's input (and by zero); the context
is recomputed (latent z unchanged), and the decoded change d(L) is fitted to an exponential
rate (1/s) on its tail above the float64 floor, reported under `memory_probe` in
`results.json` next to the design alpha and the SFM's slowest learned pole. This is the
quantity Lemma 1 bounds (`theory/m2_fading_memory.tex`). Latent injection is skipped (its
context acts through z).

Every artefact is cached under `experiments/cache/` by a hash of the config sections it
depends on, so sibling nodes share data and base maps, and a node that only changes the
evaluation re-trains nothing.

## Systems

| system | nx | ny | input enters | role | partial-knowledge model (`Plant.UNMODELLED`) |
|---|---|---|---|---|---|
| vdp | 2 | 1 | additive + saturating actuator | dev | missing: the nonlinear damping mu (1 - x1^2) x2 (undamped oscillator) |
| duffing | 2 | 1 | state-dependent gain | dev | missing: the cubic stiffness -x1^3 (linear spring) |
| fhn | 2 | 1 | additive | test | missing: the recovery leak -eps b w |
| bioreactor | 2 | 1 | bilinear (dilution rate) | dev | Monod saturation replaced by first-order kinetics mu_max s / Ks |
| rossler | 3 | 1 | additive, chaotic | test | missing: the folding nonlinearity x1 x3 |
| lorenz | 3 | 1 | additive, chaotic | dev | missing: the coupling -x1 x3 in the measured equation |
| quad_tank | 4 | 2 | distributed, non-minimum phase | dev | square-root orifice law linearised at the nominal equilibrium |
| cartpole | 4 | 2 | matrix-valued G(x), balancing loop | test | small-angle pendulum (sin th = th, cos th = 1, no om^2) |
| two_link_arm | 4 | 2 | M(q)^{-1} e1 | test | missing: Coriolis and centrifugal terms |
| highway_traffic | 5 | 3 | ramp inflow | test | free-flow fundamental diagram q = v_free rho (no congestion) |
| battery | 5 | 2 | current (SOC + 2-RC + thermal; 1-min time unit; Vm and thermistor measured) | dev | textbook 1-RC circuit (slow branch R2 \|\| C2 missing) |
| network | 10 | 3 | node 0 of a ring of 5 FHN neurons (nodes 0, 1, 3 measured) | test | ring edge 2-3 missing (graph partly unknown) |

Every plant additionally passes through a saturating back-EMF actuator, so no plant is
input-affine. Design decisions use `dev` systems only (each of dev and test spans 2-D,
3-D chaotic and >= 4-D plants); `test` systems are evaluated once with the frozen recipe.

Stress conditions are deployment-time departures that NO observer is told about: the
learned observers are trained on the nominal sensor model, and the EKF / UKF / PF keep the
nominal noise covariances (oracle values for the `clean` condition only).

## Layout

```
hyperkkl/           the library (module map in hyperkkl/__init__.py)
configs/            base.yaml (all defaults, commented) + systems/*.yaml
tests/              regression tests
experiments/        tree/ (nodes, runs, reports), TREE.md, JOURNAL.md, queue/, cache/, code/
THEORY.md           Lemma 1, Corollary 1, Proposition 1, Lemma 2, sampled-data semantics
program.md          the autoresearch loop: how to grow the tree
TASKS.md            task list, audit findings, design decisions
```
