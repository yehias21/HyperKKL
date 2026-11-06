# hyperkkl-bench

A benchmark for **state estimation of nonlinear systems**: from the measurements `y` (and, when the system is
driven, the applied input `u`), estimate the full state `x`, including the states no sensor sees. It has two
families:

| family | what is estimated | test tiers (the headline in bold) |
|---|---|---|
| **A. autonomous** (`autonomous/...`) | systems with no exogenous input whose own dynamics keep moving (limit cycles, chaos, oscillator networks) | **`test_id`**, `test_ic_ood` (held-out shell of the initial-condition box; scored on the transient too), `test_param_shift` (plant parameters drifted after training) |
| **B. non-autonomous** (`driven/...`) | input-driven systems under **realistic operating profiles**: drive cycles, recipes, missions, stimulus protocols, forcing scenarios, applied through an actuator model (rate limit, lag, quantisation, deadband) | `test_op_id` (the training operators, new episodes), **`test_op_ood`** (a different operator, route or schedule on the same device, same actuator box), and the v1 tiers as `test_legacy_id`, `test_legacy_shape_ood`, `test_legacy_sequence_ood` |

Both families also test **incomplete model knowledge** (a missing physics term, wrong parameters, coloured
heavy-tailed noise) and **sensor conditions** (outliers, outages, dropout, poor initialisation). Family B adds
**input-record conditions** (the logged input is noisy, quantised, delayed, or is the operator's command instead of
what the actuator delivered).

Version `2.0.0rc1` (release candidate; licence MIT for code and simulated data, real-data tracks under their providers' terms; public hosting pending) · data `hkb-data-2.0` (+ v1 `hkb-data-1.0`
for the legacy tiers, real-data tracks `hkb-real-1.0`) · scorer `settled-nrmse-1`.

## Quick start

```python
import hyperkkl_bench as hkb
tr = hkb.load("driven/battery/realistic", "train", recipe="mixed")   # 600 operating-profile + 600 event episodes
te = hkb.load("driven/battery/realistic", "test_op_ood")             # aggressive-highway and congested drivers
ta = hkb.load("autonomous/vdp/partial", "train")                     # u = 0
rep = hkb.evaluate_v2(my_estimator, "driven/battery/realistic")      # every tier x condition, bench scorer
hkb.load("vdp", "realistic", "test_id")                              # v1 call: unchanged (= the legacy tier)
```

```bash
hyperkkl-bench list --family autonomous           # plants per family
hyperkkl-bench evaluate-v2 ekf_gated driven/vdp/realistic --n-episodes 8
hyperkkl-bench card driven/two_link_arm_v2
hyperkkl-bench check-autonomous vdp --perturb 0   # inclusion check 4a (u = 0)
```

## The two families

**Inclusion rule "non-trivial unforced dynamics"** (family A; `hyperkkl-bench check-autonomous`). At u = 0, from 64
initial conditions over 5 T, the most active state must keep a time-std of at least 5 % of its reference std late in
the run, without decaying (`hyperkkl-bench check-autonomous`). This must hold for the nominal plant and, for the mismatch / realistic levels, for the
perturbed one. A forced plant that comes to rest at u = 0 (battery, bioreactor, cartpole_b, quad_tank, highway,
two_link_arm, duffing) would only ask for an equilibrium, so it is a family-B plant only. Lorenz enters family A at
the nominal-data levels only: its +0.3 perturbed plant has a stable fixed point. Rossler too: from the held-out IC shell its
+0.3 plant leaves the operating envelope, so its shell and parameter-shift tiers cannot be built at +0.3.

| family A plant | n_x | measured | dynamics |
|---|---|---|---|
| vdp | 2 | x1 | relaxation limit cycle |
| fhn | 2 | v | FitzHugh-Nagumo spiking |
| rossler | 3 | x1 | chaotic |
| lorenz (exact, partial) | 3 | x2 | chaotic |
| lorenz96 / lorenz96_n10 | 20 / 10 | every 2nd node | chaotic, spatially extended |
| network | 10 | 3 of 5 v nodes | ring of 5 heterogeneous FHN oscillators |

**HyperKKL and other input-conditioned observers in family A.** With u = 0 an input context is a constant, so input
conditioning is inactive. HyperKKL then acts as an autonomous KKL observer whose decoder also sees a strided window of
past latents (its z-window channel, a delay embedding), and latent-injection methods reduce to their autonomous part.
Family A compares estimators as autonomous observers. It cannot show a benefit of input conditioning.

| family B plant | input (physical) | training operators | OOD operators |
|---|---|---|---|
| battery | cell current, 1-Hz drive cycles | urban, suburban drivers, charger session | aggressive highway, congested stop-and-go |
| bioreactor | dilution rate (feed pump) | chemostat D-steps, shift-up/down | A-stat ramp, feed interruption |
| quad_tank | pump voltages | small operator moves, recipe | grade change, pump trip |
| highway_v2 | on-ramp inflow 180-1800 veh/h | time-of-day demand, pre-timed metering | upstream-signal platoons, event let-out |
| two_link_arm_v2 | shoulder torque +-12 N m | Fourier excitation, pick-and-place | task cycle, teleoperation |
| cartpole_b | cart force | repositioning | shuttle duty |
| vdp, duffing | shaker force | stepped sine, band-random | swept sine, burst random |
| fhn, network | injected current | long-square steps, ramp protocol | synaptic noise, pulse trains |
| lorenz, rossler, lorenz96 | forcing | climatology, regime persistence | trend, volcanic-like pulses |

`*_v2` plants are new versions: the v1 two_link_arm could apply only +-1 N m against a shoulder gravity torque of up to
19.6 N m (arm stretched horizontally), and the
v1 highway box clipped 23-36 % of realistic minute counts. Their `test_legacy_*` tiers are regenerated on the v2
plant with the v1 tier generators. All other `test_legacy_*` tiers are the frozen v1 test files.

## Training data (both families, both recipes shipped)

| split | family A | family B |
|---|---|---|
| `train` / `val` (standard recipe; 600 / 64) | u = 0, ICs in the core of the IC box | operating-profile episodes of the training operators |
| `train_event` / `val_event` (600 / 64) | u = 0 with impulsive state kicks | event-structured identification inputs (segments of the v1 families, rests, pulses, ramps; same box) |
| `recipe="mixed"` (the reference recipe of the v2 study) | both | both |
| `val_ood` (64, selection only) | a second held-out IC shell | the v1 val-OOD generators |

## Results (v2 study, equal data, 3 seeds)

All bundled and research baselines on the same mixed training episodes (the phase-1 zero-input episodes were used
additionally by the HyperKKL family and the UDE/NODE filters only: a stated equal-data deviation), validation-only
selection, one scorer, seeds 0-2 (3 seeds: reference numbers, not headline claims in the sense of PROTOCOL 5)
(full study: `reference/v2/RESULTS_v2.md`; exact reproduction from stored per-seed values:
`python reference/v2/reproduce_v2.py --check`). Settled unmeasured NRMSE on the headline tier, lower is better.

| family | cells | best method per cell | HyperKKL default (median rank, top 3) | paired vs HyperKKL (better / tie / worse, Holm) |
|---|---|---|---|---|
| driven (realistic level) | 13 | corrected grey-box / NODE 7, HyperKKL (default or base) 5, GRU 1 | 3, top 3 in 7 of 13 | multi-step UDE+UKF 3 / 7 / 3; NLOX-M 0 / 5 / 8; DKKL 1 / 0 / 12; tuned EKF 1 / 1 / 11 |
| autonomous (partial level) | 7 | multi-step UDE 5, HyperKKL 2 (rossler, vdp) | 5, top 3 in 3 of 7 | multi-step UDE+UKF 4 / 3 / 0; NLOX-M 0 / 4 / 3; DKKL 0 / 2 / 5 |
| real tracks | 4 | GRU (battery_pf, battery_lg, silverbox), UDE (emps) | HyperKKL A3 (no C5 in the real-data trainer): 2nd-3rd except emps outages | 3 seeds, see RESULTS_v2.md |

Notable: on Lorenz-96 the multi-step UDE is about 35x better than HyperKKL on equal data. In family A, HyperKKL acts
as an autonomous KKL observer (constant context) and is compared as such. The op_ood tier is not harder than op_id
for most methods (a different operator, not a harder one).

## Data, licence, versions, hidden test

As in v1 (see DATASHEET.md, RELEASE.md): the v2 export lives in `data_v2/`; hidden splits (`hidden_op_id`,
`hidden_op_ood`, `hidden_id`, `hidden_ic_ood`, 32 episodes each) ship without labels. A change to a frozen split or to
the scorer bumps the major version, hence 2.0.

---

# v1 reference: legacy tiers, deployment, scaling and real-data tracks

Everything below is the v1.x documentation. It stays valid: the v1 API, tiers (now `legacy_*` in v2) and tracks are unchanged.

A benchmark for **state estimation of input-driven (non-autonomous) nonlinear systems**: from the
applied input `u` and noisy measurements `y`, estimate the full state `x`, including the states no
sensor sees. Its focus is **generalisation to new inputs**. The test inputs stay inside the same
actuator box as the training inputs; only their shape or their event sequence changes. It also tests
**incomplete model knowledge**: a missing physics term, wrong parameters, and colored heavy-tailed
noise.

Version `1.2.0rc1` (release candidate: licence and public hosting pending) · data `hkb-data-1.0` + real-data tracks `hkb-real-1.1` · scorer `settled-nrmse-1` (see `hyperkkl-bench version`).

## Quick start

```bash
pip install -e .                                    # from this folder; needs numpy, scipy, pyarrow, torch (neurobem, calce: pip install -e .[real])
export HYPERKKL_BENCH_DATA=/path/to/hf_dataset      # folder with data/ and metadata/ (auto-found inside the research repo)
hyperkkl-bench evaluate ekf_gated vdp realistic --n-episodes 8      # smoke run of a reference filter
```

```python
import hyperkkl_bench as hkb
train = hkb.load("vdp", "realistic", "train")         # EpisodeSet: u (E, N+1), y (E, N+1, n_y), x (E, N+1, n_x), dt
est = hkb.baselines.GRUObserver(steps=1500)           # or any object with fit / reset / step
report = hkb.evaluate(est, "vdp", "realistic")        # fit on train/val, score test tiers x conditions + cost
print(report.table()); report.save("my_method_vdp_realistic.json")
```

Notebook: `notebooks/quickstart.ipynb` (load, train a small GRU on one plant, evaluate; about 2 min on a CPU).

## What is in it

| | |
|---|---|
| plants | 12 frozen **core** plants (+ 2 deployment-track and 5 scaling-track plants, `hyperkkl-bench list --all`): 6 **dev** (battery, bioreactor, duffing, lorenz, quad_tank, vdp) and 6 held-out **test** (cartpole_b, fhn, highway_traffic, network, rossler, two_link_arm). n_x from 2 to 10, one scalar input. Plus `legacy_cartpole`, which is excluded (unbounded) and kept only to reproduce old numbers. Cards: `docs/cards/<plant>.md`, `hkb.plant_card(p)` |
| knowledge levels | `exact` (true model), `partial` (one physical term missing), `mismatch` (true structure, parameters off by 30 %), `realistic` (partial model, 30 % mismatch, AR(1) colored Student-t noise) |
| splits | `train` (600), `val` (64), `phase1` (400 zero-input), `val_ood` (64, **public OOD validation**: 4 input generators used by no train or test split), `test_id` / `test_shape_ood` / `test_sequence_ood` (64 each, labels public), `hidden_id` / `hidden_shape_ood` / `hidden_sequence_ood` (32 each, **labels withheld**) |
| conditions | `clean`, `outliers` (2 % at 12 sigma), `burst_outage` (sensor freezes for 2 t_ref), `poor_init` (prior at a far corner of the IC box) |
| metrics | headline: **settled unmeasured NRMSE** (second half of each episode; errors normalised by a fixed per-plant x_std). Also all-state NRMSE, divergence rate, episode-bootstrap CIs, the input-OOD gap (tier / id), calibration if a covariance is returned (68/95 % coverage, ANEES, NEES-in-95 %), convergence time under `poor_init`, and cost: parameters, persistent bytes, FLOPs per step, an **MCU latency estimate** from FLOPs (480 MHz Cortex-M7 model of the research deployment study, used for budget verdicts), and the measured Python per-step latency at batch 1 on 1 CPU thread (reported, never used for verdicts) |
| tracks | `real`: measured data of real input-driven systems (see **Real-data tracks** below). `core`: 12 plants x 4 levels. `input_ood`: the OOD tiers and the OOD gap; select models on `val_ood`. `deployment`: drone `quad_planar` with multi-rate GPS / baro / IMU and a 150 us / 512 KB per-step budget; `cstr_exo` trained from **sparse lab assays**, so its training `x` is NaN between assays. `scaling`: FHN ring `fhnring_n4` to `fhnring_n64`. All five are frozen and runnable (`hkb.tracks.run_track`) |
| difficulty axes | per plant, each with provenance (`src/hyperkkl_bench/resources/difficulty.json`): information bound on the unmeasured NRMSE (3/alpha window vs full past), repairability of the partial model, largest Lyapunov exponent, observability Gramian conditioning, inclusion flags |
| reference baselines | standalone: `ekf`, `ekf_gated` (**the reference yardstick**: the published research gate), `ekf_huber` (Huber, no gate; reported next to it, PITFALLS P12), `ekf_gated_tuned` / `ukf_gated_tuned` (Q/R inflation chosen on `val`), `ude_ekf_gated` / `ude_ukf_gated` / `node_ekf_gated` (standalone corrected grey-box: MLP residual trained through RK4, PITFALLS P1), `ukf`, `ukf_gated`, `gru`, `prior_mean`. Reports on every frozen cell: `reference/runs/<baseline>/`. With the research repo: `StoredModel("ude5k1_ekf" / "ude5k1_ukf" / "node5k_ekf" / "gru140", seed)` loads trained checkpoints without retraining, and `StoredHyperKKL("C5_j3_s0.3_k0.5" / "base" / "repro_A3_D0", seed)` runs **the paper's method** (HyperKKL checkpoints of experiments/jitter_fix) on battery/partial, cartpole_b/realistic and quad_planar/realistic. It is batch-only, runs in a subprocess, and reproduces the stored NRMSE to <= 3e-4 on CPU. `reference/REFERENCE.md` is regenerated from 1440 stored per-run numbers by `reference/reproduce_reference.py` (exact match checked) |

## The estimator contract

```python
class MyEstimator:
    name = "mine"
    def fit(self, train, val, context): ...      # train/val: EpisodeSet; context: what you may know (dt, sensors,
                                                 # x_std, nominal noise, prior, the modeller's model of this level)
    def reset(self, x0_prior, P0_prior=None): ...
    def step(self, u, y): return x_hat           # or (x_hat, P) to be scored on calibration
    # optional: run_batch(U, Y, X0, P0) -> X_hat (fast path, must equal the step loop), n_params(),
    #           state_bytes(), flops_per_step()
```

`step(u_k, y_k)` returns x̂_k. It may use y_0..y_k and u_0..u_k, nothing else. `evaluate()` calls
`fit` itself; for an estimator you already fitted, pass `fit=False`. The full contract
is in `src/hyperkkl_bench/estimator.py`. Rules for a valid result (equal data, tuning only on
validation, gating, seeds) are in **PROTOCOL.md**. Known ways to get a misleading comparison are in
**PITFALLS.md**.

## Run time (CPU, 1 thread per process; measured on a shared 96-core host under load ~110)

| what | time |
|---|---|
| `evaluate(ekf_gated, vdp, ...)`, full cell (3 tiers x 4 conditions x 64 episodes) | ~2.5-3 min |
| the same on two_link_arm (N = 7729) / fhnring_n32 | ~20 min / ~13 min |
| `ude_ekf_gated` on vdp (4 learning rates x 3000 steps + evaluation, clean only) | ~6 min |
| quick-start notebook | ~1.5 min |
| `pytest -m "not slow"` / slow tests | ~1 min / ~30 min |
| real: `prepare battery_pf` / `battery_lg` (model replays) / `emps` / `silverbox` / `neurobem` / `calce` | ~8 / ~25 / <1 / <1 / ~1 / ~25 min (calce: + ~2 min to parse the workbooks once) (+ ~3 min for a re-freeze, ~11 min for the EKF noise grid) |
| real: EKF on all battery_pf test splits (481 episodes) / emps / silverbox | ~1 h / ~1 min / ~2 min |
| real: HyperKKLTrain (A3, full budget, 1 GPU) on battery_pf / emps / silverbox | ~10 / ~6 / ~6 min |

`evaluate(threads=1)` is the default: small batched filters slow down 5 to 10x with many torch
threads. Run cells in parallel processes instead (`tools/run_reference_filters.py`).

## Command line

```
hyperkkl-bench list                      # plants
hyperkkl-bench card vdp --md             # plant card
hyperkkl-bench evaluate ekf_gated vdp realistic [--n-episodes 8] [--out r.json]
hyperkkl-bench check-plant my_plant.json [--fast]     # inclusion check for a contributed plant
hyperkkl-bench verify-data [--plant vdp] # SHA-256 of every data file
hyperkkl-bench score-hidden submission/ --labels /private/labels    # leaderboard host only
hyperkkl-bench fetch | prepare | real-evaluate | real-card <track>  # real-data tracks
```

## Data

The episodes are the Parquet export of the research runs (`experiments/hf_benchmark_export`, 5.2 GB,
12 plants x 3 data variants). It is mirrored as a **private** Hugging Face dataset,
`Yahias21/HyperKKL-benchmark`; `hkb.paths.fetch_remote(plants=[...])` reads it with a token. This
package never uploads anything. The `val_ood` and hidden splits (0.43 GB public, plus 0.24 GB of
labels) were generated by `tools/build_extra_splits.py`. It uses the same generator, plant and noise
code, on CPU, with public seeds for `val_ood` and secret seeds for the hidden splits. Checksums:
`metadata/<plant>.json` (export) and `data_extra/MANIFEST.json`.

**Hidden test.** The public files under `data_extra/data/<plant>/<variant>/hidden_*.parquet` have
no `x` and no `x0`. The labels and the seed secret live in `hidden_labels/`, which **must not be
published**. It is excluded from any release; see `RELEASE.md`. A participant runs
`hkb.hidden.predict(est, plant, knowledge, "submission/")`; the host runs `score-hidden`.

## Results so far

`reference/LEADERBOARD.md` holds the package reports of `ekf_gated`, `ekf_huber`, `ekf_gated_tuned`
and `ude_ekf_gated` on every frozen cell, with track scores. On the core track, against ekf_gated:

| baseline | geo-mean ratio | cells beating ekf_gated |
|---|---|---|
| ekf_gated_tuned | 0.82 | 27 of 48 |
| ekf_huber | 1.01 | 17 of 48 |
| ude_ekf_gated | 0.11 | 34 of 48 |

ude_ekf_gated (one seed) diverges on two_link_arm. It is regenerated by
`tools/run_all_references.sh`. `reference/REFERENCE.md` holds the research results (5-seed means of
stored runs, incl. HyperKKL A3 and the learned baselines), regenerated without training by
`reference/reproduce_reference.py`.

## Real-data tracks

Six tracks on measured data of real input-driven systems (`hyperkkl_bench.real`, track `real`). **The raw
data are never redistributed by this package**: `fetch` downloads them from the original host (or copies a
local copy) and checks every file against its recorded SHA-256; `prepare` builds the episodes locally and
deterministically.

```bash
hyperkkl-bench fetch battery_pf [--from-local DIR] [--dry-run]     # Mendeley Data, SHA-256 checked
hyperkkl-bench prepare battery_pf                                  # raw -> frozen episodes (~10 min CPU)
hyperkkl-bench real-evaluate ekf battery_pf --n-episodes 4         # any bundled baseline
hyperkkl-bench real-card battery_pf                                # relevance, task, model, licence, results
```
```python
import hyperkkl_bench.real as hr
train = hr.load("battery_pf", "train")      # RealEpisodeSet: u, y, x (NaN = no label), x_fill, prior, label_mask
rep = hr.evaluate(est, "battery_pf")        # fit on train/val, run every test split from the track's prior, score
```

| track | system (licence; data fetched, not shipped) | input u | measured y | hidden, scored | input shift train -> test | partial model (identified on train only) |
|---|---|---|---|---|---|---|
| `battery_pf` (required) | Panasonic 18650PF cell, Kollmeyer, Mendeley doi:10.17632/wykht8y7tg.1 (CC BY 4.0) | current | case temperature, filtered voltage | SOC (lab coulomb counter); settled SOC RMSE, prior 20 % off | Cycles 1-3, UDDS, LA92 -> US06, HWFET, NN; + 0 degC | 2-RC ECM + thermal (research pipeline) |
| `battery_lg` | LG 18650HG2 cell, Kollmeyer et al., Mendeley doi:10.17632/b5mj79w5w9.2 (CC BY 4.0) | current | as above | SOC; as above | UDDS, LA92, Mixed1-6 -> US06, Mixed8; + 0 / 10 / 40 degC | same ECM, this cell |
| `emps` | electro-mechanical positioning axis, Janot et al. 2019 (licence not stated) | motor force | encoder position | velocity during 100-ms encoder outages (offline zero-phase label) | bang-bang -> + force pulses | rigid axis + Coulomb / viscous friction (providers' IDIM-LS) |
| `silverbox` | electronic Duffing oscillator, Wigren & Schoukens 2013 (licence not stated) | input voltage | output voltage | velocity (offline zero-phase label) | multisine -> noise of growing amplitude, beyond the training range | Duffing, LS + output-error |
| `neurobem` | agile quadrotor, Bauersfeld et al., RSS 2021, doi:10.15607/RSS.2021.XVII.042, 95 flights (no licence text found, citation only, redistribution not granted: fetch-only) | 4 rotor speeds (telemetry) | gyro, accelerometer (raw Betaflight logs) | body velocity, tilt (Vicon); settled, in m/s and deg | train / test_id: circles, lemniscates, oscillations, random points -> test_ood: CPC race lines, satellite, vertical oscillation (held out by flight) | rigid body, quadratic rotor map, lumped rotor drag (no BEM) |
| `calce` | A123 APR18650M1A LiFePO4 cell, CALCE (Univ. of Maryland; licence checked by the authors, citation requested, fetch-only) | current | case temperature, filtered voltage | SOC (lab coulomb counter, own-cell C/20 capacity); settled SOC RMSE, prior 20 % off | DST -> US06, FUDS (one training profile family); + 0 / 10 / 40 / 50 degC | 2-RC ECM, OCV = C/20 discharge branch (flat LFP plateau, 52-mV hysteresis) |

- **battery_pf reproduces the research pipeline exactly**: the windows, labels and priors equal those of
  experiments/real value for value (`tests/test_real.py`), the re-identified ECM equals the frozen one, and
  the bundled EKF reproduces the research EKF to 1e-13. `python reference/reproduce_reference.py --real`
  regenerates the research results (HyperKKL variants, EKF / UKF, grey-box, Koopman, NLOX, GRU, NCDE;
  `reference/real/REFERENCE_battery_pf.md`) from stored per-episode outputs, without training (349 checks, all match).
- **`u` may be a vector.** `neurobem` has 4 inputs, so its `u` is (E, N+1, 4) where the other tracks have (E, N+1). The bundled
  filters, UDE and GRU, `run_episodes` and HyperKKLTrain accept both; your estimator's `step` gets an array there, not a float.
- **Every track runs end to end** with the model-based filters, the corrected grey-box `ude_ekf_gated`, the
  GRU and HyperKKL (`baselines.hyperkkl_train.HyperKKLTrain`: the A3 recipe trained by the research code in a
  subprocess; needs the research repository). Smoke results (1 seed): `docs/cards/REAL_SMOKE.md`.
- Correctness checks and plots: `python tools/real_check.py <track>` -> `docs/real/`. Cards:
  `docs/cards/real_<track>.md`. Candidates that were not added, with reasons: `docs/cards/REAL_EXCLUDED.md`.
- Real data are inherently "realistic": the model is partial (identified, structurally incomplete) and the
  noise is real. Read PITFALLS P13-P18 (unlabelled states, quantised sensors, derived labels, identification
  on training data only, temperature shift, priors) before comparing methods.

## Tests and CI

`pytest -m "not slow"` takes ~1 min with the data; `pytest` takes 10-30 min, depending on host load. Without the data, only
the data-free tests and `tests/test_mini.py` run; the latter is an end-to-end run on a 0.8 MB
fixture (`tools/make_mini_dataset.py`). `.github/workflows/tests.yml` is prepared for a public repo.

## Repository layout

```
src/hyperkkl_bench/   package (data, evaluate, metrics, conditions, hidden, check, tracks, cards, baselines/, plants/)
tests/                pytest suite
reference/            reproduce_reference.py, reference_results.json, REFERENCE.md
tools/                maintainer tools (build_extra_splits.py, build_difficulty.py, build_cards.py)
docs/cards/           plant cards (generated)
notebooks/            quickstart.ipynb
data_extra/           val_ood + hidden public files (+ MANIFEST.json)
hidden_labels/        PRIVATE: hidden labels + seed secret (never release)
archive/              NOT for release: preserved research checkpoints (HyperKKL A3, copied from /tmp)
```

## Roadmap / maintenance

- v1.1: Lorenz-96 (n = 10, 20) as a chaos track (research cache `experiments/chaos_deep`, 65 GB).
- Datasheet: `DATASHEET.md`. Machine-readable metadata: `croissant.json`.
- The leaderboard host, licence, citation and authorship are open decisions for the authors (see STATUS.md).
- Versioning: a change to a frozen split or to the scorer bumps the MAJOR version. Every report carries
  `bench_version`, `data_version` and `scorer_version`. Changes are logged in `CHANGELOG.md`.
