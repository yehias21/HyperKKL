# Datasheet: hyperkkl-bench data (hkb-data-1.0)

This datasheet follows Gebru et al., "Datasheets for Datasets". Machine-readable metadata is in
`croissant.json` (core schema) and `metadata/<plant>.json` (per plant, with SHA-256 of every file).

## Motivation
- **Purpose.** To evaluate state estimators (filters, observers, learned sequence models) for
  nonlinear systems driven by an exogenous input. The question is whether they generalise to input
  signals of a new shape or a new event sequence inside the same actuator box, and to an incomplete
  model of the plant.
- **Creators / funding.** To be filled in by the authors (author decision; placeholder in CITATION.cff).

## Composition
- **Instances.** One instance is a simulated episode: the applied input `u` (zero-order hold), noisy
  measurements `y`, the true state `x`, the sample time `dt`, the initial condition `x0`, the input
  family and the JSON parameters of the input realisation.
- **Plants.**
  - 12 core plants: 6 dev and 6 test.
  - 2 deployment-track plants: the planar quadrotor `quad_planar` with multi-rate GPS / baro / IMU, and
    the exothermic CSTR `cstr_exo` with sparse lab assays.
  - 5 scaling-track FitzHugh-Nagumo rings: `fhnring_n4` to `fhnring_n64`.
  - 1 excluded legacy plant (`legacy_cartpole`).
- **Variants.** Each plant has 3 data variants: `white` (nominal plant, white noise), `mismatch30`
  (real plant parameters off by exp(+-0.3)) and `realistic` (mismatch plus AR(1) Student-t(3)
  noise). Each variant is offered at 4 knowledge levels.
- **Counts.** Per core plant and variant: 600 train, 64 val, 400 phase1, 64 val_ood, 3 x 64 test, and
  3 x 32 hidden episodes. Rossler and two_link_arm have fewer where the generator dropped diverging
  episodes (rossler hidden: 21 per tier). Counts per file are in the plant cards and the metadata.
  The deployment and scaling plants have their own sizes (cards).
- **Labels.** `x` is the full simulated state, with no label noise. Exception: `cstr_exo` training
  splits, where C_A exists only at the lab-assay samples (assay error 5 % x_std), NaN elsewhere.
- **Withheld data.** The hidden splits ship without `x` and `x0`. Their labels and the generator
  secret are kept by the leaderboard host.
- **Sensitive content.** None. All data are simulated. No people and no personal data.
- **Known limitations.**
  - Simulation only, with one scalar input.
  - The generator is public: with the plant code and the real parameters (`mismatch30_params`) anyone
    can rebuild the real plants. The hidden split hides the seeds, not the physics.
  - Noise models are parametric (white Gaussian, or AR(1) Student-t).
  - Robustness conditions are applied by the package at evaluation time; they are not stored.

## Collection process
- **Generator.**
  - float64 RK4 with a zero-order-hold input.
  - Euler-Maruyama process noise (0.5 % x_std per sqrt(s); 2 % for the drone).
  - Then the sensor model.
  - Stored as float32.
- **Inputs.**
  - Families: 11 training families, 4 shape-OOD waveforms, the stop-and-go sequence-OOD driver, and
    4 val-OOD generators (regen_decay, gated_sine, spline, urban_cycle).
  - All inputs are bounded in [-1, 1] times u_scale.
- **ICs.** Sobol samples in the IC box. Episodes leaving 1.5x the operating envelope are redrawn
  (up to 6 times) and then dropped.
- **Provenance.**
  - Core train / val / test: Parquet export of the research cache (`experiments/hf_benchmark_export/EXPORT.md`).
  - val_ood / hidden: `tools/build_extra_splits.py` (CPU, same generator).
  - Deployment / scaling: `tools/export_casestudies.py` and `tools/export_scaling.py`. These read the
    research caches without changing them; a cache miss is an error, never a regeneration.
- **Timeframe.** Generated in October 2026.

## Preprocessing
None beyond the float32 cast. No subsampling, smoothing or normalisation is applied in the files.
Normalisation constants (`x_std`, `y_std`) are in the metadata and used only by the scorer.

## Uses
- **Intended.** Benchmarking state estimators under input shift and model incompleteness; studying
  data efficiency, robustness and cost.
- **Not intended.** Claims about a specific physical device (the battery and the drone are generic
  simulations), or safety certification.
- **Rules for valid results.** See PROTOCOL.md (equal data, tuning on validation only, gating, seeds)
  and PITFALLS.md.

## Distribution
- Planned: Hugging Face dataset (currently PRIVATE: `Yahias21/HyperKKL-benchmark`, core export only),
  plus this package. **The licence is not yet decided** (LICENSE placeholder; author decision).
- The hidden labels are never distributed.

## Maintenance
- The authors (contact to be added) maintain the benchmark. Errata and new versions go in CHANGELOG.md.
- A change to a frozen split or to the scorer bumps the major version.
- Contributed plants must pass `hyperkkl-bench check-plant` (PROTOCOL.md section 4).

## Real-data tracks (hkb-real-1.1, package 1.2.0rc1)

The real-data tracks (`hyperkkl_bench.real`) are **not part of the distributed data**. The package ships
only code, file lists with SHA-256, licences, citations and the constants identified on the training splits
(`src/hyperkkl_bench/resources/real_<dataset>.json`). `hyperkkl-bench fetch <dataset>` downloads the raw
files from the original host (or copies a local copy, `--from-local`) and verifies every hash;
`hyperkkl-bench prepare <dataset>` builds the episodes locally and deterministically.

| track | source (licence) | measured | hidden (label) | splits | size fetched |
|---|---|---|---|---|---|
| battery_pf | Kollmeyer, Panasonic 18650PF, Mendeley doi:10.17632/wykht8y7tg.1 (CC BY 4.0) | current (input), case temperature, terminal voltage | SOC (lab coulomb counter); V1, V2 unlabelled | train Cycles 1-3, UDDS, LA92; val Cycle 4; test US06, HWFET, NN (25 degC); 9 cycles at 0 degC | 21 files, 46 MB |
| battery_lg | Kollmeyer, Naguib, Skells, LG 18650HG2, Mendeley doi:10.17632/b5mj79w5w9.2 (CC BY 4.0) | as battery_pf | as battery_pf | train UDDS, LA92, Mixed1-6; val Mixed7; test US06, Mixed8 (25 degC); US06 / HWFET / Mixed8 at 0 and 10 degC, US06 / HWFET at 40 degC | 20 files, 31 MB |
| emps | Janot, Gautier, Brunot, EMPS (licence not stated) | motor force (input), encoder position | velocity (offline zero-phase derivative) | train / val: bang-bang record [200, 20000) / [20000, 24641); test: pulses record | 1 zip, 1.9 MB |
| silverbox | Wigren & Schoukens, Silverbox (licence not stated) | input voltage, output voltage | velocity (offline zero-phase derivative) | train / val: multisine; test: held-out multisine, arrow (in range), arrow (amplitude extrapolation) | 1 zip, 5.8 MB |
| neurobem | Bauersfeld et al., NeuroBEM, RSS 2021, doi:10.15607/RSS.2021.XVII.042, 95 flights (no licence text found; citation requested) | 4 rotor speeds (input), gyro, accelerometer (raw Betaflight logs) | body velocity, gravity direction (Vicon); yaw and position out of scope | by flight: train 48, val 12, test_id 16, test_ood 19 (CPC, satellite, vertical oscillation held out) | 2 zips + 3 text files, 1.1 GB |
| calce | CALCE battery group (Univ. of Maryland), A123 APR18650M1A LiFePO4, 2 cells, DST / US06 / FUDS at 0-50 degC + C/20 OCV tests (licence checked by the authors; citation requested; not redistributed) | current (input), voltage, case temperature; SOC label = lab coulomb counter | windows of 30 min at 1 s; train 238, val 32, tests 54-70 each |

- **Composition.** Episodes are windows of the recordings (battery: 30 min at 1 s; EMPS: 2 s at 1 kHz;
  Silverbox: 2048 samples at 610 Hz; NeuroBEM: 10 s at 100 Hz) with per-episode priors. The full definitions, checks and plots are in
  `docs/cards/real_<dataset>.md`, `docs/real/<dataset>_checks.json` and `docs/real/<dataset>_episodes.png`.
- **People / sensitive content.** None: laboratory measurements of a battery cell, a positioning axis, an
  electronic circuit and a quadrotor in a motion-capture hangar.
- **Limitations.** One cell per battery dataset; labels are themselves estimates (PITFALLS P15);
  battery V1, V2 have no label (P13); the mechanical datasets were recorded in closed loop (EMPS) or with
  one input design per split (Silverbox), so input-shift coverage is limited to what the providers recorded.
- **Distribution / licences.** CC BY 4.0 (battery datasets) would allow redistribution with attribution, but
  the package fetches them anyway, for one code path. EMPS and Silverbox state no licence: they must not be
  rehosted without the providers' permission. NeuroBEM states no licence text either, only a citation request:
  redistribution is not granted, so it is fetch-only, and written terms should be obtained before a public release (the author
  states he checked the licence and approved adding it; that check is not recorded on the source). Datasets considered and not added (with reasons) are listed
  in `docs/cards/REAL_EXCLUDED.md`.

## Datasheet: v2 additions (hkb-data-2.0, package 2.0.0rc1)
The v1 sections above stay valid for the legacy tiers.

## Motivation
v1 measured generalisation to input waveforms that no operator produces (identification families). v2 measures
state estimation in two families: autonomous systems (no input) and input-driven systems under realistic operating
profiles. It keeps the v1 tiers as `legacy_*` so every v1 number stays reproducible.

## Composition
* **Family A (autonomous)**: vdp, fhn, rossler and lorenz (both nominal-data levels only), lorenz96 (n 20), network,
  lorenz96_n10. Each has data variants white / mismatch30 / realistic (lorenz and rossler white only).
* **Family B (driven)**: battery, bioreactor, duffing, lorenz, quad_tank, vdp, cartpole_b, fhn, highway_v2, network,
  rossler, two_link_arm_v2, lorenz96, each with the 3 data variants. Deployment, scaling and real tracks are as in
  v1.1 (unchanged files).
* **Counts per cell and variant**: train 600, train_event 600, val 64, val_event 64, val_ood 64, phase1 400, test
  tiers 64 each, hidden tiers 32 each. Exception: rossler (both families) and some phase-1 sets keep fewer episodes,
  because the generator drops episodes that leave 1.5x the operating envelope after 6 redraws. Driven rossler, for
  example, has 483 train_event episodes (MANIFEST.json per cell, `n`, `rejected`). The hardest episodes are
  therefore under-sampled there, as in v1.
* **Labels**: the full simulated state, no label noise. Profile test tiers also store the operator's command `cmd`
  (needed by `input_cmd_vs_applied`; never an input otherwise).
* **Withheld**: hidden splits without x, x0, cmd. Labels and the seed secret live in hidden_labels/ (never released).

## Collection process
* Same generator as v1 (float64 RK4, zero-order hold, Euler-Maruyama process noise 0.5 % x_std per sqrt(s), sensor
  model of the variant, stored float32), same plant configs as the v1 export (experiments/tree run configs).
  lorenz96 uses the chaos_deep configuration.
* **Inputs (family B)**: per-plant operating-profile generators (`plants/profiles.py`, from the input review of
  2026-11-01) through an actuator model (box, rate limit, first-order lag, quantisation, deadband), in physical time
  units. Training and OOD operators are disjoint and share the actuator and box. Some figures were checked against
  primary sources: US06 600 s, max acceleration 3.76 m/s^2; UDDS 1369 s trace, 1.48 m/s^2; FHWA metering 240-900
  veh/h; PeMS 30-s detector data; PX4 MPC_XY_CRUISE 5 m/s and MPC_ACC_HOR 3 m/s^2. Profiles marked `uncited` in
  the card (shaker test sequences, minimum-jerk moves, the A-stat ramp duration, the cstr deadband) are benchmark
  choices with no paper attached.
* **Inputs (family A)**: u = 0. The event-structured recipe adds impulsive state kicks: at segment boundaries
  (log-U[3, 12] t_ref), with probability 0.45, a kick of U[0.2, 1] x_std in a random direction, clipped to the
  envelope.
* **ICs**: Sobol in the IC box (family B); family A train / val / test_id in the core (centre +- 0.6 half-width),
  val_ood in the shell 0.6-0.8 and test_ic_ood in the shell 0.8-1.0 (normalised inf-norm), so the OOD tiers are
  disjoint from train / train_event / val / val_event by construction. NOT from `phase1`: its zero-input episodes
  start anywhere in the operating envelope (as in v1) and therefore include shell ICs. A method that uses phase1 must
  say so next to any test_ic_ood number (PITFALLS P30).
* **New plant versions**: two_link_arm_v2 (two_link_arm_b with a +-12 N m shoulder torque box; the v1 +-1 N m could
  not move a 1 m / 1 kg arm whose shoulder gravity torque reaches 19.6 N m) and highway_v2 (on-ramp box 0.05-0.5 veh/s).
* Generated 2026-11-03 by experiments/bench_v2/code/build_data.py (GPU, float64 integration), exported to Parquet
  by tools/export_v2.py; SHA-256 per file in metadata.

## Uses
Intended: comparing state estimators across autonomous and driven systems, under model incompleteness, realistic
operation and sensor / input-record faults. Not intended: claims about a particular physical device.
