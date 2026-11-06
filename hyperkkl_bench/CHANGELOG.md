# Changelog

## 2.0.0rc1 (2026-11-03/04, Dubai): two families of state estimation, operating-profile tiers

Major version: new frozen splits and a new top-level structure. Every v1 number stays reproducible (legacy tiers and
`hkb.load(plant, level, split)` unchanged).

- **Two families of state estimation.** `autonomous/...` (no input; vdp, fhn, rossler and lorenz (both nominal-data levels only),
  lorenz96, lorenz96_n10, network) and `driven/...` (13 input-driven plants + deployment, scaling, real tracks).
  API: `hkb.load("family/plant/level", split, recipe=)`, `hkb.context_v2`, `hkb.evaluate_v2`; CLI `list --family`,
  `evaluate-v2`, `check-autonomous`, `card family/plant`; `hkb.families.predict_v2` / `score_hidden_v2`.
- **New inclusion check** "non-trivial unforced dynamics" (family A; `hyperkkl-bench check-autonomous [--perturb 0,0.3]`). Forced plants that
  rest at u = 0 are excluded from family A by rule.
- **Operating-profile tiers (family B)**: `test_op_id`, `test_op_ood` (different operator / route / schedule, same
  actuator and box), with an actuator model and three input-record conditions (`input_realistic`,
  `input_cmd_vs_applied`, `input_quant`). The v1 tiers become `test_legacy_*`.
- **Family A tiers**: `test_id`, `test_ic_ood` (held-out IC shell), `test_param_shift`; sensor conditions noise10,
  outliers, dropout, burst_outage, poor_init.
- **Recipes**: standard (`train`, `val`) and event-structured (`train_event`, `val_event`) per family; `mixed` =
  both, the reference recipe.
- **New plant versions**: `two_link_arm_v2` (+-12 N m shoulder torque; v1's +-1 N m could not move the arm against up to 19.6 N m of gravity) and
  `highway_v2` (on-ramp box 180-1800 veh/h). Lorenz-96 joins the core chaotic set (both families).
- **Literature** (experiments/bench_v2/design/LITERATURE.md): every "(verify)" figure of the input review checked; corrected
  (US06 600 s) or removed (A-stat acceleration range, stiction 0.5-5 %). Profiles whose source was not read (Kerschen,
  Flash & Hogan) are marked `uncited`; references whose content was seen only via secondary sources (Swevers, Lorenz 1990,
  Palmer, Destexhe beyond the title) are cited with that caveat.
- **Fix**: `TunedKF` on real-data tracks (PITFALLS P28) (needed `context["prior_mean"]`; now uses the per-episode track prior).
- **Hidden test** for both families (`hidden_op_id`, `hidden_op_ood`, `hidden_id`, `hidden_ic_ood`).
- **Reference results**: all bundled and research baselines on equal data, 3 seeds (reference/v2/RESULTS_v2.md; study: experiments/bench_v2).
- Licence MIT (code and simulated data) kept; real-data tracks stay under their providers' terms.

## 1.3.0rc1 (2026-11-03, Dubai): CALCE LiFePO4 real track

- **New real track `calce`** (`hyperkkl_bench.real.calce`, shared machinery `real/battery_field.py`): SOC of an A123 APR18650M1A
  LiFePO4 cell (CALCE, University of Maryland). Training on one profile family (DST: cell A1-007 at 20 / 25 / 30 degC, A1-008 at
  20 / 30 degC), val DST A1-008 25 degC, tests US06 / FUDS at 25 degC (headline) and at 0 / 10 / 40 / 50 degC (reported apart).
  The flat LFP plateau (0.9 mV per 1 % SOC) makes voltage nearly uninformative. **Licence checked by the authors, 2026-11-02**
  (the source asks for a citation; no redistribution grant): fetch-only, 16 files SHA-256 checked. Workbooks parsed with openpyxl
  (cell values only; new in the `[real]` extra).
  - Partial model: 2-RC ECM + thermal on the training DST. OCV = the C/20 discharge branch, ECM on SOC >= 10 %. Both were chosen on
    val after the averaged OCV biased the EKF (val SOC RMSE 21.1 -> 12.3 -> 11.0 %). PITFALLS P22.
  - Smoke (1 seed; settled SOC RMSE at 25 degC): prior_mean 19.1, coulomb 20.0, EKF 7.93, UDE+EKF 9.29, GRU 5.08, HyperKKL A3 15.4
    (val 5.6: A3 does not generalise from one training profile family; PITFALLS P23).
- Candidates evaluated and NOT added (`docs/cards/REAL_EXCLUDED.md`, evidence in experiments/real_candidates/VERDICTS.md and
  DIAG_REAL.md): NanoBench and NanoDrone (a linear FIR on past sensors matches or beats every estimator), BMW i3 (BMS-estimate
  labels, no licence stated; kept staged as experimental), RACECAR (no actuator topics), BETTY and F-16 (not downloadable from this
  host).
- Diagnosis of HyperKKL A3 on the real tracks (DIAG_REAL.md): emps fails only during encoder outages (the deployed observer is
  driven by y alone and never saw an outage); neurobem fails above the training tilt range (train max 98.6 deg); no shared code
  changed, so no earlier result moves.
- `REAL_VERSION` hkb-real-1.2. The five earlier tracks are unchanged.

## 1.2.0rc1 (2026-11-03, Dubai): NeuroBEM real track

- **New real track `neurobem`** (`hyperkkl_bench.real.neurobem`): body velocity and tilt of an agile quadrotor (Bauersfeld et al.,
  RSS 2021, doi:10.15607/RSS.2021.XVII.042), 95 flights. u = 4 measured rotor speeds (DShot telemetry); y = onboard gyro and
  accelerometer; hidden = v_b and g_b against Vicon. Yaw and position are unobservable from these sensors and are out of scope.
  **No licence text on the source, citation only, redistribution not granted: fetch-only** (SHA-256 checked, from
  download.ifi.uzh.ch). Scorer `settled-quad-hidden-1`: settled hidden NRMSE (headline = mean of test_id, test_ood), plus RMSEs in
  m/s and deg, a manoeuvre-type breakdown, `diverged` on the settled window and `transient_peak_std`.
  - Sensor check (`docs/cards/neurobem_datacheck.md`): the processed CSVs hold a smoothed, partly non-causal state; the real
    sensors are in the raw Betaflight logs. New pure-Python decoder `real/blackbox.py` (0 dropped frames on 95 logs; validated
    against the providers' CSVs). Per-flight clock fit (offset + 2800-4100 ppm drift). Quaternion convention measured (x, y, z, w;
    Hamilton; body to world).
  - Splits by flight, fixed in `resources/real_neurobem.json` before any run: train 48, val 12, test_id 16, test_ood 19 flights
    (CPC, satellite and vertical oscillation held out entirely). 10-s windows at 100 Hz: 567 / 98 / 106 / 110.
  - Nominal model `plants/systems_real.NeuroBEMNominal`: rigid body, quadratic rotor map, lumped linear rotor drag, first-order
    flapping and trim moments, accelerometer calibration and lag. Identified on train only. No blade-element aerodynamics, no
    NeuroBEM residual. Open-loop unstable: good for one step (val 0.018 m/s, 0.16 deg) and about 0.1 s (0.35 m/s, 4.6 deg).
  - EKF noise inflation chosen on val (q x 3, r x 1).
- **Vector inputs.** `run_episodes`, `EKF` / `UKF`, `UDEFilter`, `GRUObserver` and `HyperKKLTrain` accept `u` of shape
  (E, N+1, n_u). Scalar tracks are unchanged (full suite). HyperKKL runner: `nu > 1` (local RK4 copy, context encoder with n_u
  inputs via a wrapper; research code not edited) and `replay = "real_windows"` for open-loop-unstable plants. On neurobem, A3 only:
  mixed data and C5 need a simulator of the true plant.
- Smoke runs (1 seed; `docs/cards/REAL_SMOKE.md`): ude_ekf_gated 0.216, EKF 0.355, GRU 0.366, HyperKKL A3 0.402, prior_mean 1.117,
  `imu_static` (new no-dynamics floor) 1.22. Tools: `tools/build_real_neurobem_resources.py`, `tools/rescore_hyperkkl.py`;
  `real_smoke.py --seed`; `real_check.py neurobem` (checks + `docs/real/neurobem_episodes.png`).
- Docs: card `docs/cards/real_neurobem.md`; README table, run times, install extra `.[real]` (pandas); DATASHEET; RELEASE item 6;
  PITFALLS P19-P21; NeuroBEM removed from `REAL_EXCLUDED.md`; Croissant entries; REVIEW_6.md. `REAL_VERSION` hkb-real-1.1 (the
  four earlier tracks are unchanged).
- Tests: 9 new + neurobem in the loader-contract parametrisation (decoder encodings, split lists, quaternion convention, causal resampling, plant, vector-input baselines, alignment,
  episodes by flight, scorer units).

## 1.1.0rc1 (2026-11-01, Dubai): real-data tracks

- **New track `real`** (`hyperkkl_bench.real`, `TRACKS["real"]`): state estimation on measured data of real
  input-driven systems. The raw data are **never redistributed**: `hyperkkl-bench fetch <track>` downloads
  them from the original host (or `--from-local DIR`; `--dry-run` checks the URLs) and verifies every file
  against its SHA-256 (Mendeley's own hashes for the battery data); archives are unpacked with a path check;
  `hyperkkl-bench prepare <track>` builds the episodes locally and deterministically.
  - `battery_pf` (required): Panasonic 18650PF (Kollmeyer, CC BY 4.0). The research preprocessing
    (experiments/real) is vendored; windows, labels, priors and the model-replay placeholders equal the
    research pipeline's value for value, the re-identified ECM equals the frozen one (0 relative difference),
    and the bundled EKF reproduces the research EKF to 1e-13. Scorer `settled-soc-rmse-1`.
  - `battery_lg`: LG 18650HG2 (Kollmeyer, Naguib, Skells, CC BY 4.0): same task on a second cell, split and
    design fixed a priori; ECM identified on its training cycles with physical bounds on the RC time constants.
  - `emps` (licence not stated; fetched only): velocity of a positioning axis; friction model by the
    providers' IDIM-LS on the training samples (matches the published values within 0.5 %). The plain
    velocity task proved degenerate (a causal finite difference scores 0.0008), so the headline split is
    `test_pulses_outage` (four 100-ms encoder outages, scored during the outages).
  - `silverbox` (licence not stated; fetched only): velocity of an electronic Duffing oscillator under
    amplitude extrapolation; Duffing model by least squares + output-error refinement on training samples.
- `RealEpisodeSet` (EpisodeSet + `prior`, `prior_std`, `label_mask`, `x_fill` placeholders, `meta`,
  `score_mask`); `real.evaluate` (fit on train / val, per-episode priors, per-split context, e.g. the 0 degC
  battery model); `RealReport`. Partial models in `plants/systems_real.py` (Pan18650PF, EMPSAxis,
  SilverboxDuffing) with the same interface as the simulated plants.
- **HyperKKL trains through the benchmark API**: `baselines.hyperkkl_train.HyperKKLTrain` runs the A3 recipe
  (beat_greybox package copy: SOAP, trainable base decoder, strided latent-window context; lora r = 4) in a
  subprocess on any dataset whose context declares a latent design and a partial model; masked loss on the
  labelled states. The mixed-data and C5 parts of the default recipe need a simulator of the true plant and do
  not apply to real data (stated in `fit_info`).
- Bundled baselines on real data: filters read `context["hold_missing"]` (quantised sensors, PITFALLS P14);
  `UDEFilter` uses the split's placeholders for never-labelled states and per-episode priors for its
  deployed-validation score; `PriorMean` returns 0 for never-labelled states.
- `reference/reproduce_reference.py --real`: the research battery results (HyperKKL variants, EKF / UKF,
  grey-box, Koopman, NLOX, GRU, NCDE) regenerated from stored per-episode outputs, 349 aggregates checked,
  all match (`reference/real/REFERENCE_battery_pf.md`). Smoke runs of every method on every track:
  `tools/real_smoke.py`, `docs/cards/REAL_SMOKE.md`. Checks + plots: `tools/real_check.py` -> `docs/real/`.
- Docs: README "Real-data tracks", cards `docs/cards/real_<track>.md` (relevance, task, model, licence,
  citation), `docs/cards/REAL_EXCLUDED.md` (8 candidates not added, with reasons), PITFALLS P13-P18,
  DATASHEET section, Croissant `cr:FileObject` entries with source URLs, hashes and licences.
- CLI: `fetch`, `prepare`, `real-evaluate`, `real-card`, `list --real`. Tests: `tests/test_real.py`.
- Version 1.1.0rc1 (new track, no change to any frozen core split or scorer). `REAL_VERSION = hkb-real-1.0`.

## 1.0.0rc4, round 4 (2026-11-01, Dubai): fixes for REVIEW_3

- **B14:** deployment budget verdicts use `cost.mcu_us_est`, a FLOP-based estimate on the research
  MCU model (480 MHz, 0.5 FLOP/cycle, 40 cycles per transcendental, x1.5 overhead).
  - Example: the gated EKF on quad_planar is 69 us here vs 84 us in the research study.
  - Measured Python latency is reported separately and never used for verdicts.
  - `transcendentals_per_step` hooks for GRU and UDE.
- **A18:** `.github/workflows/tests.yml` (prepared, not pushed). `tests/fixtures/mini` (0.8 MB) plus
  `tests/test_mini.py` give a data-free end-to-end run. An explicit `HYPERKKL_BENCH_DATA` is never
  silently overridden.
- `Report.table()` no longer prints an empty OOD-gap line.
- **Reference filter: back to the published gate (PITFALLS P12).**
  - The full-tier regenerations showed that every safeguard fixes some cells and breaks others:
    the round-3 warm-up, a Huber warm-up, and P re-opening with a Huber forced update.
  - `EKF(gate=True)` is again exactly the research gating rule, plus stale-sample skipping (default
    for every filter; fixes burst-outage drift, e.g. rossler 1.84 -> 0.06).
  - The safeguards stay as options, off by default. New `ekf_huber` (no gate) is reported next to it.
  - All reference runs were regenerated. The 3-t_ref warm-up described in the round-3 entry is
    superseded.

## 1.0.0rc3, round 3 (2026-11-01, Dubai): fixes for REVIEW_2

- **A12:** `baselines.stored.StoredHyperKKL` runs the paper's method (HyperKKL `repro_A3_D0`
  equal-data, `base`, `C5_j3_s0.3_k0.5`) from the experiments/jitter_fix checkpoints.
  - Cells: battery/partial, cartpole_b/realistic, quad_planar/realistic.
  - It runs on CPU in a subprocess with the experiment's own package copy.
  - Stored per-episode NRMSE reproduced to <= 3e-4 relative (quad 3.6e-5, cartpole 1.4e-4,
    battery 2.8e-4).
  - `cost.measure` reports batch-only estimators as "latency not measured" instead of failing.
- **New finding, gate lock-out (PITFALLS P12).** The reference gated EKF (no warm-up) lost lock on
  lorenz/exact: one episode at 2.2, cell mean 0.28 instead of 0.006. The gate is now off for the
  first 3 t_ref (`gate_warmup_tref=3.0`, default for every gated filter, incl. TunedKF and UDEFilter).
  - Round 4 correction: a plain warm-up let early outliers wreck the filter (battery 9.3 under
    outliers). The warm-up now uses a Huber update (`warmup_huber=True`), which matches the
    no-warm-up gate on 4 plants and fixes clean lorenz.
  - All reference runs were regenerated (`tools/run_all_references.sh`).

  - PROTOCOL §3 updated.
- `tools/build_leaderboard.py` writes `reference/LEADERBOARD.md`: track scores of every stored
  baseline vs ekf_gated, and per-cell values.
- The beat_greybox A3 checkpoints, until now only in a /tmp session scratch folder, were copied to
  `archive/beat_greybox_scratch/` (not for release) so they survive.
- **A11 (round 2):** the standalone `UDEFilter` follows the research ude5k1 recipe exactly
  (K = 1, 3000 steps, batch 1024, lr grid selected on deployed val NRMSE). Validated on vdp/partial.

## 1.0.0rc2, round 2 (2026-11-01, Dubai): fixes for REVIEW_1

- **A3 (tracks):** the deployment and scaling tracks are now frozen and runnable.
  - The deployment track (quad_planar, cstr_exo) was exported read-only from the case-study cache by
    `tools/export_casestudies.py`. In cstr_exo the training splits carry sparse lab-assay labels
    (NaN elsewhere).
  - The scaling track (fhnring_n4 to fhnring_n64) was exported from cache_rev3 by `tools/export_scaling.py`.
  - The ring plant code is vendored in `plants/systems_scaling.py` and the case-study plants in
    `plants/systems_casestudy.py`.
  - Plant-specific robustness conditions apply to the deployment plants: 1 % outliers at 8 sigma,
    and GPS / sensor loss on channel 0.
- **A4:** `DATASHEET.md` (Gebru et al.) and `croissant.json` (Croissant 1.0 core).
- **A5:** `metadata()` drops the real-plant parameters unless `include_real_plant=True`. `context` never
  had them. A test guards both.
- **A6 / A7:**
  - The standalone gated EKF was run on every frozen cell: `reference/runs/ekf_gated/*.json` and
    `SUMMARY.md`.
  - `tracks.reference_reports()` loads them as the track_score yardstick.
  - REFERENCE.md no longer lists the flawed research grey-box; the JSON flags it `flawed_P1`.
- **B1:** `TunedKF` / `ekf_gated_tuned` / `ukf_gated_tuned` choose the Q/R inflation on `val` only.
- **B2:**
  - Filters use the physical per-sensor sigmas when the metadata declares them.
  - Filters skip repeated (held) samples on multi-rate sensors (`hold_missing`, on by default only
    for plants with held sensors).
  - Filters no longer raise on singular innovation matrices of diverged episodes: `solve_ex` /
    `inv_ex`, and non-finite P is clipped.
- **C1:** README run-time table. `evaluate(threads=1)` by default. `slow` pytest marker:
  `pytest -m "not slow"` takes ~1.5 min.
- CLI: `run-track`, `list --all`. `Report.load()`. Plant cards list the val_ood / hidden counts and
  cover the deployment and scaling plants.

## 1.0.0rc1, round 1 (2026-10-31, Dubai)

- New package `hyperkkl_bench`, installed with `pip install -e .`. Dependencies: numpy, scipy, pyarrow, torch.
- API:
  - `load(plant, knowledge, split, n=, seed=)` returns an EpisodeSet.
  - `list_plants()`, `plant_card()`.
  - The `Estimator` protocol: `fit`, `reset`, `step`, and an optional `run_batch`.
  - `evaluate()` returns a `Report`, with `.table()` and `.save()`.
  - `data_efficiency()`, `tracks.run_track()`, `tracks.track_score()`.
- Data:
  - Reads the 12-plant Parquet export (local folder, or the private HF dataset read-only).
  - New frozen splits from `tools/build_extra_splits.py`: `val_ood` (64 episodes per plant and data
    variant; 4 generators that no train or test split uses; public seeds) and `hidden_{id,shape_ood,sequence_ood}`
    (32 per tier; secret seeds; labels in a separate private root).
- Scoring:
  - Settled unmeasured NRMSE (the headline) and all-state NRMSE, with divergence and episode-bootstrap CIs.
  - Input-OOD gap.
  - Calibration (coverage 68/95, ANEES, NEES-in-95).
  - Robustness conditions: outliers, burst_outage, poor_init, with t_conv.
  - Cost: parameters, bytes, FLOPs/step, and measured latency at batch 1 on 1 thread.
- Plant metadata and difficulty axes, with provenance (`resources/difficulty.json`). Plant cards in `docs/cards/`.
- `check-plant` CLI: interface, observability Gramian and CRB proxy, 5T boundedness, and the latent design gate.
- Baselines:
  - Standalone EKF/UKF (gated and ungated, Huber option), GRU, prior mean.
  - `StoredModel` adapters for the baselines2 checkpoints (corrected grey-box ude5k1, UDE, NODE, GRU).
- `reference/reproduce_reference.py` regenerates the reference table from the stored per-run JSONs.
  All 1440 aggregate numbers match the stored aggregates exactly.
- Docs: README, PROTOCOL (with a pre-registration template), PITFALLS (11 items), RELEASE checklist,
  and placeholders for LICENSE and CITATION.
- Tests: pytest suite covering load, splits, scorer vs stored trajectories, reference, hidden test, protocol,
  check-plant, CLI, and the notebook.
