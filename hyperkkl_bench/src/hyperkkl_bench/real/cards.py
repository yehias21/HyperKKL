"""Cards of the real-data tracks: relevance, task, model, licence (text) + the frozen identification and the
smoke-run results (generated). ``hyperkkl-bench real-card <dataset>``; tools/build_real_cards.py writes
docs/cards/real_<dataset>.md."""

from __future__ import annotations

import json
from pathlib import Path

from .registry import spec

TEXT = {
    "battery_pf": {
        "title": "Panasonic 18650PF Li-ion cell: SOC on held-out drive cycles (REQUIRED real track)",
        "relevance": (
            "The cell is driven by a measured, exogenous, strongly time-varying input: the drive-cycle current "
            "(a scaled vehicle power profile, -20 to +20 A). Its state (state of charge, two RC polarisation "
            "voltages, temperature, filtered voltage) evolves only because of that input, so the system is "
            "genuinely non-autonomous: with u = 0 it rests. SOC is HIDDEN (no sensor; the measured outputs are "
            "the case temperature and the terminal voltage), and it has a reference (the lab's coulomb counter "
            "from a known full charge). Input conditions differ between splits: training is Cycles 1-3, UDDS, "
            "LA92; the test cycles US06 (aggressive), HWFET (highway) and NN are new event SEQUENCES (the paper's "
            "sequence_ood notion; Cycles 1-4 contain segments of every profile, disclosed), and the 0 degC "
            "recordings add a temperature shift."),
        "task": "From u (current) and y = (dT, Vm), starting from a prior 20 % SOC off, estimate SOC. Score: settled SOC RMSE [% SOC] (t >= 15 min of 30-min windows); headline = mean of US06 / HWFET / NN at 25 degC. Secondary: the 9 cycles at 0 degC (models fitted at 25 degC).",
        "model": (
            "2-RC Thevenin equivalent circuit + lumped thermal model + first-order voltage acquisition filter "
            "(plants/systems_real.Pan18650PF; the standard BMS model, e.g. Plett, Battery Management Systems "
            "Vol. 1, Artech House 2015, ch. 2). OCV(SOC) from the C/20 test (mean of discharge and charge "
            "branches, degree-7 polynomial + two exponential edge terms); R0, R1, C1, R2, C2 by least squares "
            "on the bin-mean terminal voltage of the TRAINING cycles; C_th, hA by least squares on dT; "
            "Arrhenius R(T) with E_a/R = 3000 K (literature value, not identifiable at one temperature). No "
            "hysteresis, no SOC- or rate-dependent resistance: a partial model. V1, V2 have no labels; they "
            "carry the model's replay as placeholders (x_fill), never a loss target or a score."),
        "excluded": "",
    },
    "battery_lg": {
        "title": "LG 18650HG2 Li-ion cell: SOC on held-out drive cycles and temperatures",
        "relevance": (
            "Same physics and task as battery_pf on a different cell (3 Ah LG HG2, NMC), a different lab "
            "(McMaster) and different drive cycles: the input is the measured cell current of vehicle drive "
            "cycles, SOC is hidden with a coulomb-counter reference. Input conditions differ between splits: "
            "training on UDDS, LA92 and the random mixes Mixed1-6; validation Mixed7; test US06 (aggressive) "
            "and Mixed8 at 25 degC, and US06 / HWFET / Mixed8 at 0 and 10 degC and US06 / HWFET at 40 degC "
            "(temperature shift in both directions; the 25 degC HWFET file is a rest and the 40 degC Mixed8 file "
            "is truncated, both excluded). It tests whether a result on battery_pf transfers to a second cell without any "
            "re-design (the latent design, windows and noise settings are battery_pf's, fixed a priori)."),
        "task": "As battery_pf. Headline = mean of US06 / Mixed8 at 25 degC; secondary: 0, 10, 40 degC.",
        "model": (
            "The same 2-RC ECM structure, identified on THIS cell's training cycles + its own C/20 test "
            "(identify_lg(): battery_pf's procedure with physical bounds tau1 in [1, 100] s, tau2 in [100, 3600] s; "
            "unbounded, the fit put tau2 at 30 days, an integrator absorbing an OCV / capacity mismatch; frozen in "
            "resources/real_battery_lg.json). The EKF noise is battery_pf's "
            "validation-tuned setting reused a priori (no tuning on this cell)."),
        "excluded": "",
    },
    "emps": {
        "title": "EMPS electro-mechanical positioning system: velocity from position under force pulses",
        "relevance": (
            "A prismatic axis (DC motor + ball screw + load) driven by a measured force u = g_tau * vir [N] "
            "that varies continuously in time (bang-bang acceleration profiles). The plant has a pure integrator "
            "and nonlinear (Coulomb) friction, so position and velocity evolve only because of the input. Only "
            "the motor-encoder position is measured; the VELOCITY is hidden. Input conditions differ between "
            "splits: train / val are the bang-bang record; the test record adds force pulses (a disturbance "
            "input absent from training). The data were taken in closed loop (PD position control); the "
            "estimator treats the measured force as the exogenous input, as the providers' own models do. Caveat: "
            "the force is the PD controller output nu = kp kv (q_ref - q) - kv q_dot, so it carries velocity "
            "information itself; learned methods can exploit the control law (REVIEW_5 B21)."),
        "task": ("From u (force) and y = q (position), prior (q_0, 0), estimate the velocity. Label: the providers' "
                 "offline velocity (zero-phase Butterworth 4 @ 100 Hz + central difference, the IDIM-LS 'bandpass "
                 "filtering process'). HEADLINE split test_pulses_outage: the encoder is frozen (last value held) for "
                 "four 100-ms bursts in the second half of every 2-s window, and the velocity is scored during the "
                 "bursts only (velocity NRMSE; position NRMSE also reported). The plain split test_pulses (settled "
                 "window) is kept as a sanity check only: with a 50000-count encoder a causal finite difference of y, "
                 "no model, scores 0.0008 there, so it is a DEGENERATE estimation task (found in the first smoke runs, "
                 "before any comparison; the outage split was added then)."),
        "model": (
            "Rigid-body direct dynamic model of the providers (Janot, Gautier, Brunot 2019, eq. 3): M q'' = tau "
            "- Fv q' - Fc sign(q') - offset, sign smoothed by tanh(v / 1 mm/s) (plants/systems_real.EMPSAxis). "
            "M, Fv, Fc, offset by the providers' IDIM-LS on the TRAINING samples only; they agree with the "
            "published full-record estimates within 0.5 % (M), 0.5 % (Fv), 0.5 % (Fc), 3 % (offset). HyperKKL "
            "latent design: n_z = 4, alpha = 20 /s (max|lambda| dt = 0.05)."),
        "excluded": "",
    },
    "silverbox": {
        "title": "Silverbox electronic Duffing oscillator: velocity under amplitude extrapolation",
        "relevance": (
            "An electronic mass-spring-damper with a cubic hardening spring, driven by a measured input voltage "
            "(V1) and observed through the displacement voltage (V2) only; the velocity is hidden. It is a "
            "forced nonlinear oscillator: without input it decays to rest. Input conditions differ by the "
            "benchmark's design: training on odd random-phase multisines; tests on a held-out multisine, on "
            "filtered Gaussian noise of growing amplitude inside the training amplitude range, and on its last "
            "part, BEYOND the training amplitude (extrapolation)."),
        "task": "From u (V1) and y (V2), prior (y_0, 0), estimate the velocity of V2. Score: settled velocity NRMSE (second half of 2048-sample windows) against an offline zero-phase derivative (Butterworth 4 @ 250 Hz); headline = mean of the three test splits.",
        "model": (
            "Forced Duffing oscillator y'' = -d y' - k1 y - k3 y^3 + g u + b0 (the structure of Wigren & "
            "Schoukens 2013; time unit 10 ms; plants/systems_real.SilverboxDuffing). No published parameter "
            "values: equation-error least squares on zero-phase derivatives of the TRAINING samples, refined by "
            "output-error multiple shooting (200-sample segments). The circuit has further dynamics the "
            "second-order model misses (validation simulation NRMSE ~0.24 over 3.4-s windows): a partial model. "
            "HyperKKL latent design: n_z = 4, alpha = 0.2 per 10 ms (20 rad/s), the largest Butterworth bandwidth "
            "that passes the design gate max|lambda| dt <= 0.1 at 610 Hz; it is slower than the ~70 Hz "
            "resonance (stated). A causal finite difference of y (no model) scores ~0.34: the floor to beat."),
        "excluded": "",
    },
    "calce": {
        "title": "CALCE A123 LiFePO4 cell: SOC on held-out dynamic profiles, with a flat voltage plateau",
        "relevance": (
            "The cell is driven by measured, exogenous, time-varying current profiles (DST, US06, FUDS; -3.9 to +2.1 A) from a full "
            "CC-CV charge to 2.0 V, so the state moves only because of the input. SOC is hidden (measured: current, terminal voltage, "
            "case temperature) and has a reference: the lab's coulomb counter from the end of the charge, with the cell's own C/20 "
            "capacity. What is new against battery_pf / battery_lg: a LiFePO4 plateau (0.9 mV per 1 % SOC between 20 and 80 % SOC, "
            "ten times flatter than the LG cell) and a 52-mV charge/discharge hysteresis, so the voltage carries little SOC "
            "information; and training on ONE profile family (DST) with testing on two new ones (US06, FUDS), the benchmark's "
            "input-generalisation question in its hardest real form. Temperature splits at 0, 10, 40 and 50 degC move the plant."),
        "task": ("From u (current, discharge positive) and y = (dT, Vm), prior SOC0 +- 0.2, estimate SOC. Score: settled SOC RMSE "
                 "[% SOC] over t >= 15 min of 30-min windows; headline = mean of test_US06 / test_FUDS at 25 degC (both cells). "
                 "Temperature splits reported separately (PITFALLS P17)."),
        "model": ("2-RC ECM + thermal (plants/systems_real.Pan18650PF; the class name is historical) on cell A1-007's training DST at "
                  "25 degC: OCV = the C/20 DISCHARGE branch (every profile is a discharge from full; the averaged branch biased the EKF "
                  "by ~10 points of SOC on val), ECM fitted on SOC >= 10 % (the end-of-discharge collapse is outside a 2-RC model). Both "
                  "choices made on val by gated-EKF SOC RMSE (21.1 -> 12.3 -> 11.0 %). R0 = 157 mOhm is high for this cell and not "
                  "explained. Cell A1-008's capacity is 2.3 % above the model's (a real partial-model mismatch)."),
        "extra": ("## Licence\n\n" + "Licence checked by the author, 2026-11-02. Source page: 'open access to our experimental test data'; 'Use of "
                    "this data for publication purposes should include references to the CALCE article(s)'. No licence file or "
                    "redistribution grant: fetch-only." + "\n\n## Splits (fixed before any estimator ran)\n\n" + "| split | recordings | windows |\n|---|---|---|\n"
                   "| train | DST: A1-007 at 20, 25, 30 degC; A1-008 at 20, 30 degC | 238 |\n| val | DST: A1-008 at 25 degC | 32 |\n"
                   "| test_US06, test_FUDS | both cells, 25 degC | 62, 66 |\n| test{0,10,40,50}_{US06,FUDS} | both cells | 54-70 each |\n\n"
                   "Files: one workbook per cell and temperature holds DST (step 8), US06 (step 16), FUDS (step 24), confirmed by the "
                   "logged rest intervals 361 / 601 / 1373 s. -10 degC not used (A1-008 truncated); the 20 degC 'newprofile' test not used."),
        "excluded": "",
    },
    "neurobem": {
        "title": "NeuroBEM agile quadrotor: body velocity and tilt from the onboard IMU and rotor speeds",
        "relevance": (
            "A quadrotor is input-driven in the strictest sense: the four rotor speeds are the only thing that moves it, they vary "
            "continuously over the whole flight (250 to 2760 rad/s here), and with the rotors stopped the vehicle falls. Thrust, "
            "torque and rotor drag all depend on u at every instant, so an observer that ignores u has no way to separate "
            "commanded motion from drag. The rotor speeds are measured by bidirectional-DShot telemetry and are exogenous to the "
            "observer. The velocity and attitude are hidden: no onboard sensor reads them, and Vicon provides the reference. "
            "Input conditions differ between splits: test_ood holds out whole manoeuvre types (time-optimal CPC race lines, "
            "'satellite' figures, vertical oscillations), which is the benchmark's 'different operator, same drone' condition. "
            "Caveat: the flights are closed loop. Betaflight's rate controller and the trajectory tracker set the rotor speeds from "
            "the same gyro the estimator reads, so u carries state information; learned methods can exploit that, and the input "
            "is exogenous to the observer but not independent of the state (as on `emps`)."),
        "task": (
            "From u (4 rotor speeds, minus 1350 rad/s) and y (gyro 3, accelerometer 3), both causal 10-ms means on a 100-Hz grid, "
            "starting from the blind prior v_b = 0, level (g_b = e_z), estimate the body velocity v_b [m/s] and the gravity direction "
            "g_b (roll and pitch). Yaw and position are unobservable from these sensors and are not in the task. Episodes: 10-s "
            "windows inside a Vicon-valid segment, every 2.5 s (train) or 5 s (val, test); segments under 10 s give no window. Score: "
            "settled (t >= 5 s) hidden NRMSE; headline = mean of test_id and test_ood. Each report also gives per-axis velocity RMSE "
            "[m/s], tilt, roll and pitch RMSE [deg], and a breakdown by manoeuvre type. Labels are Vicon-derived and smoothed offline "
            "(non-causal), so they are estimates too (PITFALLS P15), though far better than the IMU. The accelerometer is the hard "
            "part: after calibration its residual against the mocap-derived specific force is 1.3 / 1.7 / 2.5 m/s^2 (x / y / z)."),
        "model": (
            "Nominal rigid body in the body frame, without yaw or position (plants/systems_real.NeuroBEMNominal; state v_b, g_b, "
            "w, a_m). Structure: Mahony, Kumar, Corke, IEEE RAM 2012 (rigid body, quadratic rotor map); Faessler, Franchi, Scaramuzza, "
            "IEEE RA-L 2018 and Leishman et al., IEEE CSM 2014 (lumped linear rotor drag, without which the accelerometer shows no "
            "tilt or horizontal speed). J is the Readme value; the rest is identified on the TRAIN flights only (48 flights): thrust "
            "and drag by least squares on the mocap-derived specific force; control effectiveness from 2-Hz high-passed signals "
            "(Smeur, Chu, de Croon, JGCD 2016), then a first-order blade-flapping moment and a trim moment on the remainder (a plain "
            "fit gives B too small by a factor ~8, because at low frequency the rotors balance the steady aerodynamic moment); "
            "accelerometer affine calibration and lag. The calibration gains are 0.60-0.78 with a 2.6 m/s^2 z offset: unexplained, "
            "taken as measured. NOT in the model: blade-element aerodynamics, the NeuroBEM residual (never used or copied), motor "
            "reaction torque (so yaw acceleration fits poorly), speed-dependent thrust loss, battery and ground effects. It is open-loop "
            "UNSTABLE: useful for one step and about 0.1 s, then it tumbles, so HyperKKL's phase-1 trajectories use the real training "
            "windows instead of a model replay (`replay = real_windows`). HyperKKL latent design: n_z = 13, alpha = 1.2 /s (max|lambda| "
            "dt = 0.0996), fixed before any run; A3 recipe. 'Mixed data' and C5 are NOT used: they need a simulator of the true plant."),
        "extra": (
            "## Licence and redistribution\n\n"
            "**No licence text was found** on the project page, the Readme, the download host or the code archive (checked 2026-11-02). "
            "The only stated term is **citation only**: 'If you use this work in your research, please cite the following paper'. "
            "arXiv's non-exclusive licence covers the paper, not the data. The author states he checked the licence personally and "
            "approved adding the data; that check is not recorded on the source. **Redistribution is not granted**, so all rights must "
            "be assumed reserved. The track is **fetch-only**: this package ships no raw data and no episodes, and `fetch` downloads "
            "from the original host with SHA-256 checks. Before a public release: ask the UZH RPG providers for written terms, keep "
            "`data_real/` and `logs/hkkl_train/*/data.npz` out of it, and decide about the derived items that do ship: "
            "`docs/real/neurobem_episodes.png` (plots of the data) and the identified constants, alignment and split lists in "
            "`resources/real_neurobem.json`.\n\n"
            "## Data\n\n"
            "**95 flights** (the project page says 96; `Flights.txt`, `raw_data` and `processed_data` all hold 95), 247 segments, 4496 s. "
            "u and y come from the raw Betaflight logs (`real/blackbox.py`, a new pure-Python decoder), labels from the processed CSVs. "
            "Every flight has its own clock fit (offset + 2800-4100 ppm drift; gyro residual <= 0.018 rad/s). Full evidence: "
            "`docs/cards/neurobem_datacheck.md`. Checks and plots: `docs/real/neurobem_checks.json`, `neurobem_episodes.png`.\n\n"
            "## Splits (by flight, fixed in resources/real_neurobem.json; never within a flight)\n\n"
            "| split | flights | 10-s windows | manoeuvre types |\n|---|---|---|---|\n"
            "| train | 48 | 567 | circle, wobbly circle, 3D circle, battery-test circle, lemniscate, linear oscillation, random points, ellipse |\n"
            "| val | 12 | 98 | 3D circle, battery-test circle, lemniscate, linear oscillation, random points |\n"
            "| test_id | 16 | 106 | same types as train, held-out flights |\n"
            "| test_ood | 19 | 110 | CPC (9), satellite (8), vertical oscillation (2): held out entirely |\n\n"
            "Rule, fixed before any estimator ran: the three held-out types go to test_ood; within each other type, flights in time "
            "order, index i % 5 == 2 -> test_id, i % 5 == 4 -> val, else train. Aggressive flight is on both sides: train lemniscates and "
            "linear oscillations reach 15.9-17.7 m/s and 46-49 m/s^2, the held-out satellites and CPC 12-15 m/s and 43-46 m/s^2. The "
            "providers' `testset.txt` is not used: it holds out segments of flights that also train. Flight 2021-02-18-16-43-54 is "
            "164 Hz in the CSV and is interpolated; flight 2021-02-23-19-15-11 notes 'battery moved' (centre of mass shift) and is kept."),
        "excluded": "",
    },
}

EXCLUDED = [
    ("Cascaded Tanks (Schoukens, Mattsson, Wigren, Noel; CC BY-SA 4.0)",
     "input-driven and relevant, but the hidden upper-tank level has no measurement and no physically derived "
     "reference (it can only be inferred through the model itself), and each split is a single 1024-sample "
     "record (one train, one test): no well-defined hidden-state score and too little data for a learned-"
     "estimator track. A held-out-sensor variant is possible later. The official 4TU file URL also timed out from "
     "this host (the Google Drive mirror works)."),
    ("Wiener-Hammerstein 2009 and WH with process noise (Schoukens, Suykens, Ljung)",
     "block-oriented electronic circuit: its internal states have no physical meaning and no reference, so there "
     "is no hidden quantity to estimate beyond output simulation (WH-2009 licence not stated; WH-PN CC BY-SA 4.0)."),
    ("Coupled electric drives (Wigren & Schoukens 2017)",
     "500 samples per record, a sign-insensitive speed sensor, and no reference for any hidden state (belt "
     "tension, motor speeds): no well-defined estimation task; licence not stated."),
    ("CALCE INR 18650-20R (NMC) dynamic set", "the A123 LiFePO4 set is the `calce` track; the NMC set repeats battery_pf / battery_lg and has no temperature channel."),
    ("Sandia / batteryarchive.org", "licence not stated; full CSVs only on e-mail request; not used."),
    ("Blackbird UAV dataset (MIT)", "4.9 TB; the data host did not resolve from this machine; data licence unclear (the MIT licence covers the code repository)."),
    ("NanoDrone system-identification benchmark (IDSIA; licence checked by the author)",
     "evaluated as a roll-channel task (experiments/real_candidates): a linear causal FIR on the past sensors alone (0.346) beats every "
     "estimator (GRU 0.395, EKF 0.423, HyperKKL 0.76-0.93); every published channel is zero-phase filtered (non-causal) and the velocity "
     "reference is the mocap-aided onboard filter; 15 flights; redundant with neurobem."),
    ("NanoBench (Ullah, Baca 2026; BSD 3-Clause)",
     "evaluated as the same roll-channel task: a linear causal FIR on past gyro + accelerometer (0.473) equals the GRU (0.480) and beats the "
     "model-based filters, so there is no nonlinear estimation task left; the motor commands add 2-8 %. The repository has 107 flights (README: "
     "172), 20 exact duplicates and 18 with an onboard gyro that does not match Vicon. Redundant with neurobem."),
    ("BMW i3 real driving (Steinstraeter et al., IEEE DataPort; login, no licence stated)",
     "staged as experimental only: the logged SoC is the BMS estimate, a coulomb counter of the logged current within each trip (Q = 61.7 Ah, "
     "residual < 0.1 %), so the score is agreement with the BMS, not true charge; 5 trips have broken labels; licence not confirmed."),
    ("RACECAR (Indy Autonomous Challenge; CC BY-NC 4.0)", "no throttle, brake or steering topic in any of the 23 bag topic lists: no measured input."),
    ("BETTY (CMU et al., 32 TB)", "throttle / brake / steering are logged per the paper, but the data host timed out from this machine and the site's data link is disabled."),
    ("F-16 ground vibration test (Noel, Schoukens; CC BY-SA 4.0)", "not downloadable from this host (figshare 504, 4TU unreachable); no physical hidden state has a reference, so only a held-out-sensor task would be possible."),
]


def smoke_results(dataset: str) -> dict:
    from ..paths import CHECKOUT
    out = {}
    for f in sorted((Path(CHECKOUT) / "reference" / "real" / "smoke").glob(f"{dataset}__*.json")):
        d = json.loads(f.read_text())
        out[f.stem.split("__", 1)[1]] = {"headline": d["meta"].get("headline"),
                                         "splits": {s: b["summary"].get(d["meta"]["headline_metric"]) for s, b in d["results"].items()}}
    return out


def neurobem_table() -> list:
    """Smoke results of neurobem in physical units, per state (settled, t >= 5 s)."""
    from ..paths import CHECKOUT
    order = ["prior_mean", "imu_static", "ekf", "ekf_gated", "ude_ekf_gated", "gru", "hyperkkl"]
    L = ["| method | split | hidden NRMSE | v_x [m/s] | v_y [m/s] | v_z [m/s] | |v| [m/s] | roll [deg] | pitch [deg] | tilt [deg] | diverged |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        f = Path(CHECKOUT) / "reference" / "real" / "smoke" / f"neurobem__{m}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for sp in ("test_id", "test_ood"):
            r = d["results"][sp]["summary"]
            L.append(f"| {m} | {sp} | {r['nrmse_hidden']:.3f} | {r['vel_x_ms']:.2f} | {r['vel_y_ms']:.2f} | {r['vel_z_ms']:.2f} | "
                     f"{r['vel_rmse_ms']:.2f} | {r['roll_rmse_deg']:.1f} | {r['pitch_rmse_deg']:.1f} | {r['tilt_rmse_deg']:.1f} | {r['diverged']:.3f} |")
    return L


def card_markdown(dataset: str) -> str:
    t = TEXT[dataset]
    sp = spec(dataset)
    src, fr = sp["source"], sp.get("frozen", {})
    L = [f"# Real-data track `{dataset}`: {t['title']}", "",
         f"**Source.** {src['title']} ({src.get('creator', '')}); {src.get('repository', '')}. "
         f"{'DOI ' + src['doi'] + '. ' if src.get('doi') else ''}Landing page: {src['landing']}",
         f"**Licence.** {src['license']}. **Redistribution:** {src['redistribution']}",
         f"**Cite.** {src['citation']}", "",
         "**Get the data.** `hyperkkl-bench fetch " + dataset + "` (from the original host; every file checked "
         "against its SHA-256), then `hyperkkl-bench prepare " + dataset + "`. `--from-local DIR` copies a local copy instead; "
         "`--dry-run` only checks the URLs.", "",
         "## Why it belongs in this benchmark", "", t["relevance"], "", "## Task and score", "", t["task"], "",
         "## Partial model (f_known)", "", t["model"], ""]
    if fr:
        L += ["## Frozen identification (training split only)", "", "```json",
              json.dumps({k: v for k, v in fr.items() if k not in ("n_episodes", "n_windows", "filter_tuning", "alignment")}, indent=1,
                         default=float)[:6000 if dataset == "neurobem" else 4000], "```", ""]
    if t.get("extra"):
        L += [t["extra"], ""]
    sm = smoke_results(dataset)
    if sm:
        L += ["## Smoke runs (1 seed, short training; NOT benchmark results)", "",
              "| method | headline | " + " | ".join(next(iter(sm.values()))["splits"]) + " |",
              "|---|---|" + "---|" * len(next(iter(sm.values()))["splits"])]
        for m, r in sm.items():
            L.append(f"| {m} | {r['headline']:.4g} | " + " | ".join(f"{v:.4g}" for v in r["splits"].values()) + " |")
        L.append("")
    if dataset == "neurobem" and sm:
        L += ["### In physical units (mean over episodes of the settled RMSE; test splits)", ""] + neurobem_table() + [""]
    return "\n".join(L)


def excluded_markdown() -> str:
    L = ["# Real-data candidates that were NOT added", "", "| candidate | reason |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in EXCLUDED]
    return "\n".join(L) + "\n"
