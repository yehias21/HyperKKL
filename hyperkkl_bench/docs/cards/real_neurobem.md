# Real-data track `neurobem`: NeuroBEM agile quadrotor: body velocity and tilt from the onboard IMU and rotor speeds

**Source.** NeuroBEM quadrotor flight dataset (L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza (Robotics and Perception Group, University of Zurich)); University of Zurich, https://download.ifi.uzh.ch/rpg/NeuroBEM/. DOI 10.15607/RSS.2021.XVII.042. Landing page: https://rpg.ifi.uzh.ch/NeuroBEM.html
**Licence.** no licence text on the source (project page, Readme, download host and code archive checked on 2026-11-02; the only stated term is 'If you use this work in your research, please cite the following paper'). Use approved by the author, who states he checked the licence personally. **Redistribution:** NOT GRANTED by any text on the source, so all rights must be assumed reserved: do not rehost the raw data or the prepared episodes. `fetch` downloads from the original host. Check the terms with the UZH RPG providers before a public release
**Cite.** L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza, NeuroBEM: Hybrid Aerodynamic Quadrotor Model, Robotics: Science and Systems (RSS), 2021, doi:10.15607/RSS.2021.XVII.042, arXiv:2106.08015

**Get the data.** `hyperkkl-bench fetch neurobem` (from the original host; every file checked against its SHA-256), then `hyperkkl-bench prepare neurobem`. `--from-local DIR` copies a local copy instead; `--dry-run` only checks the URLs.

## Why it belongs in this benchmark

A quadrotor is input-driven in the strictest sense: the four rotor speeds are the only thing that moves it, they vary continuously over the whole flight (250 to 2760 rad/s here), and with the rotors stopped the vehicle falls. Thrust, torque and rotor drag all depend on u at every instant, so an observer that ignores u has no way to separate commanded motion from drag. The rotor speeds are measured by bidirectional-DShot telemetry and are exogenous to the observer. The velocity and attitude are hidden: no onboard sensor reads them, and Vicon provides the reference. Input conditions differ between splits: test_ood holds out whole manoeuvre types (time-optimal CPC race lines, 'satellite' figures, vertical oscillations), which is the benchmark's 'different operator, same drone' condition. Caveat: the flights are closed loop. Betaflight's rate controller and the trajectory tracker set the rotor speeds from the same gyro the estimator reads, so u carries state information; learned methods can exploit that, and the input is exogenous to the observer but not independent of the state (as on `emps`).

## Task and score

From u (4 rotor speeds, minus 1350 rad/s) and y (gyro 3, accelerometer 3), both causal 10-ms means on a 100-Hz grid, starting from the blind prior v_b = 0, level (g_b = e_z), estimate the body velocity v_b [m/s] and the gravity direction g_b (roll and pitch). Yaw and position are unobservable from these sensors and are not in the task. Episodes: 10-s windows inside a Vicon-valid segment, every 2.5 s (train) or 5 s (val, test); segments under 10 s give no window. Score: settled (t >= 5 s) hidden NRMSE; headline = mean of test_id and test_ood. Each report also gives per-axis velocity RMSE [m/s], tilt, roll and pitch RMSE [deg], and a breakdown by manoeuvre type. Labels are Vicon-derived and smoothed offline (non-causal), so they are estimates too (PITFALLS P15), though far better than the IMU. The accelerometer is the hard part: after calibration its residual against the mocap-derived specific force is 1.3 / 1.7 / 2.5 m/s^2 (x / y / z).

## Partial model (f_known)

Nominal rigid body in the body frame, without yaw or position (plants/systems_real.NeuroBEMNominal; state v_b, g_b, w, a_m). Structure: Mahony, Kumar, Corke, IEEE RAM 2012 (rigid body, quadratic rotor map); Faessler, Franchi, Scaramuzza, IEEE RA-L 2018 and Leishman et al., IEEE CSM 2014 (lumped linear rotor drag, without which the accelerometer shows no tilt or horizontal speed). J is the Readme value; the rest is identified on the TRAIN flights only (48 flights): thrust and drag by least squares on the mocap-derived specific force; control effectiveness from 2-Hz high-passed signals (Smeur, Chu, de Croon, JGCD 2016), then a first-order blade-flapping moment and a trim moment on the remainder (a plain fit gives B too small by a factor ~8, because at low frequency the rotors balance the steady aerodynamic moment); accelerometer affine calibration and lag. The calibration gains are 0.60-0.78 with a 2.6 m/s^2 z offset: unexplained, taken as measured. NOT in the model: blade-element aerodynamics, the NeuroBEM residual (never used or copied), motor reaction torque (so yaw acceleration fits poorly), speed-dependent thrust loss, battery and ground effects. It is open-loop UNSTABLE: useful for one step and about 0.1 s, then it tumbles, so HyperKKL's phase-1 trajectories use the real training windows instead of a model replay (`replay = real_windows`). HyperKKL latent design: n_z = 13, alpha = 1.2 /s (max|lambda| dt = 0.0996), fixed before any run; A3 recipe. 'Mixed data' and C5 are NOT used: they need a simulator of the true plant.

## Frozen identification (training split only)

```json
{
 "params": {
  "Jx": 0.0025,
  "Jy": 0.0021,
  "Jz": 0.0043,
  "dx": 0.30647292924065084,
  "dy": 0.2995742689329794,
  "dz": 0.49008286785420996,
  "tau_a": 0.05,
  "bT0": 4.2228687717283,
  "bT1": 3.688360607531423,
  "bT2": 3.6381930310263617,
  "bT3": 3.3655053041642566,
  "Mx": 3.2216869665506893,
  "My": -5.892899939476724,
  "c0x": 0.1618934817644185,
  "c0y": 3.9848624459529387,
  "c0z": 0.021382530902984524,
  "Bx0": -70.78315492934115,
  "Bx1": -67.11471233191608,
  "Bx2": 64.3360918121524,
  "Bx3": 73.51563162239,
  "By0": 65.33079835246056,
  "By1": -63.61223398607301,
  "By2": 49.788760261595655,
  "By3": -73.31869587543171,
  "Bz0": 2.0649357275287588,
  "Bz1": -0.8020436168316107,
  "Bz2": -2.276238987364489,
  "Bz3": 1.003834926564831,
  "Cxx": 0.6899511675632606,
  "Cxy": 0.018918064754375984,
  "Cxz": 0.005424307495468515,
  "Cyx": -0.0021997080643100927,
  "Cyy": 0.6036609512847642,
  "Cyz": -0.02814195360769889,
  "Czx": -0.2199966990010597,
  "Czy": 0.2804492170960913,
  "Czz": 0.7753791717561275,
  "bx": -0.014614027306358189,
  "by": 0.7027438892845858,
  "bz": 2.561005424545731
 },
 "identification": {
  "x_std": [
   3.7922473623240087,
   4.23316108925397,
   2.482135087644111,
   0.4503573901843291,
   0.3997432039850391,
   0.2772895290508506,
   1.4752996773577713,
   1.4393420791955813,
   0.26955173634802965,
   1.6804181762915427,
   1.9753061630973185,
   6.876440826745613
  ],
  "x_mean": [
   -0.009988958696546233,
   0.08246635228172602,
   0.5507665044246113,
   0.015623743779912765,
   0.0068629536575720825,
   0.748467038385985,
   0.020467037616269276,
   -0.006573241290500025,
   -0.007791856333260598,
   0.1095345802051258,
   0.2684549258688517,
   14.635368755311843
  ],
  "y_mean": [
   0.020467037616269276,
   -0.006573241290500025,
   -0.007791856333260598,
   0.1095345802051258,
   0.2684549258688517,
   14.635368755311843
  ],
  "r_y": [
   0.06348788703405367,
   0.07729803827683412,
   0.011715597767501949,
   1.3410113422896475,
   1.6645848907285987,
   2.548708855238443
  ],
  "u_std": [
   359.46338947811375,
   335.4427542732128,
   349.7790921984079,
   329.3145926355757
  ],
  "u_mean": [
   58.0306325639402,
   -49.424905322133846,
   52.203966235141955,
   -36.22460233652499
  ],
  "u_scale": 1041.6374440725372,
  "specific_force_residual_std": [
   0.5955335651921911,
   0.46393635506100034,
   1.2992018919383437
  ],
  "specific_force_std": [
   1.512876373933108,
   1.7069817704121582,
   8.349991702257517
  ],
  "angular_acc_residual_std": [
   8.592744029187417,
   12.14872641569372,
   1.7676561304631182
  ],
  "angular_acc_std": [
   12.118106769074195,
   15.136179509812948,
   1.758183150259432
  ],
  "accelerometer_residual_std": [
   1.3410113422896475,
   1.6645848907285987,
   2.548708855238443
  ],
  "n_train_samples": 184639,
  "n_train_segments": 48,
  "arm_check": {
   "B_note": "thrust/torque consistency: rotor arm implied by B and bT with the Readme J, m",
   "ly_m": 0.05987098895526894,
   "lx_m": 0.045969352330984495
  },
  "model_fit": {
   "train": {
    "onestep": {
     "v_ms": 0.016805138274688727,
     "tilt_deg": 0.13547937459992213,
     "w_rads": 0.09691402416618591,
     "am_ms2": 0.7496330976420466
    },
    "n_windows": 300,
    "open_loop_0.1s": {
     "windows_not_tumbled": 1.0,
     "v_ms": 0.29228874170848634,
     "tilt_deg": 3.457214969061559,
     "w_rads": 0.8091667108970524
    },
    "open_loop_0.5s": {
     "windows_not_tumbled": 1.0,
     "v_ms": 3.86491784573447,
     "tilt_deg": 55.66845406881355,
     "w_rads": 4.430059665609612
    },
    "open_loop_1s": {
     "windows_not_tumbled": 1.0,
     "v_ms": 5.117302715949964,
     "tilt_deg": 85.22205520351704,
     "w_rads": 10.906369276480506
    },
    "open_loop_10s": {
     "windows_not_tumbled": 0.7933333333333333,
     "v_ms": 15.510017463078073,
     "tilt_deg": 98.14468881688099,
     "w_rads": 31.318120513676277
    }
   },
   "val": {
    "onestep": {
     "v_ms": 0.018284500398996914,
     "tilt_deg": 0.16256639091133349,
     "w_rads": 0.10253458515742618,
     "am_ms2": 0.6853063130661066
    },
    "n_windows": 98,
    "open_loop_0.1s": {
     "windows_not_tumbled": 1.0,
     "v_ms": 0.3513197587665156,
     "tilt_deg": 4.604849024551169,
     "w_rads": 1.0843157488504467
    },
    "open_loop_0.5s": {
     "windows_not_tumbled": 1.0,
     "v_ms": 4.202671597280901,
     "tilt_deg": 66.59783287807556,
     "w_rads": 4.50528582164476
    },
    "open_loop_1s": {
     "windows_not_tumbled": 1.0,
     "v_ms": 5.264852072584691,
     "tilt_deg": 88.45399711657068,
     "w_rads": 10.046625337729582
    },
    "open_loop_10s": {
     "windows_not_tumbled": 0.8571428571428571,
     "v_ms": 15.987039006226356,
     "tilt_deg": 100.59854732635308,
     "w_rads": 31.984527789914967
    }
   }
  }
 }
}
```

## Licence and redistribution

**No licence text was found** on the project page, the Readme, the download host or the code archive (checked 2026-11-02). The only stated term is **citation only**: 'If you use this work in your research, please cite the following paper'. arXiv's non-exclusive licence covers the paper, not the data. The author states he checked the licence personally and approved adding the data; that check is not recorded on the source. **Redistribution is not granted**, so all rights must be assumed reserved. The track is **fetch-only**: this package ships no raw data and no episodes, and `fetch` downloads from the original host with SHA-256 checks. Before a public release: ask the UZH RPG providers for written terms, keep `data_real/` and `logs/hkkl_train/*/data.npz` out of it, and decide about the derived items that do ship: `docs/real/neurobem_episodes.png` (plots of the data) and the identified constants, alignment and split lists in `resources/real_neurobem.json`.

## Data

**95 flights** (the project page says 96; `Flights.txt`, `raw_data` and `processed_data` all hold 95), 247 segments, 4496 s. u and y come from the raw Betaflight logs (`real/blackbox.py`, a new pure-Python decoder), labels from the processed CSVs. Every flight has its own clock fit (offset + 2800-4100 ppm drift; gyro residual <= 0.018 rad/s). Full evidence: `docs/cards/neurobem_datacheck.md`. Checks and plots: `docs/real/neurobem_checks.json`, `neurobem_episodes.png`.

## Splits (by flight, fixed in resources/real_neurobem.json; never within a flight)

| split | flights | 10-s windows | manoeuvre types |
|---|---|---|---|
| train | 48 | 567 | circle, wobbly circle, 3D circle, battery-test circle, lemniscate, linear oscillation, random points, ellipse |
| val | 12 | 98 | 3D circle, battery-test circle, lemniscate, linear oscillation, random points |
| test_id | 16 | 106 | same types as train, held-out flights |
| test_ood | 19 | 110 | CPC (9), satellite (8), vertical oscillation (2): held out entirely |

Rule, fixed before any estimator ran: the three held-out types go to test_ood; within each other type, flights in time order, index i % 5 == 2 -> test_id, i % 5 == 4 -> val, else train. Aggressive flight is on both sides: train lemniscates and linear oscillations reach 15.9-17.7 m/s and 46-49 m/s^2, the held-out satellites and CPC 12-15 m/s and 43-46 m/s^2. The providers' `testset.txt` is not used: it holds out segments of flights that also train. Flight 2021-02-18-16-43-54 is 164 Hz in the CSV and is interpolated; flight 2021-02-23-19-15-11 notes 'battery moved' (centre of mass shift) and is kept.

## Smoke runs (1 seed, short training; NOT benchmark results)

| method | headline | test_id | test_ood |
|---|---|---|---|
| ekf | 0.3548 | 0.3269 | 0.3826 |
| ekf_gated | 0.3552 | 0.3274 | 0.3831 |
| gru | 0.3662 | 0.1989 | 0.5334 |
| hyperkkl | 0.4024 | 0.2167 | 0.588 |
| imu_static | 1.22 | 1.084 | 1.356 |
| prior_mean | 1.117 | 0.9993 | 1.235 |
| ude_ekf_gated | 0.2157 | 0.1932 | 0.2382 |

### In physical units (mean over episodes of the settled RMSE; test splits)

| method | split | hidden NRMSE | v_x [m/s] | v_y [m/s] | v_z [m/s] | |v| [m/s] | roll [deg] | pitch [deg] | tilt [deg] | diverged |
|---|---|---|---|---|---|---|---|---|---|---|
| prior_mean | test_id | 0.999 | 3.56 | 3.89 | 2.07 | 5.91 | 32.5 | 28.6 | 42.0 | 0.000 |
| prior_mean | test_ood | 1.235 | 4.86 | 4.91 | 2.37 | 7.50 | 49.4 | 33.3 | 55.2 | 0.000 |
| imu_static | test_id | 1.084 | 2.49 | 3.41 | 2.09 | 4.87 | 49.7 | 33.3 | 54.1 | 0.000 |
| imu_static | test_ood | 1.356 | 3.04 | 3.92 | 2.51 | 5.70 | 66.1 | 42.2 | 68.0 | 0.000 |
| ekf | test_id | 0.327 | 1.10 | 0.94 | 1.42 | 2.13 | 6.3 | 4.5 | 5.9 | 0.000 |
| ekf | test_ood | 0.383 | 1.28 | 1.16 | 1.64 | 2.46 | 9.5 | 5.3 | 7.6 | 0.000 |
| ekf_gated | test_id | 0.327 | 1.11 | 0.94 | 1.42 | 2.13 | 6.3 | 4.5 | 5.9 | 0.000 |
| ekf_gated | test_ood | 0.383 | 1.28 | 1.16 | 1.64 | 2.47 | 9.6 | 5.3 | 7.6 | 0.000 |
| ude_ekf_gated | test_id | 0.193 | 0.61 | 0.64 | 0.78 | 1.22 | 4.9 | 2.7 | 3.9 | 0.000 |
| ude_ekf_gated | test_ood | 0.238 | 0.77 | 0.94 | 0.85 | 1.51 | 7.6 | 3.9 | 5.7 | 0.000 |
| gru | test_id | 0.199 | 0.60 | 0.66 | 0.60 | 1.10 | 7.8 | 4.8 | 6.8 | 0.000 |
| gru | test_ood | 0.533 | 1.11 | 1.33 | 1.25 | 2.18 | 27.5 | 15.2 | 23.3 | 0.000 |
| hyperkkl | test_id | 0.217 | 0.67 | 0.75 | 0.60 | 1.19 | 8.9 | 5.8 | 7.9 | 0.000 |
| hyperkkl | test_ood | 0.588 | 1.20 | 1.44 | 1.42 | 2.40 | 30.7 | 18.5 | 27.6 | 0.000 |
