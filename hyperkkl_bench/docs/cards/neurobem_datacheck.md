# NeuroBEM sensor check (step 1, stop gate)

Date: 2026-11-02, Dubai. Host: https://download.ifi.uzh.ch/rpg/NeuroBEM/ (Readme.md, Flights.txt, testset.txt,
processed_data.zip, raw_data.zip, code.tar.xz inspected; pdf.zip and predictions.tar.xz not downloaded).

## Verdict: PASS. Real measured channels exist, separate from the mocap state.

No sensor is simulated. The processed CSVs alone would NOT have passed: they hold a fused / smoothed state. The real
sensors are in the raw Betaflight logs, which the provider's Readme calls "only provided for completeness".

| channel | where | rate | what it is |
|---|---|---|---|
| `gyroADC[0..2]` | `raw_data/*.BFL` | ~1051 Hz logged | onboard gyro after the firmware filters (dynamic LPF 200-500 Hz, 250 Hz LPF, RPM notches), deg/s |
| `accSmooth[0..2]` | `raw_data/*.BFL` | ~1051 Hz logged | onboard accelerometer (`acc_1G` 4096) after the firmware LPF (`acc_lpf_hz` 1000 = 10.00 Hz) |
| `debug[0..3]` (`debug_mode` 45, RPM telemetry) | `raw_data/*.BFL` | ~1051 Hz logged | bidirectional DShot rotor eRPM / 100: measured motor speed, not the command |
| `motor[0..3]` | `raw_data/*.BFL` | ~1051 Hz logged | commanded motor output (not used) |
| `BaroAlt` | `raw_data/*.BFL` | ~13 Hz | barometer. NOT used: median correlation with mocap height 0.34, difference std 1.3 m (airspeed and propwash) |
| position, quaternion | `processed_data/*.csv` (from `raw_data/*.bag`) | 400 Hz | Vicon: GROUND TRUTH only |
| velocity, acceleration | `processed_data/*.csv` | 400 Hz | derived from Vicon offline (non-causal smoothing): label only |
| `ang vel`, `mot` | `processed_data/*.csv` | 400 Hz | NOT independent truth: zero-phase Savitzky-Golay smoothing of the logged gyro and rotor telemetry (`MergeAndPreprocessData.m`, `log_smoother.m`). `mot` is also rescaled per flight by hover offsets. They use future samples, so neither is given to an estimator |

## How this was checked

- `.BFL` is a compressed binary format. `real/blackbox.py` is a new pure-Python decoder (data version 2), written from
  the published format. All 95 logs decode with 0 dropped frames (168,035 I + 5,543,634 P frames).
- **Decoder validation.** An independent third-party decoder could not be run in this environment (running externally
  cloned code was refused by the session's safety policy, and it was not pursued). The check used instead is stronger in one
  respect: the providers' CSVs were made with the official `blackbox_decode`, so the decoded gyro, smoothed the way their
  scripts smooth it, must reproduce their `ang vel`. After the clock fit below it does on all 95 flights: residual RMS
  median 0.0076, max 0.018 rad/s, against a signal std of about 1 rad/s. Rotor speed correlates 0.998 with `mot`. This
  validates gyro and rotor telemetry directly. `accSmooth` has no such direct reference; it decodes through the same
  frames, and its physics is checked in the track's data checks.
- **Clock.** The log clock and the merged clock differ by an offset and a drift of 2786-4060 ppm (median 3774 ppm). A
  single offset fails: 0.3 s of drift over a 2-min flight, and periodic flights give false correlation peaks. Each flight
  is fitted by a global (offset, slope) search over the whole flight, then refined. The providers' own scripts do the
  equivalent (`correctClockDrift`).
- **Quaternion.** Order (x, y, z, w), Hamilton, body to world: d(pos)/dt = R(q) v_body has MSE 2.4e-5 (m/s)^2, against 11.4
  with the conjugate. The Readme does not state the convention; this is measured.
- **Frames and accelerations.** `vel` is body frame, front-left-up. `acc` is the specific force in the body frame:
  a = dv/dt + w x v + 9.81 g_b holds to MSE 0.15 (m/s^2)^2, against about 7-10 without gravity.

## Discrepancies with the source's description

- 95 flights in `Flights.txt`, `raw_data` and `processed_data`, not the 96 quoted on the project page. Duration matches:
  4496 s (1 h 15 min), 1.80 M samples, 247 segments.
- Flight 2021-02-18-16-43-54 is sampled at 164 Hz in the CSV, not 400 Hz. It is resampled like the rest and flagged.
- The Readme says body rates are "filtered using both onboard and Vicon measurements". In the code the published `ang vel`
  is the smoothed onboard gyro only (`merged(:,2:7) = data(:,2:7)`).
- `processed_data.zip` contains one stray vim `.swp` file (ignored, never opened).
- The published `testset.txt` lists 13 segments of flights that also have training segments, so it splits WITHIN flights.
  It is not used (the track splits by flight).

## What the sensors can and cannot see

An IMU and rotor speeds give no absolute position or heading: **yaw and position are unobservable** and are not in the task.
A pure thrust model makes the accelerometer read only thrust, so tilt and horizontal velocity would be unobservable too.
They become observable through rotor drag, which couples body velocity into the accelerometer (Leishman, Macdonald, Beard,
McLain, IEEE CSM 2014). The data show it: specific force correlates -0.93 / -0.96 with body velocity on x / y. The onboard
accelerometer is much noisier than the mocap-derived acceleration, so this is a hard real estimation problem, not a
filtered copy of the truth.
