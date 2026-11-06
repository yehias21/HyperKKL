# Real-data track `emps`: EMPS electro-mechanical positioning system: velocity from position under force pulses

**Source.** Data Set and Reference Models of EMPS (Electro-Mechanical Positioning System) (A. Janot, M. Gautier, M. Brunot (ONERA / LS2N)); nonlinearbenchmark.org (benchmark page https://www.nonlinearbenchmark.org/benchmarks/emps); file hosted on Google Drive by the benchmark site. Landing page: https://www.nonlinearbenchmark.org/benchmarks/emps
**Licence.** not stated by the data provider (no licence on the benchmark page, in the zip or in the description). **Redistribution:** UNKNOWN: this package does not redistribute the data; `fetch` downloads them from the benchmark site's link. Ask the authors before rehosting.
**Cite.** A. Janot, M. Gautier, M. Brunot, Data Set and Reference Models of EMPS, Workshop on Nonlinear System Identification Benchmarks, Eindhoven, 2019

**Get the data.** `hyperkkl-bench fetch emps` (from the original host; every file checked against its SHA-256), then `hyperkkl-bench prepare emps`. `--from-local DIR` copies a local copy instead; `--dry-run` only checks the URLs.

## Why it belongs in this benchmark

A prismatic axis (DC motor + ball screw + load) driven by a measured force u = g_tau * vir [N] that varies continuously in time (bang-bang acceleration profiles). The plant has a pure integrator and nonlinear (Coulomb) friction, so position and velocity evolve only because of the input. Only the motor-encoder position is measured; the VELOCITY is hidden. Input conditions differ between splits: train / val are the bang-bang record; the test record adds force pulses (a disturbance input absent from training). The data were taken in closed loop (PD position control); the estimator treats the measured force as the exogenous input, as the providers' own models do. Caveat: the force is the PD controller output nu = kp kv (q_ref - q) - kv q_dot, so it carries velocity information itself; learned methods can exploit the control law (REVIEW_5 B21).

## Task and score

From u (force) and y = q (position), prior (q_0, 0), estimate the velocity. Label: the providers' offline velocity (zero-phase Butterworth 4 @ 100 Hz + central difference, the IDIM-LS 'bandpass filtering process'). HEADLINE split test_pulses_outage: the encoder is frozen (last value held) for four 100-ms bursts in the second half of every 2-s window, and the velocity is scored during the bursts only (velocity NRMSE; position NRMSE also reported). The plain split test_pulses (settled window) is kept as a sanity check only: with a 50000-count encoder a causal finite difference of y, no model, scores 0.0008 there, so it is a DEGENERATE estimation task (found in the first smoke runs, before any comparison; the outage split was added then).

## Partial model (f_known)

Rigid-body direct dynamic model of the providers (Janot, Gautier, Brunot 2019, eq. 3): M q'' = tau - Fv q' - Fc sign(q') - offset, sign smoothed by tanh(v / 1 mm/s) (plants/systems_real.EMPSAxis). M, Fv, Fc, offset by the providers' IDIM-LS on the TRAINING samples only; they agree with the published full-record estimates within 0.5 % (M), 0.5 % (Fv), 0.5 % (Fc), 3 % (offset). HyperKKL latent design: n_z = 4, alpha = 20 /s (max|lambda| dt = 0.05).

## Frozen identification (training split only)

```json
{
 "params": {
  "M": 95.07049640750232,
  "Fv": 204.50122482109964,
  "Fc": 20.299777612282895,
  "offset": -3.082205059863071,
  "v_eps": 0.001
 },
 "identification": {
  "q_v": 0.0007889373202981617,
  "force_residual_rel": 0.044144317241009964,
  "published_IDIM_LS_full_record": {
   "M": 95.1089,
   "Fv": 203.5034,
   "Fc": 20.3935,
   "offset": -3.1648
  },
  "u_scale": 152.0498356124539,
  "r_y": 2.2247482731453213e-08,
  "v_std": 0.08712020248426035,
  "y_std": 0.08296872508029252,
  "y_mean": 0.11862925024747474,
  "v_mean": 0.003695340885001371,
  "label_noise_floor_nrmse": 0.0002397542520871435,
  "model_fit": {
   "train": {
    "sim_nrmse_pos": 0.036003304434173246,
    "sim_nrmse_vel": 0.0456905965867821,
    "onestep_nrmse_vel": 0.0002708463366872962
   },
   "val": {
    "sim_nrmse_pos": 0.052517999147102354,
    "sim_nrmse_vel": 0.0478100907799594,
    "onestep_nrmse_vel": 0.00028921145443926404
   }
  }
 }
}
```

## Smoke runs (1 seed, short training; NOT benchmark results)

| method | headline | test_pulses_outage | test_pulses |
|---|---|---|---|
| ekf | 0.01883 | 0.01883 | 0.009371 |
| ekf_gated | 0.01883 | 0.01883 | 0.009371 |
| finite_diff | 0.9791 | 0.9791 | 0.0007567 |
| gru | 0.06856 | 0.06856 | 0.07414 |
| hyperkkl | 0.8908 | 0.8908 | 0.004887 |
| prior_mean | 0.9736 | 0.9736 | 0.9768 |
| ude_ekf_gated | 0.01239 | 0.01239 | 0.006578 |
