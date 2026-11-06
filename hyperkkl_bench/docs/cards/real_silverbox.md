# Real-data track `silverbox`: Silverbox electronic Duffing oscillator: velocity under amplitude extrapolation

**Source.** Silverbox (nonlinear benchmark) (T. Wigren, J. Schoukens); nonlinearbenchmark.org (benchmark page https://www.nonlinearbenchmark.org/benchmarks/silverbox); file hosted on Google Drive by the benchmark site. Landing page: https://www.nonlinearbenchmark.org/benchmarks/silverbox
**Licence.** not stated by the data provider (the ECC 2013 paper says the data are 'available for free download'). **Redistribution:** UNKNOWN: this package does not redistribute the data; `fetch` downloads them from the benchmark site's link.
**Cite.** T. Wigren, J. Schoukens, Three free data sets for development and benchmarking in nonlinear system identification, European Control Conference (ECC) 2013, pp. 2933-2938, doi:10.23919/ECC.2013.6669201

**Get the data.** `hyperkkl-bench fetch silverbox` (from the original host; every file checked against its SHA-256), then `hyperkkl-bench prepare silverbox`. `--from-local DIR` copies a local copy instead; `--dry-run` only checks the URLs.

## Why it belongs in this benchmark

An electronic mass-spring-damper with a cubic hardening spring, driven by a measured input voltage (V1) and observed through the displacement voltage (V2) only; the velocity is hidden. It is a forced nonlinear oscillator: without input it decays to rest. Input conditions differ by the benchmark's design: training on odd random-phase multisines; tests on a held-out multisine, on filtered Gaussian noise of growing amplitude inside the training amplitude range, and on its last part, BEYOND the training amplitude (extrapolation).

## Task and score

From u (V1) and y (V2), prior (y_0, 0), estimate the velocity of V2. Score: settled velocity NRMSE (second half of 2048-sample windows) against an offline zero-phase derivative (Butterworth 4 @ 250 Hz); headline = mean of the three test splits.

## Partial model (f_known)

Forced Duffing oscillator y'' = -d y' - k1 y - k3 y^3 + g u + b0 (the structure of Wigren & Schoukens 2013; time unit 10 ms; plants/systems_real.SilverboxDuffing). No published parameter values: equation-error least squares on zero-phase derivatives of the TRAINING samples, refined by output-error multiple shooting (200-sample segments). The circuit has further dynamics the second-order model misses (validation simulation NRMSE ~0.24 over 3.4-s windows): a partial model. HyperKKL latent design: n_z = 4, alpha = 0.2 per 10 ms (20 rad/s), the largest Butterworth bandwidth that passes the design gate max|lambda| dt <= 0.1 at 610 Hz; it is slower than the ~70 Hz resonance (stated). A causal finite difference of y (no model) scores ~0.34: the floor to beat.

## Frozen identification (training split only)

```json
{
 "params": {
  "d": 0.42603790655282636,
  "k1": 19.156918632387494,
  "k3": 81.72668511593957,
  "g": 19.23222938832401,
  "b0": -0.10244568675281619
 },
 "identification": {
  "q_v": 0.12417897439718875,
  "accel_residual_rel": 0.35428259993952194,
  "equation_error_init": {
   "d": 0.2536837433466176,
   "k1": 15.241728347416894,
   "k3": 39.099344039300966,
   "g": 10.237327902562466,
   "b0": -0.05049340342492431
  },
  "equation_error_accel_residual_rel": 0.10653275401604828,
  "output_error_segment_nrmse": 0.24055446038625738,
  "oe_segments": 552,
  "u_scale": 0.10140027374826022,
  "r_y": 7.090369927874958e-06,
  "v_std": 0.21303905583741445,
  "y_std": 0.054535261335838955,
  "y_mean": 0.0008002592028897731,
  "v_mean": -1.1819751507173648e-05,
  "label_noise_floor_nrmse": 0.08118127702003876,
  "model_fit": {
   "train": {
    "sim_nrmse_pos": 0.24159306231191358,
    "sim_nrmse_vel": 0.3159056464294987,
    "onestep_nrmse_vel": 0.1788101320756292
   },
   "val": {
    "sim_nrmse_pos": 0.2401573511138506,
    "sim_nrmse_vel": 0.3144102537695154,
    "onestep_nrmse_vel": 0.1764768083645094
   }
  }
 }
}
```

## Smoke runs (1 seed, short training; NOT benchmark results)

| method | headline | test_multisine | test_arrow | test_arrow_extrap |
|---|---|---|---|---|
| ekf | 0.2004 | 0.1832 | 0.1294 | 0.2885 |
| ekf_gated | 0.2005 | 0.1832 | 0.1294 | 0.2889 |
| finite_diff | 0.343 | 0.3097 | 0.2191 | 0.5002 |
| gru | 0.03217 | 0.01507 | 0.01101 | 0.07044 |
| hyperkkl | 0.06067 | 0.02621 | 0.02285 | 0.1329 |
| prior_mean | 1.08 | 0.9888 | 0.7063 | 1.546 |
| ude_ekf_gated | 0.1226 | 0.1125 | 0.08002 | 0.1752 |
