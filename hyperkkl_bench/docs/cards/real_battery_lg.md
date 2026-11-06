# Real-data track `battery_lg`: LG 18650HG2 Li-ion cell: SOC on held-out drive cycles and temperatures

**Source.** LG 18650HG2 Li-ion Battery Data (Phillip Kollmeyer, Mina Naguib, Michael Skells (McMaster University)); Mendeley Data, version 2. DOI 10.17632/b5mj79w5w9.2. Landing page: https://data.mendeley.com/datasets/b5mj79w5w9/2
**Licence.** CC BY 4.0. **Redistribution:** allowed by CC BY 4.0 with attribution; this package nevertheless does NOT redistribute the raw files: they are fetched from Mendeley Data and checked against Mendeley's SHA-256
**Cite.** P. Kollmeyer, M. Naguib, M. Skells, LG 18650HG2 Li-ion Battery Data, Mendeley Data, v2, 2020, doi:10.17632/b5mj79w5w9.2

**Get the data.** `hyperkkl-bench fetch battery_lg` (from the original host; every file checked against its SHA-256), then `hyperkkl-bench prepare battery_lg`. `--from-local DIR` copies a local copy instead; `--dry-run` only checks the URLs.

## Why it belongs in this benchmark

Same physics and task as battery_pf on a different cell (3 Ah LG HG2, NMC), a different lab (McMaster) and different drive cycles: the input is the measured cell current of vehicle drive cycles, SOC is hidden with a coulomb-counter reference. Input conditions differ between splits: training on UDDS, LA92 and the random mixes Mixed1-6; validation Mixed7; test US06 (aggressive) and Mixed8 at 25 degC, and US06 / HWFET / Mixed8 at 0 and 10 degC and US06 / HWFET at 40 degC (temperature shift in both directions; the 25 degC HWFET file is a rest and the 40 degC Mixed8 file is truncated, both excluded). It tests whether a result on battery_pf transfers to a second cell without any re-design (the latent design, windows and noise settings are battery_pf's, fixed a priori).

## Task and score

As battery_pf. Headline = mean of US06 / Mixed8 at 25 degC; secondary: 0, 10, 40 degC.

## Partial model (f_known)

The same 2-RC ECM structure, identified on THIS cell's training cycles + its own C/20 test (identify_lg(): battery_pf's procedure with physical bounds tau1 in [1, 100] s, tau2 in [100, 3600] s; unbounded, the fit put tau2 at 30 days, an integrator absorbing an OCV / capacity mismatch; frozen in resources/real_battery_lg.json). The EKF noise is battery_pf's validation-tuned setting reused a priori (no tuning on this cell).

## Frozen identification (training split only)

```json
{
 "Q_Ah": 2.7782,
 "thermocouple_offset_K": -1.2341699999999989,
 "params_25C": {
  "Q": 2.7782,
  "R0": 0.01906882955340247,
  "R1": 0.012433439900373707,
  "C1": 1177.1379725213321,
  "R2": 0.039343389133670935,
  "C2": 91502.03069107315,
  "C_th": 132.06924528106424,
  "hA": 0.47422292766118307,
  "Ea_R": 3000.0,
  "dUdT": -0.0001,
  "T_amb": 296.91582999999997,
  "tau_f": 2.0,
  "ow0": 0.02,
  "ow1": 0.02,
  "o0": 3.1352467642380937,
  "o1": 2.351076471970083,
  "o2": 0.24694698995004183,
  "o3": -29.500667833501037,
  "o4": 98.17993671675784,
  "o5": -140.5842043775464,
  "o6": 96.36347752609801,
  "o7": -26.094390699439902,
  "o8": -0.25294690498368444,
  "o9": 0.05178034724898833
 },
 "ecm_fit": {
  "R0": 0.01906882955340247,
  "R1": 0.012433439900373707,
  "C1": 1177.1379725213321,
  "R2": 0.039343389133670935,
  "C2": 91502.03069107315,
  "tau1_s": 14.63587423579174,
  "tau2_s": 3599.999999999992,
  "v_rmse_mV": 25.195712463007105,
  "v_max_mV": 182.81144588012532,
  "nfev": 10,
  "active_bounds": [
   0,
   0,
   0,
   0,
   1
  ]
 },
 "thermal_fit": {
  "C_th": 132.06924528106424,
  "hA": 0.47422292766118307,
  "tau_th_s": 278.49611981524316,
  "dT_rmse_K": 0.16985358449349292
 },
 "ocv_fit_rms_V": 0.004269076706023687,
 "x_std": [
  0.2351632368044536,
  0.01661132701440195,
  0.010135336375370904,
  0.3614967652473921,
  0.23123130462186026
 ],
 "x_mean": [
  0.5345148567488437,
  0.012985770543782665,
  0.02598825798403268,
  0.42287614333493645,
  0.01761829885547643
 ],
 "y_std": [
  0.3614967652473921,
  0.23123130462186026
 ],
 "filter_tuning_note": "battery_pf's validation-tuned EKF noise reused a priori (no tuning on this cell)",
 "open_loop_model_fit_train_val": {
  "25/UDDS": {
   "vm_rmse_mV": 23.723998170156186
  },
  "25/LA92": {
   "vm_rmse_mV": 23.919494010891103
  },
  "25/Mixed1": {
   "vm_rmse_mV": 26.207405516586793
  },
  "25/Mixed2": {
   "vm_rmse_mV": 24.585681137397415
  },
  "25/Mixed3": {
   "vm_rmse_mV": 26.119908546589727
  },
  "25/Mixed4": {
   "vm_rmse_mV": 23.9097388382784
  },
  "25/Mixed5": {
   "vm_rmse_mV": 27.175691598685788
  },
  "25/Mixed6": {
   "vm_rmse_mV": 26.896007447743003
  },
  "25/Mixed7": {
   "vm_rmse_mV": 27.790576521579325
  }
 }
}
```

## Smoke runs (1 seed, short training; NOT benchmark results)

| method | headline | test_US06 | test_Mixed8 | test0_US06 | test0_HWFET | test0_Mixed8 | test10_US06 | test10_HWFET | test10_Mixed8 | test40_US06 | test40_HWFET |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coulomb | 20.01 | 20 | 20.01 | 20.04 | 20 | 20 | 20.02 | 20 | 20 | 20 | 20 |
| ekf | 1.845 | 1.697 | 1.994 | 2.634 | 3.489 | 4.055 | 2.321 | 2.199 | 2.147 | 3.822 | 3.046 |
| ekf_gated | 1.845 | 1.697 | 1.994 | 2.634 | 3.489 | 4.055 | 2.321 | 2.199 | 2.147 | 3.822 | 3.046 |
| gru | 0.9105 | 1.104 | 0.717 | 14.81 | 13.7 | 10.24 | 8.622 | 6.438 | 5.172 | 2.823 | 2.385 |
| hyperkkl | 1.03 | 1.3 | 0.7597 | 11.82 | 13.98 | 9.952 | 7.98 | 6.596 | 5.176 | 2.366 | 1.914 |
| prior_mean | 22.72 | 24.1 | 21.34 | 22.74 | 18.35 | 22.5 | 22.07 | 19.32 | 21.04 | 23.26 | 20.6 |
| ude_ekf_gated | 1.891 | 2.522 | 1.259 | 6.55 | 7.531 | 3.809 | 1.491 | 5.033 | 1.76 | 3.418 | 4.787 |
