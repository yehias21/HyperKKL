# Real-data track `calce`: CALCE A123 LiFePO4 cell: SOC on held-out dynamic profiles, with a flat voltage plateau

**Source.** CALCE battery data: A123 APR18650M1A (LiFePO4) dynamic profiles (DST, US06, FUDS) and low-current OCV tests (CALCE battery group, University of Maryland (M. Pecht et al.)); https://calce.umd.edu/battery-data (files on web.calce.umd.edu). Landing page: https://calce.umd.edu/battery-data
**Licence.** checked by the author, 2026-11-02. The source states only 'open access to our experimental test data' and 'Use of this data for publication purposes should include references to the CALCE article(s)'; no licence file. **Redistribution:** not granted explicitly: this package does not redistribute the data; `fetch` downloads them from the source
**Cite.** Y. Xing, W. He, M. Pecht, K. L. Tsui, State of charge estimation of lithium-ion batteries using the open-circuit voltage at various ambient temperatures, Applied Energy 113 (2014) 106-115; F. Zheng, Y. Xing, J. Jiang, B. Sun, J. Kim, M. Pecht, Influence of different open circuit voltage tests on state of charge online estimation for lithium-ion batteries, Applied Energy 183 (2016) 513-525

**Get the data.** `hyperkkl-bench fetch calce` (from the original host; every file checked against its SHA-256), then `hyperkkl-bench prepare calce`. `--from-local DIR` copies a local copy instead; `--dry-run` only checks the URLs.

## Why it belongs in this benchmark

The cell is driven by measured, exogenous, time-varying current profiles (DST, US06, FUDS; -3.9 to +2.1 A) from a full CC-CV charge to 2.0 V, so the state moves only because of the input. SOC is hidden (measured: current, terminal voltage, case temperature) and has a reference: the lab's coulomb counter from the end of the charge, with the cell's own C/20 capacity. What is new against battery_pf / battery_lg: a LiFePO4 plateau (0.9 mV per 1 % SOC between 20 and 80 % SOC, ten times flatter than the LG cell) and a 52-mV charge/discharge hysteresis, so the voltage carries little SOC information; and training on ONE profile family (DST) with testing on two new ones (US06, FUDS), the benchmark's input-generalisation question in its hardest real form. Temperature splits at 0, 10, 40 and 50 degC move the plant.

## Task and score

From u (current, discharge positive) and y = (dT, Vm), prior SOC0 +- 0.2, estimate SOC. Score: settled SOC RMSE [% SOC] over t >= 15 min of 30-min windows; headline = mean of test_US06 / test_FUDS at 25 degC (both cells). Temperature splits reported separately (PITFALLS P17).

## Partial model (f_known)

2-RC ECM + thermal (plants/systems_real.Pan18650PF; the class name is historical) on cell A1-007's training DST at 25 degC: OCV = the C/20 DISCHARGE branch (every profile is a discharge from full; the averaged branch biased the EKF by ~10 points of SOC on val), ECM fitted on SOC >= 10 % (the end-of-discharge collapse is outside a 2-RC model). Both choices made on val by gated-EKF SOC RMSE (21.1 -> 12.3 -> 11.0 %). R0 = 157 mOhm is high for this cell and not explained. Cell A1-008's capacity is 2.3 % above the model's (a real partial-model mismatch).

## Frozen identification (training split only)

```json
{
 "Q_Ah": 1.063561667212343,
 "params_25C": {
  "Q": 1.063561667212343,
  "R0": 0.1574974902705519,
  "R1": 0.02242516404453669,
  "C1": 914.3676921867241,
  "R2": 0.00010000000000000026,
  "C2": 999999.9999999986,
  "C_th": 889.2603013541079,
  "hA": 1.2174850505880344,
  "Ea_R": 3000.0,
  "dUdT": -0.0001,
  "T_amb": 298.15,
  "tau_f": 2.0,
  "ow0": 0.01,
  "ow1": 0.02,
  "o0": 2.93215382810015,
  "o1": 4.34747444632358,
  "o2": -29.968982057874893,
  "o3": 117.56053665489588,
  "o4": -259.2609230852193,
  "o5": 316.37756477996675,
  "o6": -198.22666077549673,
  "o7": 49.521820864778405,
  "o8": -0.813572416675526,
  "o9": 0.20096212133685515
 },
 "thermocouple_offset_K": 0.0,
 "arrhenius_condition": true,
 "ecm_fit": {
  "R0": 0.1574974902705519,
  "R1": 0.02242516404453669,
  "C1": 914.3676921867241,
  "R2": 0.00010000000000000026,
  "C2": 999999.9999999986,
  "tau1_s": 20.504845494311716,
  "tau2_s": 100.00000000000013,
  "v_rmse_mV": 7.116309537986659,
  "v_max_mV": 69.87524719047977,
  "nfev": 17,
  "active_bounds": [
   0,
   0,
   0,
   -1,
   -1
  ]
 },
 "thermal_fit": {
  "C_th": 889.2603013541079,
  "hA": 1.2174850505880344,
  "tau_th_s": 730.4075733205949,
  "dT_rmse_K": 0.15547818568714658
 },
 "ocv_fit_rms_V": 0.017136545562237898,
 "x_std": [
  0.22721527027332322,
  0.01137712552264587,
  1.9492941790410143e-05,
  0.2081149229945073,
  0.1541038827828376
 ],
 "x_mean": [
  0.5159694435546038,
  0.011337889779678435,
  5.045695374757311e-05,
  0.10116485840150874,
  -0.520465405641053
 ],
 "y_std": [
  0.2081149229945073,
  0.1541038827828376
 ],
 "filter_tuning_note": "battery_pf's validation-tuned EKF noise reused a priori",
 "open_loop_model_fit_train_val": {
  "train/A1-007_DST_20C": {
   "vm_rmse_mV": 72.35005978813035,
   "soc_end_err_pct": 3.2002178684820137e-12
  },
  "train/A1-007_DST_25C": {
   "vm_rmse_mV": 58.08361163518662,
   "soc_end_err_pct": 3.4017927363905187e-12
  },
  "train/A1-007_DST_30C": {
   "vm_rmse_mV": 42.32399380083242,
   "soc_end_err_pct": 2.8477220581635265e-12
  },
  "train/A1-008_DST_20C": {
   "vm_rmse_mV": 45.37795778943138,
   "soc_end_err_pct": -2.198531445220727
  },
  "train/A1-008_DST_30C": {
   "vm_rmse_mV": 18.02231550171307,
   "soc_end_err_pct": -2.2320074154361347
  },
  "val/A1-008_DST_25C": {
   "vm_rmse_mV": 40.89868517003863,
   "soc_end_err_pct": -2.1959977629401486
  }
 },
 "identification_note": "LFP hysteresis (charge - discharge C/20 voltage, mid SOC) 51.7 mV; OCV = DISCHARGE branch, ECM on SOC >= 0.1 (chosen on val)"
}
```

## Licence

Licence checked by the author, 2026-11-02. Source page: 'open access to our experimental test data'; 'Use of this data for publication purposes should include references to the CALCE article(s)'. No licence file or redistribution grant: fetch-only.

## Splits (fixed before any estimator ran)

| split | recordings | windows |
|---|---|---|
| train | DST: A1-007 at 20, 25, 30 degC; A1-008 at 20, 30 degC | 238 |
| val | DST: A1-008 at 25 degC | 32 |
| test_US06, test_FUDS | both cells, 25 degC | 62, 66 |
| test{0,10,40,50}_{US06,FUDS} | both cells | 54-70 each |

Files: one workbook per cell and temperature holds DST (step 8), US06 (step 16), FUDS (step 24), confirmed by the logged rest intervals 361 / 601 / 1373 s. -10 degC not used (A1-008 truncated); the 20 degC 'newprofile' test not used.

## Smoke runs (1 seed, short training; NOT benchmark results)

| method | headline | test_US06 | test_FUDS | test0_US06 | test0_FUDS | test10_US06 | test10_FUDS | test40_US06 | test40_FUDS | test50_US06 | test50_FUDS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coulomb | 20.03 | 20.03 | 20.04 | 20.04 | 20.04 | 20.04 | 20.04 | 20.03 | 20.03 | 20.03 | 20.03 |
| ekf | 7.925 | 8.401 | 7.45 | 29.57 | 26.98 | 19.67 | 18.67 | 15.71 | 14.45 | 24.33 | 23.7 |
| ekf_gated | 7.925 | 8.401 | 7.45 | 28.45 | 24.17 | 19.67 | 18.67 | 15.71 | 14.48 | 24.4 | 22.92 |
| gru | 5.075 | 3.616 | 6.535 | 26 | 20.66 | 14.16 | 8.761 | 15.21 | 16.37 | 18.99 | 15.6 |
| hyperkkl | 15.42 | 5.626 | 12.26 | 18.58 | 27.94 | 19.99 | 19.5 | 16.51 | 16.68 | 24.06 | 18.41 | 23.47 |
| prior_mean | 19.1 | 19.02 | 19.18 | 16.15 | 17.1 | 17.5 | 17.18 | 19.67 | 19.86 | 19.02 | 20.57 |
| ude_ekf_gated | 9.29 | 7.617 | 10.96 | 21.02 | 20 | 11.33 | 9.733 | 10.61 | 12.61 | 12.05 | 11.07 |
