# Real-data track `battery_pf`: Panasonic 18650PF Li-ion cell: SOC on held-out drive cycles (REQUIRED real track)

**Source.** Panasonic 18650PF Li-ion Battery Data (Phillip Kollmeyer (University of Wisconsin-Madison)); Mendeley Data, version 1 (2018-06-21). DOI 10.17632/wykht8y7tg.1. Landing page: https://data.mendeley.com/datasets/wykht8y7tg/1
**Licence.** CC BY 4.0. **Redistribution:** allowed by CC BY 4.0 with attribution; this package nevertheless does NOT redistribute the raw files: they are fetched from Mendeley Data and checked against Mendeley's SHA-256
**Cite.** P. Kollmeyer, Panasonic 18650PF Li-ion Battery Data, Mendeley Data, v1, 2018, doi:10.17632/wykht8y7tg.1

**Get the data.** `hyperkkl-bench fetch battery_pf` (from the original host; every file checked against its SHA-256), then `hyperkkl-bench prepare battery_pf`. `--from-local DIR` copies a local copy instead; `--dry-run` only checks the URLs.

## Why it belongs in this benchmark

The cell is driven by a measured, exogenous, strongly time-varying input: the drive-cycle current (a scaled vehicle power profile, -20 to +20 A). Its state (state of charge, two RC polarisation voltages, temperature, filtered voltage) evolves only because of that input, so the system is genuinely non-autonomous: with u = 0 it rests. SOC is HIDDEN (no sensor; the measured outputs are the case temperature and the terminal voltage), and it has a reference (the lab's coulomb counter from a known full charge). Input conditions differ between splits: training is Cycles 1-3, UDDS, LA92; the test cycles US06 (aggressive), HWFET (highway) and NN are new event SEQUENCES (the paper's sequence_ood notion; Cycles 1-4 contain segments of every profile, disclosed), and the 0 degC recordings add a temperature shift.

## Task and score

From u (current) and y = (dT, Vm), starting from a prior 20 % SOC off, estimate SOC. Score: settled SOC RMSE [% SOC] (t >= 15 min of 30-min windows); headline = mean of US06 / HWFET / NN at 25 degC. Secondary: the 9 cycles at 0 degC (models fitted at 25 degC).

## Partial model (f_known)

2-RC Thevenin equivalent circuit + lumped thermal model + first-order voltage acquisition filter (plants/systems_real.Pan18650PF; the standard BMS model, e.g. Plett, Battery Management Systems Vol. 1, Artech House 2015, ch. 2). OCV(SOC) from the C/20 test (mean of discharge and charge branches, degree-7 polynomial + two exponential edge terms); R0, R1, C1, R2, C2 by least squares on the bin-mean terminal voltage of the TRAINING cycles; C_th, hA by least squares on dT; Arrhenius R(T) with E_a/R = 3000 K (literature value, not identifiable at one temperature). No hysteresis, no SOC- or rate-dependent resistance: a partial model. V1, V2 have no labels; they carry the model's replay as placeholders (x_fill), never a loss target or a score.

## Frozen identification (training split only)

```json
{
 "Q_Ah": 2.99491,
 "thermocouple_offset_K": 0.6307000000000009,
 "params_25C": {
  "Q": 2.99491,
  "R0": 0.03294095796692829,
  "R1": 0.0229216257668172,
  "C1": 1101.302691027656,
  "R2": 0.10503677081447058,
  "C2": 11101.297402847651,
  "C_th": 102.88137941601232,
  "hA": 0.22595038859702596,
  "Ea_R": 3000.0,
  "dUdT": -0.0001,
  "T_amb": 298.78069999999997,
  "tau_f": 2.0,
  "ow0": 0.02,
  "ow1": 0.1,
  "o0": 3.611427913967818,
  "o1": -8.2829148258448,
  "o2": 95.67415237130271,
  "o3": -470.6991162814208,
  "o4": 1231.5133303449925,
  "o5": -1780.5770931930115,
  "o6": 1352.6294110469103,
  "o7": -429.6373916982886,
  "o8": -0.8623639967762259,
  "o9": 10.008588335977004
 },
 "ecm_fit": {
  "R0": 0.03294095796692829,
  "R1": 0.0229216257668172,
  "C1": 1101.302691027656,
  "R2": 0.10503677081447058,
  "C2": 11101.297402847651,
  "tau1_s": 25.24364813972464,
  "tau2_s": 1166.0444310461862,
  "v_rmse_mV": 34.121369452734164,
  "v_max_mV": 514.7826405866862,
  "nfev": 9
 },
 "thermal_fit": {
  "C_th": 102.88137941601232,
  "hA": 0.22595038859702596,
  "tau_th_s": 455.3272957609177,
  "dT_rmse_K": 0.414589749408396
 },
 "ocv_fit_rms_V": 0.004484842458350129,
 "arrhenius_factor_0C": 2.5014387174692816,
 "x_std": [
  0.22533669759123362,
  0.017576006834873488,
  0.024882262430029552,
  0.5808985883211857,
  0.2240314338985075
 ],
 "x_mean": [
  0.5441909726214019,
  0.015172380865960058,
  0.0664358811498928,
  0.8607988750493644,
  -0.020166575349106815
 ],
 "y_std": [
  0.5808985883211857,
  0.2240314338985075
 ],
 "open_loop_model_fit": {
  "25/Cycle_1": {
   "vm_rmse_mV": 35.68569788495829,
   "dT_rmse_K": 0.7798864543451479
  },
  "25/Cycle_2": {
   "vm_rmse_mV": 36.73667042326419,
   "dT_rmse_K": 0.4641267793914809
  },
  "25/Cycle_3": {
   "vm_rmse_mV": 30.375488992380554,
   "dT_rmse_K": 0.37068947335415114
  },
  "25/Cycle_4": {
   "vm_rmse_mV": 72.40302197145083,
   "dT_rmse_K": 0.53750558333112
  },
  "25/US06": {
   "vm_rmse_mV": 69.74593226165791,
   "dT_rmse_K": 0.6517379982351011
  },
  "25/HWFTa": {
   "vm_rmse_mV": 56.333227766046456,
   "dT_rmse_K": 0.5652186919607632
  },
  "25/HWFTb": {
   "vm_rmse_mV": 57.60521995910904,
   "dT_rmse_K": 0.5700906576979964
  },
  "25/UDDS": {
   "vm_rmse_mV": 33.95499310959997,
   "dT_rmse_K": 0.17941430984958404
  },
  "25/LA92": {
   "vm_rmse_mV": 24.47519195123345,
   "dT_rmse_K": 0.2527881842585687
  },
  "25/NN": {
   "vm_rmse_mV": 23.17653099008128,
   "dT_rmse_K": 0.4690497778374577
  },
  "0/Cycle_1": {
   "vm_rmse_mV": 220.1676214250161,
   "dT_rmse_K": 0.9509940334815066
  },
  "0/Cycle_2": {
   "vm_rmse_mV": 220.76676632122738,
   "dT_rmse_K": 1.007015491801302
  },
  "0/LA92": {
   "vm_rmse_mV": 80.92902102200976,
   "dT_rmse_K": 0.38448878376760737
  },
  "0/NN": {
   "vm_rmse_mV": 94.17701901333064,
   "dT_rmse_K": 0.6019688886156187
  },
  "0/Cycle_3": {
   "vm_rmse_mV": 112.47370214477606,
   "dT_rmse_K": 1.2315691051238147
  },
  "0/Cycle_4": {
   "vm_rmse_mV": 117.58349144379477,
   "dT_rmse_K": 0.7376786522163067
  },
  "0/US06": {
   "vm_rmse_mV": 175.23056004539802,
   "dT_rmse_K": 1.3477403692278442
  },
  "0/HWFET": {
   "vm_rmse_mV": 121.13576987978769,
   "dT_rmse_K": 0.4331024079628834
  },
  "0/UDDS": {
   "vm_rmse_mV": 57.85507905206714,
   "dT_rmse_K": 0.2761641771836073
  }
 }
}
```

## Smoke runs (1 seed, short training; NOT benchmark results)

| method | headline | test_US06 | test_HWFET | test_NN | test0_Cycle_1 | test0_Cycle_2 | test0_LA92 | test0_NN | test0_Cycle_3 | test0_Cycle_4 | test0_US06 | test0_HWFET | test0_UDDS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coulomb | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| ekf | 3.996 | 8.851 | 1.191 | 1.947 | 9.667 | 9.373 | 4.132 | 7.762 | 8.48 | 6.734 | 22.21 | 9.785 | 3.609 |
| ekf_gated | 4 | 8.851 | 1.201 | 1.947 | 9.666 | 9.033 | 4.132 | 7.752 | 8.48 | 6.722 | 22.24 | 9.785 | 3.607 |
| gru | 1.176 | 1.915 | 1.084 | 0.5292 | 12.78 | 12.62 | 8.748 | 8.414 | 12.09 | 13.34 | 17.46 | 16.02 | 9.678 |
| hyperkkl | 1.157 | 1.984 | 0.9761 | 0.5121 | 12.62 | 12.31 | 8.933 | 9.102 | 11.67 | 12.9 | 15.55 | 15.9 | 9.656 |
| prior_mean | 19.07 | 19.46 | 19.76 | 18 | 22.78 | 19.28 | 26.42 | 28.01 | 11.83 | 17.61 | 16.64 | 14.68 | 15.77 |
| ude_ekf_gated | 1.895 | 1.913 | 1.518 | 2.254 | 11.81 | 11.64 | 6.75 | 6.559 | 9.274 | 10.46 | 22.21 | 11.69 | 9.055 |
