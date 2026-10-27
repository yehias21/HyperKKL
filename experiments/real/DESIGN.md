# Real-data experiment: battery SOC estimation on held-out drive cycles (pre-registration)

Written 2026-10-25 before any observer was trained or evaluated on these data. Before writing it
we had only (i) surveyed file durations, current/voltage/temperature ranges and (ii) identified
the ECM on the TRAINING cycles (the identification is part of the method, so it is fixed here).
Changes after this point are listed under "Deviations" in `report.md`, with reasons.

## Question

Two claims of the paper, tested on real hardware, with a test set of input sequences
(drive cycles) never used for training or model selection:

- **C3 (incomplete model)**: a learned KKL observer beats a model-based filter when the model is
  incomplete. Here the model is a real BMS engineer's model: a 2-RC equivalent circuit with
  constant parameters identified by least squares from the training cycles (it cannot be exact:
  real cells have hysteresis, SOC- and rate-dependent resistance and diffusion dynamics).
- **C1 (where the input enters)**: conditioning the decoder's weights on a fading-memory input
  context (lora / hyper) beats injecting the input into the latent dynamics (latent).

## Data (DATA.md)

Panasonic 18650PF, Kollmeyer, Mendeley DOI 10.17632/wykht8y7tg.1 (CC BY 4.0), one cell,
chamber at 25 degC (primary) and 0 degC (secondary). Every drive-cycle file is a discharge from
full charge to 2.5 V (25 degC) or 80 % DOD (0 degC).

- Resampled to dt = 1 s. Input u_k = mean current over [t_k, t_k + 1 s) (charge-preserving
  zero-order hold), discharge positive.
- Outputs y = (dT, Vm): dT = case temperature - (ambient + thermocouple rest offset; offset =
  median of the first 30 s of the training files minus the nominal ambient); Vm = the bin-mean
  terminal voltage minus 3.7 V behind the plant's first-order acquisition filter (tau_f = 2 s,
  exact ZOH discretisation, applied in software). This is the quantity our battery plant
  (`systems_extra.LiIonCell`) measures, so every observer uses the unchanged y = x[:, sensors]
  contract.
- Target: SOC = 1 + Ah_lab / Q, the lab's coulomb counter (reset at the start of each file, which
  starts at full charge) with Q = 2.995 Ah, the measured C/20 discharge capacity at 25 degC.
  Limitations: (a) this is the standard reference in the SOC literature but it is itself an
  estimate (current-sensor accuracy of the lab tester, capacity measured once; the C/20 test is
  dated 2017-05-08, after the 25 degC drive cycles of March, so the cell had aged by up to ~40
  cycles; the nominal 2.9 Ah would shift the reference by up to 2.8 % SOC at the end of a
  discharge); (b) "full" = end of the CC-CV charge, by definition SOC = 1.
- States x = (SOC, V1, V2, dT, Vm), as the simulated battery plant. Labelled on real data: SOC
  (reference), dT, Vm (measured). V1, V2 have no labels.

## Split (by drive cycle, 25 degC)

| role | recordings | use |
|---|---|---|
| train | Cycle 1, Cycle 2, Cycle 3, UDDS, LA92 | ECM identification; phase-2 training windows |
| validation | Cycle 4 | learned-model selection (best step), lora-vs-hyper choice, filter Q/R tuning, nothing else |
| test | US06 (aggressive), HWFET (HWFTa + HWFTb, two runs of the highway cycle, pooled), NN | reported once |

Disclosed overlap of SHAPES: per the dataset readme, Cycles 1-4 are random mixes of US06, HWFET,
UDDS, LA92 and NN segments, and NN combines portions of US06 and LA92. The test cycles are
therefore new SEQUENCES (event order, dwell, pure repetition of one aggressive / highway profile
from full to empty; the paper's `sequence_ood` notion) on real hardware, not wholly new waveform
shapes. No test sample is in training (checked: disjoint files and disjoint absolute time
intervals, `realdata.check_split`).

Secondary analyses (reported separately, no decision rule):

- **Temperature shift**: the same models (trained at 25 degC, ECM identified at 25 degC with
  Arrhenius R(T), E_a/R = 3000 K) evaluated on every 0 degC drive cycle (US06, HWFET, NN, and the
  training-family cycles UDDS, LA92, Cycle 1-4, which there differ only in temperature).
- **Strict shape split** (only if time permits): train on UDDS and LA92 only.

## Episodes

Windows of T = 30 min (1800 steps), the battery plant's horizon. Training: stride 2 min over the
training recordings. Validation and test: stride 5 min (overlapping; the statistics below treat
them as dependent). Each window starts at its own SOC (range 1.0 to ~0.1 over a recording).

Wrong initial SOC: every observer gets the same prior x0_hat = (SOC0 + e, 0, 0, dT_0, Vm_0),
e in {+0.2, -0.2}, each sign kept when SOC0 + e lies in [0, 1] (so every episode starts exactly
20 % SOC off). Learned observers start from z0 = T0(x0_hat), filters from x0_hat.

## Observers (same information for all: u, y, the prior)

Learned KKL (default recipe of `configs/base.yaml` unless stated): nz = 7 (nx + 2, as the battery
plant), Butterworth A with alpha = 1.3 /min (the largest value <= the battery plant's 1.5 /min that
passes the design gate max|lambda| dt <= 0.1 at dt = 1 s), time unit 1 min, sfm context,
budget 140k, 3000 phase-2 steps, seeds 0, 1, 2. Variants: autonomous, latent, concat, lora, hyper.

- Phase 1 (base maps T0, T0+) on the IDENTIFIED model: the standard phase 1 (labels from
  zero-input episodes + PDE residual on Sobol collocation states), with simulated forced
  episodes = the training windows' measured currents replayed on the identified model from
  perturbed initial states (so the base maps cover the states real currents reach).
- Phase 2 on REAL training windows, `train.phase2` unchanged, with the loss restricted to the
  labelled states (SOC, dT, Vm): `realdata.masked_scales` sets sigma = inf for V1, V2 (their error
  and gradient are exactly zero) and rescales the labelled sigmas so the loss is the masked mean.
  The latent variant's rollout start T0(x_k0) needs V1, V2 at k0: we use the identified model's
  replay of the measured current (placeholders; never a loss target).
- Model selection: phase 2's own best-validation checkpoint (Cycle 4 windows, masked loss).
- **WS-best**: per seed, whichever of lora / hyper has the lower validation SOC RMSE (settled
  window, wrong-prior protocol above). Chosen on validation only.

Model-based (the standard BMS estimators) on the identified ECM (`observers.EKF`, `observers.UKF`,
unchanged): P0 = diag(0.2, std V1, std V2, 0.5 K, 10 mV)^2. Q, R chosen on the validation cycle
by settled SOC RMSE (oracle-tuned like `ekf_oracle`; separately for EKF and UKF) over the grid
q_SOC in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} /sqrt(min), process-noise rate of the other states
c * std_j with c in {0.01, 0.1, 1}, r_Vm in {3, 10, 30, 100} mV; r_dT = 0.3 K.

Reference: coulomb counting from the wrong prior (SOC0 + e - integral u dt / (3600 Q)); its error
stays at 20 % by construction (with the right prior it is the reference itself).

## Metrics (SOC in %)

Per episode: settled RMSE (t >= 15 min; settle_frac 0.5 as in the paper; PRIMARY), whole-window
RMSE, settled max |error|, convergence time (first time after which |error| <= 5 % SOC to the end
of the window; censored at 30 min). Per test cycle: means over its episodes. Also whole-recording
runs (from full charge with prior SOC 0.8), RMSE over t >= 15 min. Learned methods: mean and
seed spread over 3 seeds.

Uncertainty for a paired comparison A vs B on one test cycle: d_w = mean over the window's
episodes (both prior signs) and over seeds of RMSE_A - RMSE_B; 95 % CI of mean_w d_w by a moving
block bootstrap over window start times (block = 6 windows = 30 min, 10 000 resamples). Seed
consistency: the per-seed means of d.

## Decision rules (fixed now)

- **H1 (C3 on real data)**: WS-best beats the tuned EKF if, on >= 2 of the 3 test cycles (US06,
  HWFET, NN), the 95 % CI of the settled-RMSE difference (WS-best - EKF) lies below 0 AND all 3
  seeds have a negative mean difference. If the same holds with the opposite sign, the EKF wins;
  otherwise inconclusive.
- **H2 (C1 on real data)**: WS-best beats latent, with the same rule (WS-best - latent).
- Everything else (UKF, concat, autonomous, whole-window RMSE, max error, convergence time,
  whole-recording runs, 0 degC, strict split) is secondary and descriptive.

Whatever the outcome, it is reported.

## Compute

GPUs 5, 6, 7 only (user constraint, see STATUS.md), one seed per GPU, < 10 GB each; filters on CPU.
Code: `hyperkkl/realdata.py` (data, ECM, episodes), `hyperkkl/realdata_run.py` (pipeline);
no module hashed by `cache.py` is edited, and the shared cache is not used (everything is
written under `experiments/real/`).

## Addendum A (2026-10-26 02:27 Dubai): data-driven baselines (written BEFORE fitting or evaluating them)

The main results (H1, H2) were already computed when this addendum was written; the addendum
adds baselines and one secondary comparison and changes nothing above.

Baselines (`hyperkkl/baselines_data.py`, unchanged code), fitted per seed (0, 1, 2) on exactly the
learned observers' real training data (the training recordings Cycle 1-3, UDDS, LA92; transitions
taken once from each recording, i.e. the union of the training windows without duplicates);
selection / early stopping / ridge choice on the validation recording (Cycle 4) only:

- `koopman_kf`, `koopman_kf_bilinear`: EDMD with inputs (5 states + 400 thin-plate RBFs) and a
  Kalman filter on the lifted state.
- `greybox_ekf`, `greybox_ukf`: the identified ECM + a residual MLP fitted to one-step transition
  residuals, inside the unchanged EKF / UKF.
- `nlox` (Woelk et al.), trained with the same masked loss as the learned KKL observers.

Unlabelled states: the state vector for every baseline is the same x as for the learned observers,
(SOC_ref, V1, V2, dT_meas, Vm_meas), where V1, V2 are the identified model's replay of the measured
current (no measurement of them exists). Consequence, stated: for the grey-box residual the V1, V2
targets are the model's own replay, so their residual is ~0 by construction and the MLP effectively
corrects only the SOC, dT and Vm equations; the Koopman model learns V1, V2 dynamics that are the
identified model's. NLOX's loss is masked to SOC, dT, Vm (sigma = inf for V1, V2), as phase 2.

Noise tuning on validation, same wrong-prior protocol and score (settled SOC RMSE) as the EKF:
grey-box EKF / UKF on the same 72-point (q_SOC, c, r_Vm) grid as the EKF (so they correspond to the
repository's `greybox_ekf_oracle`, there being no nominal sensor model on real data); Koopman KF:
Q = the regression-residual Q times s, s in {0.1, 1, 10, 100}, r_Vm in {3, 10, 30, 100} mV, r_dT
0.3 K. NLOX has no noise parameters (its own best-validation checkpoint).

Secondary comparison (pre-stated, same CI rule as H1/H2): **WS-best vs the best data-driven
baseline per test cycle** (US06, HWFET, NN at 25 degC), where "best" = the data-driven baseline
(koopman_kf, koopman_kf_bilinear, greybox_ekf, greybox_ukf, nlox) with the lowest seed-mean settled
RMSE ON THAT TEST CYCLE (an oracle choice that favours the baselines). WS-best is "better" on a
cycle if the 95 % block-bootstrap CI of (WS-best - baseline) lies below 0 and all 3 seeds' mean
differences are negative (seed s of WS-best paired with seed s of the baseline); verdict over >= 2
of 3 cycles as for H1. Also reported: the same comparison against the baseline chosen on validation,
the whole-recording runs and 0 degC.
