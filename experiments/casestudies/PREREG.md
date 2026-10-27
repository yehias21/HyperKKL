# Deployment case studies: pre-registration

Written 2026-10-27 13:15 Dubai (Asia/Dubai, UTC+4), BEFORE any observer was trained at full size or scored
(only tiny-step smoke runs, whose numbers are not used, and the plant / inclusion / information-budget
computations, which involve no observer). Later changes go into timestamped addenda at the end; the text
above them is never edited. Situations: `CASE_A.md` (quadrotor), `CASE_B.md` (exothermic CSTR).

## 1. Cells, methods, seeds, metric

* Cells: 2 cases x 5 knowledge levels (exact, mm10, mm30, partial, realistic; CASE_*.md tables).
  Tiers id, shape_ood, sequence_ood (64 episodes each, protocol v2 input box); robustness conditions on
  the id tier (poor_init, outliers, gps_loss / sensor_loss). Seeds 0-4 for every method.
* Methods (code/run_cs.py): **HyperKKL A3** (primary: SOAP + unfrozen base decoder at lr x0.1 +
  strided z-window context; beat_greybox recipe, unchanged), HyperKKL **A4+D3** (secondary; realistic
  level only), HyperKKL **A0** (old default, reference); **EKF, UKF** with the practitioner's model;
  **grey-box EKF / UKF / EKF-oracle** (model + one-step residual MLP, baselines_data); **UDE + EKF / UKF**
  (model + trajectory-trained residual, ~5k params); **NODE + EKF / UKF** (model-free, ~5k); **GRU**
  (~140k); **NLOX** (capacity-matched ~140k); **climatology** (training mean).
  The learned baselines get a 2-point lr grid {1e-3, 1e-2} (baselines2 used 4 points; reduced for
  compute), chosen on validation; HyperKKL arms get no grid (frozen recipes).
* Data budgets and sparse labels as in CASE_*.md. Case B: masked losses where a method can use them;
  one-step grey-box gets linearly interpolated assays (CASE_B.md).
* Primary metric: settled NRMSE (second half of each episode) on the UNMEASURED states
  (`nrmse_unmeasured`, mean of the per-state NRMSE), seed mean over 64 episodes. Primary tier for the
  verdicts: **sequence_ood**; all tiers reported.

## 2. Practitioner's requirements (pass / fail per method and level)

* **Case A accuracy:** settled RMSE vx <= 0.15 m/s, vz <= 0.10 m/s, pitch <= 0.035 rad (2 deg) on the
  id tier; 1.5x those on shape_ood / sequence_ood; **no divergence** in any tier or condition.
* **Case A deployment:** <= 150 us per 100 Hz estimator step on the Cortex-M7 (estimated from FLOPs and
  transcendental counts, code/latency_cs.py; not measured on hardware) and <= 512 KB float32 weights.
  CPU (1 thread, batch 1) latency reported as measured.
* **Case B accuracy:** settled RMSE C_A <= 0.05 mol/L on id; <= 0.075 mol/L on the OOD tiers; no divergence.
* **Case B deployment:** <= 100 ms per step on one CPU core, <= 50 MB.
* Robustness (both): after a poor initial guess, converge (paired forget time reported); outage
  recovery time reported (A: 5 s GNSS loss; B: 30 min T loss).

## 3. Predictions (scored mechanically in RESULTS.md)

**Bits rule** (experiments/information): HyperKKL CANNOT beat a comparator whose error eps is below
A, the unmeasured Bayesian PCRB over the 3/alpha window under the cell's true plant and actual sensor
model (code/infolib_cs.py: BUDGET_FIX method + per-output noise + multi-rate GNSS/baro). The A values
are computed by code/budget_cs.py before any observer run and listed in Addendum 1 (written before the
observer launch). Pre-registered form of the prediction: per (case, level, tier), **CANNOT if A > eps**,
eps = the best grey-box/UDE filter's seed-mean unmeasured NRMSE (oracle pick of greybox_ekf/ukf/
ekf_oracle, ude5k_ekf/ukf); a violation = CANNOT and HyperKKL A3 significantly better (Holm). The rule is
one-sided: "can" is not a prediction of a win.
My point forecasts (so the rule is tested where it bites): Case A, the window bound is small because
the IMU (gyro at 100 Hz; accelerometer carrying the drag velocity signal on the TRUE plant) informs the
velocities quickly -> expect "can" at every level; Case B, the 30-min window over temperature-only
measurements of a slow (~60 min) plant -> expect A close to or above the best filter's eps at the
exact / mm10 levels (CANNOT), "can" at partial / realistic where the filters are model-limited.

**Who wins (A3 vs best grey-box/UDE, sequence_ood, Holm over levels):**
* P1 (Case A): HyperKKL A3 **loses** at exact, mm10, mm30 (the model-based filters with an exact or
  near-exact rigid-body model and 100 Hz IMU are hard to beat with 34 min of data) and **does not win**
  at partial or realistic (the drag residual is a simple, well-identified one-step function: the
  grey-box / UDE recover it from dense mocap).
* P2 (Case B): A3 **loses** at exact / mm10 / mm30; at partial / realistic it is **tied or loses** to
  the best of grey-box/UDE (with ~270 assays it has too few labels for its decoder). UDE+EKF is the best
  grey-box/UDE variant at partial / realistic (it is the only physics method that uses the assays
  directly, not interpolated).
* P3 (both): A3 beats A0 (Holm) on most (>= 6/10) case x level cells (sequence_ood), as in beat_greybox.
* P4 (both): the plain model filters (EKF/UKF) are worst-case-ratio poor (worst-case ratio across
  levels > 2) because of the partial levels; the knowledge-agnostic winner (smallest worst-case ratio,
  sequence_ood) is a grey-box or UDE filter in both cases, not HyperKKL.
* P5 (robustness): under GNSS loss (A) the filters (which skip stale samples and predict through
  the model) recover faster than every learned sequence observer (HyperKKL, GRU, NLOX); with a poor
  initial guess the KKL observers forget the prior within ~3/alpha (2 s / 30 min), the filters faster.

**Deployment (Case A):** expected MCU-estimate ranking (fastest first): climatology < EKF < grey-box
EKF ~ UDE-EKF ~ NODE-EKF < UKF variants < NLOX-140k ~ HyperKKL A0 < HyperKKL A3 < GRU-140k; **HyperKKL
A3/A0, GRU-140k and NLOX-140k do NOT meet the 150 us / 512 KB budget** (>= 140k weights =
560 KB f32 by the recipe's own conditioner budget); the EKF family does. On the host CPU with torch
(batch 1, 1 thread) the per-call overhead dominates: the KKL observers and the GRU (few large ops) are
expected to be FASTER than the EKF/UKF (many small ops, vmap Jacobian, matrix exponential), i.e. the
CPU ranking is NOT the MCU ranking.

**Practitioner's pick (forecast):** Case A: **grey-box EKF** (or UDE+EKF): best or near-best accuracy at
every level, fits the MCU budget, recovers from GNSS loss. Case B: **UDE+EKF** (uses the sparse
assays properly, physics-anchored, interpretable residual; the budget is irrelevant here).

## 4. Analysis (mechanical, code/analyze_cs.py)

* Per case x tier x level: seed mean and 95% t-CI of the primary metric for every method.
* Paired comparisons A3 vs each comparator: per seed r_s = (mean_A3 - mean_c) / mean_c on the same
  episodes; mean, t-CI over seeds, one-sample t-test; **Holm across the 5 levels** of one (case, tier,
  comparator) family. Win / lose = Holm p < 0.05 and sign; else tie.
* Knowledge-agnostic summary: for each method and (case, tier), the ratio of its seed-mean error to the
  best method of each level; **average log-ratio (geometric mean ratio) and worst-case ratio** across
  the 5 levels.
* Robustness: NRMSE per condition, divergence rate, poor-init convergence (paired forget time) and
  outage recovery time (metrics.block_scorecard).
* Bits rule table and violations; requirement pass/fail; latency / FLOPs / footprint table.
* Calibration of filter covariances vs a split-conformal HyperKKL band: optional, only if time permits.

## Addenda

### Addendum 1 (2026-10-27 14:28 Dubai) -- information budgets A, before the observer launch
Computed by code/budget_cs.py on the runs' own (GPU-generated, cached) evaluation episodes, true plant,
the cell's sensor model (budgets/A.json, budgets/B.json; logs/budget_*.log). A = unmeasured Bayesian
PCRB over 3/alpha (A: 2 s, B: 30 min), sigma-normalised; id / sequence_ood / shape_ood:

| case | exact | mm10 | mm30 | partial | realistic |
|---|---|---|---|---|---|
| A | 0.0145 / 0.0102 / 0.0134 | 0.0151 / 0.0107 / 0.0140 | 0.0164 / 0.0116 / 0.0152 | 0.0145 / 0.0102 / 0.0134 | 0.0165 / 0.0116 / 0.0153 |
| B | 0.0089 / 0.0086 / 0.0106 | 0.0102 / 0.0097 / 0.0112 | 0.0098 / 0.0092 / 0.0104 | 0.0089 / 0.0086 / 0.0106 | 0.0103 / 0.0096 / (pending) |

The windowed bound is essentially the full-past bound in both cases (the 3/alpha window already holds
the information). So the CANNOT branch of the rule needs a comparator below ~0.01-0.017 NRMSE. My
Case-B forecast in section 3 ("A close to or above eps at exact / mm10") is therefore now expected to be
WRONG unless the exact-model EKF reaches ~0.01; I keep it as written (it is scored as a forecast).
Other pre-launch changes, all before any full-size observer run: cstr_exo kinetics frozen above 400 K
and IC box shrunk (STATUS.md 13:5x: numerical runaway of the fixed-step RK4 from hot IC corners).
Shape_ood B realistic value is filled from budgets/B.json in RESULTS.md.
