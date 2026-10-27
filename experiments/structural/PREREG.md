# Structural model error: pre-registration

Written 2026-10-26, finished 16:58 Dubai (13:58 CET), BEFORE any training, filtering or evaluation
run of this campaign. Before this file, only (i) open-loop simulations of the true plants
(32 episodes per plant and tier, no observer; used to check that the structural plants stay inside
the envelope and that the hidden state changes the trajectories) and (ii) unit tests were run. No
observer was trained or evaluated. Later changes are appended as timestamped addenda and never edit
the text above them. Anything decided after seeing a result of this campaign is labelled post hoc.

## 0. Question and hypothesis

The grey-box baselines (`greybox_ekf`, `greybox_ukf`, `greybox_ekf_oracle`: the modeller's model +
a 2x64 SiLU residual MLP r(x, u) fitted to the one-step defect, inside an EKF/UKF;
`hyperkkl/baselines_data.py`) beat HyperKKL on window-starved plants and lose on information-rich
ones (experiments/information/RESULTS.md, P2). All existing model errors are "residual-friendly"
(wrong parameters, one smooth missing term). Hypothesis: when the model is **structurally** wrong
(hidden dynamics that no function of the model's state and input can represent), the grey-box's
model becomes a biased prior and HyperKKL wins even on window-starved plants: who wins is decided by
information budget x model repairability.

## 1. Tiers (both run here, same code, same seeds)

* **ref** = the C3-realistic setting exactly (TEST-C3 for cartpole and highway_traffic): modeller's
  model = `f_known` at nominal parameters (knowledge partial), real plant 30% off nominal
  (`perturbed(0.3)`), coloured Student-t sensor noise (AR(1) rho 0.9, nu 3, 3%), input protocol v2
  tiers id / shape_ood / sequence_ood, 64 evaluation episodes per tier, 600 training + 400 phase-1 +
  64 validation episodes. The run config is the `config` stored in the existing tree run's
  results.json (seed k uses the seed-k config). Data files already in the live cache are read
  read-only (identical keys), so the ref episodes are exactly the C3-realistic / TEST-C3 episodes.
* **struct** = ref + one structural error in the REAL plant only (table below). The modeller's model,
  the actuator, the sensor model, the input protocol, the NRMSE normalisation (scales from the
  reference plant: `system_scales` is computed with the structure removed), the envelope and the
  rejection rule are unchanged. Hidden states are appended after the visible ones; their initial
  value is drawn per episode from the declared range (own seed stream); labels and the NRMSE use
  the visible (model) state only; y is computed from the full state. Hidden-mechanism constants are
  not physical parameters of the reference plant, so the 30% perturbation is exactly the ref one.

| plant | structural error (class in code/structural.py) | constants | hidden IC | why a residual r(x, u) on the model state cannot fix it |
|---|---|---|---|---|
| quad_tank | **hidden fifth tank**: pump 1's cross flow (1-gam1) k1 v1 enters an unmodelled tank 5 that drains into tank 4 through an orifice (sqrt law) | A5 = 28 cm^2, a5 = 0.071 cm^2 (a copy of tank 3; linearised time constant ~32 s at the operating point, ~2 t_ref) | h5 ~ U[1, 6] cm | tank 4's inflow is a function of h5, i.e. of the past of v1 (a nonlinear low-pass of u), not of (h1..h4, u_k) |
| battery | **OCV hysteresis** (one-state model, Plett 2004): hy' = -gamma \|I\|/(3600 Q) (hy + M tanh(I/I_w)), terminal voltage + hy | M = 10 mV, gamma = 50 per unit SOC, I_w = 0.02 A | hy ~ U[-M, M] | hy depends on the sign history of the current (and is held at rest); at u_k = 0 any value in [-M, M] is possible for the same (SOC, V1, V2, dT, Vm) |
| cartpole | **motor / drive lag** on the cart force inside the balancing loop: F' = (F_cmd(x, u) - F)/tau_m | tau_m = 0.05 s | F_0 = F_cmd(x_0, 0) | F is a filtered past of the commanded force; the model applies F_cmd(x_k, u_k) instantly |
| highway_traffic | **on-ramp queue**: the ramp demand reaches cell 3 through a vehicle queue n' = d - n/tau_r, ramp inflow n/tau_r | tau_r = 20 s | n ~ U[3, 10] veh | the ramp inflow is the queue (a filtered past of the demand), not the current demand |
| bioreactor | **growth lag** (metabolic adaptation): the specific growth rate m follows Monod kinetics with lag, m' = (mu(s) - m)/tau_g | tau_g = 2 h | m ~ U[0.15, 0.6] 1/h | m is a filtered past of the substrate trajectory; for the same (x1, x2, u) any m is possible |
| vdp | **sensor dynamics**: the position sensor is first order, y = s, s' = (x1 - s)/tau_s | tau_s = 0.25 s | s_0 = x1_0 + U[-0.5, 0.5] | the error is in the measurement channel; the grey-box residual acts on the dynamics only and assumes y = x1 |
| lorenz | **regime switch**: rho -> 1.25 rho (smooth step, width 0.5) at a hidden random time | delta = 25%, width 0.5 | switch time ~ U[10, 40] (T = 50) | the regime is a hidden discrete state; r(x, u) can only fit the average of the two regimes |

The magnitudes were fixed from physics before any observer run (a-priori values above) and are not
tuned. Open-loop check (no observer; struct vs ref with the same IC/input, RMS difference / x_std
after washout, id tier): quad_tank 0.16-0.69, battery 0.01-0.08, cartpole 0.02-0.28, highway 0.03-0.20,
bioreactor 0.37-0.50, vdp 0.11-0.16 (y: 0.25), lorenz (chaotic, not meaningful). 0-7 of 32 episodes
were redrawn by the existing rejection rule (quad_tank 7, bioreactor 3), others 0.

## 2. Plants

Window-starved (W): quad_tank, battery, cartpole, highway_traffic. Information-rich (R): bioreactor,
vdp, lorenz. (duffing excluded per author decision.) Seeds 0, 1, 2 for every (plant, tier).
highway_traffic is the most expensive; it is run last in the queue. If it has not finished all six
jobs by the end of the campaign, W = {quad_tank, battery, cartpole} and the thresholds below use the
"of 3" variant (stated in RESULTS).

## 3. Methods (all trained on the same labelled data of the job, evaluated on the same episodes)

* **HyperKKL A4** (primary learned method) = the A4 stack of experiments/beat_greybox/PREREG.md, which
  is: lora code path rank 4, SFM context, + S: SOAP (phase 1 and 2, lr 1e-2) + P: base decoder
  trainable in phase 2 at lr x 0.1 + H: strided z-window (20 samples, stride round(6/(alpha dt)/20))
  as extra hypernetwork-context channel + R: base decoder width 256, conditioner budget 280k,
  6000 phase-2 steps. Code: a copy of the beat_greybox scratch package (live package + soap.py +
  beat.py; sha1 list in code/pkg_snapshot.sha1). The beat_greybox campaign has not selected its final
  arm yet; A4 is used because it is the full stack containing SOAP + unfrozen base, as instructed.
* HyperKKL A0 (secondary, reported, no verdict): the default lora recipe (Adam, frozen base).
* Grey-box: `greybox_ekf`, `greybox_ukf`, `greybox_ekf_oracle` (unchanged package code; the oracle's
  (q, r) inflation tuned per job on validation episodes by `job.tune_oracle`; in the struct tier its
  model-error rate is computed against the structural plant with the hidden state drawn from its
  range). **Best grey-box** per (plant, tier, eval tier) = the one of these three with the lowest
  seed-mean NRMSE (oracle pick favouring the baseline, as beat_greybox).
* Plain model-based filters: `ekf`, `ukf` on the modeller's model (nominal Q, R, P0).
* **Residual-capacity control** `gbw_ekf`, `gbw_ukf` (code/wingb.py): the same residual MLP class
  (2 x 64 SiLU), same targets/optimiser/early stopping, whose input additionally holds a multi-scale
  window of the past: [(xhat_{k-l} - x_mean)/x_std, u_{k-l}/u_scale] for l in {1, 2, 4, ..., 2^m <=
  round(2 t_ref/dt)}. Trained with true past states (teacher forcing, Gaussian noise 0.05 x_std on
  the window states); deployed with the filter's own past posterior means. Same Q, R, P0 as
  greybox_ekf/ukf.
* Climatology: the constant mean training state (true labels after washout).

## 4. Metric and statistics (fixed)

Per episode the settled NRMSE of `metrics.episode_metrics` (settle 0.5 N, normalised by the
reference plant's x_std, all visible states); per seed the mean over 64 episodes of a tier.
Primary eval tier: **sequence_ood**; id and shape_ood reported. Comparison A vs B within a tier:
`metrics.paired_relative` (per-seed relative change of the mean NRMSE on the same episodes,
Student-t 95% CI over 3 seeds). "Beats" = CI entirely below 0; "significantly worse" = entirely above 0.

## 5. Pre-registered predictions (verdicts computed mechanically by code/analyze.py)

* **P-S1**: under structural mismatch HyperKKL (A4) beats the best grey-box on >= 3 of the 4 (or 3 of
  3) window-starved plants (paired CI excludes 0, sequence_ood), whereas under the reference
  (parametric/partial) tier it loses there. Operationally PASS iff (a) struct: A4 beats the best
  grey-box on >= 3 of W, AND (b) ref: A4 is significantly worse than the best grey-box on >= 3 of W.
  (a) and (b) are reported separately.
* **P-S2**: on rich plants HyperKKL's margin does not shrink. Margin m = paired relative change of A4
  vs the best grey-box (per seed, sequence_ood; negative = HyperKKL better). PASS iff on none of the
  3 rich plants m_struct - m_ref is significantly > 0 (Welch t-interval, 95%, on the 3 + 3 per-seed
  margins; the two tiers have different episodes, so seeds are not paired across tiers).
* **P-S3**: the grey-box degradation correlates with the a-priori non-repairability. Degradation
  D = (best-grey-box NRMSE struct - best-grey-box NRMSE ref) / best-grey-box NRMSE ref (seed means,
  sequence_ood; the best grey-box chosen per tier). Repairability rho = 1 - MSE_gb / MSE_model of the
  open-loop H-step prediction (H = round(t_ref/dt)) of the measured output from the TRUE state on
  the struct tier's VALIDATION episodes (64 random start times after washout per episode), errors in
  y_std units clipped at 100: MSE_model with the modeller's model, MSE_gb with model + the fitted
  grey-box residual (seed mean). It uses no evaluation episode and no filter output. PASS iff
  Spearman(D, 1 - rho_struct) >= 0.6 over the 7 (or 6) plants.

Reported without verdict: the same counts on id and shape_ood; P-S1 also against the best of all
five grey-box variants (incl. gbw_ekf/ukf: does the window control rescue the grey-box?); A0;
unmeasured-state NRMSE; ekf/ukf; climatology; rho_ref; the residual fits' validation R^2.

## 6. Compute and isolation

GPUs: 5, 6, 7 and (author permission received 16:50 Dubai) 0-4, at most 2 of our processes per GPU,
yielding 0-4 if a non-project process appears. Code: experiments/structural/code (package copy under
code/pkg, the run-time patches in structural.py; nothing in hyperkkl/ is edited). Outputs:
experiments/structural/runs, cache experiments/structural/cache. Smoke tests (tiny step counts)
precede the full run; they are used only to fix crashes.

## Addendum 1 (2026-10-26 19:20 Dubai): cartpole exclusion variant (coordinator / author decision), before any verdict

Received from the coordinator: a pre-registered inclusion check (experiments/cartpole_bounded/INCLUSION.md)
found that the original cartpole violates KKL's bounded-state assumption (cart random walk, end-spread
ratio 9.4); the testbed replaces it by cartpole_b. Written before any verdict of this campaign was computed
(at this time 30/42 jobs had finished; only per-job progress lines had been looked at, analyze.py had
been run once on incomplete data, where it produced no verdict input because no plant had 3 seeds in both tiers).
The cartpole cells are kept and reported. Verdicts are reported both ways:

* (a) **as pre-registered** (section 5);
* (b) **cartpole excluded (post-hoc, external reason)**: W = {quad_tank, battery, highway_traffic}; P-S1 uses
  the pre-registered "of 3" variant exactly: PASS iff struct: A4 beats the best grey-box on >= 3 of 3 AND
  ref: A4 significantly worse on >= 3 of 3. P-S3 is also reported over the remaining 6 plants (threshold
  unchanged, >= 0.6; the n >= 6 requirement is met). P-S2 is unaffected (cartpole is not a rich plant).

cartpole_b is not run here (future work).
