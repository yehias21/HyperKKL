# Pitfalls checklist

Each item below was found in the HyperKKL experiments. Each one silently changed a conclusion before
it was caught. Tick every box before you report a comparison.

- [ ] **P1. The grey-box residual is fitted in one form and deployed in another.** The residual of a
  grey-box / UDE model was fitted as a correction to the *discrete one-step map*
  (Phi + dt r). It was then deployed *inside RK4* as f_model + r, a different model, and
  8 to 470x less accurate. Every "beats the grey-box" verdict was against a weakened baseline.
  Fit the residual through the same integrator the filter uses. The corrected baseline is
  `ude5k1`. (`experiments/baselines2/AUDIT_UDE.md`)
  *Check:* the one-step prediction error of the deployed model on `val` matches the error
  measured during fitting.
- [ ] **P2. The EKF or UKF is ungated under outliers.** A plain Kalman update diverges on 12 sigma
  outliers. Without a gate, 2274 filter episodes diverged; a fixed 99.9 % chi-square gate removed
  96.7 % of the divergences. With the gate, the gated UDE+EKF was more accurate than gated
  HyperKKL in 12 of 14 outlier/outage cells (8 significant). "Learned observers never diverge" was an artefact.
  Use `EKF(gate=True)`. (`experiments/gating/RESULTS.md`)
- [ ] **P3. The plant is unbounded.** The original cart-pole's cart position is a random walk.
  Its spread grows 9.4x from T to 5T, so the NRMSE depends on the episode length and the
  normalisation. `check-plant` rejects it (bounded rule). Use `cartpole_b`.
  (`experiments/cartpole_bounded/INCLUSION.md`)
- [ ] **P4. The latent input matrix is ill-conditioned.** For KKL-type designs, the latent filter
  (A, B) must pass the design gate (controllability Gramian cond < 1e12). One arm (beat_greybox D1:
  doubled n_z with a cosine B) relaxed the gate. Its cond was 3.2e14 on battery, so its "harmful"
  verdict is confounded. Never relax the gate. The cosine B fails at large n_z (cond 1e14 to 1e18);
  use a block / bank design. (`experiments/fairness_battery/RESULTS.md` Task 2, `audit_B.md`)
- [ ] **P5. A physics loss is inactive.** A time-varying PDE term was off by default
  (`freeze_encoder_deltas=True`). The runs that were meant to enable it passed the string
  `'False'`, which is truthy, so the term stayed off. Test that every loss term you claim
  actually has a non-zero gradient. (`experiments/tv_physics/AUDIT.md`)
- [ ] **P6. Training data is unequal.** A learned method trained on a richer mix (event-structured
  episodes) was compared with filters whose residuals saw only the standard set. Give every method
  the same episodes, or report both numbers. (`experiments/fairness_battery`,
  `experiments/equaldata_mix`; PROTOCOL.md section 2)
- [ ] **P7. The input is applied inside a step differently from the data.** The data hold u_k
  constant on [t_k, t_k+1) (zero-order hold). Feeding a filter (u_k + u_k+1)/2 was wrong at every
  switch: bioreactor EKF NRMSE was 0.187 instead of 0.006. (`hyperkkl/data.py` docstring)
- [ ] **P8. Selection is done on test data.** Use `val` / `val_ood` only. `val_ood` exists so that
  input-OOD robustness can be selected for without looking at the test tiers.
- [ ] **P9. The real parameters leak.** The metadata files contain `mismatch30_params`, the real
  plant's parameters. They are there for documentation and must not reach a model-based method at
  `mismatch` or `realistic`. Use `context["model"]`.
- [ ] **P10. The headline is the all-state NRMSE.** The measured states are nearly free, since a
  filter just passes y through. The headline is the **unmeasured** NRMSE. Report both.
- [ ] **P11. Only one seed is reported, or bootstrap CIs are used as if they were seed CIs.** See
  PROTOCOL.md section 5.
- [ ] **P12. A fixed innovation gate is fragile, and every simple fix breaks another cell.** This
  was found while building the reference filters (full 64-episode seq-OOD tier, 9 cells x clean /
  outliers / burst outage):
  - A chi-square gate active from the first sample locks out during the transient from the broad
    prior (lorenz/exact 0.14 instead of 0.006).
  - Switching it off for 3 t_ref lets early 12-sigma outliers wreck the filter (battery 9.3,
    network 81).
  - A Huber warm-up fixes battery but not the lock-outs later on (lorenz 0.026, two_link_arm/mismatch
    2.0, rossler burst 1.8).
  - Re-opening P at a forced update fixes lorenz and two_link_arm but lets outliers in on
    vdp/mismatch (7.7).
  - A Huber-only filter never locks out but loses clean lorenz (0.038).

  - The full regeneration with all three safeguards then made the model-mismatch cells worse
    (highway_traffic/mismatch 0.81 vs 0.10; fhnring under outliers 5-90).

  The reference `ekf_gated` therefore keeps the **published rule** of the research gating study
  (no warm-up, no re-opening), plus stale-sample skipping. Its known failure (lorenz/exact) is
  documented, and `ekf_huber` is reported next to it, because **no fixed rule dominates**. One network / fhnring_n4 episode still diverges under outliers with every
  variant (cell mean ~1.57 = one clipped episode out of 64). Report the divergence rate, report
  gated filters on the clean condition too, and never pick the gate variant on test data.

## Real-data tracks (v1.1, v1.2)

These were found or guarded against while building the real-data tracks (`hyperkkl_bench.real`).

- [ ] **P13. Unlabelled states are scored or trained on.** On real batteries only SOC, dT and Vm have
  labels; the RC voltages V1, V2 do not. They are NaN in `x` (`label_mask`), and `x_fill` carries the
  partial model's replay as a *placeholder* (what the research did). Never score a placeholder, never use
  it as a loss target, and never call a method "full-state" on real data. The research masked the loss
  (`realdata.masked_scales`); the bundled UDE and GRU mask it the same way, and HyperKKLTrain passes the mask.
- [ ] **P14. A quantised sensor is mistaken for a frozen one.** The battery thermocouple moves in ~0.2 K
  steps and the EMPS encoder in 1/50000 rev, so exact repeats are normal. The core filters' stale-sample
  rule (skip a sample equal to the previous one) would then throw most temperature updates away. Real
  tracks set `context["hold_missing"] = False`; do the same in your own filters.
- [ ] **P15. The label is itself an estimate.** Battery SOC is the lab's coulomb counter from a full
  charge (capacity measured once, after ageing of up to ~40 cycles: up to ~3 % SOC bias late in a
  discharge). EMPS / Silverbox velocities are offline zero-phase derivatives of the measured position
  (non-causal; never an input). Differences below the label noise floor (card: `label_noise_floor_nrmse`)
  are not differences.
- [ ] **P16. The partial model is identified on test data.** The ECM, the EMPS friction model and the
  Silverbox Duffing model are identified on the TRAINING split only (`freeze` / `identify`), and frozen in
  `resources/real_<dataset>.json`. Re-identifying on all data (as the EMPS reference script does on its full
  record) leaks the test inputs into the model.
- [ ] **P17. Temperature shift is reported as input shift.** The battery 0 / 10 / 40 degC splits change the
  plant (resistances x 2.5 at 0 degC), not only the input. The model-based context moves the 25 degC model
  by its Arrhenius factor; learned methods keep their 25 degC fit. Report these splits separately from the
  25 degC headline (the research: the grey-box EKF wins at 0 degC, learned observers fail).
- [ ] **P18. Per-episode priors leak the label.** Real-track priors are SOC0 +- 0.2 (battery) and (y_0, 0)
  (mechanics), the same rule for every method and episode. Starting a filter from the true SOC0, or
  tuning on which sign of the prior is used, is a different (easier) task.
- [ ] **P19. Smoothed "measurements" leak the future (neurobem).** The published `ang vel` and `mot` columns are zero-phase
  smoothings of the logged gyro and rotor telemetry, over about +-60 ms and more. Do not use them as sensors or inputs: the
  track's `y` and `u` are causal 10-ms means of the raw logs. The published `acc` is Vicon-derived and is a label, not an IMU.
- [ ] **P20. Flying attitude is not observable from a thrust-only model (neurobem).** An accelerometer on a quadrotor reads
  thrust, not gravity. Tilt and horizontal speed reach it only through rotor drag, so a filter without a drag term has no tilt
  information however good its tuning. Yaw and position are not observable at all from these sensors: do not report them.
- [ ] **P21. Closed-loop inputs and identification (neurobem, emps).** At low frequency the flight controller's rotor differential
  balances the steady aerodynamic moment, so regressing angular acceleration on rotor speeds gives a control effectiveness too small
  by a factor ~8. And the rotor speeds are computed from the gyro: u carries state information, learned methods will exploit it.
- [ ] **P22. A hysteretic cell's OCV is averaged (calce).** LiFePO4 has a flat plateau (0.9 mV per 1 % SOC) and a 52-mV gap between
  its C/20 charge and discharge curves. On discharge-only profiles the averaged OCV is half a hysteresis off, worth ~29 % SOC: the
  first calce EKF scored 18 % (prior: 19 %). The discharge branch, chosen on val, gave 7.9 %. Check the OCV branch against the
  direction of the profiles, and choose it on val.
- [ ] **P23. One input family in training (calce, and A3 in general).** Trained on one drive-cycle family, A3 fits it (calce val on
  DST 5.6 %) and fails on new ones (US06 / FUDS 12-19 %). On battery_lg, UDDS-only training raises its val error from 0.63 to 3.32 %,
  while 202 random windows instead of 487 leave it at 0.64 % (experiments/real_candidates/DIAG_REAL.md). Report the number of input
  families in training next to any learned-observer result.

## Two families and operating profiles (v2)

- [ ] **P24. An "autonomous" task on a plant that comes to rest.** A forced plant at u = 0 (battery, bioreactor,
  cartpole_b, quad_tank, highway, two_link_arm, damped duffing) converges to an equilibrium. Estimating it is
  trivial and says nothing about observers. Family A admits only plants that pass the "non-trivial unforced
  dynamics" check. Lorenz passes at nominal parameters but not at +0.3 (stable fixed point); rossler's +0.3 plant
  escapes the envelope from the held-out IC shell, so both are nominal-only in family A. (`experiments/bench_v2/design/autonomy_scan/`)
- [ ] **P25. Crediting input conditioning in an autonomous cell.** At u = 0 an input context is constant, so a
  conditioned decoder no longer depends on the input; HyperKKL is then an autonomous KKL observer whose decoder also
  sees a strided latent window (a delay embedding). Any gain of HyperKKL, concat or FiLM over an autonomous KKL in
  family A comes from training and that window (budget, fine-tuning, C5), not from conditioning.
- [ ] **P26. Testing on inputs nobody deploys.** v1 test tiers are identification waveforms (am, sawtooth, PRBS)
  rescaled in time. A method can win there and lose under a real operator, or the reverse. Report the
  operating-profile tiers as the headline and the legacy tiers separately.
- [ ] **P27. OOD that is really "larger amplitude".** A tier with more authority than training (wider box, faster
  slews) mixes extrapolation in state with a change of operator. The op_ood tier uses the same actuator and box as
  training (checked by the profile tests). A different driver may still spend more time at the box edge, which is
  intended and stated.
- [ ] **P28. A tuned filter selected from a prior the track does not use.** hyperkkl_bench 1.1 `TunedKF` builds
  its validation prior from `context["prior_mean"]`. The real-data tracks give a per-episode prior instead, and the
  tuned filters crashed there. Fixed in 2.0: TunedKF selects on each validation episode's own track prior.
- [ ] **P29. Equal data in name only.** "Mixed" means train + train_event for EVERY method. A comparator trained on
  the standard split alone sees half the episodes and none of the event structure (PITFALLS P6).
- [ ] **P30. phase1 is not held out from the initial-condition OOD tier.** The family-A `test_ic_ood` shell is disjoint
  from train / train_event / val, but `phase1` (400 zero-input episodes over the whole envelope, as in v1) starts
  56-100 % of its episodes in that shell. In the v2 reference study, the HyperKKL family (phase 1) and the UDE / NODE
  filters (transitions) used phase1, and GRU, NCDE, NLOX-M and DKKL did not. Their test_ic_ood numbers are
  therefore not out-of-distribution (marked * in reference/v2/extras.md), and using phase1 at all departs from the
  equal-data rule (P6, P29). Report whether phase1 was used.
