# Pre-registration: bounded held-out plants cartpole_b, two_link_arm_b

Fixed 2026-10-26 18:51 Dubai, BEFORE any observer (learned or filter) was run on either plant. Post-hoc testbed
change (disclosed); old cartpole / two_link_arm results stay in the record (appendix).

## Runs (no tuning on the new plants)
* Tree node **TEST-C1-B** (parent TEST-C1, its config unchanged): methods autonomous, latent, concat,
  film, lora, hyper; baseline ekf; seeds 0-2.
* Tree node **TEST-C3-B** (parent TEST-C3: knowledge partial, mismatch 0.3, sensor colored_t 0.03 /
  rho 0.9 / nu 3): methods latent, lora, hyper; baselines ekf, ukf, pf, kkl_mb, nlox, ekf_huber,
  ukf_huber, ekf_oracle, ukf_oracle, koopman_kf, greybox_ekf, greybox_ukf, greybox_ekf_oracle;
  seeds 0-2 (climatology = NRMSE 1 reference, as in every report).
* SOAP arms **A2** and **A4** (beat-greybox campaign recipe, harness = p2_rescore/code/run_p2.py copied to
  code/run_soap.py with pkg + systems_bounded) on TEST-C3-B, seeds 0-2.
Jobs run directly (python -m hyperkkl.job), same code as the tree worker; smoke test first.

## Primary comparison
Per plant and tier (id, sequence_ood), metric = unmeasured-state NRMSE (also total):
HyperKKL = A4, A2 and the frozen-recipe best of lora/hyper, each reported; grey-box = lowest seed-mean
of greybox_ekf / greybox_ukf / greybox_ekf_oracle. Fold = HyperKKL mean / grey-box mean, 95 % CI
paired over seeds 0-2 (t-interval, df 2, on log ratios) = p2_rescore's analysis. Win: CI upper < 1;
lose: CI lower > 1; else tie.

## Bits-rule prediction (Part A method: whitened AR(1) Student-t PCRB, true plant with 0.3 mismatch)
Budget on CPU-generated TEST-C3-setting episodes (`budget_new.py` -> budget_*.json):

| plant | tier | A = unm lb, window 3/alpha | full-past unm lb | per-state window lb |
|---|---|---|---|---|
| cartpole_b | id | 0.0085 | 0.0075 | x 0.0045, v 0.0081, th 0.0058, om 0.0089 |
| cartpole_b | sequence_ood | 0.0083 | 0.0074 | |
| two_link_arm_b | id | 0.0090 | 0.0090 | q1 0.0008, q2 0.0025, dq1 0.0037, dq2 0.0143 |
| two_link_arm_b | sequence_ood | 0.0081 | 0.0081 | |
(old cartpole under the same method: A 0.0648, full 0.0105 — the bounded plant is information-rich:
window bound ~ full-past bound.)

Rule: HyperKKL **CANNOT win** in a tier if A > eps (eps = best grey-box unmeasured NRMSE of that tier);
otherwise the information budget does not rule a win out. Prediction: for both plants and both tiers
the rule says "CANNOT win" only if the grey-box reaches an unmeasured NRMSE below ~0.008-0.009, i.e.
within ~1.1x of the full-past bound. On the old test plants the best grey-box sat 2-5x above the
full-past total bound (cartpole 5.1x, fhn 2.3x), so **predicted: "can win" (not ruled out) on both
plants, both tiers**. A violation = predicted "CANNOT win" and HyperKKL wins.
Not a prediction of a win: the rule is one-sided (fhn: "can win", HyperKKL loses).

## Addendum 1 (2026-10-26 19:07 Dubai, before any observer run on two_link_arm_b)
two_link_arm_b revision 2 (DESIGN.md: IC box q1 centred on the hanging pose). Its budget replaces the
table row above: A = **0.0098** (id) / **0.0092** (sequence_ood), full-past 0.0095 / 0.0090; same
prediction ("can win" unless the grey-box reaches ~0.009-0.010). two_link_arm_b is an optional extra
plant (the original two_link_arm passes the inclusion rule and stays); it runs at lowest priority on
GPU slots cartpole_b does not need. cartpole_b analysis is unaffected.
