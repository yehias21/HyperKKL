# cartpole_b results (post-hoc testbed change, disclosed) — 2026-10-26 20:30 Dubai

Pre-registration: PREREG.md (fixed before any observer run). Full tables: tables.md / results.json
(analyze.py). Tree reports: experiments/tree/TEST-C1-B/report.md (TEST-C3-B report after the optional
two_link_arm_b runs finish). Old cartpole results are unchanged in TEST-C1 / TEST-C3 / p2_rescore.

## Primary comparison (TEST-C3 setting, seeds 0-2, unmeasured NRMSE, fold = HyperKKL / best grey-box)
| tier | grey-box (mean) | A4 fold [95% CI] | A2 fold | frozen lora fold | old cartpole A4 fold |
|---|---|---|---|---|---|
| id | greybox_ekf 0.0343 | **0.33 [0.27, 0.41] win** | 0.65 [0.52, 0.82] win | 2.40 [2.27, 2.54] lose | 5.71 [1.11, 23.4] lose |
| sequence_ood | greybox_ukf 0.0728 | **0.71 [0.61, 0.83] win** | 0.94 [0.80, 1.10] tie | 2.66 [2.09, 3.39] lose | 8.28 [1.64, 33.0] lose |
Total NRMSE: A4 0.38 [0.31, 0.47] (id), 0.77 [0.65, 0.91] (seq): win. Per state (A4 / grey-box, seq):
x 0.011/0.012, v 0.017/0.039, th 0.018/0.025, om 0.086/0.106. Every other filter (EKF/UKF/PF/Huber/
oracle on the partial model, NLOX, Koopman-KF) is 4-9x worse than the grey-box on id unmeasured; kkl_mb is worse than climatology (= 1).

## TEST-C1-B (exact model, white noise)
lora/hyper beat concat (-46..-48 %, CI excludes 0) but not latent (-17..-20 %, n.s.) on sequence_ood;
EKF with the exact model 0.006 (as on every plant).

## Bits rule (corrected budget, Part A method)
A = 0.0085 (id) / 0.0083 (seq) << eps 0.034 / 0.073: predicted "can win" on both tiers (pre-registered);
outcome win (A4) on both -> consistent, no violation. Old cartpole: A 0.065 > eps 0.048 on id
("CANNOT win") -> lose, consistent.

## Verdict
On the bounded cart-pole, the SOAP HyperKKL (A4) beats the best grey-box filter on both tiers
(3x on id, 1.4x on sequence_ood), where on the unbounded cartpole it lost 6-8x. The frozen recipe still
loses (2.4-2.7x). This supports the diagnosis that the old cartpole's failure was caused by the
testbed (unbounded cart random walk), not by an information limit.
