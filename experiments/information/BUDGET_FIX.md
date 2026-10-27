# Budget fix for the TEST-C3 comparison — POST HOC

Written 2026-10-26 18:27 Dubai. Everything in this file is post hoc (computed after the TEST-C3 and p2_rescore
outcomes were known). Code: `infolib_c3.py` (recursion), `budgets_c3.py` (-> `budgets_c3.json`,
log `budgets_c3.log`), `bits_rule_c3.py` (-> `bits_rule_c3.json`). Old files untouched.

## What was wrong
1. **Noise model.** `budgets.py` computed the window bound A (unmeasured Bayesian PCRB, window
   3/alpha) for the test plants under the TEST-C1 setting: white Gaussian sensor noise (0.03 x y_std).
   The comparison it is used against (TEST-C3) generates data with AR(1)-coloured Student-t noise
   (`data.sensor`: noise 0.03, kind colored_t, rho 0.9, nu 3).
2. **Plant.** It linearised the nominal plant; TEST-C3 data come from the true plant
   `build_plant(scfg, perturb=0.3)` (hyperkkl.data.eval_data). The fix uses the TEST-C3 cached
   episodes (true plant) and Jacobians of the true plant.
3. **Tier.** `bits_rule_posthoc.py` used the eval_id budget for both tiers; sequence_ood now uses
   the eval_seq distribution.

Dev plants: their comparison (C3-regimes, condition clean) uses white Gaussian 0.03, mismatch 0
(checked in all six run configs) = the C1-where setting of budgets.json, and the eval cache key
excludes `system.knowledge`, so the episodes are identical. Not recomputed; only fix 3 applies.

## Method (whitening + Student-t Fisher factor)
Sensor model (hyperkkl.data.measure): y_k = h(x_k) + v_k, v_k = rho v_{k-1} + sqrt(1-rho^2) s e_k,
v_0 = s e_0, e_k iid unit-variance t(nu), s = 0.03 y_std. Pre-whitened measurement (conditioning on y_k):
y_{k+1} - rho y_k = h(x_{k+1}) - rho h(x_k) + sqrt(1-rho^2) s e_{k+1}, with independent noise, so the
joint density is a chain whose k-th factor couples (x_k, x_{k+1}); the Tichavsky-Muravchik-Nehorai
recursion holds with
  D11 = E[F'Q^-1 F] + rho^2 E[H_k' J_e H_k],  D12 = -E[F']Q^-1 - rho E[H_k' J_e H_{k+1}],
  D22 = Q^-1 + E[H_{k+1}' J_e H_{k+1}],       J_{k+1} = D22 - D12'(J_k + D11)^-1 D12,
  J_0 = P0^-1 + E[H_0' J_v H_0],  J_v = J_t / s^2,  J_e = J_t / ((1 - rho^2) s^2).
Student-t Fisher factor (location family, unit variance):
  **J_t = (nu+1)/(nu+3) * nu/(nu-2)**  (= 2 for nu = 3; -> 1 as nu -> inf).
Heavy tails make the bound *lower* (an efficient robust estimator can exploit them); colouring makes
it higher (slow noise looks like a slow state). At a window start J_t/s^2 is used for the marginal of
v (Stam's inequality: the true marginal Fisher information is smaller), which can only lower the
bound. Self-test: rho = 0, J_t = 1 reproduces infolib exactly (rel. err 0). Prior = climatology of the
settled TEST-C3 eval_id states; process noise as before.

Guarantee check: all 184 (test plant x method x tier) cells of TEST-C3 (achieved.json, partial) have
per-state RMS >= 0.95 x the new full-past per-state bound: **0 violations**.

## New numbers (A = unmeasured lb, window 3/alpha; old vs new)
Decomposition: old -> same white noise on the true plant -> coloured Gaussian -> coloured t (= new).
| plant | dist | A old (C1 white, exact plant, eval_id) | A white, true plant | A colored Gauss | **A C3 (colored-t, true plant)** | full-past unm old | full-past unm C3 | bits 3/a old | bits 3/a C3 |
|---|---|---|---|---|---|---|---|---|---|
| cartpole | eval_id | 0.0415 | 0.0397 | 0.0761 | **0.0648** | 0.0107 | 0.0105 | 27.0 | 23.8 |
| cartpole | eval_seq | 0.0415 | 0.0396 | 0.0760 | **0.0647** | 0.0106 | 0.0105 | 27.0 | 23.8 |
| fhn | eval_id | 0.0078 | 0.0079 | 0.0081 | **0.0081** | 0.0077 | 0.0079 | 14.8 | 14.7 |
| fhn | eval_seq | 0.0077 | 0.0078 | 0.0080 | **0.0080** | 0.0076 | 0.0078 | 15.0 | 14.8 |
| highway_traffic | eval_id | 0.0512 | 0.0773 | 0.1061 | **0.0965** | 0.0186 | 0.0203 | 18.7 | 18.9 |
| highway_traffic | eval_seq | 0.0513 | 0.0775 | 0.1059 | **0.0965** | 0.0186 | 0.0203 | 18.7 | 18.9 |
| network | eval_id | 0.0079 | 0.0083 | 0.0088 | **0.0087** | 0.0076 | 0.0081 | 48.3 | 42.2 |
| network | eval_seq | 0.0078 | 0.0083 | 0.0087 | **0.0087** | 0.0076 | 0.0081 | 48.4 | 42.3 |
| rossler | eval_id | 0.0014 | 0.0013 | 0.0013 | **0.0013** | 0.0014 | 0.0013 | 27.0 | 28.7 |
| rossler | eval_seq | 0.0014 | 0.0013 | 0.0013 | **0.0013** | 0.0014 | 0.0013 | 27.0 | 28.8 |
| two_link_arm | eval_id | 0.0456 | 0.0193 | 0.0201 | **0.0201** | 0.0455 | 0.0199 | 21.0 | 28.5 |
| two_link_arm | eval_seq | 0.0361 | 0.0171 | 0.0177 | **0.0176** | 0.0360 | 0.0175 | 21.9 | 28.9 |

Reading: colouring roughly doubles the window bound on cartpole (0.040 -> 0.076; the slow cart states
and the rho = 0.9 noise share the same band) and on highway_traffic; the t factor recovers ~15 %.
two_link_arm's bound halves because the true (mismatched) plant is better excited than the nominal
one; fhn, network, rossler barely move. Full-past bounds barely move (except two_link_arm): the grey-box,
which uses the whole past, is not limited by the window.

## Corrected rule table (HyperKKL "cannot win" if A > eps; eps = best grey-box unmeasured error)
```

== tier id
plant              A_old   A_new full_new  eps_gb log2(A/eps)   pred_old   pred_new   fold  src outcome
battery           0.0084  0.0084   0.0036  0.0274       -1.71        can        can   0.54   A4 win
bioreactor        0.0085  0.0085   0.0085  0.2700       -4.98        can        can   0.03   A4 win
duffing           0.0113  0.0113   0.0053  0.0095       +0.26     CANNOT     CANNOT   7.24  old lose
lorenz            0.0014  0.0014   0.0014  0.2390       -7.42        can        can   0.03   A4 win
quad_tank         0.0297  0.0297   0.0198  0.0196       +0.60     CANNOT     CANNOT   1.69   A4 tie
vdp               0.0038  0.0038   0.0038  0.0248       -2.71        can        can   0.26   A4 win
cartpole          0.0415  0.0648   0.0105  0.0478       +0.44        can     CANNOT   5.71   A4 lose
fhn               0.0078  0.0081   0.0079  0.0140       -0.79        can        can   1.77   A2 lose
highway_traffic   0.0512  0.0965   0.0203  0.0434       +1.15     CANNOT     CANNOT   4.21  old lose
rossler           0.0014  0.0013   0.0013  0.6577       -8.94        can        can   0.37  old win
two_link_arm      0.0456  0.0201   0.0199  0.3148       -3.97        can        can   1.22  old lose

== tier sequence_ood
plant              A_old   A_new full_new  eps_gb log2(A/eps)   pred_old   pred_new   fold  src outcome
battery           0.0084  0.0084   0.0038  0.0192       -1.18        can        can   1.06   A2 tie
bioreactor        0.0085  0.0085   0.0085  0.2696       -4.99        can        can   0.05   A4 win
duffing           0.0113  0.0111   0.0054  0.0101       +0.13     CANNOT     CANNOT   7.39  old lose
lorenz            0.0014  0.0014   0.0014  0.2386       -7.42        can        can   0.03   A4 win
quad_tank         0.0297  0.0296   0.0198  0.0185       +0.67     CANNOT     CANNOT   2.28   A4 lose
vdp               0.0038  0.0036   0.0036  0.0327       -3.19        can        can   0.23   A4 win
cartpole          0.0415  0.0647   0.0105  0.0712       -0.14        can        can   8.28   A4 lose
fhn               0.0078  0.0080   0.0078  0.0135       -0.76        can        can   3.58   A2 lose
highway_traffic   0.0512  0.0965   0.0203  0.0334       +1.53     CANNOT     CANNOT   5.64  old lose
rossler           0.0014  0.0013   0.0013  0.7091       -9.09        can        can   0.34  old win
two_link_arm      0.0456  0.0176   0.0175  0.3165       -4.17        can        can   1.19  old lose
```
(fold = HyperKKL / best grey-box unmeasured NRMSE; src = SOAP arm from p2_rescore where present, else
the original lora run; outcome by paired CI where available. network: no grey-box run, omitted.)

## Changes and violations
* **cartpole, id: prediction flips "can win" -> "CANNOT win"** (A 0.0648 > eps 0.0478, log2 +0.44);
  outcome lose (A4 fold 5.7) -> consistent. sequence_ood stays "can win" (A 0.0647 < eps 0.0712,
  log2 -0.14, marginal); outcome lose (fold 8.3) -> not a violation (the rule is one-sided: "can win" is
  a necessary condition, not a prediction of a win), but the margin is under 10 %.
* **fhn: unchanged** ("can win", A 0.008 vs eps 0.014); loses with A2 (fold 1.8 / 3.6): the loss is not
  explained by the information budget.
* highway_traffic: "CANNOT" with a larger margin (A 0.097 vs eps 0.043/0.033). two_link_arm: "can win"
  with a larger margin. Dev plants: unchanged except sequence_ood now uses eval_seq (no flips).
* **Violations (predicted CANNOT, observed win): none**, before or after the fix.
