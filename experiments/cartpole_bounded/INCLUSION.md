# Inclusion check (pre-registered rule) — POST HOC with respect to the original plant selection

Rule fixed 2026-10-26 18:23 Dubai, BEFORE any of the check's numbers were computed and before any observer
result on cartpole_b / two_link_arm_b exists. Plants: battery, bioreactor, duffing, lorenz, quad_tank,
vdp, cartpole, fhn, highway_traffic, network, rossler, two_link_arm, cartpole_b, two_link_arm_b.
Both criteria come from the KKL theory's standing assumptions (trajectories in a compact set; the
state is (backward-distinguishable) observable from the output history). Code: `inclusion.py`.

## (1) OBSERVABLE (at least weakly)
Setting: each plant's own C1 setting (exact model, white Gaussian sensor noise 0.03 x y_std,
process noise 0.005), eval_id episodes (dev: C1-where cache; test: TEST-C1 cache; new plants:
episodes generated with the same generator on CPU).
* (1a) full-past Bayesian PCRB (experiments/information, `infolib.pcrb_bayes_full`) on every
  unmeasured state j, sigma-normalised: crb_j must be finite and < 1 (below climatology).
* (1b) local observability along trajectories: linearised observability Gramian over a window of
  3/alpha s, W_o = sum_k Phi_k^T H_k^T R^{-1} H_k Phi_k (Phi_k = state-transition Jacobian, states in
  sigma-normalised units), at 12 times x 32 episodes. Pass if lambda_min/lambda_max > 1e-10 at
  >= 90 % of the (episode, time) points (numerical full rank). Median condition number reported.
* OBSERVABLE = (1a) and (1b). "weak" is noted when max_j crb_j >= 0.1 or the median
  lambda_min/lambda_max < 1e-6.

## (2) BOUNDED under the protocol's input box
Long rollouts of horizon 5 T (T = the plant's episode length), RK4 + the data generator's process
noise, zero-order-hold inputs from the protocol (v(t) in [-1, 1] x u_scale), no episode rejection.
Input sets: training families (regime "train"), id, shape_ood, sequence_ood families; 64 episodes per
set, ICs Sobol in the IC box. Raw (unwrapped) states, exactly what the observer must reconstruct.
* S_j(t) = std over episodes of x_j, averaged over the window [t - T/10, t].
* (2a) end spread: S_j(5T) <= 2 S_j(T) for every state j and every input set.
* (2b) no secular growth: the per-state spread over [4T, 5T] does not exceed 2x the spread over
  [T, 2T] (same window statistic), and all rollouts finite.
* BOUNDED = (2a) and (2b) for every state and input set. (A random walk gives S(5T)/S(T) = 2.24.)

Verdict table: plant x (observable?, bounded?, evidence). Plants failing a criterion are flagged; the
original results are NOT removed (disclosed post-hoc testbed change, old results kept in appendix).

## Results (computed 2026-10-26 19:07 Dubai with inclusion.py after the rule above was fixed)
| plant | observable? | bounded? | evidence: max unmeasured full-past CRB; Gramian median lmin/lmax (frac full rank) | evidence: worst S(5T)/S(T), worst secular ratio (state, input set) |
|---|---|---|---|---|
| battery | yes | yes | 0.0064; 3.3e-04 (100%) | 1.96 (x5, shape_ood); 1.60 (x5, shape_ood) |
| bioreactor | yes | yes | 0.0085; 1.8e-01 (100%) | 1.16 (x1, train); 1.14 (x2, train) |
| duffing | yes | yes | 0.0053; 1.2e-02 (100%) | 1.05 (x1, train); 1.08 (x1, train) |
| lorenz | yes | yes | 0.0016; 1.9e-04 (100%) | 1.02 (x3, train); 1.00 (x3, train) |
| quad_tank | yes | yes | 0.0245; 4.9e-03 (100%) | 1.33 (x2, train); 1.28 (x2, train) |
| vdp | yes | yes | 0.0038; 1.4e-01 (100%) | 1.01 (x1, shape_ood); 1.00 (x1, shape_ood) |
| cartpole | yes | NO | 0.0145; 3.6e-05 (100%) | 9.62 (x1, id); 4.54 (x1, id) |
| fhn | yes | yes | 0.0077; 1.7e-01 (100%) | 1.13 (x2, id); 1.10 (x2, id) |
| highway_traffic | yes | yes | 0.0195; 2.1e-03 (100%) | 1.12 (x3, train); 1.19 (x4, train) |
| network | yes | yes | 0.0113; 1.7e-04 (100%) | 1.58 (x2, shape_ood); 1.09 (x2, train) |
| rossler | yes (weak) | yes | 0.0018; 3.5e-08 (100%) | 1.01 (x3, shape_ood); 1.01 (x3, sequence_ood) |
| two_link_arm | yes | yes | 0.0651; 1.4e-04 (100%) | 1.00 (x2, train); 1.00 (x4, id) |
| cartpole_b | yes | yes | 0.0086; 1.2e-04 (100%) | 1.20 (x4, shape_ood); 1.14 (x4, shape_ood) |
| two_link_arm_b | yes | yes | 0.0114; 2.7e-02 (100%) | 0.96 (x2, id); 1.06 (x2, train) |

(two_link_arm_b = revision 2; revision 1 failed nothing formally but wound the shoulder, see DESIGN.md.)

**Verdict.** Every plant is observable (rossler weakly: Gramian ratio 3.5e-8, PCRB still 0.0018).
**Only the original cartpole fails**, on boundedness: cart-position spread grows 9.6x from T to 5T
(secular ratio 4.5; a ballistic/random-walk drift). **The original two_link_arm passes both criteria**
(its elbow winds early from the initial energy, |q2| up to ~21 rad, but the spread does not grow
after T: ratios <= 1.0), so by this pre-registered rule it stays in the test set unchanged; only
cartpole is replaced (by cartpole_b, which passes both). two_link_arm_b is kept only as an optional
extra plant.
