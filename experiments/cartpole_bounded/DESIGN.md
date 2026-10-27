# Bounded test plants: cartpole_b and two_link_arm_b — design (POST HOC testbed change, disclosed)

Written 2026-10-26 18:51 Dubai, before any observer was trained or evaluated on either plant. Design used only
physical / control criteria (linearisation, protocol rollouts, observability); no learned-observer or
filter result on these plants was looked at. Code: `hyperkkl/systems_bounded.py`
(sha1 d94a8b44b5ae01f846fb35e1058a5c9f58edd421, registered lazily from `hyperkkl/__init__.py` like
systems_extra; not in cache.DATA_MODULES, so no existing cache key changes; must not be edited once
runs exist), configs `configs/systems/{cartpole_b,two_link_arm_b}.yaml`, checks `design_check.py`
(-> design_check_*.json), `inclusion.py` (-> inclusion/*.json), tests `tests/test_systems_bounded.py`
(7/7 pass: registry, dims, perturbed() keeps the class, f - f_known = exactly the declared terms,
boundedness over 5T random-input rollouts at the box edges).

## Why
cartpole closes only the pole loop (K_BAL has zero cart gains): the cart is a random walk with drift
(x_std 75 m, training spans +-485 m, angle std 0.065 rad): a ~1000x scale gap in the latent and a
violation of the compact-set assumption of KKL theory. The inclusion check (INCLUSION.md) confirms:
cart position spread grows 9.4x from T to 5T (rule: <= 2) -> NOT bounded.

## cartpole_b
Same physics, sensors (x, th), partial model (UNMODELLED large-angle terms), actuator (saturating,
u_bar 2, back-EMF k 0.25 on om, automatic design unchanged), T 30 s, nz 6, alpha 1.5, t_ref 1.
Change: the stabilising loop F = f_scale (u - K x) gets weak cart feedback,
K = (k_x, k_v, k_th, k_om) = (**-0.8, -0.6**, -9.889688, -2.181384) (pole gains = original K_BAL).
IC box x [-1.5, 1.5] m, v [-1, 1] m/s (was +-5, +-3: a box on the track); th, om unchanged.
Criteria and how the gains were chosen (grid k_x in {-0.4..-1.2}, k_v in {-0.2..-1.0}; linearisation +
48-episode protocol rollouts, nothing else):
* closed-loop stable: eigenvalues -7.85 +- 0.75j (pole loop, tau 0.13 s), **-0.345 +- 1.07j (centering,
  tau 2.9 s, zeta 0.31)**: centering 22x slower than the pole loop, several seconds, so the cart keeps
  moving under the input (not driven to the origin);
* static gain x_ss/u = 1/|k_x| = 1.25 m per unit command: the protocol box |v| <= 1 maps to +-1.25 m,
  leaving margin on a gym-like +-2.4 m track. Protocol rollouts (horizon T): |x| p99 2.0-2.04 m on all
  four input sets; absmax 3.3-3.8 m (initial transients from the IC box);
  k_x = -0.5 (x_ss/u 2 m) gave p99 above 2.4 m; stronger gains (|k_x| >= 1) made the centering faster
  than 2 s;
* settled state stds (id): x 0.72 m, v 0.45 m/s, th 0.051 rad, om 0.10 rad/s; sequence_ood: 0.72,
  0.89, 0.13, 0.26: all four non-trivial (cartpole: 75 m, 5.5 m/s, 0.065, 0.065);
* the t_ref rule (median 1/|Re eig| along the zero-input trajectory = 1.5 s -> t_ref = 1) is unchanged;
* observability: full-past Bayesian PCRB on the unmeasured (v, om) 0.0056 / 0.0086 (C1 setting);
  linearised observability Gramian over 3/alpha: lambda_min/lambda_max median 1.2e-4 (condition
  8.6e3), full rank at 100 % of 384 points (cartpole: 3.6e-5);
* boundedness (INCLUSION.md rule, 5T rollouts): worst S(5T)/S(T) = 1.18, worst secular ratio 1.14: PASS.

## two_link_arm_b
Same physics, sensors (q1, q2), partial model (UNMODELLED Coriolis / centrifugal), actuator (j = dq1,
k 0.1), IC box, T 100 s, nz 6, alpha 2, t_ref 2. Change: **elbow torsion spring tau_2 = -k_e q2,
k_e = m2 g lc2** (4.905 N m/rad nominal; scales with the mismatched parameters like a spring sized for
the link), rest pose = straight elbow, in both f and f_known.
* physical sizing: the spring is as stiff as link 2's own gravity stiffness, so the static gravity sag
  of the elbow is <= 1 rad and the link cannot wind; elbow natural frequency 3.8 rad/s vs shoulder
  pendulum ~3 rad/s;
* checked alternatives k_e x {0.25, 0.5} (16-episode 5T rollouts): the settled elbow excitation
  barely changes (q2 std 0.04-0.05 vs 0.03 rad) but transients reach |q2| 5.3-5.8 rad (x0.25) or
  2.0 rad (x0.5), i.e. near-winding; x1 keeps |q2| <= 0.99 rad. Kept x1;
* protocol rollouts (horizon T): |q2| <= 0.99 rad, q1 within [-4.5, 1.0] (no shoulder winding);
  settled stds q1 0.18, q2 0.03 rad, dq1 0.48, dq2 0.10 rad/s (two_link_arm: 0.83, 6.36, 0.71, 0.86).
  Disclosed weakness: the elbow is only weakly excited by the shoulder torque in the input band
  (its motion is largely the decay of the initial energy), so q2/dq2 are small-signal states;
* the protocol's t_ref rule would now give 7 (median 1/|Re eig| 21 s: the lightly damped elbow mode);
  t_ref kept at 2 so that the input sequences equal the original arm's (disclosed deviation);
* observability: full-past PCRB on (dq1, dq2) 0.0037 / 0.0144 (C3 setting) — Gramian: see
  inclusion/two_link_arm_b.json (filled below before the runs);
* note on the original two_link_arm: it PASSES the pre-registered boundedness rule (spread of q2 does
  not grow after T: the elbow winds early from the IC energy and then stops), but its raw q2 reaches
  |q2| ~ 21 rad, so its operating set is large rather than unbounded.

## two_link_arm_b, revision 2 (2026-10-26 19:07 Dubai; still before any observer run on it)
The inclusion check of revision 1 (IC box q1 in [-1, 1], i.e. around the horizontal) showed the
shoulder winding over the top from the initial potential energy (|q1| up to 16.6 rad under shape_ood;
saved as inclusion/two_link_arm_b.v1_icbox_original.json.bak). Fix (YAML only, code unchanged): q1 IC
box centred on the hanging pose, [-pi/2 - 1, -pi/2 + 1]; other boxes unchanged; the actuator's x_ref
(dq1 box centre 0) is unaffected. Result (5T rollouts, 64 episodes x 4 input sets): |q1| <= 2.8 rad,
|q2| <= 1.0 rad, no winding; settled stds (id, horizon T) q1 0.08, q2 0.02 rad, dq1 0.20, dq2 0.07
rad/s (small-signal plant; disclosed); full-past PCRB (C1) on dq1/dq2 0.0046/0.0114; Gramian
lambda_min/lambda_max median 0.027 (condition 37, far better than revision 1's 1.7e4); bounded ratios
<= 0.96. Budget (TEST-C3 method): A = 0.0098 (id) / 0.0092 (sequence_ood); full-past 0.0095 / 0.0090.
**Status after the inclusion verdict (coordinator decision):** the original two_link_arm passes both
pre-registered criteria and stays in the test set unchanged; two_link_arm_b is an optional extra plant
only (lowest priority), not a replacement.
