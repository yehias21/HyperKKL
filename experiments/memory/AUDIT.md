# C2 memory audit: theory, implementation, and what the pre-registered nodes actually tested

Date: 2026-10-25 23:40 CET (2026-10-26 02:40 Dubai). Auditor: memory-audit agent. Scope: the C2 results in
`theory/m2_fading_memory.tex` (with the shared assumptions of `m1_existence_location.tex`), the checks in
`theory/checks/m2_checks.py`, the code paths `hyperkkl/nets.py` (SFMEncoder, WindowEncoder), `design.py`,
`conditioning.py`, `job.method_cfg`, `train.phase2`, `evaluate.memory_probe`/`probe_curves`, `metrics.fit_decay`,
`data.py` (washout, settle window), `analysis/evidence.py` (C2 rules), and the node configs
`experiments/tree/C2-*/node.yaml`. All numerical checks below were run on the CPU
(`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4`), on synthetic encoders and on the cached trained models of C2-floor-05
and C1-where (lora, seed 0). Scripts: scratchpad `check_sfm_window.py`, `check_real_probe2.py`, `check_lorenz_pairs.py`
(outputs quoted inline). No hashed module was edited (the queue was non-empty throughout: 0 pending / 15-32 running).

Classification key: **BUG** (code wrong), **MISMATCH** (code right, but the node tests something other than what the
theorem states), **THEORY ERROR** (statement wrong or imprecise), **OK**.

## Summary

1. **No BUG was found in the encoders, the probe mechanics, or the training path.** SFM `seq()` and `step()` agree to
   2e-16; the floor is applied in both (same `sigma()`); the impulse response decays at exactly the slowest pole;
   `xi_k` uses `u_{<k}` only; the window encoder uses exactly `u_{k-L+1..k}` causally with zero history; parameter
   counts match the docstrings.
2. **The pre-registered C2 rules test quantities the theorems do not assert** (four MISMATCHes, M1-M4 below):
   the probe-rate check ("fitted rate >= floor") tests a local log-slope that Lemma m2-sfm(c) does not bound (it is an
   upper envelope); the window rule was pre-registered without `ln l` and without the design constant
   `ln(kappa |B| ybar/alpha)`, both of which the theorem carries and both plant-dependent; "within 5% of e(512)" is a
   relative-change criterion with an effective eta that is 10-50x smaller than the pre-registered eta = 0.05;
   and the floor intervention changes exponential attenuation, not truncation, so Theorem m2-window (a truncation
   statement) does not predict "no loss at c = 0.5" unless `ln(1/eta) >= ln l + ln(design const)`, which fails on battery.
3. **Two statistics artefacts, not bugs, produced the two headline anomalies:** (i) probe ratios 0.73-0.96 below the
   floor are the finite-window log-slope of a mixture of modes seen through the psi Jacobian (cross terms of opposite
   sign between the floored blocks and the 0.66 alpha block); the pure filter tail `|Ad^l Bd|` fits at 1.00-1.02x the
   floor on the same trained models; (ii) the flat lorenz curve is the MEAN over 128 (episode, time) pairs dominated by
   4 pairs from one episode (74% of the mean at L = 512); the MEDIAN pair decays 8e-3 -> 3e-5 over 5.7 s, i.e. at
   rate 1.0/s = alpha_sfm exactly.
4. **Two THEORY-text imprecisions** (T1, T2): the SFM fill-in pole statement "alpha/1.5^b, b = 1..5, slowest 0.132 alpha"
   holds for n_z = 5 only (n_z = 7: b = 1..4, slowest 0.198 alpha; n_z = 6: b = 1..5); and the corollary's refined
   rule as used in `tab:m2-window-rule` omits the design constant, which is 8.5-32x z_scale on these plants
   (adds 70-1360 samples to L*). No proof is wrong.
5. **Capacity confound in C2-window:** the window encoder has 64 L FIR parameters (5184 at L = 1, 37888 at L = 512,
   trainable, outside the 140k budget because `budget_includes_context: false`). Minor relative to 140k, but the
   encoder is where the input enters, and bioreactor's non-monotone curve (min at 128, 2.4x worse at 512) is an
   optimisation effect of exactly this kind. The Stage-2 design (truncated SFM history, identical parameters for every
   L) removes it.
6. **quad_tank cannot be tested on the grid** (fixed dt = 0.05, alpha dt = 0.005: L0 = 599 > 512); the window sweep's
   grid was shorter than the pre-registered L* for one of six systems before any run.

Verdict: the theory's statements are correct as stated after revision 3; the experiments C2-floor-* and C2-window-*
did not test them. Stage 2 (below, `PREREG.md`) tests the sufficient half of Corollary m2-rule with the constants it
carries, with a fixed-architecture truncated-history context on the six held-out test systems.

---

## A. Theory: what each result claims

Notation: alpha = design bandwidth = spectral abscissa of the Butterworth A (`system.alpha x alpha_scale`); kappa = 1;
dt = 0.1 sin(pi/2n_z)/alpha unless the system fixes dt (quad_tank: 0.05 s). Hence alpha dt = 0.0309 (n_z = 5),
0.0259 (6), 0.0223 (7), 0.0131 (12), and 0.0050 for quad_tank. L in the theory is in SECONDS; a context of L samples
covers (L-1) dt seconds of history that acts on the latent (Prop. m2-current(a): u_k has not acted at t_k).

| result | bounded quantity | units / constants | status |
|---|---|---|---|
| Lemma m2-tail (a) | `\|T^0(x(t),t;u) - T_L(x(t),u_t)\| <= eps(L)/2`, eps(L) = 2 kappa \|B\| ybar e^{-alpha L}/alpha: the IDEAL time-varying map vs its L-second truncation (zero history before the window) | raw latent units; L in s; rate exactly alpha (slowest latent pole), NOT a latent pole alpha_k | proven; used by every benchmark |
| Lemma m2-tail (b) | same map under two inputs equal on [t-L, t]: difference = e^{AL}[...], rate exactly alpha on the slow left eigenvector | needs the counterfactual backward solution in X: unavailable for Lorenz/Rossler | proven under its hypothesis |
| Thm m2-window (a) | worst-case error of the IDEAL decoder D = T_L^* is <= (l/2) eps(L); so L*(eta) <= [ln(1/eta) + ln l + ln(kappa\|B\|ybar/alpha)]/alpha | eta = ABSOLUTE worst-case state error (in whatever units l is expressed: with l = l_dec_q95 in sigma-units per raw z, eta is in sigma units); l = Lipschitz constant of the inverse of the windowed map (A4), a property of the plant+filter, not of a learned decoder | proven (sufficient only) |
| Thm m2-window (b) | fibre diameter delta(L) between r_q e^{-alpha_k L}/\|P\| and 2 kappa R_q e^{-alpha L}/sigma_min(P): any decoder errs >= delta/2 | linear plants, (H1) washout, (H2) overlapping reachable sets; alpha_k >= alpha is a latent-pole rate (2.618 alpha at n_z = 5, 4.05 alpha at n_z = 7) | proven for linear plants; nonlinear: conjecture |
| Cor m2-rule | L*(eta) ~ [ln(1/eta) + ln l]/alpha "plus the design constant ln(kappa\|B\|ybar/alpha)/alpha" | same as (a); the design constant is explicitly part of the bound | proven (restatement); its use for the sweep is labelled post hoc |
| Lemma m2-sfm (a,b) | per-lag Jacobian `\|dc_k/du_{k-l}\| <= l_psi \|B_d\| e^{-alpha_xi (l-1) dt}/u_ref`; l_E | alpha_xi = slowest SFM block rate (floor x alpha with the floor); samples | proven; bound attained |
| Lemma m2-sfm (c) | history swap: `\|c_k - c'_k\| <= l_psi \|B_d\| 2 ubar e^{-alpha_xi L dt}/(u_ref (1-e^{-alpha_xi dt}))` | an UPPER ENVELOPE in L; says nothing about the local log-slope of the measured curve | proven |
| Prop m2-current | (a) T^0 at t_k independent of u_k; (b) linear SISO: (z, u_t) never determines x; (c) local nonlinear version | - | proven |

Important consequences for what an experiment can test:

* (a)/(a') are one-sided (sufficient). A sweep can only confirm that the error has plateaued BY L* (plus the learned
  decoder's own floor); it cannot fail because the knee is earlier (duffing knee 32 samples vs L* 281).
* The theorem's eta is an absolute error of the ideal decoder. A learned decoder has a floor e_inf (its own fit error,
  NRMSE 0.01-0.1) that is unrelated to truncation; the truncation term adds to it (roughly in quadrature).
* `l` is not measured anywhere; `l_dec_q95` (random-direction quotient of the STATIONARY base decoder, sigma-units per
  raw z, step 1e-2 z_std) is a proxy of a related constant. Its units make eta a sigma-normalised error and make the
  design constant `|B| ybar/alpha` (raw z units, ~ z_scale) NOT omittable: the dimensionless argument of the log is
  `l x |B| ybar/(alpha eta)`.

Design-constant values on the 12 systems (from the runs' `scales`; `|B| ybar/alpha` vs `z_scale`): battery 10.7 (32x
z_scale), bioreactor 8.9 (10x), duffing 0.85 (9x), lorenz 52 (16x), quad_tank 897 (6x), vdp 4.7 (9x), fhn 11 (7x),
rossler 60 (8.5x), cartpole 748 (23x), two_link_arm 64 (31x), highway_traffic 2.1 (12x), network 82 (13x). In samples at
eta = 0.05 (l = median phase-1 `l_dec_q95` of the existing runs):

| system | n_z | alpha dt | l_dec_q95 | L0 = ln(1/eta)/(alpha dt) | L* = (ln(1/eta)+ln l)/(alpha dt) | L*_full (+ design const) | settle index (0.5 N) |
|---|---|---|---|---|---|---|---|
| battery | 7 | 0.0223 | 815 | 135 | 436 | 542 | 1011 |
| bioreactor | 5 | 0.0309 | 43 | 97 | 219 | 289 | 809 |
| duffing | 5 | 0.0309 | 299 | 97 | 281 | 276 | 2427 |
| lorenz | 7 | 0.0223 | 9 | 135 | 234 | 412 | 2247 |
| quad_tank | 6 | 0.0050 | 14 | 599 | 1123 | 2483 | 3000 |
| vdp | 5 | 0.0309 | 23 | 97 | 199 | 249 | 1214 |
| fhn (test) | 5 | 0.0309 | 8 | 97 | 163 | 241 | 809 |
| rossler (test) | 7 | 0.0223 | 138 | 135 | 356 | 540 | 3595 |
| cartpole (test) | 6 | 0.0259 | 76 | 116 | 283 | 539 | 869 |
| two_link_arm (test) | 6 | 0.0259 | 224 | 116 | 325 | 485 | 3864 |
| highway_traffic (test) | 12 | 0.0131 | 49719 | 230 | 1058 | 1115 | 3831 |
| network (test) | 12 | 0.0131 | 40 | 230 | 512 | 849 | 1149 |

(`memory_s` in `scales`, i.e. `design.memory_length` at eps = 1e-3 z_scale with l = 1, equals `t_burn` and is
293-726 samples for alpha_scale 1 and 1735 for quad_tank: the code's own window estimate already exceeds the C2 grid
on quad_tank and is 60-90% of it elsewhere.)

### Mismatches between what was tested and what was proven

**M1 (C2-floor-*, probe rule "fitted rate >= floor"): MISMATCH.** Lemma m2-sfm(c) bounds `|c - c'|` by an envelope
`C e^{-alpha_xi L dt}`. The fitted least-squares log-slope of the measured curve over L in [128, 512] samples is not
bounded by the lemma in either direction: a mixture of modes with rates >= floor seen through a fixed linear map (the psi
Jacobian) has cross terms of either sign, so the local slope can be below the floor while the envelope holds. Checked on
the trained C2-floor-05 models (lora, seed 0): the pure filter tail `|Ad^l Bd|` over lags 100-512 fits at 1.02x (battery),
1.00x (vdp), 1.02x (lorenz) alpha_sfm, but `|J_psi(0) Ad^l Bd|` fits at 0.71x, 0.86x, 1.01x; the floored blocks of
battery sit at exactly 0.500 alpha (three blocks) next to one at 0.664 alpha, with learned rotations omega = 0.07-0.08 alpha
on the floored blocks (beat period ~50 s): the opposite-sign cross terms shallow the slope at lags 100-200 (0.69x) and
less at 400-512 (0.84x). The probe's own `alpha_sfm` field (0.75006 for floor 0.5 x 1.5) already certifies the floor; the
rule should have compared the ENVELOPE (`d(L) <= C e^{-floor L dt}`, or the per-lag Jacobian of Lemma (a) at small lags)
or the pole itself. Reported ratios 0.73-0.96 therefore do not test the lemma. The lemma is correct.

**M2 (C2-floor-*, main rule "no loss at c = 0.5"): MISMATCH.** The hypothesis "memory ln(20)/(0.5 alpha) covers the
theory window" uses L0 = ln(1/eta)/alpha and drops both `ln l` and the design constant. The SFM floor also attenuates
(e^{-0.5 alpha tau}) rather than truncates, and the un-floored SFM's slow fill-in modes (0.13-0.2 alpha at init, 0.14-0.18
alpha after training; battery after training: 0.739, 0.471, 0.295, 0.184 alpha) approximate the integrator a coulomb-counting
state needs. With the floor at 0.5 alpha the battery SFM keeps 0.664 and three blocks at 0.500 alpha (measured): the two
slowest modes are gone, and the error rises +23-36% (TOST fails). Corollary m2-rule with l = 815 predicts L* = 436
samples (542 with the design constant) for battery, i.e. 2.2-2.7x the e-fold count of the floored SFM over 97 samples:
the corollary predicts the battery loss and the node's rule contradicted its own theory. A floor of c alpha corresponds to
a window only in the loose sense `ln(1/eta)/(c alpha)`; the Stage-2 design truncates instead.

**M3 (C2-window-a1, rule "e(L >= L*) within 5% of e(512)"): MISMATCH (constants and eta).** (i) L* was pre-registered
as ln(1/eta)/alpha without `ln l` and without the design constant; the theorem's sufficient window at eta = 0.05 is
L*_full = 249-542 samples on the dev systems (2483 on quad_tank), not 97-135. (ii) "Within 5% of e(512)" is a relative
criterion on a learned decoder whose floor e(512) is 0.015-0.06 NRMSE: a truncation term t adds roughly in quadrature,
`sqrt(e^2 + t^2) <= 1.05 e  <=>  t <= 0.32 e ~ 0.005-0.02`, so the effective eta is 0.005-0.02 sigma, 2.5-10x smaller than
the pre-registered 0.05 (adds 30-75 samples to L*). (iii) The sufficient direction can only be falsified by a plateau
reached AFTER L*; on battery the curve is still falling at 512 (e(256)/e(512) = 1.73), consistent with L*_full = 542 and
with the grid being too short, not with a wrong rate. (iv) quad_tank: L0 = 599 > 512, untestable by construction.
(v) Capacity confound: the window encoder's trainable parameters grow 5184 -> 37888 with L (budget excludes the
encoder); bioreactor is non-monotone (min at L = 128, 2.4x worse at 512 on sequence_ood), which no truncation bound
describes and which a growing FIR bank can produce. (vi) `l_dec_q95` differs by seed by <= 10% and by node not at all
(same phase 1); fine.

**M4 (C2-window-a2, rule "knee in samples changes <= 2x at 2 alpha"): MISMATCH (statistic).** The knee is "smallest L
with e(L')/e(512) - 1 <= 5% for all L' >= L" on point estimates with 3 seeds whose paired CIs are +-10-20% wide; a
5% threshold on such curves is not identifiable (vdp: knee 32 at alpha, 128 at 2 alpha although e(64..512) are within
+-7% of each other at both bandwidths). The theory's prediction (knee in samples invariant with dt ~ 1/alpha) is about the
rate alpha of the ideal map's truncation; with the learned floor e(512) itself changing with alpha (different A, different
phase 1), the test confounds the two. A log-interpolated knee at a 2x-wider threshold, or a direct fit of the truncation
term e(L)^2 - e(inf)^2, would be the matching statistic. Not re-run in Stage 2 (dev-only; secondary).

**M5 (C2 probe as a test of Lemma m2-tail): MISMATCH, already stated in the tex.** The probe holds z fixed and tests the
ENCODER's memory (Lemma m2-sfm), not the map's (Lemma m2-tail). The tex says so; the node titles ("Corollary 1 ...
Lemma 2 check") conflate them.

### Theory-text issues (no proof affected)

**T1 (THEORY ERROR, minor, `sec:m2-ctx` "At initialisation").** "fill-in poles alpha/1.5^b, b = 1..5, slowest
alpha/1.5^5 = 0.132 alpha, next 0.198 alpha ... 7.6x" holds for n_z = 5 (3 unique poles, 5 fill-ins). For n_z = 7
(battery, lorenz, rossler: 4 unique poles) b = 1..4 and the slowest initial pole is 0.198 alpha (5.1x); for n_z = 6
(quad_tank, cartpole, two_link_arm: 3 pairs) b = 1..5 (0.132 alpha); for n_z = 12 b = 1..2 (0.444 alpha). The measured
"0.18 alpha on battery and lorenz, between the two slowest initial poles" is in fact just below their initial 0.198 alpha
(trained: battery 0.184, lorenz 0.200). Repair: state the count as `8 - ceil(n_z/2)` fill-ins and give the n_z = 7 values;
the conclusion "inherited, not learned" is unchanged (and strengthened). `m2_checks.sfm_history_swap` asserts
alpha/1.5^5 at n_z = 5 only (correct for what it tests).

**T2 (THEORY ERROR, presentation, `cor:m2-rule` / `tab:m2-window-rule`).** The corollary states the design constant and the
table drops it ("the design constant is omitted") although it is 8.5-32 x z_scale here, i.e. +70 (duffing, where
|B| ybar/alpha = 0.85 raw units with l in per-raw-z units ... the constant is dimensionally required, not optional) to
+1360 (quad_tank) samples. With l = l_dec_q95 (sigma per raw z) the dimensionless combination is l |B| ybar/alpha; dropping
the constant is not a small-constant approximation. Repair: either carry the constant in the table (L*_full column above)
or express l in units of z_scale (`l_dec_q95 x z_scale`) and say so. The "n.t." conclusions get stronger (battery, duffing,
lorenz, quad_tank all beyond the grid), bioreactor (289) and vdp (249) remain testable and pass.

**T3 (OK, but worth a sentence).** Thm (b)'s rate alpha_k and the "needle" are latent-pole statements for linear plants;
the C2 nodes never measured a decay rate of the error vs L, so nothing in the data bears on (b). The tex is explicit.

## B. Implementation: findings and numerical checks

| item | finding | class |
|---|---|---|
| `SFMEncoder.sigma()` = -(floor + softplus(raw)); used by `_ad_bd()` which both `impulse()` (seq) and `step()` call | floor applied identically in seq and step; `alpha()` >= floor for any parameters (checked after perturbing raw_sigma, w_in: 0.50004 alpha) | OK |
| fill-in poles `p.real.max()/1.5^(b-len(p)+1)` | slowest 0.1317 alpha (n_z = 5), 0.198 alpha (n_z = 7), 0.132 (n_z = 6), 0.444 (n_z = 12); with floor c the slow blocks clip to c alpha + 1e-4 | OK (tex T1) |
| impulse response `H[l] = Ad^l Bd` | tail rate 0.7501 = slowest pole exactly (synthetic, floor 0.5 x 1.5); `|Ad^l Bd|` of trained models 1.00-1.02x alpha_sfm | OK |
| seq vs step | SFM 2.2e-16, window 4.4e-16 (float64, N = 400); `test_core`, `test_verify_models` cover float32/CUDA | OK |
| causality | SFM: perturbing u[200:] changes c from k = 200 (direct path u_k), xi from k = 201; window L = 16: perturbing u[100] changes c on k in [100, 115] exactly | OK |
| `u_ref` | every encoder divides by `u_ref = system.u_scale`; the probe swaps raw u, so Delta u <= 2 ubar as in Lemma (c) | OK |
| window `L` semantics | window = L samples INCLUDING u_k, so (L-1) dt seconds of acting history; tex says so | OK |
| window encoder parameters | 5184 + 64 (L-1): 5184 (L = 1) ... 37888 (L = 512), trainable, outside the 140k budget (`budget_includes_context: false`); `match_width` sizes the conditioner only | MISMATCH (capacity confound, M3 v) |
| episode start | window: zero history (`F.pad`); SFM: xi_0 = 0; both identical to deployment (`init_state` zeros); training samples k >= burn (293-1735 samples, = `memory_s`), so the zero-history transient is excluded from training and from the metric (settle = 0.5 N >= burn everywhere) | OK |
| `method_cfg('lora@window:window=L')` | overrides phase2.context.{kind, window} per method; phase-2 cache key includes it | OK |
| `probe_curves`: keep mask `k > t - L` | keeps u_{t-L+1..t} (L samples incl. the current); differing lags are >= L; corresponds to Lemma (c) with L-1 (off by one sample, lenient, irrelevant for a rate) | OK |
| probe holds z = `z_obs[:, t]` fixed, recomputes c on the spliced input, decodes with the trained conditioner in float64 | as documented; isolates the context path | OK |
| probe L range | `probe_lengths(min(512, settle))`; settle = 0.5 N >= 809 >> 512: the largest L never reaches the un-settled start | OK |
| `fit_decay`: log-linear LSQ on points with d > 1e-9 and L >= 0.25 L_max (i.e. L in [128, 512] samples) | a local slope; see M1. Also the probe statistic is the MEAN over (episode, time) pairs of per-pair RMS | MISMATCH (M1) + artefact (below) |
| lorenz flat curve | reproduced bit-for-bit on the cached lora model (16 x 8 pairs): mean d(512) = 0.0073, median 0.00003, max 0.33; top 4 pairs (episode 8, four times, plus episode 9 at t = N) carry 74% of the mean; their `\|Delta c\|` at L = 512 is 0.08-0.16 (10x the mean pair) and the local decoder gain is 0.8-0.9 sigma per unit c vs median 0.15; the median pair decays 8e-3 -> 3e-5 over 5.7 s = rate 1.0/s = alpha_sfm. On 8 x 4 pairs the mean itself decays at 0.81-0.94x | artefact of the mean statistic, not a bug; report the median or the log-mean |
| `memory_length`, `washout_time` (design.py) | both = Lemma-tail window at eps = 1e-3 z_scale with l = 1, kappa measured = 1; used for burn and recorded as `memory_s`; consistent with the tex's "311-467 samples" | OK |
| `alpha_scale` (C2-window-a2) | rescales alpha, hence dt (alpha dt invariant), poles, scales cache key, SFM floor (absolute = c x alpha_scale x alpha) | OK |
| analysis `window_curves` | `Lstar = ln(1/eta)/(alpha dt)`, `Lref` adds ln(l_dec_q95); no design constant; "not testable" handled | MISMATCH (M3 i) |
| analysis `analyse_floor` probe check | `rate_other >= c alpha_design` strict, per seed and method, all must pass | MISMATCH (M1) |

No fix to a hashed module is required by the audit. The one code addition is the Stage-2 feature (a `history` option
on the SFM context; `nets.py` only), applied after the queue empties.

### Why the probe ratios and the flat lorenz curve are not bugs: the numbers

Trained C2-floor-05 models (lora, seed 0), probe recomputed with `\|Delta c\|` (encoder output) and `\|Delta xhat\|/sigma`
separated, 8 episodes x 4 times, plus a linearised perturbation (u + 1e-4 N(0,1) everywhere older than L) and the per-lag
Jacobian `\|dc_k/du_{k-l}\|` (Lemma (a) quantity):

| system | alpha_sfm | rate of `\|Delta c\|` other / zero / linearised (x alpha_sfm) | rate of `\|Delta xhat\|` (x alpha_sfm) | Jacobian tail rate lags 100-512 (x alpha_sfm) | `\|Ad^l Bd\|` tail (x alpha_sfm) |
|---|---|---|---|---|---|
| lorenz | 1.000 | 0.91 / 0.94 / 0.91 | 0.81 / 0.94 / 1.00 | 0.90 | 1.02 |
| vdp | 0.750 | 0.84 / 0.84 / - | 0.82 / 0.85 / - | - | 1.00 |
| battery | 0.750 | 0.80 / 0.83 / 0.79 | 0.73 / 0.81 / 0.76 | 0.71 | 1.02 |

The encoder's state forgets at exactly its slowest pole; the context (after psi) and the decoded state show the same
sub-floor local slope even for an infinitesimal perturbation, so it is mode mixing in psi's Jacobian, not saturation or
the decoder. The pre-registered statistic is the wrong one; the lemma holds.

## C. What the audit changes for the paper (recommendations to the lead; no tex edited)

1. `sec:m2-ctx`: fix T1 (fill-in count per n_z) and say the probe rule tested a local slope the lemma does not bound;
   report the probe envelope or the median curve; replace "lorenz unexplained" by the pair-breakdown above.
2. `tab:m2-window-rule`: add the design-constant column (L*_full) or express l in z_scale units (T2). The post-hoc reading
   becomes: every dev system's L*_full is beyond or at the edge of the 512 grid except bioreactor (289) and vdp (249),
   both of which pass.
3. Section "Tested by": state the four mismatches (M1-M4) as the reason the pre-registered rules could not test the
   theorems, and point to the Stage-2 node (C2-trunc-test) as the pre-registered test of Corollary m2-rule's sufficient
   half with its constants.

## D. Coordination note

The Stage-2 code change touches `hyperkkl/nets.py` only (SFMEncoder `history` option + `build_context` passing it),
plus new tests in `tests/test_memory_history.py` and new (unhashed) analysis code. The equal-budget tuning agent edits
`conditioning.py`, `train.py`, `job.py`: no file overlap. Both edits change `module_hash(MODEL_MODULES)`, so every new
job retrains phase 1 (~130 s per (system, seed)) regardless of which lands first; the shared tests (`test_core`,
`test_verify_models`, `test_verify_train`, `test_verify_pipeline`, `test_scorecard`, `test_graphs`) are to be re-run after
both edits, on a GPU in {5, 6, 7}. Status of the edit: see the end of this file.

## E. Status log

- 2026-10-25 23:40 CET: audit complete; queue 0 pending / 15 running; no hashed module edited yet.
- 2026-10-25 23:50 CET: `PREREG.md` written (L*_i for all 12 systems); nodes `C2-trunc-test` / `C2-trunc-dev` written
  (planned, not queued); `tests/test_memory_history.py`, `experiments/make_tree_memory.py`, `experiments/memory/verdict.py`
  and `analysis.evidence.analyse_trunc` (registered in `analyse_all`; smoke-tested: 40 nodes, new ones PENDING) added.
- 2026-10-26 00:00 CET: the `nets.py` change was developed and verified on a scratch copy of the package (13/13 new
  tests pass; diff = SFMEncoder `history` option + `build_context` passing it; reads use `getattr(self, "history",
  None)` so modules pickled before the option still load). Waiting for the queue (2 TEST-C3 network jobs running).
  Baseline of the shared test suite started on the unpatched package (GPU 1).
- 2026-10-26 01:20 CET: lead approved the edit with pending = 0 (running jobs and the reeval shards had imported their
  code; reeval checks the DATA-module hash only). `hyperkkl/nets.py` patched (byte-identical to the verified scratch
  copy; backup in the session scratchpad `nets.py.before`); new MODEL hash e6b1572442; `tests/test_memory_history.py`
  13/13 on GPU 2. Shared suite started: group A (core, verify_models, verify_train) on GPU 2, group B (verify_pipeline,
  scorecard, graphs) on GPU 3. Queuing follows group A.
- 2026-10-26 02:03 CET: shared suite after the edit: 268 passed, 1 xfailed (group A 193 in 45 min on GPU 2, group B 75
  in 5 min on GPU 3) = the baseline. `C2-trunc-test` (18 jobs) and `C2-trunc-dev` (18 jobs) queued; the worker
  (--gpus 0-7) started them immediately (test node first by priority).
- 2026-10-26 04:08 CET: all 36 jobs done (no errors, no job hit the 05:47 cut); `verdict.py` -> `RESULTS.md`:
  C2-trunc-test FAIL ((S) 2/6, (T) 1/6), C2-trunc-dev FAIL ((S) 3/6, (T) 3/6) by the pre-registered rule; cell level:
  sufficiency within +5% on 43/48 test cells and 48/48 within +10%, 33/36 with the design constant; tightness on 13/24
  cells; H = 1 is +112% (test median) worse than the full context. Interpretation in `READING.md` (appended to RESULTS.md).
- GPU rules in force: any idle GPU until 06:00 CET (05:45 CET hard stop for anything on GPUs 0-4; the worker is
  restarted with `--gpus 5-7` at 05:47 CET); from 06:00 CET only GPUs 5, 6, 7. Jobs of C2-trunc killed by the 05:47
  enforcement will be re-queued with `python -m hyperkkl queue <node>` (idempotent: only missing results).
