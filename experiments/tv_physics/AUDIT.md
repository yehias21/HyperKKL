# AUDIT — physics (PDE) losses in HyperKKL: archive vs current, and the time-varying term

2026-10-26, no GPU used for this part. Line numbers: archive = `archive/HyperKKL/src/training2.py`
(identical to `archive/hyperkkl_code/src/training2.py`), current = `hyperkkl/hyperkkl/train.py`,
campaign = beat_greybox `pkg/hyperkkl/beat.py` (copied to `code/pkg/hyperkkl/beat.py`).
Conventions everywhere: latent z' = A z + B y (A Hurwitz, real Jordan/Butterworth; B from
`design.input_matrix`), unnormalised y = h(x) (data.integrate l.106, LatentStep l.36-51). Signs below
were checked against these.

## 0. Bottom line

| # | term | where | verdict |
|---|---|---|---|
| P1 | autonomous PDE on T0 (phase 1) | current train.py:90-96; archive 140-149, 257-279 | **math correct** (form, sign, A/B, modeller's f, smooth SiLU, jvp). Design flaw: the 1/‖f0(x)‖² weight puts 63-72% of the PDE weight on 1% of collocation points (battery, bioreactor, vdp) and mixes state units |
| P1u | forced PDE on the autonomous T0 (`pde_u_scale>0`) | archive 259-263 | **mis-specified** (forces J_T0 G ≈ 0); never used in any archived run |
| INJ | injection target Φ* = J_T0 (f(x,u) − f(x,0)) | archive 488-510 | correct (Bernard-Andrieu Thm-4 stationary-T form); not a TV PDE; uses the TRUE f |
| TV-a | archive time-varying residual of `dyn` (3 modes) | archive 1099-1120 | chain-rule term present in `jacobian` mode, but (i) it is on the ENCODER, which the deployed observer never evaluates, with no inversion/consistency term to the decoder, (ii) **off by default** (`freeze_encoder_deltas=True`, config.py:298) and the runs meant to switch it on passed the string 'False' (truthy, still frozen), (iii) `jacobian` mode differences ċ forward in time, O(dt) and spurious at input switches, (iv) fixed weight 1, uses the TRUE f |
| TV-c | current pipeline phase 2 | train.py:260-352, beat.py:190-281 | **absent**: no f anywhere in phase 2 (decoder loss only, train.py:190-195 / beat.py:170-178) |

**Answer to "is it implemented correctly"**: the phase-1 autonomous PDE is correct. The archived
time-varying loss has the right chain-rule structure only in `jacobian` mode, and it never
constrained the deployed map. It was also inactive in the default recipe and in the runs that
were meant to test it. The current pipeline has no time-varying loss at all. A correct version
for the current architecture is derived in section 3, implemented in `code/pkg/hyperkkl/tv.py`
and unit-tested in `code/pkg/tests/test_tv.py`.

## 1. Phase-1 autonomous PDE (current, and archive)

Code (current train.py:91-96):
```
zj, Jf = jvp(T0, (xc[j],), (fc[j],))            # Jf = J_T0(x) f(x,0),  f = modeller's model
R      = (Jf - zj @ A.T - yc[j] @ B.T) @ W.T    # W = -A^{-1}
l_pde  = mean_j || R_j / ||f(x_j,0)|| ||^2,  loss += lambda_pde * min(1, s/(0.25 steps)) * l_pde
```
Required, from the autonomous KKL PDE: J_T0(x) f(x,0) = A T0(x) + B h(x). This matches term for
term, with the same sign as `integrate` and `LatentStep`. Further checks:
* f is `data.model_plant(scfg)` (data.py:469-472): nominal parameters, and `f_known` under
  knowledge = partial. The actuator wrapper is included (`SaturatingActuator.f(x,0)` contains the
  back-EMF feedback u_eff = u_bar tanh(−k(x_j−x_ref)/u_bar)). The zero-input label episodes
  (`p1`) come from the same wrapped plant, so labels and PDE describe the same "zero-input"
  system. Under C3-partial/mismatch the labels follow the REAL plant and the PDE follows the
  modeller's model. They conflict by design: that is the test condition, not a bug.
* W = −A⁻¹ maps a residual to the steady latent error (ξ' = Aξ + r ⇒ ξ_ss = −A⁻¹r): correct.
* Collocation: an independent scrambled Sobol set over the operating envelope (data.py:466). That
  is correct and better than the shipped code's trajectory halves (archive docstring l.7-11).
  Caveat (theory, not code): for states whose backward zero-input flow leaves the domain (for
  example outside the vdp limit cycle), T0 exists only after the Andrieu-Praly modification of f
  outside the domain. The residual there is a soft constraint on an extension.
* Smoothness: T0 is SiLU (nets.py:4-6), so the second derivatives through the jvp are exact.
* Unit test `test_phase1_pde_term_is_zero_for_the_exact_linear_kkl_map_and_sign_sensitive` calls
  the real `train.phase1_diagnostics`. eps_pde_rms < 1e-4 for T0 = Px with PF − AP = BC, and it
  is large for the wrong-sign Sylvester solution.

**Discrepancy (design, not math): the normalisation `R / ||f(x,0)||`.** This weights each point by
1/‖f0‖² (clamped at 1e-6), with ‖·‖ in raw, unscaled state units. Measured on the cached
C3-partial collocation sets: max/median weight 4e4 (battery), 3.5e5 (bioreactor), 7e4 (vdp),
3e3 (lorenz), 2.4e2 (quad_tank). The top 1% of points carry 63%, 72%, 68%, 31% and 14% of the
total PDE weight. The PDE therefore effectively sees about 1% of its collocation set, near the
equilibria. Battery's ‖f0‖ is dominated by one state (|f0|≈5.3 against σ=0.11), so the weight
ignores the other states' scales. Suggested fix (NOT applied, because it changes cached models):
divide by ‖f0 ⊘ σ_x‖ with a floor (for example its 5% quantile), or drop the division and keep −A⁻¹.

## 2. Archive physics terms

**P1/P1u (archive 140-149, 257-279).** Same residual as current (`pde_residual_collocation`), with
the same −A⁻¹ weighting (`weight_residual`, default True) and the same 1/‖f0‖ division (l.266).
f0 comes from `system.function` = the TRUE plant (episodes.py:660): the archive never had
modeller-model error. `pde_u_scale>0` (l.259-263) replaces f0 by f0 + G u with u ~ U(−s,s) but
keeps T0 = T0(x). The forced KKL PDE has a u-independent solution only if J_T0 G ≡ 0, so this term
pulls T0 toward a least-squares compromise that is wrong for every u. The comment ("the residual
is affine in u, so a u = 0-only loss leaves the forcing term unenforced", l.143-146) misreads the
theory: the forcing term must be absorbed by ∂T/∂t (time-varying T) or by an injection term
(Thm 4 of Bernard & Andrieu), not by T0. This term was 0.0 in all 808 archived runs (EVIDENCE.md §0).

**INJ (archive 457-510, `train_injection`).** Target Φ*(x,u) = J_T0(x)[f(x,u) − f(x,0)], computed
with `xdot`, `xdot_auto` from the TRUE plant (episodes.py:184). With z = T0(x) and an exact T0,
z' = A z + B y + J_T0 G u. This is the stationary-T + injection observer of Bernard & Andrieu
(TAC 2019, Thm 4), the same structure as the current `kkl_mb` baseline (observers.py:134-150).
The math is correct, and the docstring's (A) vs (B) argument is right. It is not a TV PDE.

**TV-a (archive 1099-1120, `train_dynamic`).** T_mod(x,c) = encoder with hypernetwork LoRA deltas
from c; the deployed map is the decoder T⁺_mod(z,c) (l.1098, deployment loss l_dep).
* `characteristic` (default mode): r = T(x_{k+1},c_{k+1}) − e^{AΔ}T(x_k,c_k) − ∫e^{A(Δ−s)}B y ds
  (quadratic interpolation of y, 16-point Gauss, l.89-120). This is the exact integral form of the
  TV PDE. Correct.
* `jacobian`: r = J_x T f(x,u) + J_c T ċ − A T − B y via one jvp over (x,c) (l.1105-1113). The
  chain-rule term is present, with the right sign. But ċ = (c_{k+1} − c_k)/Δ (l.1110) is a
  forward difference paired with tangents at k, so it is O(Δ). The window context contains u_k,
  so at every input switch (square, PRBS: half the training pool) ċ ~ Δu/Δ is an artefact of
  sampling, not a derivative.
* `finite_difference`: forward Euler on z, O(Δ). Never run.
* Shared defects of the three modes:
  1. **Wrong object.** The residual acts on the encoder T_mod. Inference discards the encoder, and
     no term ties it to the decoder (no T⁺(T(x,c),c) = x consistency). The decoder is affected
     only through the shared hypernetwork trunk and context encoder. Physics never constrains
     the deployed estimate.
  2. **Inactive.** `freeze_encoder_deltas=True` by default (config.py:298) skips the term entirely
     (l.1101). The composed recipe ran with it on (`FA_*`), but the dedicated `RESchar/RESjac`
     ablations passed the STRING 'False', which is truthy, so the encoder stayed frozen. They are
     bit-identical to their control (EVIDENCE.md §0).
  3. Weight fixed at 1 with RMS(z) normalisation (l.1116, l.1149). There is no λ and no −A⁻¹
     weighting, unlike phase 1.
  4. `xdot` = TRUE f (oracle physics).
* Archive evidence (EVIDENCE.md §3): the only clean on/off contrast (`FD_loo_frz` vs `FA_wt`,
  9 pairs) gives **turning the residual OFF −19.6% [−29.0, −9.0] on graded OOD** (i.e. the residual
  hurt), +6.3% [−2.5, +15.9] on waveform OOD and +2.8% [−2.8, +8.7] on ID. `jacobian` mode was
  worse than `characteristic` (waveform +21.6% [+11.1, +33.0]).

## 3. What the TV term SHOULD be for the current architecture

Bernard & Andrieu (TAC 2019, eq. 3): T(x,t) with ∂T/∂t + ∂T/∂x f(x,u(t)) = A T + B h(x). Then
z(t) − T(x(t),t) → 0 for the filter z' = A z + B y. In HyperKKL, t-dependence enters through a
causal summary of the input, T(x,t) = T̃(x, c(t)). If T is the encoder, then ∂T/∂t = ∂T̃/∂c · ċ:
**the chain-rule term cannot be argued away** unless c is constant (u ≡ const after washout) or
Bernard & Andrieu Thm 4's hypotheses hold and an injection term replaces it.

The current (and campaign) deployed observer has NO encoder in phase 2. It is
x̂_k = T⁺(z_k; c_k), c_k = [ψ(ξ_k, u_k), g(z_k, z_{k−s}, …, z_{k−19s}, zs_k)] (SFM context +
z-window + slow bank; beat.py:114-178). If T⁺(T(x,t); c(t)) = x along trajectories, then
differentiating gives the **inverse TV PDE**, which constrains exactly the deployed map:

  R = ∂T⁺/∂z (A z + B y) + ∂T⁺/∂c · ċ − f(T⁺(z,c), u) = 0.                       (TV)

Every context channel is an ODE driven by known signals, so ċ is exact between samples (ZOH
u = u_k), with no finite differences:
ξ' = A_ξ ξ + B_ξ u_k/u_ref (the generator of SFMEncoder's ZOH-exact (Ad,Bd); A_ξ =
blockdiag([[s,w],[−w,s]]), B_ξ = w_in); d/dt z_{k−j} = A z_{k−j} + B y_{k−j}; zs' = A_s zs + B y_k;
d u_k/dt = 0. One forward-mode jvp of the whole decoder over (z, ξ, window, zs) with these
tangents returns ∂T⁺/∂z ż + ∂T⁺/∂c ċ exactly. f = the modeller's model (same as every
model-based component). The residual is sampled on training trajectories, because (z, c) pairs
are only meaningful on the reachable set: independent collocation in (z, c) would ask the
decoder to obey physics at latent/context combinations no state produces. Normalisation:
R/(σ_x α). Choices made and why:
* y in ż is the measured y, so the residual is "the deployed estimate, moving with the deployed
  latent ODEs, has derivative f". Noise adds J_z T⁺ B v, which penalises noise gain. h(x̂) would be
  the noise-free alternative.
* The direct-path input u_k inside ψ jumps at switches. (TV) holds between samples, and the
  integral form below also penalises jumps.
* Exploratory variant (`tv_form="step"`): the integral/characteristic analogue
  R_k = (x̂_{k+1} − Φ^{RK4}_f(x̂_k, u_k, Δ))/Δ. It is derivative-free and exact across ZOH switches.

**Irreducible floor.** For a perfect decoder (x̂ = x), R = f_true(x,u) − f_model(x,u). Measured on
validation episodes for the trained OFF (A4+D3) models (`tv_diag`): the OFF decoder's TV residual
**equals the floor to 2-3 digits in 9/10 cells**. Examples: bioreactor/partial 12.7 vs 12.7,
lorenz/partial 7.89 vs 7.90, battery/mm30 0.614 vs 0.610 (units σ_x·α). The exception is
quad_tank/partial (0.73 vs floor 0.007). So the data-trained decoder already is as
"physics-consistent" as the true state. Under C3 the remaining residual is model error, and
reducing it means biasing x̂ toward the wrong model. This is the main reason to expect no gain,
and it was stated in PREREG.md before any ON run.

## 4. Is this the right way of doing it? (verified literature only; LIT.md)

* **Full TV PDE** (Bernard & Andrieu, IEEE TAC 64(1):270-281, 2019, eq. 3) is the theoretically
  right object for T(x,t). Their explicit solution needs backward integration under the input.
  Any learned T(x, c(t)) must include ∂T/∂c·ċ, which is our (TV). For a decoder-only deployment,
  the inverse form (TV) on trajectories is the consistent choice. An encoder-side residual (the
  archive) is right only together with an inversion/consistency loss that ties it to the decoder.
* **Stationary T + injection** (Bernard & Andrieu Thm 4: control-affine, instantaneously uniformly
  observable, drift strongly differentially observable) avoids ∂T/∂t entirely:
  ξ' = Aξ + By + φ(ξ)u. This route is taken by Peralez & Nadri (L4DC 2024, PMLR 242:929-940;
  model-based Ψ = T(f(T⁻¹z,u)) − T(f(T⁻¹z,u₀)), discrete time), Woelk, Morris & Tang (arXiv
  2602.24252, NLOX, model-free ω(z)u; already a baseline here) and the archive's omega/phi heads,
  and by our `kkl_mb` baseline. Miao & Gatsis (L4DC 2023, PMLR 211:208-219) test inputs only with
  an autonomously trained observer. In an appendix they tried a stationary PDE residual and found
  little benefit.
* **Autonomous PINN-KKL**: Niazi et al. ACC 2023 (arXiv 2210.01476; residual on trajectory "physics
  points", joint T/T*) and Niazi, Cao, Barreau & Johansson (arXiv 2501.11655; sequential: T with
  data + PDE on trajectory states, then the inverse), the template for our phase 1. They state
  non-autonomous systems are out of scope. Buisson-Fenet et al. (IFAC 2023, arXiv 2204.00318)
  condition T* on the gain ω_c, a parameter, not time.
* No verified prior work learns T(x, input history) with a ∂T/∂t (or ∂T/∂c·ċ) residual. (TV) is,
  as far as we could verify, new. The answer is therefore: the archive's way was not right (wrong
  object, finite-difference ċ, inactive). (TV) is the principled form for this architecture. Its
  value is bounded by model quality (floor above), and the experiment tests exactly that, with an
  oracle-f arm to separate method from model error.

## 5. Implementation and tests (code/pkg, a copy; the live package is untouched)
`hyperkkl/tv.py`: `sfm_xi_dot`, `TVInputs` (primals + exact tangents), `tv_residual` (one jvp),
`step_residual`, `phase2_tv` (= `beat.phase2_bg` when λ = 0; otherwise the same loop + λ·ramp·L_TV;
selection unchanged), `tv_diagnostic` (residual RMS, model-error floor). `tv.py` is NOT in
`cache.MODEL_MODULES`, so module hash b843eb68d0 = campaign: the campaign's phase-1/phase-2
models are reused read-only, and OFF reproduces the stored A4+D3 test NRMSE exactly (10/10 cells).
Tests (`tests/test_tv.py`): continuous generator ↔ discrete SFM filter (expm), analytic linear
KKL inverse T⁺ = P⁺(z − ξ) has TV residual < 1e-9 and is NOT zero without the ċ term, phase-1
residual zero for the exact linear map and sign-sensitive, jvp derivative = central difference of
the deployed estimate sequence (all context channels, Δ = 1e-3), `gradcheck` of the loss w.r.t.
context parameters through the jvp, default-off bit-identical to `phase2_bg` (state dicts, val),
and the ON path trains (both forms) while the TV loss is a working descent objective (Adam on the term alone halves it). All 7 pass (`logs/pytest_tv_gpu_final.log`; the fixture tests ran with TV_TEST_DEV=cuda:1 because the CPU run was too slow under load). An earlier version of the last test ("λ = 10 reduces the residual of the SELECTED model after 80 steps") failed. Its premise was wrong: Adam/SOAP updates do not depend on the loss scale and the gradient is clipped, so λ ≥ 10 already gives the same TV-dominated direction, and a data-only decoder reduces the residual just as fast. It was replaced.
