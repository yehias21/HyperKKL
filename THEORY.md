# Theory: pointer

The theory is maintained in **`theory/`** (`main.tex` -> `main.pdf`; sources `m1_existence_location.tex`,
`m2_fading_memory.tex`, `m3_error_bounds.tex`; checks `theory/checks/*.py`; revision record
`theory/REVISION.md`). This file is only an index. Statements, assumptions, constants and numbers are
those of `theory/main.pdf`; where this file and the tex differ, the tex is right. Empirical verdicts quoted
in the tex come from the mechanical results layer `paper/generated/RESULTS.md` / `numbers.tex`.

Build: `cd theory && pdflatex main && bibtex main && pdflatex main && pdflatex main` (0 undefined
references; Sections 1-3 + Appendix A). Checks (CPU, ~5 min total):

```
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python theory/checks/check_theory.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python theory/checks/m1_location.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python theory/checks/m2_checks.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python theory/checks/m3_checks.py
```

## One assumption block

(A1) forward invariance of a compact `X`; (A2) `||h|| <= y_bar`; (A3) Hurwitz `A` with
`||e^{At}|| <= kappa e^{-alpha t}` (`kappa = 1` for the normal Butterworth design); (A4) uniform (in `L >= L_0`)
Lipschitz injectivity of the windowed map `T_L`, constant `l` (assumed; its proxy `l_dec_q95` is reported per
system). Latent convention: label latents start at `z(0) = 0`; the deployed observer starts at
`T0_hat(x0_hat)`, carried as the `Delta z_0` term. Chaotic plants: (A4)'s constant is not under control when
`alpha` is below the backward expansion rate; counterfactual-past statements do not apply to them.

## Statements and status

| label (tex) | statement | status | tested by |
|---|---|---|---|
| `lem:m1-window`, `prop:m1-suff` | decoder target `G*(T_L, E(w))`, sufficient contexts exist (linear: `c = Q_L w`), history is necessary | proven | - |
| `rem:m1-koopman` | KKL map = resolvent of the Koopman generator; latent injection / NLOX = lifted (bilinear) Koopman with inputs; decoder side = input-parametrised left inverse | positioning only | - |
| `prop:m1-where` | (a) latent injection: contraction margin on a forward-invariant ball; (b) decoder side: stable for every input, pays `l_z(c), l_c(c)` at unseen contexts (both families c-uniformly bounded, very different constants) | proven | no node measures the margin / gains yet (eval-only) |
| `cor:m1-amplitude` | margin `>= alpha - kappa l_Phi`, shrinks with the observer's context; sufficient only (vacuous for trained injections) | proven | - |
| `prop:m1-tight` | bounded observable plant with exact injection whose latent observer diverges; windowed-context decoder within `(l/2) eps(L)` | proven | check |
| `cor:m1-ood` | OOD excess <= Lipschitz extrapolation in context distance (+ `(l/2) eps(L)`); no learner does better without structure | proven | context distance per tier not yet reported |
| `lem:m2-tail` | (a) finite window within `eps(L)/2` of the finite-start map; (b) counterfactual pasts forgotten at exactly rate `alpha` | (a) proven; (b) proven, inapplicable to chaotic plants | - |
| `thm:m2-window` | (a) `L* <= (ln(1/eta) + ln l + const)/alpha`; (b) lower bound with rate `alpha_k in [alpha, alpha_max]`, fibre computed by support function | (a) proven; (b) proven for LINEAR plants under (H1)-(H2); nonlinear (b) conjecture | C2-window-a1 FAIL (id 2/5, seq 3/5), C2-window-a2 FAIL (1/2) |
| `cor:m2-rule` | refined rule `L* ~ (ln(1/eta) + ln l)/alpha`; integrator-type states = large `l` | proven (rewrite of (a)); its use on the sweep is POST HOC | explains the a1 failures post hoc (table in tex) |
| `prop:m2-current` | current input is not enough (linear SISO: for every `n_z`) | proven | `L = 1` row of C2-window-a1 |
| `lem:m2-sfm` | SFM forgets at a guaranteed rate; (c) history-swap bound = the memory probe | proven | probe rate >= floor: 0/6 at 0.5 alpha (ratios 0.73-0.96, lorenz flat) |
| `prop:m3-iss` | ISS decomposition relative to the nominal observer; (iii) is vacuous a priori (15x-1400x) | (i),(ii),(iv) exact; (iii) decomposition, not a certificate | realised gains (Def. `m3-measured`) not run by any node |
| `cor:m3-faults` | sensor faults bounded and forgotten at rate `alpha` (qualitative) | proven | C3-faults: filters win in all 15 exact-model conditions |
| `prop:m3-mismatch` | persistent model error: filter bias `(I-(I-KC)F)^{-1}(I-KC) d_bar`, bias-variance floor under Q-inflation, learned error has no model term; learned wins iff floor > fit error | proven for linear plants; empirical reading on the benchmarks | C3-mismatch10/20/30/50, C3-partial, C3-realistic PASS; crossover < 10 % (hypothesis 10-30 % FAILS); model-error ordering 30/36; TEST-C3 inconclusive |
| `rem:m3-kf` | KF beats KKL on linear plants, also with outliers | proven | - |
| `lem:D` | sampled-data observer: exact step, `O(dt^2)` interpolation, merged kernels, injection step-size condition | proven | checks |
