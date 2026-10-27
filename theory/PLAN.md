# Theory plan — what the paper needs, and nothing more

The paper's thesis has three claims; each theorem must serve one of them. Anything that
does not change what a reader believes about a claim is cut or moved to a remark.

| claim | result that carries it | where |
|---|---|---|
| **why a short window suffices** (C2) | Thm m2-window: sufficient window `(ln(1/η) + ln l)/α` for every plant (A1-A4); matching lower bound with rate `α_k ∈ [α, α_max]` proven for LINEAR plants only (nonlinear: conjecture). Cor m2-rule: the plant enters only through `ln l` (integrator-type states = large `l`). Pre-registered C2 tests FAILED; the refined rule is a post-hoc explanation | main text (one statement, modest) + appendix |
| **where the input should enter** (C1) | Prop m1-where: latent injection needs a contraction margin on a forward-invariant ball (depends on the observer's context); decoder-side conditioning is stable for every input and pays `l_z, l_c` at unseen contexts (OOD = context distance under protocol v2) | main text |
| **when learned KKL beats filters** (C3) | Prop m3-mismatch (linear plants): a wrong model gives every fixed-gain filter a non-decaying bias; Q-inflation trades it against noise gain, a data-independent floor; the learned decoder pays its fit error on the TRUE plant and has no model term. Prediction: learned wins iff floor > fit error (crossover in the relative model error). Sensor-fault ISS (Cor m3-faults) is qualitative and NOT a comparative advantage: filters win on exact models | main text |
| what the deployed sampled observer is | one discretisation lemma: exact latent step, `O(dt²)` interpolation error, ZOH input and context | appendix |
| why SFM rather than GRU | Lemma m2-sfm (SFM forgets at a guaranteed rate; (c) = the memory probe) + GRU counterexample | appendix |
| error decomposition of the deployed observer | Prop m3-iss: exact attribution (ii), stochastic form (iv); the a-priori bound (iii) is vacuous (15x-1400x) and is presented as a decomposition, never a certificate | appendix |
| existence of the time-varying map | **cited**, not re-proved (Bernard & Andrieu 2019); assumptions stated once | appendix |

Status tags on every statement: [proven] / [proven, linear plants] / [conjecture] / [empirical]. Empirical verdicts
come from `paper/generated/RESULTS.md` (pre-registered rules applied mechanically), never from a digest.
Revision record of the editor reviews: `theory/REVISION.md`.

## Three reviewer hats (what each will ask)

**Control theorist.** Are the assumptions those of the KKL literature, stated once and
cited (backward distinguishability, forward completeness, bounded output, Hurwitz `A`)?
Is the latent-injection result the classical small-gain argument, credited as such? Is
the sampled-data observer the one analysed? Is any constant claimed but not defined?

**Deep-learning practitioner.** Does each statement predict something checkable about
the architectures (amplitude dependence of the latent variant; window knee ∝ 1/α;
robustness ranking by measured `l_c`)? Expressivity is NOT the story: every decoder-side
variant is a universal approximator given capacity (one remark with citations), so C1
differences are inductive bias and optimisation at a matched budget — say exactly that.

**ML theorist.** No vacuous guarantees: the bound is a decomposition into measurable
terms, not a certificate; say which constants are estimated (and that empirical
Lipschitz quantiles are lower bounds). Out-of-distribution: the only thing theory can say
is Lipschitz extrapolation in the context — state it in one line and test it.

## Style (Occam's razor)

- Fewest assumptions that make the statement true; one assumption block, reused.
- Continuous-time statements in the main text, each proof ≤ half a page; sampled-data
  constants in one appendix lemma that every main result points to.
- No per-architecture universality proofs; no high-probability certification machinery;
  no restating of cited theorems' proofs.
- Every result: statement → one-sentence intuition → proof → what it predicts → which
  experiment node tests it.
