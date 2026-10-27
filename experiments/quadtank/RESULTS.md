# quad_tank: persistent excitation vs observability. Stage-2 results

Pre-registration: `PREREG.md` (written 11:20 CET, before any run; addendum 1 at 11:40 CET, before
the first real run, lists implementation-only changes). Stage-1 diagnostics: `DIAG.md`. Runs:
2026-10-26 11:40-13:36 CET, GPUs 5-7 (HyperKKL), CPU (data generation, grey-box refits).
Per-run JSON in `runs/`, aggregate in `analysis.json`, training-set measures in `data_diag.json`,
timings / phase-1 diagnostics in `extra.json`, code in `code/` (scratch copy of the beat_greybox
package; the only package change is `code/beat_valfix.patch`).

## Verdict

**Not PE-limited. The data are consistent with the observability-limited explanation, plus a small
data-quantity effect.** Changing only the training data moves HyperKKL's primary error by at most
-9%. More slow-band energy on its own (X2) does not help and makes it worse. The gap to the
grey-box shrinks only from 2.5x to 2.3x (C3-partial) and from 1.7x to 1.6x (C3-mismatch30).
Retrained on the same enriched data, the grey-box filter does not change.

**The pre-registered label is "mixed / partially PE-limited", and only on a technicality.**
The rule says "observability-limited" only if both X2 and X3 move the error by less than 10% in
both scenarios. That held in C3-partial (X2 +9.8%, X3 -8.8%). In C3-mismatch30, X2's point
estimate is +14.8%, a deterioration that is not significant (CI [-16, +46]). So the rule fell into
"mixed" because of a non-significant worsening, not because any arm improved by 10% or more. The
PE criterion is far from met: it needs <= -30% (CI below 0) and a gap closure of 0.5. The best arm
reaches -8.8% and a gap closure of 0.15-0.20.

Primary tier sequence_ood. HyperKKL = the campaign recipe A2 (SOAP + unfrozen base), seeds 0-2.
Changes are `paired_relative` with a Student-t 95% CI over seeds; * marks a CI that excludes 0.

| scenario | arm | training data | HyperKKL NRMSE | vs X0 | gap closure vs stored grey-box | HyperKKL vs best stored grey-box | retrained grey-box (best) | grey-box vs its X0 refit |
|---|---|---|---|---|---|---|---|---|
| C3-partial | X0 | 600 x 300 s, 11 families | 0.0353 | | | +150% [+124, +175]* | 0.0143 | |
| | X1 | 600 x 900 s | 0.0331 | -6.3% [-16.0, +3.4] | 0.11 | +134% [+131, +136]* | 0.0143 | -0.3% [-1.7, +1.1] |
| | X2 | 300 orig + 300 slow, 300 s | 0.0388 | **+9.8% [+0.8, +18.8]*** | -0.16 | +174% [+162, +186]* | 0.0143 | -0.3% [-1.0, +0.4] |
| | X3 | 600 orig + 600 slow, 900 s | 0.0322 | **-8.8% [-14.8, -2.9]*** | 0.15 | +127% [+115, +140]* | 0.0144 | +0.6% [-0.2, +1.4] |
| C3-mismatch30 | X0 | as above | 0.0224 | | | +73% [+49, +97]* | 0.0128 | |
| | X1 | | 0.0210 | -6.6% [-25.1, +12.0] | 0.15 | +62% [+8, +116]* | 0.0132 | +3.3% [-5.2, +11.7] |
| | X2 | | 0.0256 | +14.8% [-16.0, +45.5] | -0.34 | +99% [+47, +151]* | 0.0128 | -0.2% [-9.2, +8.8] |
| | X3 | | 0.0205 | -8.4% [-30.0, +13.3] | 0.20 | +58% [+37, +80]* | 0.0131 | +2.7% [-4.0, +9.3] |

The best stored grey-box on sequence_ood is greybox_ekf: 0.0142 (C3-partial) and 0.0129
(C3-mismatch30). Gap closure is (e_X0 - e_Xk) / (e_X0 - e_greybox). HyperKKL loses to the
grey-box filter in every arm and scenario on sequence_ood, whether against the stored grey-box or
against the one retrained on the arm's own data.

## What the arms did to the data (seed 0; same measures as DIAG.md)

| arm | input fluctuation energy below 1/49 s | below alpha | within-episode std h3 / h4 (cm), C3-partial | h3 / h4 variance not explained by (h1, h2, input history) | seq_ood states beyond the validation p99 NN distance |
|---|---|---|---|---|---|
| X0 | 13.6% | 62% | 0.151 / 0.154 | 7.8% / 0.35% | 1.8% |
| X1 | 23.6% | 65% | 0.144 / 0.187 | 2.9% / 0.38% | 1.6% |
| X2 | 26.8% | 85% | 0.180 / 0.229 | 6.7% / 0.24% | 1.5% |
| X3 | **47.4%** | 83% | 0.193 / **0.283** | 2.5% / 0.25% | 2.0% |

(sequence_ood itself: 2.7% of its energy below 1/49 s; within-episode h3 / h4 std 0.197 / 0.299 cm.)

The intervention did what it was designed to do. X3 more than triples the sub-corner input
energy. That is above battery's 43% at its own corner, which removes the one PE-leaning signal of
Stage 1. X3 also nearly doubles the within-episode motion of the upper tanks, up to the level
sequence_ood shows. The error barely moves anyway.

## Secondary results

**Other tiers.** Same picture on C3-partial: X3 -7.0% [-9.4, -4.6]* on id and -5.3% (ns) on
shape_ood; X2 is +0.9% / +3.9% (ns). On C3-mismatch30, where HyperKKL was already within 8-12% of
the grey-box on id / shape_ood, X3 (-6.3%* on id, -5.3% on shape) brings it level with the grey-box:

- **id:** +1.2% [+0.9, +1.6] against the stored grey-box, +0.7% [-6.7, +8.1] against the retrained one.
- **shape_ood:** +5.3% [-4.6, +15.2] against the retrained one.

This is a few-percent effect on a small gap. It is not a closure on the primary tier.

**Per state (sequence_ood, seed mean, NRMSE of h1..h4).**

| scenario | arm | HyperKKL h1 / h2 / h3 / h4 | grey-box h1 / h2 / h3 / h4 |
|---|---|---|---|
| C3-partial | X0 | 0.0092 / 0.0072 / **0.0632** / 0.0280 | 0.0057 / 0.0056 / 0.0227 / 0.0147 |
| | X3 | 0.0086 / 0.0070 / **0.0571** / 0.0261 | 0.0057 / 0.0056 / 0.0229 / 0.0147 |
| C3-mismatch30 | X0 | 0.0082 / 0.0065 / 0.0263 / **0.0342** | 0.0057 / 0.0056 / 0.0189 / 0.0147 |
| | X3 | 0.0076 / 0.0062 / 0.0223 / **0.0324** | 0.0057 / 0.0056 / 0.0198 / 0.0147 |

The richest data cut the upper-tank errors by 5-15%. h3 (C3-partial) and h4 (C3-mismatch30) stay
2.2-2.5x above the grey-box. They sit at the level of the 10-30 s Cramer-Rao bound of DIAG.md (c),
where h3 is 0.080 x_std at 10 s and 0.017 at 30 s.

**Why X2 hurts.** X2 keeps the episode count and replaces half of the original families with slow
ones. The primary tier, sequence_ood, is faster than training (stop-and-go: 2.7% of its energy is
below the corner). Removing fast content therefore moves the training distribution away from the
evaluation. That is the opposite of what a PE deficit in the slow band would predict. X1 (length
only, original families) and X3 (length, slow content and 2x episodes) give the same -6 to -9%. So
the small gain comes with more and longer labelled data, not with slow-band content.

**Reproduction checks.**

- **HyperKKL X0 vs the campaign's arm A2** (same code and data): per-seed sequence_ood
  0.0356 / 0.0363 / 0.0340 against 0.0356 / 0.0363 / 0.0340 (C3-partial), and
  0.0227 / 0.0213 / 0.0232 against 0.0227 / 0.0213 / 0.0232 (C3-mismatch30). Equal to 4 digits.
- **Grey-box refit on X0 (CPU) vs the stored C3 grey-box (GPU):** sequence_ood +1.0%
  [-1.2, +3.2] (C3-partial) and -1.1% [-6.3, +4.2] (C3-mismatch30); id and shape_ood within
  ±0.5%.
- **The X1 grey-box fitted twice** (in-job and by a separate CPU job, same seed): identical to
  5 digits.

**Phase 1 and cost.** Phase-1 diagnostics do not change with the data:

| scenario | eps_dec | l_dec_q95 |
|---|---|---|
| C3-partial | 0.002-0.005 | 19-25 |
| C3-mismatch30 | 0.008-0.012 | 14-16 |

Validation NRMSE (original 300-s validation set) is flat across arms: 0.027-0.031 on C3-partial
and 0.0135-0.0144 on C3-mismatch30.

Wall-clock on the shared GPUs:

| | phase 1 | phase 2 |
|---|---|---|
| X0 / X2 (300-s episodes) | 545-705 s | about 780 s |
| X1 / X3 (900-s episodes) | 545-705 s | 850-940 s |

No episode was rejected in any arm.

## Interpretation

Stage 1 predicted "observability-limited, < 10% change", and the outcome agrees:

- The evaluation states are inside the training support.
- The training inputs are persistently exciting over three slow time constants.
- The error sits on h3/h4, which the measured lower tanks reveal only over 30-50 s. The KKL
  latent (nz 6, alpha 0.1, memory about 10 s) stores h3/h4 in directions about 1e-5 as strong as
  those carrying h1/h2.

Tripling the sub-corner input energy, tripling the episode length and doubling the number of
episodes leaves HyperKKL 2.3x (C3-partial) and 1.6x (C3-mismatch30) behind the grey-box. The
grey-box needs no extra data, because it carries the mass-balance structure and integrates
innovations with unbounded memory.

Levers that act on the information channel are not data levers. Candidates:

- a slower or larger latent;
- the 600-sample z-window of ERROR_FLOOR (0.046 -> 0.038);
- a model-based innovation correction.

The data levers tested here are exhausted at about -9%.

Caveats:

- 3 seeds, so the C3-mismatch30 CIs are wide (±20-30%).
- Only one recipe was tested (A2). The campaign's final arm, not yet selected, adds the H window
  on some cells, which could interact with data length.
- Validation was kept at the original 300-s episodes in every arm (pre-registered). Selection
  therefore favours the original distribution slightly. Validation NRMSE is flat across arms, so
  this does not explain the null.
- The step count was fixed, so X3 sees each episode half as often as X0.
