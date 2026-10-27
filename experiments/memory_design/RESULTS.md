# Measurement memory for HyperKKL: results

Pre-registration: `PREREG.md` (17:10 Dubai, before any run; addendum 1 at 17:35, implementation facts
from unit and smoke tests only). Runs: 2026-10-26 17:03-18:48 Dubai, GPUs 0-4 (shared with other
HyperKKL agents, 2 of our processes per GPU), phase-1 data on the CPU. Mechanical analysis:
`code/analyze_md.py` -> `analysis.json`, `tables.md` (all tiers). Per-run JSON in `runs/`, logs in
`logs/`, code in `code/` (copied campaign package unchanged + `pkg/hyperkkl/memory.py`, `run_md.py`).

## Verdict

**The hypothesis fails as stated. All four pre-registered predictions FAIL.** Longer memory through
slower latent poles (M1 poles / 3, M2 log-spaced down to alpha/10, M4 context-chosen scaling) does
not help on any window-starved cell. It is significantly worse than M0 on 4 of 8 cells for each of
M1, M2 and M4 on sequence_ood. The one memory mechanism that helps is the strided z-window into the
hypernetwork context (M3). It is significantly better than M0 on 5 of 8 cells and never
significantly worse. It cuts quad_tank/partial by 35%, but it leaves HyperKKL 1.6-1.7x behind the
grey-box on both quad_tank cells. It closes none of the quad_tank/mismatch30 gap. M3 is in effect the
campaign's A3: the A3 numbers are the same within noise, so the plant-specific window span and the
parameter matching change nothing.

| prediction | rule | outcome | verdict |
|---|---|---|---|
| P-M1 | quad_tank, best memory arm (validation): e_up <= -30% (CI < 0) AND n3, n4 <= 1.5 x CRB_30, both scenarios | partial (M5): e_up **-30.3% [-40.3, -20.4]** (part a met), but n3 0.0385 vs 0.0258 and n4 0.0289 vs 0.0203 (part b missed); mm30 (M3): -0.9% [-12.6, +10.8], n4 0.0339 vs 0.0255 | **FAIL** |
| P-M2 | >= 3 of 4 window-starved cells: significantly better than M0 AND gap closure >= 0.5 (or still ahead of the grey-box where M0 already wins) | 1/4: only battery/mm30 (M3 -23%*, already ahead of the grey-box, now -50%*). quad_tank/partial closure 0.46 (M5), quad_tank/mm30 0.02, battery/partial: M3 +5% (ns) | **FAIL** |
| P-M3 | rich cells: every arm's CI upper bound vs M0 <= +10% (20/20) | 11/20. Violations are bioreactor (M1 +130%* / +164%*, M2 +16% / +28%*, M4 +21% / +23%*) and wide CIs (vdp/mm30 M4, bioreactor/mm30 M3, M5) | **FAIL** |
| P-M4 | M4 not significantly worse than the best fixed-memory arm, point <= +5%, on >= 3/4 starved AND 4/4 rich | 0/4 starved, 0/4 rich (M4 is +27% to +145% vs the best fixed arm, 6 of 8 significant) | **FAIL** |

Reproduction: M0, re-evaluated from the campaign's cached A2 models, equals the stored A2 per seed
on all 8 cells (max deviation 0.000%).

## Primary table: sequence_ood (seed-mean NRMSE, 3 seeds; paired Student-t 95% CI; **bold** = significantly better, *italic* = significantly worse)

"val" = seed-mean deployed validation NRMSE, which was used to select the best memory arm (marked).
A3 and A4 are the campaign's stored arms, shown for reference (no verdict). A4+D3 was not complete
for any of these cells when the analysis ran.

| cell | best grey-box | arm | NRMSE | val | vs M0 | vs grey-box |
|---|---|---|---|---|---|---|
| quad_tank/partial | ekf 0.0142 | M0 (A2) | 0.0353 | 0.0297 | | *+150% [+124, +175]* |
| | | M1 poles/3 | 0.0347 | 0.0232 | -2% [-13, +9] | *+145% [+93, +197]* |
| | | M2 log-spaced | 0.0366 | 0.0245 | +4% [-17, +24] | *+158% [+131, +185]* |
| | | M3 window W* | 0.0230 | 0.0188 | **-35% [-39, -31]** | *+62% [+42, +82]* |
| | | M4 s(c) | 0.0348 | 0.0234 | -2% [-30, +27] | *+146% [+56, +236]* |
| | | M5 = M2+M3 (best) | 0.0255 | 0.0183 | **-28% [-38, -18]** | *+80% [+58, +103]* |
| | | A3 / A4 | 0.0233 / 0.0324 | | -34%* / -8% | +65%* / +129% |
| quad_tank/mm30 | ekf 0.0129 | M0 (A2) | 0.0224 | 0.0142 | | *+73% [+49, +97]* |
| | | M1 | 0.0261 | 0.0148 | +17% [-1, +35] | *+102% [+98, +105]* |
| | | M2 | 0.0331 | 0.0181 | *+48% [+11, +86]* | *+156% [+115, +197]* |
| | | M3 (best) | 0.0222 | 0.0139 | -1% [-12, +11] | *+72% [+42, +101]* |
| | | M4 | 0.0317 | 0.0166 | *+42% [+31, +53]* | *+145% [+117, +174]* |
| | | M5 | 0.0222 | 0.0141 | -1% [-12, +10] | *+72% [+33, +111]* |
| | | A3 / A4 | 0.0228 / 0.0227 | | +2% / +1% | +77%* / +75%* |
| battery/partial | ekf 0.0191 | M0 | 0.0192 | 0.0150 | | +2% [-45, +49] |
| | | M1 | 0.0331 | 0.0157 | *+73% [+46, +100]* | +76% [-3, +156] |
| | | M2 | 0.0349 | 0.0283 | *+82% [+30, +135]* | +86% [-6, +178] |
| | | M3 (best) | 0.0201 | 0.0121 | +5% [-14, +23] | +7% [-37, +50] |
| | | M4 | 0.0355 | 0.0285 | *+85% [+52, +118]* | +91% [-31, +212] |
| | | M5 | 0.0200 | 0.0135 | +4% [-22, +29] | +6% [-47, +59] |
| | | A3 / A4 | 0.0241 / 0.0265 | | +25% / +38%* | +28% / +40%* |
| battery/mm30 | ekf 0.0151 | M0 | 0.0097 | 0.0089 | | **-36% [-43, -28]** |
| | | M1 | 0.0132 | 0.0077 | *+36% [+16, +56]* | -12% [-35, +10] |
| | | M2 | 0.0183 | 0.0143 | *+89% [+84, +94]* | *+22% [+6, +37]* |
| | | M3 (best) | 0.0075 | 0.0068 | **-23% [-28, -18]** | **-50% [-58, -43]** |
| | | M4 | 0.0183 | 0.0147 | *+88% [+74, +103]* | +22% [-1, +44] |
| | | M5 | 0.0094 | 0.0072 | -3% [-12, +7] | **-37% [-49, -26]** |
| | | A3 / A4 | 0.0071 / 0.0068 | | -26%* / -30%* | -53%* / -55%* |
| vdp/partial | ekf 0.0257 | M0 | 0.0094 | 0.0086 | | **-63% [-73, -53]** |
| | | M1 | 0.0081 | 0.0063 | **-14% [-17, -10]** | **-68% [-77, -59]** |
| | | M2 | 0.0091 | 0.0064 | -3% [-6, +0] | **-64% [-75, -54]** |
| | | M3 (best) | 0.0068 | 0.0063 | **-27% [-30, -25]** | **-73% [-81, -66]** |
| | | M4 | 0.0092 | 0.0064 | -2% [-5, +1] | **-64% [-73, -55]** |
| | | M5 | 0.0082 | 0.0063 | **-13% [-15, -10]** | **-68% [-76, -60]** |
| vdp/mm30 | ekf 0.0098 | M0 | 0.0092 | 0.0084 | | **-6% [-12, -0]** |
| | | M1 | 0.0088 | 0.0069 | -4% [-11, +3] | **-10% [-16, -4]** |
| | | M2 | 0.0091 | 0.0068 | -1% [-8, +6] | -7% [-17, +3] |
| | | M3 (best) | 0.0071 | 0.0066 | **-22% [-23, -21]** | **-27% [-32, -22]** |
| | | M4 | 0.0091 | 0.0069 | -0% [-12, +11] | -7% [-15, +2] |
| | | M5 | 0.0085 | 0.0068 | **-8% [-14, -2]** | **-13% [-17, -10]** |
| bioreactor/partial | ukf 0.2813 | M0 | 0.0147 | 0.0068 | | **-95% [-97, -93]** |
| | | M1 | 0.0335 | 0.0087 | *+130% [+0, +260]* | **-88% [-93, -83]** |
| | | M2 | 0.0169 | 0.0071 | +16% [-22, +55] | **-94% [-94, -94]** |
| | | M3 | 0.0132 | 0.0067 | **-11% [-15, -7]** | **-95% [-97, -93]** |
| | | M4 | 0.0177 | 0.0074 | +21% [-17, +60] | **-94% [-94, -93]** |
| | | M5 (best) | 0.0117 | 0.0067 | -20% [-40, +1] | **-96% [-96, -95]** |
| bioreactor/mm30 | ekf_oracle 0.0410 | M0 | 0.0121 | 0.0059 | | **-70% [-71, -69]** |
| | | M1 | 0.0320 | 0.0078 | *+164% [+104, +223]* | **-22% [-37, -7]** |
| | | M2 | 0.0155 | 0.0066 | *+28% [+23, +33]* | **-62% [-64, -60]** |
| | | M3 (best) | 0.0118 | 0.0059 | -3% [-22, +16] | **-71% [-78, -65]** |
| | | M4 | 0.0149 | 0.0063 | *+23% [+19, +27]* | **-64% [-64, -63]** |
| | | M5 | 0.0118 | 0.0060 | -2% [-61, +57] | **-71% [-89, -53]** |

Counts against M0 over the 8 cells (significantly better / significantly worse):

| arm | sequence_ood | shape_ood | id |
|---|---|---|---|
| M1 | 1 / 4 | 4 / 4 | 4 / 4 |
| M2 | 0 / 4 | 3 / 4 | 3 / 3 |
| M3 | **5 / 0** | **5 / 0** | **5 / 0** |
| M4 | 0 / 4 | 3 / 5 | 3 / 3 |
| M5 | 3 / 0 | 5 / 0 | 4 / 0 |

Full id and shape_ood tables: `tables.md`. Against the grey-box, M3 beats it on the same cells
where M0 already wins (5/8 on sequence_ood). No memory arm beats the grey-box on either quad_tank
cell or on battery/partial, in any tier.

## quad_tank: upper-tank errors vs the Cramer-Rao bound (sequence_ood, seed means, x_std units)

CRB = empirical-Gramian bound on x(t0) from a window of measurements (DIAG.md (c); no prior, no
process noise).

| scenario | | h3 | h4 |
|---|---|---|---|
| partial | CRB 10 s / 30 s | 0.0801 / 0.0172 | 0.0449 / 0.0135 |
| | M0 | 0.0632 | 0.0280 |
| | M1 / M2 / M4 | 0.0598 / 0.0572 / 0.0544 | 0.0281 / 0.0355 / 0.0350 |
| | **M3** / M5 | **0.0378** / 0.0385 | **0.0235** / 0.0289 |
| | grey-box EKF | 0.0223 | 0.0147 |
| mismatch30 | CRB 10 s / 30 s | 0.0710 / 0.0174 | 0.0558 / 0.0170 |
| | M0 | 0.0263 | 0.0342 |
| | M1 / M2 / M4 | 0.0306 / 0.0316 / 0.0305 | 0.0367 / 0.0534 / 0.0503 |
| | M3 / **M5** | 0.0259 / **0.0223** | **0.0339** / 0.0351 |
| | grey-box EKF | 0.0192 | 0.0147 |

* On partial, the window moves h3 from 0.79x the 10-s CRB to 2.2x the 30-s CRB (-40%), but not to
  1.5x. The grey-box sits at 1.3x / 1.1x the 30-s CRB.
* On mismatch30, h4 does not move with any arm (0.034 = 2.0x CRB_30). The slow-pole arms make it
  worse (0.050-0.053). This scenario's residual is not a memory deficit the latent can fix.

## What the memory arms actually did (post-hoc, not pre-registered)

1. **M4 chose not to slow down.** The learned pole scaling stayed at s ~ 1 on every cell: the seed
   mean of s over validation is 0.92-1.00, and the 5th percentile is >= 0.88. Gradient training
   preferred the fastest allowed latent. M4's results are therefore M2's with an extra
   time-varying gain, and its losses against M3 follow from M2's.
2. **Slower poles make the latent easier to decode but worse to estimate from.** The phase-1 base
   decoder's Lipschitz constant falls 5-10x on quad_tank (l_dec_q95 23 -> 2.2-3.6 for M2), and the
   validation NRMSE improves on quad_tank/partial (0.030 -> 0.023-0.025 for M1/M2/M4). On the
   evaluation tiers, though, sequence_ood is not better and is often worse. A slow mode averages
   the measured levels over 30-100 s, which smears exactly the fast input transitions that
   sequence_ood is made of.
3. **It is not prior forgetting** (`posthoc_prior_quad_tank.json`, `code/posthoc_prior.py`).
   Re-running the trained quad_tank observers with an oracle prior x0_hat = x(0) leaves the
   settled error of every arm unchanged on mismatch30 (M2 0.0331 -> 0.0326). On partial it gets
   worse for M2 / M4 / M5 (0.037 -> 0.048), because those decoders were trained with the
   deployment latent z0 = T0(x_center) and learned its transient. In the error-vs-time profile,
   M2's late-episode error (last 50%) is 0.9-1.1x M0's on partial and 1.4-1.6x on mismatch30. So the slow designs lose at steady state,
   not by remembering a bad prior.
4. **The window works by speeding convergence and fixing h3.** In the error-vs-time deciles on
   quad_tank/partial, M3's RMS at 30-40% of the episode is 0.034 vs M0's 0.082. Its h3 error falls
   from 0.063 to 0.038. The window gives the decoder explicit access to the latent's past (50 s).
   That carries the slow information without slowing the latent itself. On rich plants it also
   helps (vdp -22% to -27%), so it is a general-purpose improvement, not a fix specific to
   window-starved plants.
5. **No room for the optional "best stack" pass.** The validation-best memory arm is M3 on 6/8
   cells and M5 on 2/8. Both are window arms, and A4 already contains a window (6/alpha). So "M* on
   A4" is A4 itself (stored: quad_tank +129% / +75% vs grey-box). The A4+D3 wave of the campaign
   had not finished on these cells.

## Parameters and compute

Phase-2 trainable parameters (conditioner + context + base decoder + extra module) are matched to M0
within 1% on every cell:

| plant | M0 | M3/M5 | M4 |
|---|---|---|---|
| quad_tank | 181,804 | -0.15% | +0.16% |
| battery | 182,417 | +0.54% | +0.16% |
| vdp, bioreactor | 180,886 | -0.84% | +0.16% |

M1 and M2 match exactly. Steps are identical: phase 1 6000 + 6000, phase 2 3000. Wall-clock per
seed on shared GPUs is 250-600 s for phase 1 and 250-600 s for phase 2 (`tables.md`). New training
in this study totals about 16 GPU-process-hours on shared GPUs; M0 was not retrained. The long
phase-1 episode generation took about 5 CPU-minutes in total.

## Caveats

* 3 seeds, so the CIs on bioreactor and quad_tank/partial M4 are wide.
* The slow designs needed a new labelling rule: longer phase-1 episodes, and phase-2 latents started
  from the deployment initial condition (PREREG 2.2). This is a faithful but not the only way to
  train a KKL observer whose washout exceeds the episode. A design that labels the slow modes
  differently might fare better.
* The window goes into the hypernetwork context, not the base decoder (as in the campaign).
* The CRBs ignore the prior and process noise. For mismatch30 they are computed on the perturbed
  true plant.

## Plain insight (5 lines)

1. Longer memory helps HyperKKL only when it comes as an explicit window of past latents (M3, -35% on quad_tank/partial). Slowing the latent poles does not help.
2. Slower poles make the latent smoother and easier to decode, but they blur the fast input changes, and the observer gets worse (M1/M2 significantly worse on 4/8 cells).
3. Given the choice, the network keeps its poles fast (learned s ~ 1), so context-chosen memory (M4) brings nothing.
4. Even the best memory leaves quad_tank 1.6-1.7x behind the grey-box. h3 stays 2.2x above the 30-s bound, and the mismatch30 cell does not move, so a memory budget alone does not explain the gap.
5. The window is a cheap general improvement that also helps rich plants; the grey-box's advantage on quad_tank needs model structure (Stage B) rather than more memory.
