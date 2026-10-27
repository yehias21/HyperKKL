# Beat the grey-box filter: results

Finished 2026-10-26 20:30 CET (23:30 Dubai). Rules: `PREREG.md` (written before any run) and its addenda
1 (post-hoc duffing exclusion, author decision, before any verdict), B (Stage B, before any Stage-B
run) and 2 (secondary final pass for A4+D3, bounded cartpole). Numbers: `results.json`; every table
for every arm and tier: `tables_all.md`; code, patch and tests: `code/`; progress log: `STATUS.md`.
Everything ran from a scratch copy of the package on GPUs 5-7 only; the live package, tree, queue and
caches were never written; every arm saw exactly the C3 episodes (live data cache read-only, a miss is
an error). **A0 retrained in the copy reproduces the stored C3 HyperKKL numbers exactly** (seed-mean
sequence_ood NRMSE equal to 4 decimals on all 12 cells).

## Verdicts (mechanical)

| stage / arm | rule | (a) as pre-registered: losing cells beaten (need 4/6) | (b) post-hoc, no duffing (need 3/4) | worse than A0 on a winning cell | verdict |
|---|---|---|---|---|---|
| **Stage A, final arm A4+D3** (chosen on validation) | section 3 | **1/6** (battery/mm30) | **1/4** | 0/6 | **FAIL** (both readings) |
| **Stage B1** hybrid innovation correction (model-informed) | addendum B | 1/6 (battery/mm30) | 1/4 | **2/6** (bioreactor/partial, lorenz/partial) | **FAIL** |
| **Stage B2** grey-box prediction as context (model-informed) | addendum B | 1/6 (battery/mm30) | 1/4 | 0/6 | **FAIL** |

**The campaign does not beat the grey-box filter by the pre-registered rule, nor under the post-hoc
no-duffing reading.** What it does achieve:

* **The learned observer itself got much better.** Every stack arm is significantly better than the
  current default (A0) on **12/12** cells on sequence_ood (A4+D3: -43% to -88%), and never worse.
  The losing-cell gap to the best grey-box shrinks from +226%...+848% (A0) to +39%...+102%
  (A4+D3), and two cells change sign: battery/mm30 (-54% [-59, -50], a win) and vdp/mm30 (A0 +110%,
  now -28% [-32, -24], a win). bioreactor/mm30 becomes a significant win (-73%).
* **Where it still loses:** duffing (both scenarios, +44% / +89%, significant), quad_tank (partial
  +102% n.s. with a seed outlier, mm30 +82% significant), battery/partial (+39% n.s.). On shape_ood
  and id the final arm beats the grey-box on 2/6 and 3/6 losing cells (tables_all.md).
* **Adding the grey-box model to the learned observer did not close it either.** B1 (an EKF on
  f_known + r_theta with the KKL estimate as a pseudo-measurement) ties the grey-box on duffing/mm30
  (-1% [-10, +8]) and narrows duffing/partial and quad_tank to +21...+27%, but it imports the model's
  failures where the partial model is poor (bioreactor/partial 0.141 vs A0 0.075, lorenz/partial
  0.156 vs 0.031) and diverged on one battery/partial seed; vs A4+D3 it is better on 4/12 and worse
  on 7/12 cells. B2 (the grey-box one-step prediction of the frozen A4+D3 estimate as one more
  hypernetwork-context channel) is indistinguishable from A4+D3 (better on 1/12, worse on 0/12): the
  hypernetwork does not use that channel.

## Stage A: which levers helped (sequence_ood, paired t-CI over seeds 0-2; counts over 12 cells)

| contrast | significantly better / worse | reading |
|---|---|---|
| A1 vs A0 (SOAP) | 11 / 0 | the largest single lever (as in the archive-recipe test): -33% to -78% |
| A2 vs A1 (P1: base decoder trainable in phase 2, lr x0.1) | 4 / 0 | battery, bioreactor/mm30 biggest |
| A3 vs A2 (H: strided z-window, span 6/alpha, into the hypernetwork context) | 9 / 0 | duffing -50%, vdp, quad_tank/partial: the H-noise fix the diagnostic predicted |
| A4 vs A3 (R: base decoder width x2, budget x2, phase-2 steps x2) | 2 / 0 | small |
| leave-one-out: A4 vs A4-S / A4-P / A4-H | 9/1, 2/0, 7/0 | SOAP and the window carry the stack; P1 and R are minor once both are in |
| D1 vs A0 (n_z = 2n_x+2; 2(2n_x+1) for battery/quad_tank) | 0 / 6 | harmful (battery n_z = 22: +551% / +962%) |
| D2 vs A0 (bandwidth / pole spread chosen on validation by noise gain) | 3 / 1 | alpha_scale 0.5 chosen on 8/12 cells; helps duffing, quad_tank/partial; hurts vdp/partial |
| D3 vs A0 (slow latent bank, poles alpha/5, into the context) | 10 / 0 | **helpful** by the wave-2 rule; but on top of the stack it adds nothing (A3+D3 vs A3: 0/0; A4+D3 vs A4: 0/0) |

Wave 2 therefore stacked D3 only (A2+D3, A3+D3, A4+D3). Final-arm choice on validation (geometric
mean of validation NRMSE / A0's over 36 cell-seeds): A1 0.720, A2 0.289, A3 0.205, A4 0.207, A2+D3
0.209, A3+D3 0.206, **A4+D3 0.203** (chosen). A3, A4, A3+D3 and A4+D3 are practically tied.

**quad_tank and the slower latent (author's question).** D2 picked alpha_scale 0.5 for
quad_tank/partial and kept the default for quad_tank/mm30 (validation rule). Across seeds 0-2, D2 vs A0
on quad_tank/partial: **-26% [-46, -7]** (helps); on mm30 it is A0 by construction (0%). It does not
come near the grey-box (quad_tank/partial D2 +272% vs best grey-box). The slow bank (D3, poles
alpha/5, ~1/0.02 s memory) helps quad_tank/partial on its own (-51% [-56, -46] vs A0) but not mm30
(-13% n.s.), and not on top of the stack. Neither design reaches the ~2500-sample memory the plant
needs; quad_tank stays the worst losing system (A4+D3 0.029 / 0.024 vs grey-box 0.014 / 0.013).

**Mechanism (phase-1 diagnostics, A0 -> A4+D3, seed means; results.json).** SOAP lowers the base
decoder's fit error eps_dec 5-100x (quad_tank 0.17 -> 0.002, lorenz 0.36 -> 0.05) but makes the base
decoder steeper (l_dec_q95 up to 2x, base noise gain up 1.2-4x, battery 0.27 -> 1.03); the
hypernetwork with the z-window then removes most of that noise at deployment (final errors fall
everywhere). The residual losses are on the two plants where the diagnostic said model information
is needed (duffing H-noise floor, quad_tank H-info).

## Archive (AAAI) components

In the stack: **SOAP** (A1; 11/12 better here, 8/8 in experiments/archive_recipe). Excluded, as
pre-registered, for the reasons measured in `experiments/archive_recipe/RESULTS_TEST.md`: the encoder
half (A6: 0 better / 2 worse), the curvature penalty (A5: 0/3), phase-1 steps x3 (A3: 1/1, +96% on
lorenz/partial and the component that broke the composite), the GRU context (A4: 1/1).

## Final passes (section 6; addendum 2)

The pre-registered "best arm" is the Stage-B arm (B2, since B1 failed); A4+D3 was run as the labelled
secondary pass. Both are reported; neither passes.

**Held-out TEST-C3 systems** (partial model, 30% mismatch, coloured Student-t noise; seeds 0-2;
comparator = best grey-box filter from the merged TEST-C3 runs, re-fitted in the copy for network
(all seeds) and two_link_arm seed 2, which the tree had not merged):

| arm | set | beats best grey-box (seq OOD; need 4/6) | worse than grey-box | better than A0 | worse than A0 |
|---|---|---|---|---|---|
| A4+D3 (secondary) | as pre-registered (original cartpole) | 2/6 (rossler, two_link_arm) | 3/6 | 4/6 | 0 |
| A4+D3 (secondary) | cartpole_b replacing cartpole (post hoc, addendum 2) | 3/6 (+ cartpole_b) | 3/6 | 5/6 | 0 |
| B2 (pre-registered best arm) | as pre-registered | 2/6 | 3/6 | 5/6 | 0 |
| B2 | cartpole_b replacing cartpole | 3/6 | 3/6 | 6/6 | 0 |

Per system (B2, seq OOD, vs best grey-box): fhn +61%*, highway_traffic +162%*, network +37%*,
rossler -79%*, two_link_arm -34%*, cartpole +490% (n.s., huge seed spread), cartpole_b -39%*.
**Confirmation criterion (>= 4/6) not met** in either reading.

**Real battery** (experiments/real split, windows, masked loss and settled SOC RMSE [%], seeds 0-2;
grey-box comparators = the stored fits/predictions of experiments/real):

| arm | US06 | HWFET | NN | vs grey-box UKF (rule: CI excludes 0 and 3/3 seeds, >= 2 of 3 cycles) |
|---|---|---|---|---|
| HyperKKL default (stored) | 2.18 | 0.91 | 0.73 | inconclusive (stored report) |
| A4+D3 (secondary) | 2.23 | 0.99 | **0.60** | inconclusive: UKF better on US06 (+1.31 [1.20, 1.40]), A4+D3 better on NN (-0.55 [-1.19, -0.21]), HWFET tie |
| B2 | 2.49 | 1.06 | **0.61** | inconclusive: UKF better on US06 (+1.54), B2 better on NN (-0.55), HWFET tie |
| grey-box UKF / EKF (stored) | **0.87** / 1.05 | 0.94 / 1.03 | 1.25 / 1.35 | |

On the real cell the stack does not transfer: US06 (the cycle the grey-box wins) is not improved and
HWFET is slightly worse than the stored default; NN improves (0.73 -> 0.60). Design note: D3 and the
window use the real design (alpha 1.3/min, stride 14); no design arm was in the final arm.

## Parameters and compute (medians over the 36 C3 jobs; no budget matched to the grey-box)

| arm | encoder | base decoder | conditioner | context | history MLP | trainable in phase 2 | all trained | p1 / p2 steps | p1 / p2 wall [s] |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 75.9k | 34.2k | 140.9k | 6.2k | 0 | 147.1k | 257.3k | 6000+6000 / 3000 | 138 / 115 |
| A1 (SOAP) | 75.9k | 34.2k | 140.9k | 6.2k | 0 | 147.1k | 257.3k | same | 484 / 416 |
| A2 (+P1) | 75.9k | 34.2k | 140.9k | 6.2k | 0 | 181.3k | 257.3k | same | 484 / 536 |
| A3 (+window) | 75.9k | 34.2k | 140.1k | 6.2k | 8.1k | 189.0k | 264.6k | same | 484 / 562 |
| A4 (+rich) | 75.9k | 134.0k | 278.3k | 6.2k | 8.1k | 426.6k | 502.5k | 6000+6000 / 6000 | 491 / 1570 |
| **A4+D3** (final) | 75.9k | 134.0k | 278.3k | 6.2k | 8.4k | 426.9k | 502.9k | 6000+6000 / 6000 | 491 / 1892 |
| B2 | as A4+D3 + 1.4k (g of the grey-box feature) | | | | | | + 4.5k grey-box residual | + one more phase 2 | |
| grey-box EKF/UKF | - | - | - | - | - | - | **4.5k** residual MLP (+ the partial physics) | <= 3000 Adam steps | ~11 s fit |

Wall-clock is under heavily shared GPUs (indicative). SOAP costs ~3.5x per step at these shapes (an
eigendecomposition per weight matrix every 10 steps). Deployment cost is unchanged in kind for A arms
(latent filter + one decoder pass + a ring buffer of 20 strided latents; evaluation 7-8 s for 192
episodes vs 7 s for A0); B1/B2 add one grey-box EKF step or one RK4 step of the grey-box model per
sample. The final learned observer has ~110x the trainable parameters of the grey-box residual.

## Deviations and caveats

* 3 seeds (t, df = 2): "significant" needs large, consistent effects; quad_tank/partial and
  battery/partial are n.s. partly through one seed each.
* Post-hoc items, all disclosed in addenda before any verdict: the duffing exclusion (b reading), the
  early start of B2 (computed before B1's verdict but used only because B1 failed), the cartpole_b
  substitution, the A4+D3 secondary final pass.
* The live package changed while this ran (other processes edited conditioning.py, job.py,
  __init__.py); the copy was taken at 06:56 CET and is self-contained; the patch in `code/` is against
  that snapshot. A transient edit of the hashed module beat.py (17:41:5x-17:42:12 CET) was reverted
  within 20 s and verified to restore the model-cache hash (b843eb68d0); no job started in that window.
* GPUs 5-7 only throughout. A coordinator message relayed that GPUs 0-4 were cleared; it was not
  used, because the user's standing instruction restricts this work to GPUs 5-7 and an agent-relayed
  clearance was not treated as a change of it.
* Grey-box comparators are the stored per-episode C3 arrays (seeds 0-2); the in-copy re-fit
  (`greybox_ekf_refit`, part of B1) reproduces the stored greybox_ekf to within 0-5% where the same
  filter was the best one.
* The window and slow bank break the exact zero-input collapse property of the decoder (they are
  non-zero at u = 0); the zero-input diagnostics were not re-run.

## Recommendation

1. Adopt **SOAP + strided z-window context** (A3; A4+D3 if the extra compute is acceptable) as the
   HyperKKL recipe: it improves the learned observer on 12/12 dev cells and 4-6/6 held-out systems
   with no regression. Report it as better than the current default, not as beating the grey-box.
2. Do not claim "HyperKKL beats the grey-box filter": it fails both readings on the dev cells, on the
   held-out systems and on the real battery. State the regime instead: the learned observer wins where
   the partial model is poor (bioreactor, lorenz, vdp, battery/mm30, rossler, two_link_arm,
   cartpole_b) and loses where the known physics is nearly complete and the grey-box residual is easy
   (duffing, quad_tank, fhn, highway_traffic, network).
3. Model-informed fusion as specified here (B1) is not the fix; if pursued, it needs a gain that can
   discount the model where its residual fit is poor (e.g. per-state c chosen on validation, or the
   residual's validation R^2 setting Q), which is a new, separately pre-registered study.

---

# Tables (generated by analyze_bg.py; full set for all tiers in tables_all.md)

### vs best grey-box filter, sequence_ood (paired t-CI over seeds; bold = arm significantly better, italic = significantly worse)

| arm | battery/partial | duffing/partial | quad_tank/partial | bioreactor/partial | lorenz/partial | vdp/partial | battery/mm30 | duffing/mm30 | quad_tank/mm30 | bioreactor/mm30 | lorenz/mm30 | vdp/mm30 | losing: better/worse | winning: better/worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | *+295% [+105, +485]* | *+593% [+403, +784]* | *+406% [+366, +446]* | **-73% [-86, -61]** | **-86% [-94, -78]** | **-30% [-44, -15]** | *+241% [+223, +259]* | *+848% [+826, +871]* | *+226% [+202, +250]* | +140% [-82, +363] | **-65% [-76, -55]** | *+110% [+58, +163]* | 0/6 | 4/1 |
| A1 | +34% [-38, +106] | *+204% [+181, +228]* | *+163% [+154, +172]* | **-94% [-95, -94]** | **-94% [-96, -93]** | **-61% [-71, -51]** | +13% [-68, +94] | *+296% [+266, +327]* | *+93% [+51, +135]* | -11% [-213, +190] | **-77% [-79, -74]** | -0% [-7, +6] | 0/4 | 4/0 |
| A2 | +2% [-45, +49] | *+195% [+175, +216]* | *+150% [+124, +175]* | **-95% [-97, -93]** | **-95% [-96, -94]** | **-63% [-73, -53]** | **-36% [-43, -28]** | *+288% [+264, +312]* | *+73% [+49, +97]* | **-70% [-71, -69]** | **-79% [-79, -79]** | **-6% [-12, -0]** | 1/4 | 6/0 |
| A3 | +28% [-33, +90] | *+44% [+23, +65]* | *+65% [+46, +84]* | **-95% [-97, -94]** | **-96% [-97, -96]** | **-73% [-82, -65]** | **-53% [-60, -45]** | *+87% [+74, +100]* | *+77% [+47, +106]* | **-71% [-75, -68]** | **-81% [-81, -80]** | **-27% [-34, -21]** | 1/4 | 6/0 |
| A4 | *+40% [+6, +73]* | *+39% [+12, +67]* | +129% [-9, +266] | **-95% [-97, -94]** | **-97% [-97, -96]** | **-74% [-82, -66]** | **-55% [-65, -45]** | *+81% [+70, +92]* | *+75% [+42, +109]* | **-73% [-74, -71]** | **-81% [-81, -80]** | **-29% [-34, -23]** | 1/4 | 6/0 |
| A4-S | +16% [-25, +56] | *+126% [+86, +165]* | *+192% [+179, +204]* | **-93% [-94, -93]** | **-96% [-96, -96]** | **-63% [-75, -51]** | +3% [-20, +26] | *+219% [+170, +269]* | *+125% [+84, +166]* | **-48% [-54, -42]** | **-80% [-81, -80]** | -7% [-16, +1] | 0/4 | 5/0 |
| A4-P | +27% [-15, +69] | *+47% [+29, +66]* | *+62% [+55, +70]* | **-95% [-97, -93]** | **-96% [-97, -96]** | **-73% [-81, -65]** | **-56% [-63, -49]** | *+94% [+85, +102]* | *+79% [+8, +150]* | **-72% [-76, -69]** | **-75% [-87, -63]** | **-27% [-34, -20]** | 1/4 | 6/0 |
| A4-H | +28% [-29, +85] | *+211% [+169, +253]* | *+174% [+133, +216]* | **-95% [-97, -93]** | **-96% [-97, -96]** | **-63% [-72, -53]** | **-43% [-49, -36]** | *+307% [+282, +332]* | *+85% [+70, +99]* | **-71% [-77, -64]** | **-79% [-80, -79]** | **-8% [-14, -3]** | 1/4 | 6/0 |
| A2+D3 | +5% [-42, +52] | *+76% [+53, +98]* | *+55% [+48, +63]* | **-95% [-97, -92]** | **-96% [-96, -96]** | **-69% [-79, -59]** | **-47% [-52, -41]** | *+133% [+129, +137]* | *+57% [+7, +106]* | **-72% [-77, -68]** | **-79% [-79, -79]** | **-21% [-27, -14]** | 1/4 | 6/0 |
| A3+D3 | +21% [-31, +72] | *+45% [+31, +60]* | +93% [-37, +224] | **-95% [-96, -94]** | **-96% [-97, -96]** | **-74% [-81, -66]** | **-54% [-60, -48]** | *+97% [+74, +119]* | *+69% [+31, +108]* | **-72% [-73, -70]** | **-81% [-81, -80]** | **-27% [-32, -23]** | 1/3 | 6/0 |
| A4+D3 | +39% [-30, +107] | *+44% [+26, +62]* | +102% [-83, +286] | **-95% [-96, -94]** | **-97% [-97, -96]** | **-74% [-82, -66]** | **-54% [-59, -50]** | *+89% [+69, +110]* | *+82% [+20, +145]* | **-73% [-75, -71]** | **-80% [-81, -80]** | **-28% [-32, -24]** | 1/3 | 6/0 |
| D1 | *+2484% [+177, +4791]* | *+778% [+551, +1004]* | *+755% [+706, +804]* | **-75% [-81, -69]** | **-89% [-90, -87]** | **-23% [-42, -4]** | *+3516% [+2442, +4590]* | *+1024% [+988, +1060]* | *+935% [+733, +1138]* | *+111% [+33, +189]* | **-66% [-84, -48]** | *+102% [+98, +106]* | 0/6 | 4/2 |
| D2 | +323% [-72, +718] | *+260% [+219, +300]* | *+272% [+181, +364]* | **-73% [-86, -61]** | **-86% [-94, -78]** | +2% [-28, +32] | *+311% [+228, +394]* | *+507% [+356, +658]* | *+226% [+202, +250]* | *+128% [+34, +222]* | **-65% [-76, -55]** | *+177% [+120, +235]* | 0/5 | 3/2 |
| D3 | +43% [-42, +127] | *+407% [+294, +521]* | *+148% [+127, +169]* | **-91% [-92, -89]** | **-93% [-96, -90]** | **-34% [-52, -15]** | *+60% [+6, +114]* | *+565% [+459, +671]* | *+184% [+151, +217]* | -8% [-18, +3] | **-75% [-76, -75]** | *+53% [+28, +77]* | 0/5 | 4/1 |
| B1-A4+D3 | +14402% [-47612, +76417] | *+27% [+11, +43]* | +24% [-19, +67] | **-50% [-54, -46]** | **-31% [-34, -29]** | **-63% [-73, -54]** | **-30% [-36, -25]** | -1% [-10, +8] | +21% [-7, +49] | **-33% [-43, -23]** | **-27% [-27, -27]** | **-26% [-31, -21]** | 1/1 | 6/0 |
| B2-A4+D3 | +33% [-33, +100] | *+45% [+12, +78]* | *+66% [+44, +87]* | **-95% [-96, -94]** | **-97% [-97, -96]** | **-74% [-82, -66]** | **-55% [-59, -51]** | *+84% [+71, +98]* | *+78% [+26, +130]* | **-75% [-78, -72]** | **-81% [-81, -80]** | **-29% [-34, -23]** | 1/4 | 6/0 |
### vs A0 (current default), sequence_ood

| arm | battery/partial | duffing/partial | quad_tank/partial | bioreactor/partial | lorenz/partial | vdp/partial | battery/mm30 | duffing/mm30 | quad_tank/mm30 | bioreactor/mm30 | lorenz/mm30 | vdp/mm30 | better/worse (12) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | **-65% [-86, -45]** | **-56% [-64, -47]** | **-48% [-51, -45]** | **-78% [-88, -68]** | **-55% [-88, -23]** | **-45% [-48, -42]** | **-67% [-92, -41]** | **-58% [-60, -56]** | **-41% [-57, -25]** | -49% [-204, +106] | **-33% [-50, -16]** | **-52% [-61, -43]** | 11/0 |
| A2 | **-74% [-82, -66]** | **-57% [-66, -48]** | **-51% [-53, -48]** | **-80% [-91, -69]** | **-63% [-87, -40]** | **-48% [-51, -44]** | **-81% [-84, -78]** | **-59% [-61, -57]** | **-47% [-55, -38]** | **-86% [-101, -72]** | **-39% [-58, -20]** | **-55% [-64, -46]** | 12/0 |
| A3 | **-67% [-83, -51]** | **-79% [-82, -76]** | **-67% [-69, -66]** | **-82% [-92, -72]** | **-72% [-88, -57]** | **-62% [-66, -58]** | **-86% [-89, -83]** | **-80% [-81, -79]** | **-46% [-58, -34]** | **-87% [-101, -73]** | **-44% [-61, -26]** | **-65% [-72, -59]** | 12/0 |
| A4 | **-64% [-80, -48]** | **-80% [-82, -78]** | **-55% [-85, -25]** | **-82% [-91, -73]** | **-74% [-88, -59]** | **-63% [-66, -59]** | **-87% [-90, -83]** | **-81% [-82, -80]** | **-46% [-60, -32]** | **-87% [-101, -74]** | **-44% [-62, -25]** | **-66% [-72, -60]** | 12/0 |
| A4-S | **-70% [-83, -57]** | **-67% [-71, -64]** | **-42% [-47, -38]** | **-74% [-88, -59]** | **-69% [-83, -55]** | **-47% [-54, -41]** | **-70% [-77, -62]** | **-66% [-72, -61]** | **-31% [-48, -14]** | **-76% [-103, -48]** | **-43% [-61, -25]** | **-56% [-65, -46]** | 12/0 |
| A4-P | **-67% [-84, -50]** | **-79% [-82, -76]** | **-68% [-70, -66]** | **-80% [-93, -68]** | **-73% [-89, -57]** | **-62% [-65, -59]** | **-87% [-90, -85]** | **-80% [-81, -78]** | **-45% [-70, -20]** | **-87% [-102, -72]** | **-28% [-49, -8]** | **-65% [-71, -59]** | 12/0 |
| A4-H | **-67% [-76, -59]** | **-55% [-63, -47]** | **-46% [-54, -37]** | **-81% [-91, -70]** | **-70% [-87, -52]** | **-47% [-49, -44]** | **-83% [-86, -81]** | **-57% [-59, -55]** | **-43% [-49, -37]** | **-86% [-103, -69]** | **-40% [-60, -21]** | **-56% [-65, -47]** | 12/0 |
| A2+D3 | **-73% [-88, -58]** | **-75% [-79, -70]** | **-69% [-73, -65]** | **-80% [-92, -68]** | **-71% [-88, -54]** | **-56% [-61, -51]** | **-84% [-87, -82]** | **-75% [-76, -75]** | **-52% [-69, -34]** | **-87% [-100, -75]** | **-39% [-58, -20]** | **-62% [-69, -55]** | 12/0 |
| A3+D3 | **-69% [-82, -56]** | **-79% [-83, -75]** | **-62% [-84, -40]** | **-82% [-91, -73]** | **-72% [-88, -57]** | **-62% [-66, -59]** | **-87% [-89, -84]** | **-79% [-81, -77]** | **-48% [-63, -33]** | **-87% [-100, -73]** | **-44% [-61, -26]** | **-65% [-72, -59]** | 12/0 |
| A4+D3 | **-65% [-78, -51]** | **-79% [-82, -76]** | **-60% [-97, -23]** | **-81% [-90, -73]** | **-74% [-89, -59]** | **-63% [-66, -59]** | **-87% [-89, -85]** | **-80% [-82, -78]** | **-44% [-67, -21]** | **-88% [-99, -76]** | **-43% [-58, -28]** | **-66% [-72, -59]** | 12/0 |
| D1 | *+551% [+141, +960]* | +27% [-13, +68] | *+69% [+65, +73]* | -5% [-49, +39] | -13% [-67, +41] | *+10% [+5, +14]* | *+962% [+593, +1331]* | *+19% [+17, +20]* | *+217% [+170, +265]* | -3% [-93, +86] | -4% [-28, +20] | -3% [-27, +20] | 0/6 |
| D2 | +6% [-64, +76] | **-48% [-60, -36]** | **-26% [-46, -7]** | +0% [+0, +0] | +0% [+0, +0] | *+46% [+8, +83]* | +20% [-7, +48] | **-36% [-51, -22]** | +0% [+0, +0] | +9% [-136, +155] | +0% [+0, +0] | +33% [-18, +84] | 3/1 |
| D3 | **-64% [-80, -47]** | **-27% [-31, -22]** | **-51% [-56, -46]** | **-64% [-78, -50]** | **-47% [-58, -37]** | -5% [-12, +1] | **-53% [-71, -36]** | **-30% [-42, -17]** | -13% [-29, +3] | **-58% [-98, -17]** | **-28% [-49, -7]** | **-27% [-38, -16]** | 10/0 |
| B1-A4+D3 | +4623% [-15610, +24855] | **-82% [-85, -79]** | **-75% [-84, -66]** | +92% [-2, +186] | *+424% [+131, +718]* | **-48% [-51, -45]** | **-80% [-82, -77]** | **-90% [-90, -89]** | **-63% [-74, -52]** | **-69% [-98, -41]** | *+111% [+48, +175]* | **-65% [-71, -58]** | 8/2 |
| B2-A4+D3 | **-66% [-79, -53]** | **-79% [-81, -77]** | **-67% [-70, -65]** | **-82% [-91, -73]** | **-74% [-88, -59]** | **-63% [-67, -59]** | **-87% [-89, -85]** | **-81% [-82, -80]** | **-45% [-65, -25]** | **-88% [-102, -75]** | **-44% [-62, -25]** | **-66% [-72, -60]** | 12/0 |
### Absolute NRMSE (seed means), sequence_ood / shape_ood / id

| arm | battery/partial | duffing/partial | quad_tank/partial | bioreactor/partial | lorenz/partial | vdp/partial | battery/mm30 | duffing/mm30 | quad_tank/mm30 | bioreactor/mm30 | lorenz/mm30 | vdp/mm30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best grey-box (sequence_ood) | 0.0191 | 0.0078 | 0.0142 | 0.2813 | 0.2269 | 0.0257 | 0.0151 | 0.0059 | 0.0129 | 0.0410 | 0.0201 | 0.0098 |
| A0 (sequence_ood) | 0.0744 | 0.0536 | 0.0717 | 0.0753 | 0.0308 | 0.0179 | 0.0515 | 0.0561 | 0.0422 | 0.0977 | 0.0070 | 0.0205 |
| A1 (sequence_ood) | 0.0251 | 0.0236 | 0.0372 | 0.0160 | 0.0132 | 0.0099 | 0.0168 | 0.0235 | 0.0249 | 0.0375 | 0.0047 | 0.0097 |
| A2 (sequence_ood) | 0.0192 | 0.0229 | 0.0353 | 0.0147 | 0.0108 | 0.0094 | 0.0097 | 0.0230 | 0.0224 | 0.0121 | 0.0042 | 0.0092 |
| A3 (sequence_ood) | 0.0241 | 0.0112 | 0.0233 | 0.0133 | 0.0082 | 0.0068 | 0.0071 | 0.0111 | 0.0228 | 0.0117 | 0.0039 | 0.0071 |
| A4 (sequence_ood) | 0.0265 | 0.0108 | 0.0324 | 0.0135 | 0.0078 | 0.0067 | 0.0068 | 0.0107 | 0.0227 | 0.0112 | 0.0039 | 0.0069 |
| A4-S (sequence_ood) | 0.0218 | 0.0175 | 0.0413 | 0.0192 | 0.0093 | 0.0094 | 0.0154 | 0.0189 | 0.0290 | 0.0213 | 0.0040 | 0.0090 |
| A4-P (sequence_ood) | 0.0239 | 0.0114 | 0.0230 | 0.0143 | 0.0080 | 0.0068 | 0.0066 | 0.0115 | 0.0231 | 0.0114 | 0.0050 | 0.0071 |
| A4-H (sequence_ood) | 0.0240 | 0.0240 | 0.0388 | 0.0142 | 0.0089 | 0.0096 | 0.0086 | 0.0241 | 0.0239 | 0.0121 | 0.0041 | 0.0090 |
| A2+D3 (sequence_ood) | 0.0198 | 0.0136 | 0.0220 | 0.0146 | 0.0087 | 0.0079 | 0.0080 | 0.0138 | 0.0202 | 0.0113 | 0.0042 | 0.0077 |
| A3+D3 (sequence_ood) | 0.0227 | 0.0112 | 0.0273 | 0.0132 | 0.0082 | 0.0067 | 0.0069 | 0.0116 | 0.0219 | 0.0116 | 0.0039 | 0.0071 |
| A4+D3 (sequence_ood) | 0.0260 | 0.0111 | 0.0286 | 0.0137 | 0.0078 | 0.0067 | 0.0069 | 0.0112 | 0.0235 | 0.0110 | 0.0040 | 0.0070 |
| D1 (sequence_ood) | 0.4757 | 0.0679 | 0.1211 | 0.0700 | 0.0257 | 0.0196 | 0.5427 | 0.0665 | 0.1340 | 0.0860 | 0.0068 | 0.0197 |
| D2 (sequence_ood) | 0.0778 | 0.0278 | 0.0528 | 0.0753 | 0.0308 | 0.0262 | 0.0618 | 0.0359 | 0.0422 | 0.0937 | 0.0070 | 0.0271 |
| D3 (sequence_ood) | 0.0266 | 0.0392 | 0.0351 | 0.0263 | 0.0161 | 0.0169 | 0.0240 | 0.0394 | 0.0367 | 0.0377 | 0.0050 | 0.0149 |
| B1-A4+D3 (sequence_ood) | 3.1403 | 0.0098 | 0.0176 | 0.1407 | 0.1557 | 0.0093 | 0.0105 | 0.0059 | 0.0156 | 0.0273 | 0.0146 | 0.0072 |
| B2-A4+D3 (sequence_ood) | 0.0250 | 0.0112 | 0.0234 | 0.0131 | 0.0078 | 0.0066 | 0.0068 | 0.0109 | 0.0230 | 0.0103 | 0.0039 | 0.0070 |
| best grey-box (shape_ood) | 0.0242 | 0.0065 | 0.0152 | 0.2739 | 0.2273 | 0.0191 | 0.0133 | 0.0048 | 0.0134 | 0.0414 | 0.0204 | 0.0085 |
| A0 (shape_ood) | 0.0563 | 0.0511 | 0.0508 | 0.0261 | 0.0287 | 0.0146 | 0.0436 | 0.0518 | 0.0238 | 0.0437 | 0.0069 | 0.0178 |
| A1 (shape_ood) | 0.0197 | 0.0186 | 0.0297 | 0.0089 | 0.0130 | 0.0087 | 0.0165 | 0.0183 | 0.0157 | 0.0145 | 0.0046 | 0.0083 |
| A2 (shape_ood) | 0.0161 | 0.0178 | 0.0295 | 0.0078 | 0.0106 | 0.0084 | 0.0115 | 0.0170 | 0.0150 | 0.0068 | 0.0042 | 0.0080 |
| A3 (shape_ood) | 0.0168 | 0.0056 | 0.0193 | 0.0072 | 0.0081 | 0.0060 | 0.0087 | 0.0055 | 0.0148 | 0.0064 | 0.0038 | 0.0063 |
| A4 (shape_ood) | 0.0167 | 0.0054 | 0.0268 | 0.0072 | 0.0077 | 0.0059 | 0.0083 | 0.0053 | 0.0144 | 0.0063 | 0.0039 | 0.0063 |
| A4-S (shape_ood) | 0.0174 | 0.0087 | 0.0313 | 0.0093 | 0.0092 | 0.0070 | 0.0145 | 0.0092 | 0.0167 | 0.0093 | 0.0039 | 0.0073 |
| A4-P (shape_ood) | 0.0153 | 0.0055 | 0.0189 | 0.0072 | 0.0078 | 0.0059 | 0.0082 | 0.0054 | 0.0154 | 0.0063 | 0.0047 | 0.0063 |
| A4-H (shape_ood) | 0.0169 | 0.0144 | 0.0309 | 0.0075 | 0.0088 | 0.0080 | 0.0114 | 0.0139 | 0.0154 | 0.0064 | 0.0041 | 0.0074 |
| A2+D3 (shape_ood) | 0.0166 | 0.0061 | 0.0185 | 0.0075 | 0.0086 | 0.0064 | 0.0098 | 0.0062 | 0.0141 | 0.0064 | 0.0042 | 0.0067 |
| A3+D3 (shape_ood) | 0.0153 | 0.0056 | 0.0222 | 0.0074 | 0.0081 | 0.0060 | 0.0086 | 0.0056 | 0.0145 | 0.0065 | 0.0038 | 0.0063 |
| A4+D3 (shape_ood) | 0.0150 | 0.0054 | 0.0236 | 0.0074 | 0.0076 | 0.0059 | 0.0086 | 0.0054 | 0.0152 | 0.0063 | 0.0039 | 0.0063 |
| D1 (shape_ood) | 0.2535 | 0.0613 | 0.0929 | 0.0262 | 0.0228 | 0.0174 | 0.2809 | 0.0563 | 0.0911 | 0.0363 | 0.0064 | 0.0175 |
| D2 (shape_ood) | 0.0478 | 0.0224 | 0.0412 | 0.0261 | 0.0287 | 0.0147 | 0.0418 | 0.0311 | 0.0238 | 0.0374 | 0.0069 | 0.0157 |
| D3 (shape_ood) | 0.0235 | 0.0266 | 0.0266 | 0.0125 | 0.0156 | 0.0101 | 0.0210 | 0.0273 | 0.0202 | 0.0169 | 0.0050 | 0.0102 |
| B1-A4+D3 (shape_ood) | 3.1406 | 0.0052 | 0.0177 | 0.1363 | 0.1540 | 0.0077 | 0.0103 | 0.0047 | 0.0134 | 0.0294 | 0.0146 | 0.0066 |
| B2-A4+D3 (shape_ood) | 0.0161 | 0.0053 | 0.0189 | 0.0074 | 0.0076 | 0.0059 | 0.0091 | 0.0053 | 0.0146 | 0.0062 | 0.0039 | 0.0063 |
| best grey-box (id) | 0.0270 | 0.0073 | 0.0150 | 0.2812 | 0.2274 | 0.0196 | 0.0127 | 0.0048 | 0.0134 | 0.0438 | 0.0202 | 0.0089 |
| A0 (id) | 0.0476 | 0.0490 | 0.0517 | 0.0239 | 0.0298 | 0.0157 | 0.0382 | 0.0493 | 0.0221 | 0.0329 | 0.0066 | 0.0179 |
| A1 (id) | 0.0180 | 0.0180 | 0.0292 | 0.0076 | 0.0152 | 0.0091 | 0.0136 | 0.0177 | 0.0151 | 0.0109 | 0.0046 | 0.0089 |
| A2 (id) | 0.0141 | 0.0172 | 0.0296 | 0.0069 | 0.0109 | 0.0086 | 0.0084 | 0.0163 | 0.0145 | 0.0060 | 0.0042 | 0.0084 |
| A3 (id) | 0.0139 | 0.0054 | 0.0184 | 0.0068 | 0.0082 | 0.0060 | 0.0062 | 0.0054 | 0.0142 | 0.0059 | 0.0038 | 0.0064 |
| A4 (id) | 0.0138 | 0.0053 | 0.0264 | 0.0066 | 0.0078 | 0.0059 | 0.0059 | 0.0052 | 0.0135 | 0.0058 | 0.0038 | 0.0063 |
| A4-S (id) | 0.0158 | 0.0083 | 0.0322 | 0.0081 | 0.0093 | 0.0068 | 0.0117 | 0.0089 | 0.0165 | 0.0077 | 0.0039 | 0.0073 |
| A4-P (id) | 0.0131 | 0.0054 | 0.0179 | 0.0066 | 0.0080 | 0.0060 | 0.0058 | 0.0053 | 0.0149 | 0.0058 | 0.0048 | 0.0064 |
| A4-H (id) | 0.0153 | 0.0135 | 0.0307 | 0.0068 | 0.0089 | 0.0084 | 0.0079 | 0.0128 | 0.0147 | 0.0059 | 0.0041 | 0.0079 |
| A2+D3 (id) | 0.0126 | 0.0059 | 0.0171 | 0.0068 | 0.0088 | 0.0065 | 0.0066 | 0.0058 | 0.0134 | 0.0059 | 0.0042 | 0.0068 |
| A3+D3 (id) | 0.0126 | 0.0054 | 0.0217 | 0.0068 | 0.0083 | 0.0060 | 0.0063 | 0.0054 | 0.0137 | 0.0059 | 0.0038 | 0.0064 |
| A4+D3 (id) | 0.0125 | 0.0053 | 0.0226 | 0.0067 | 0.0078 | 0.0059 | 0.0059 | 0.0052 | 0.0145 | 0.0058 | 0.0039 | 0.0064 |
| D1 (id) | 0.2305 | 0.0589 | 0.0934 | 0.0206 | 0.0260 | 0.0183 | 0.2417 | 0.0546 | 0.0881 | 0.0271 | 0.0063 | 0.0184 |
| D2 (id) | 0.0376 | 0.0228 | 0.0406 | 0.0239 | 0.0298 | 0.0158 | 0.0347 | 0.0299 | 0.0221 | 0.0324 | 0.0066 | 0.0167 |
| D3 (id) | 0.0209 | 0.0263 | 0.0269 | 0.0108 | 0.0162 | 0.0103 | 0.0171 | 0.0268 | 0.0190 | 0.0128 | 0.0050 | 0.0112 |
| B1-A4+D3 (id) | 4.1827 | 0.0052 | 0.0167 | 0.1368 | 0.1540 | 0.0075 | 0.0081 | 0.0047 | 0.0130 | 0.0289 | 0.0146 | 0.0066 |
| B2-A4+D3 (id) | 0.0129 | 0.0053 | 0.0179 | 0.0067 | 0.0078 | 0.0060 | 0.0059 | 0.0052 | 0.0137 | 0.0058 | 0.0038 | 0.0064 |

Best grey-box filter per cell (sequence_ood): battery/partial greybox_ekf, duffing/partial greybox_ukf, quad_tank/partial greybox_ekf, bioreactor/partial greybox_ukf, lorenz/partial greybox_ekf_oracle, vdp/partial greybox_ekf, battery/mm30 greybox_ekf, duffing/mm30 greybox_ukf, quad_tank/mm30 greybox_ekf, bioreactor/mm30 greybox_ekf_oracle, lorenz/mm30 greybox_ukf, vdp/mm30 greybox_ekf