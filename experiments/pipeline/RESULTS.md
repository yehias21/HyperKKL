# Pipeline study: answer first, mechanical tables below

**Asked.** Is the two-phase, frozen-base pipeline (phase 1 PINN encoder + labels and decoder; phase 2
conditioner with the base frozen) optimal? Or should the base be unfrozen, or the whole observer
trained end to end? The rules were pre-registered in `PREREG.md`, and the Stage-1 prediction is in
`EVIDENCE.md`. 54 jobs: 3 scenarios x 6 dev systems x 3 seeds, all finished. P0 reproduces the
tree runs exactly (max relative per-episode difference 0 over 54 runs). Equal compute: 15000
loss-batch gradient evaluations per arm.

## Verdicts (sequence_ood; BETTER = paired CI in favour on >= 4/6 systems and worse on none)

| arm | C3-partial | C3-mismatch30 | C1-where (exact) |
|---|---|---|---|
| P1-lora (unfreeze base decoder, lr x0.1) | NOT BETTER (better 5/6, **worse on vdp** +22%) | NOT BETTER (better 3/6, worse 0) | **BETTER** (5/6, worse 0) |
| P1-hyper | NOT BETTER (2/6, worse 0) | **BETTER** (5/6, worse 0) | NOT BETTER (3/6, worse 0) |
| P2-lora (joint from scratch) | NOT BETTER (1/6) | NOT BETTER (1/6, worse 1) | NOT BETTER (1/6, worse 1) |
| P2-hyper | NOT BETTER (1/6, worse 1) | NOT BETTER (1/6, worse 1) | NOT BETTER (1/6, **worse 4/6**) |
| P3 (unconditioned fine-tune) | **WORSE** (worse 5/6) | **WORSE** (4/6) | **WORSE** (5/6) |
| P4-lora (grey-box PDE) | NOT BETTER (0/6, worse 0) | NOT BETTER (0/6) | - |
| P4-hyper | NOT BETTER (0/6) | NOT BETTER (1/6) | - |

**By the pre-registered rule, two-phase frozen is NOT optimal.** P1 (unfreeze the base decoder in
phase 2 at lr x0.1) is BETTER in 2 of 6 arm-kind x scenario cells (mismatch30/hyper, exact/lora).
Across all 36 P1 system-cells it is significantly better in 23 and significantly worse in 1
(C3-partial/vdp/lora, +22%). The gains are -11% to -52% on battery, bioreactor, duffing, lorenz and
quad_tank; on vdp it is neutral. It costs nothing extra: same steps, and ~35k more trainable
parameters (the base decoder).

* **P2 (end to end from scratch) does not help.** It wins only on duffing, by 27-42% in every
  scenario. It is worse or unstable elsewhere: bioreactor +80-115%, lorenz with huge seed variance,
  quad_tank +6-34%. That holds even with 5000 phase-2 evaluations against P0's 3000. The phase-1
  pretraining is worth keeping.
* **P3 rules out "it was only base repair".** An unconditioned fine-tune at the same budget is 3-9x
  worse than P0 and P1 (+180% to +720% vs P0-best). It is still 26-97% better than the autonomous
  base. So the input conditioning carries the C1 effect. P1's gain comes from letting the base
  features adapt *together with* the conditioner, not from repairing the base alone.
* **P4 (grey-box PDE) does nothing.** As predicted in Stage 1, it moves the PDE residual and eps_dec
  but not the deployed error: 0/6 significant on partial, and only bioreactor/hyper on mismatch30.

## Does anything close the gap to the data-driven baselines? No.

| scenario | loss cells (P0 significantly worse than the best data-driven) | closed by any arm | narrowed (significantly better than P0) |
|---|---|---|---|
| C3-partial | battery, bioreactor, duffing, quad_tank (4) | **0** | P1 on battery (+276% -> +135%) and duffing (+593% -> +437%); P2 on duffing (+377%) |
| C3-mismatch30 | battery, duffing, quad_tank, **vdp** (4) | **0** | P1 on battery (+241% -> +97%), duffing (+848% -> +624%) and quad_tank (+226% -> +178%); P2 on duffing |

New in this run: the grey-box/Koopman baselines are evaluated on C3-mismatch30, which the tree has
not merged yet. **On mismatch30 greybox_ekf/ukf beat KKL on battery, duffing, quad_tank and vdp**
(0.0151 / 0.0059 / 0.0129 / 0.0098 vs P0 0.0515 / 0.0561 / 0.0422 / 0.0205). The tree's
`C3-mismatch30 (b)` is therefore expected to come out FAIL once merged: KKL keeps a win only on lorenz
(a tie with NLOX), and the bioreactor cell is non-significant. All four loss cells stay loss cells
under every arm. KKL's error floor of ~0.04-0.07 moves, by at most ~45%, but it is not broken.

## Implications

* **Recommendation:** replace "frozen phase 2" with "phase 2 with the base decoder co-trained at lr
  x0.1" (P1) as the default pipeline. Keep phase 1 as pretraining. Do not go end to end (P2). The
  default should not be flipped silently, though: P1 hurts vdp under C3-partial (lora +22%), and only
  one learning-rate factor was tried.
* **Cost: the exact zero-input collapse.** P1's zero-input map leaves phase 1's T0+ by up to 0.2-3.6
  sigma (max). Structurally, Delta(c=0) is still exactly 0 *relative to its own base*. Its zero-input
  NRMSE is in fact *better* than the frozen base: e.g. lorenz 0.63 -> 0.07, bioreactor 0.066 -> 0.011
  (C3-partial). The "collapses exactly to the autonomous observer" property has to be restated as
  "collapses to the co-trained base", not dropped.
* **C1 is unchanged in direction, with one caveat.** P3 (no input) is far worse than every
  conditioned arm, and worse than the tree's concat on 5/6 and latent on 5/6 exact-model systems
  (lorenz excepted). So the C1 effect is input conditioning, not base repair, which answers Park R4
  (fine-tune half). The lora-vs-hyper tie stays a tie under P1 (|delta| <= 14% almost everywhere).
  Caveat: latent, concat and film were not re-run with an unfrozen base, so whether unfreezing changes
  the *location ranking* is untested. That needs a C1-where-P1 node (lr x0.1 for every location).
* **C3:** the headline vs model-based filters is unaffected (P1 only lowers KKL's error). C3(b), vs
  data-driven baselines, fails on partial and is now also expected to fail on mismatch30. No pipeline
  change rescues it. The losses are a KKL error floor, not base-map quality (`EVIDENCE.md` (a)).

## Caveats

3 seeds (t df = 2): "significant" needs large, consistent effects, and P2's large seed variance on
lorenz and bioreactor reads as non-significant. Vote counts are not multiplicity-corrected. The P1
factor (0.1) and the P2 lr are single, untuned settings. Best-of selections are made on the test
episodes. Wall-clock is under shared GPUs. Per user update, GPUs 0-4 were also used before 05:45 CET;
all jobs finished by 01:17 CET.

---

# Pipeline study: results (mechanical verdicts of PREREG.md)

Generated by `analyze.py` (scratch copy; mirrored in `experiments/pipeline/code/`) from `experiments/pipeline/runs/*/*/seed*/results.json` plus the tree runs they pair with. Relative change of mean episode NRMSE vs the reference, Student-t 95% CI over seeds 0-2, `*` = CI excludes 0. Negative = the arm is better.

Cells complete (all 3 seeds): 18/18.

## C3-partial

### sequence_ood

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P4-lora | P4-hyper | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0744 | 0.0717 | 1.669 | 0.0441 -40% [-60, -20]* | 0.0502 -29% [-53, -6]* | 0.0701 -5% [-36, +27] | 0.0890 +24% [-30, +78] | 0.0739 -2% [-42, +38] | 0.0766 +6% [-0, +13] | 0.4315 +510% [+291, +728]* |
| bioreactor | 0.0753 | 0.0703 | 4.550 | 0.0472 -36% [-62, -10]* | 0.0492 -28% [-60, +4] | 0.1323 +82% [-26, +190] | 0.1392 +102% [+41, +162]* | 0.0630 -14% [-64, +37] | 0.0686 -1% [-33, +31] | 0.1928 +183% [+37, +329]* |
| duffing | 0.0536 | 0.0642 | 0.313 | 0.0436 -18% [-36, -0]* | 0.0416 -35% [-41, -29]* | 0.0390 -27% [-51, -2]* | 0.0370 -42% [-49, -36]* | 0.0530 -1% [-18, +17] | 0.0612 -5% [-14, +5] | 0.2151 +302% [+241, +364]* |
| lorenz | 0.0308 | 0.0192 | 0.641 | 0.0166 -45% [-65, -24]* | 0.0164 -13% [-59, +33] | 0.0421 +45% [-194, +283] | 0.0417 +117% [-100, +333] | 0.0279 -7% [-51, +37] | 0.0158 -17% [-39, +5] | 0.0308 +63% [-29, +156] |
| quad_tank | 0.0717 | 0.0625 | 1.603 | 0.0607 -15% [-26, -5]* | 0.0601 -4% [-10, +3] | 0.0717 +0% [-5, +5] | 0.0680 +9% [-1, +19] | 0.0680 -5% [-12, +2] | 0.0638 +2% [-13, +17] | 0.3458 +454% [+397, +511]* |
| vdp | 0.0179 | 0.0199 | 0.102 | 0.0219 +22% [+5, +40]* | 0.0220 +11% [-6, +28] | 0.0193 +7% [-12, +27] | 0.0212 +7% [-7, +20] | 0.0196 +10% [-23, +42] | 0.0238 +20% [-1, +40] | 0.0759 +324% [+287, +360]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### shape_ood

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P4-lora | P4-hyper | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0563 | 0.0474 | 0.864 | 0.0376 -31% [-68, +6] | 0.0416 -12% [-33, +9] | 0.0605 +11% [-49, +71] | 0.0640 +36% [-26, +97] | 0.0569 +2% [-13, +17] | 0.0503 +6% [-9, +20] | 0.2841 +502% [+376, +629]* |
| bioreactor | 0.0261 | 0.0338 | 2.881 | 0.0187 -26% [-71, +18] | 0.0211 -38% [-42, -34]* | 0.0535 +110% [-14, +235] | 0.0527 +56% [+30, +82]* | 0.0255 -2% [-22, +19] | 0.0277 -18% [-30, -6]* | 0.0800 +215% [+54, +375]* |
| duffing | 0.0511 | 0.0531 | 0.189 | 0.0348 -31% [-56, -7]* | 0.0319 -40% [-54, -26]* | 0.0326 -35% [-64, -7]* | 0.0311 -41% [-57, -25]* | 0.0474 -7% [-30, +16] | 0.0537 +1% [-6, +8] | 0.1371 +171% [+98, +244]* |
| lorenz | 0.0287 | 0.0184 | 0.636 | 0.0163 -42% [-61, -23]* | 0.0161 -10% [-65, +46] | 0.0413 +50% [-195, +296] | 0.0415 +124% [-94, +341] | 0.0253 -10% [-49, +29] | 0.0152 -16% [-45, +13] | 0.0302 +68% [-32, +168] |
| quad_tank | 0.0508 | 0.0496 | 1.063 | 0.0503 -1% [-5, +3] | 0.0516 +4% [-1, +9] | 0.0530 +4% [+1, +8]* | 0.0511 +3% [-5, +11] | 0.0504 -1% [-4, +2] | 0.0506 +2% [-2, +6] | 0.1955 +294% [+263, +325]* |
| vdp | 0.0146 | 0.0167 | 0.074 | 0.0180 +24% [+8, +40]* | 0.0189 +13% [-5, +31] | 0.0151 +4% [-9, +17] | 0.0173 +4% [-16, +23] | 0.0163 +12% [-20, +44] | 0.0216 +29% [+7, +51]* | 0.0496 +240% [+223, +256]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### id

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P4-lora | P4-hyper | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0476 | 0.0436 | 0.900 | 0.0334 -29% [-52, -7]* | 0.0384 -11% [-36, +14] | 0.0530 +13% [-29, +54] | 0.0578 +34% [-28, +96] | 0.0473 -0% [-9, +9] | 0.0448 +3% [-12, +17] | 0.2946 +581% [+410, +752]* |
| bioreactor | 0.0239 | 0.0291 | 2.624 | 0.0150 -36% [-65, -8]* | 0.0167 -42% [-55, -30]* | 0.0408 +73% [+3, +144]* | 0.0399 +37% [+5, +70]* | 0.0176 -26% [-52, +1] | 0.0224 -23% [-36, -10]* | 0.0666 +183% [+75, +291]* |
| duffing | 0.0490 | 0.0503 | 0.212 | 0.0335 -31% [-57, -5]* | 0.0313 -37% [-51, -23]* | 0.0329 -32% [-62, -1]* | 0.0314 -37% [-52, -22]* | 0.0450 -7% [-31, +17] | 0.0511 +2% [-5, +8] | 0.1216 +151% [+71, +231]* |
| lorenz | 0.0298 | 0.0221 | 0.638 | 0.0166 -43% [-61, -26]* | 0.0166 -24% [-48, -1]* | 0.0424 +47% [-180, +275] | 0.0433 +92% [-58, +242] | 0.0274 -6% [-40, +28] | 0.0191 -13% [-30, +4] | 0.0310 +41% [+2, +79]* |
| quad_tank | 0.0517 | 0.0533 | 1.155 | 0.0509 -2% [-9, +6] | 0.0521 -2% [-5, +1] | 0.0535 +3% [+1, +5]* | 0.0535 +0% [-4, +5] | 0.0516 -0% [-2, +2] | 0.0537 +1% [-5, +7] | 0.2281 +341% [+314, +368]* |
| vdp | 0.0157 | 0.0180 | 0.082 | 0.0180 +15% [+2, +28]* | 0.0197 +9% [-0, +19] | 0.0159 +1% [-8, +10] | 0.0182 +1% [-15, +17] | 0.0167 +7% [-15, +29] | 0.0215 +19% [-2, +40] | 0.0492 +214% [+196, +232]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### Verdicts (C3-partial, sequence_ood; BETTER = CI in favour on >= 4/6 and worse on none)

| arm | better on | worse on | n systems | verdict | shape_ood better/worse | id better/worse |
|---|---|---|---|---|---|---|
| P1-lora | 5 ['battery', 'bioreactor', 'duffing', 'lorenz', 'quad_tank'] | 1 ['vdp'] | 6 | **NOT BETTER** | 2/1 | 4/1 |
| P1-hyper | 2 ['battery', 'duffing'] | 0 [] | 6 | **NOT BETTER** | 2/0 | 3/0 |
| P2-lora | 1 ['duffing'] | 0 [] | 6 | **NOT BETTER** | 1/1 | 1/2 |
| P2-hyper | 1 ['duffing'] | 1 ['bioreactor'] | 6 | **NOT BETTER** | 1/1 | 1/1 |
| P4-lora | 0 [] | 0 [] | 6 | **NOT BETTER** | 0/0 | 0/0 |
| P4-hyper | 0 [] | 0 [] | 6 | **NOT BETTER** | 1/1 | 1/0 |
| P3 | 0 [] | 5 ['battery', 'bioreactor', 'duffing', 'quad_tank', 'vdp'] | 6 | **WORSE** | 0/5 | 0/6 |

### Controls (C3-partial, sequence_ood)

| system | P3 vs autonomous | P3 vs P1-best | P1-hyper vs P1-lora | P2-hyper vs P2-lora | P0-hyper vs P0-lora |
|---|---|---|---|---|---|
| battery | -74% [-82, -66]* | +880% [+798, +961]* | +14% [-7, +35] | +27% [-35, +88] | -2% [-52, +47] |
| bioreactor | -96% [-97, -95]* | +309% [+273, +344]* | +4% [-14, +23] | +5% [-8, +18] | -1% [-103, +100] |
| duffing | -31% [-37, -26]* | +418% [+383, +452]* | -5% [-10, +1] | -5% [-23, +14] | +20% [-2, +43] |
| lorenz | -95% [-97, -93]* | +88% [+53, +123]* | -1% [-16, +13] | +12% [-144, +169] | -35% [-81, +11] |
| quad_tank | -78% [-81, -76]* | +476% [+421, +531]* | -1% [-13, +11] | -5% [-13, +3] | -13% [-18, -8]* |
| vdp | -26% [-32, -19]* | +246% [+220, +273]* | +0% [-3, +3] | +10% [-10, +31] | +11% [+1, +21]* |

### Gap to the best data-driven baseline (C3-partial, sequence_ood)

| system | best data-driven | P0-best vs it | loss cell | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|---|
| battery | greybox_ekf 0.0191 | P0-hyper +276% [+257, +295]* | yes | P1-lora 0.0441: vs base +135% [+19, +252]* narrows | P2-lora 0.0701: vs base +272% [+132, +413]* | P3 0.4315: vs base +2196% [+1252, +3141]* | P4-lora 0.0739: vs base +289% [+4, +574]* |
| bioreactor | koopman_kf 0.0271 | P0-hyper +158% [+49, +266]* | yes | P1-lora 0.0472: vs base +75% [+39, +110]* | P2-lora 0.1323: vs base +388% [+348, +428]* | P3 0.1928: vs base +612% [+497, +727]* | P4-lora 0.0630: vs base +132% [+99, +165]* |
| duffing | greybox_ukf 0.0078 | P0-lora +593% [+403, +784]* | yes | P1-hyper 0.0416: vs base +437% [+399, +474]* narrows | P2-hyper 0.0370: vs base +377% [+334, +421]* narrows | P3 0.2151: vs base +2679% [+2308, +3050]* | P4-lora 0.0530: vs base +584% [+514, +654]* |
| lorenz | nlox 0.0287 | P0-hyper -33% [-52, -14]* | no | P1-hyper 0.0164: vs base -43% [-58, -28]* | P2-hyper 0.0417: vs base +44% [-90, +178] | P3 0.0308: vs base +7% [-25, +39] | P4-hyper 0.0158: vs base -45% [-46, -43]* |
| quad_tank | greybox_ekf 0.0142 | P0-hyper +341% [+280, +402]* | yes | P1-hyper 0.0601: vs base +324% [+287, +361]* | P2-hyper 0.0680: vs base +380% [+344, +416]* | P3 0.3458: vs base +2341% [+2156, +2527]* | P4-hyper 0.0638: vs base +350% [+341, +359]* |
| vdp | greybox_ekf 0.0257 | P0-lora -30% [-44, -15]* | no | P1-lora 0.0219: vs base -14% [-43, +16] | P2-lora 0.0193: vs base -25% [-40, -10]* | P3 0.0759: vs base +198% [+110, +286]* | P4-lora 0.0196: vs base -23% [-62, +17] |

## C3-mismatch30

### sequence_ood

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P4-lora | P4-hyper | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0515 | 0.0696 | 1.102 | 0.0298 -42% [-47, -37]* | 0.0334 -52% [-67, -37]* | 0.0600 +17% [-15, +49] | 0.0755 +9% [-8, +25] | 0.0551 +7% [-10, +24] | 0.0609 -13% [-26, +0] | 0.4206 +721% [+528, +913]* |
| bioreactor | 0.0977 | 0.1226 | 7.370 | 0.0707 -22% [-89, +46] | 0.0692 -40% [-76, -5]* | 0.1203 +39% [-124, +201] | 0.1056 -8% [-80, +64] | 0.0543 -40% [-83, +3] | 0.0533 -52% [-97, -8]* | 0.1988 +122% [-78, +321] |
| duffing | 0.0561 | 0.0628 | 0.306 | 0.0451 -20% [-26, -13]* | 0.0428 -32% [-38, -26]* | 0.0394 -30% [-52, -8]* | 0.0388 -38% [-48, -29]* | 0.0549 -2% [-10, +6] | 0.0620 -1% [-7, +4] | 0.2189 +290% [+279, +302]* |
| lorenz | 0.0070 | 0.0097 | 0.013 | 0.0055 -20% [-41, +0] | 0.0057 -39% [-72, -7]* | 0.0076 +10% [-19, +39] | 0.0066 -28% [-75, +18] | 0.0074 +8% [-89, +105] | 0.0091 -1% [-80, +79] | 0.0079 +14% [-27, +55] |
| quad_tank | 0.0422 | 0.0436 | 1.766 | 0.0360 -14% [-25, -4]* | 0.0405 -7% [-15, +1] | 0.0565 +34% [+11, +56]* | 0.0507 +16% [+12, +20]* | 0.0420 -0% [-23, +23] | 0.0450 +3% [-10, +16] | 0.1840 +336% [+310, +362]* |
| vdp | 0.0205 | 0.0232 | 0.116 | 0.0190 -7% [-26, +13] | 0.0192 -17% [-31, -4]* | 0.0194 -5% [-43, +33] | 0.0204 -11% [-46, +23] | 0.0196 -4% [-21, +14] | 0.0238 +3% [-11, +17] | 0.0766 +276% [+201, +350]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### shape_ood

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P4-lora | P4-hyper | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0436 | 0.0514 | 0.675 | 0.0313 -28% [-43, -13]* | 0.0333 -35% [-57, -13]* | 0.0484 +11% [+6, +15]* | 0.0609 +18% [-4, +41] | 0.0487 +14% [-56, +84] | 0.0485 -5% [-28, +18] | 0.2936 +579% [+380, +778]* |
| bioreactor | 0.0437 | 0.0424 | 6.623 | 0.0270 -29% [-123, +65] | 0.0274 -35% [-44, -26]* | 0.0487 +35% [-205, +275] | 0.0452 +8% [-59, +76] | 0.0195 -51% [-99, -3]* | 0.0248 -41% [-62, -20]* | 0.0914 +118% [+42, +193]* |
| duffing | 0.0518 | 0.0536 | 0.199 | 0.0353 -32% [-40, -24]* | 0.0322 -40% [-45, -34]* | 0.0331 -35% [-65, -6]* | 0.0315 -41% [-53, -29]* | 0.0498 -3% [-23, +16] | 0.0540 +1% [-8, +9] | 0.1377 +167% [+119, +215]* |
| lorenz | 0.0069 | 0.0100 | 0.012 | 0.0055 -18% [-50, +13] | 0.0056 -42% [-69, -15]* | 0.0071 +5% [-37, +46] | 0.0063 -35% [-70, +1] | 0.0074 +12% [-99, +122] | 0.0095 -1% [-67, +64] | 0.0064 -4% [-48, +40] |
| quad_tank | 0.0238 | 0.0237 | 1.128 | 0.0227 -4% [-18, +9] | 0.0287 +21% [+1, +41]* | 0.0303 +27% [+12, +43]* | 0.0292 +23% [-9, +55] | 0.0259 +9% [-13, +32] | 0.0258 +9% [-14, +32] | 0.1226 +417% [+326, +508]* |
| vdp | 0.0178 | 0.0212 | 0.074 | 0.0174 -1% [-31, +29] | 0.0181 -14% [-23, -6]* | 0.0167 -6% [-45, +33] | 0.0178 -15% [-55, +25] | 0.0168 -5% [-24, +14] | 0.0218 +3% [-13, +20] | 0.0491 +179% [+95, +263]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### id

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P4-lora | P4-hyper | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0382 | 0.0481 | 0.733 | 0.0275 -28% [-31, -25]* | 0.0297 -38% [-58, -19]* | 0.0441 +16% [-12, +43] | 0.0563 +17% [-7, +41] | 0.0415 +9% [-31, +48] | 0.0421 -13% [-24, -1]* | 0.3120 +719% [+565, +873]* |
| bioreactor | 0.0329 | 0.0326 | 6.503 | 0.0193 -34% [-110, +41] | 0.0197 -39% [-51, -27]* | 0.0426 +41% [-92, +175] | 0.0327 +2% [-56, +61] | 0.0170 -44% [-87, -2]* | 0.0198 -38% [-66, -10]* | 0.0682 +112% [+26, +199]* |
| duffing | 0.0493 | 0.0507 | 0.265 | 0.0339 -31% [-42, -20]* | 0.0316 -38% [-43, -32]* | 0.0332 -32% [-65, +2] | 0.0317 -37% [-48, -27]* | 0.0471 -4% [-26, +18] | 0.0513 +1% [-7, +9] | 0.1261 +157% [+98, +216]* |
| lorenz | 0.0066 | 0.0099 | 0.012 | 0.0055 -17% [-43, +10] | 0.0057 -41% [-71, -10]* | 0.0069 +5% [-24, +34] | 0.0064 -32% [-78, +14] | 0.0071 +9% [-77, +95] | 0.0091 -4% [-69, +60] | 0.0068 +4% [-33, +41] |
| quad_tank | 0.0221 | 0.0225 | 0.998 | 0.0227 +3% [-2, +8] | 0.0287 +28% [+3, +53]* | 0.0297 +34% [+6, +63]* | 0.0293 +31% [-5, +66] | 0.0246 +12% [-8, +31] | 0.0272 +21% [+6, +36]* | 0.1148 +419% [+361, +478]* |
| vdp | 0.0179 | 0.0211 | 0.087 | 0.0172 -4% [-23, +15] | 0.0186 -12% [-18, -6]* | 0.0172 -4% [-35, +27] | 0.0181 -14% [-47, +20] | 0.0174 -2% [-14, +9] | 0.0218 +3% [-13, +20] | 0.0485 +172% [+112, +232]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### Verdicts (C3-mismatch30, sequence_ood; BETTER = CI in favour on >= 4/6 and worse on none)

| arm | better on | worse on | n systems | verdict | shape_ood better/worse | id better/worse |
|---|---|---|---|---|---|---|
| P1-lora | 3 ['battery', 'duffing', 'quad_tank'] | 0 [] | 6 | **NOT BETTER** | 2/0 | 2/0 |
| P1-hyper | 5 ['battery', 'bioreactor', 'duffing', 'lorenz', 'vdp'] | 0 [] | 6 | **BETTER** | 5/1 | 5/1 |
| P2-lora | 1 ['duffing'] | 1 ['quad_tank'] | 6 | **NOT BETTER** | 1/2 | 0/1 |
| P2-hyper | 1 ['duffing'] | 1 ['quad_tank'] | 6 | **NOT BETTER** | 1/0 | 1/0 |
| P4-lora | 0 [] | 0 [] | 6 | **NOT BETTER** | 1/0 | 1/0 |
| P4-hyper | 1 ['bioreactor'] | 0 [] | 6 | **NOT BETTER** | 1/0 | 2/1 |
| P3 | 0 [] | 4 ['battery', 'duffing', 'quad_tank', 'vdp'] | 6 | **WORSE** | 0/5 | 0/5 |

### Controls (C3-mismatch30, sequence_ood)

| system | P3 vs autonomous | P3 vs P1-best | P1-hyper vs P1-lora | P2-hyper vs P2-lora | P0-hyper vs P0-lora |
|---|---|---|---|---|---|
| battery | -62% [-64, -59]* | +1316% [+1091, +1542]* | +12% [-16, +41] | +27% [-12, +66] | +36% [+0, +71]* |
| bioreactor | -97% [-98, -97]* | +187% [+158, +217]* | -2% [-10, +6] | -10% [-60, +39] | +28% [-28, +84] |
| duffing | -29% [-32, -25]* | +411% [+385, +438]* | -5% [-8, -2]* | -1% [-18, +15] | +12% [-2, +26] |
| lorenz | -39% [-58, -20]* | +43% [+24, +61]* | +2% [-9, +13] | -13% [-18, -8]* | +38% [-24, +99] |
| quad_tank | -90% [-92, -87]* | +411% [+328, +493]* | +12% [+8, +17]* | -10% [-33, +14] | +4% [-8, +16] |
| vdp | -34% [-40, -28]* | +303% [+251, +355]* | +1% [-3, +4] | +6% [-17, +28] | +14% [-31, +59] |

### Gap to the best data-driven baseline (C3-mismatch30, sequence_ood)

| system | best data-driven | P0-best vs it | loss cell | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|---|
| battery | greybox_ekf 0.0151 | P0-lora +241% [+223, +259]* | yes | P1-lora 0.0298: vs base +97% [+82, +113]* narrows | P2-lora 0.0600: vs base +299% [+209, +388]* | P3 0.4206: vs base +2697% [+2156, +3239]* | P4-lora 0.0551: vs base +266% [+220, +312]* |
| bioreactor | nlox 0.0353 | P0-lora +194% [-123, +510] | no | P1-hyper 0.0692: vs base +104% [-23, +232] | P2-hyper 0.1056: vs base +206% [+106, +305]* | P3 0.1988: vs base +484% [+162, +807]* | P4-hyper 0.0533: vs base +57% [-51, +166] |
| duffing | greybox_ukf 0.0059 | P0-lora +848% [+826, +871]* | yes | P1-hyper 0.0428: vs base +624% [+580, +668]* narrows | P2-hyper 0.0388: vs base +556% [+434, +678]* narrows | P3 0.2189: vs base +3600% [+3405, +3795]* | P4-lora 0.0549: vs base +828% [+727, +929]* |
| lorenz | nlox 0.0068 | P0-lora +3% [-36, +41] | no | P1-lora 0.0055: vs base -19% [-30, -8]* | P2-hyper 0.0066: vs base -3% [-20, +15] | P3 0.0079: vs base +16% [-5, +37] | P4-lora 0.0074: vs base +8% [-59, +75] |
| quad_tank | greybox_ekf 0.0129 | P0-lora +226% [+202, +250]* | yes | P1-lora 0.0360: vs base +178% [+156, +201]* narrows | P2-hyper 0.0507: vs base +292% [+272, +312]* | P3 0.1840: vs base +1321% [+1141, +1502]* | P4-lora 0.0420: vs base +225% [+170, +280]* |
| vdp | greybox_ekf 0.0098 | P0-lora +110% [+58, +163]* | yes | P1-lora 0.0190: vs base +95% [+63, +128]* | P2-lora 0.0194: vs base +99% [+35, +162]* | P3 0.0766: vs base +685% [+645, +726]* | P4-lora 0.0196: vs base +101% [+87, +115]* |

## C1-where

### sequence_ood

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P3 |
|---|---|---|---|---|---|---|---|---|
| battery | 0.0719 | 0.0761 | 1.497 | 0.0416 -40% [-72, -9]* | 0.0465 -38% [-58, -19]* | 0.0988 +40% [-29, +108] | 0.1038 +36% [+3, +69]* | 0.4329 +518% [+252, +783]* |
| bioreactor | 0.0617 | 0.0510 | 4.700 | 0.0363 -41% [-45, -37]* | 0.0319 -37% [-54, -20]* | 0.1229 +101% [+40, +162]* | 0.1098 +115% [+101, +129]* | 0.2311 +356% [+229, +482]* |
| duffing | 0.0570 | 0.0620 | 0.302 | 0.0446 -22% [-34, -9]* | 0.0421 -32% [-37, -27]* | 0.0384 -32% [-50, -15]* | 0.0397 -36% [-45, -27]* | 0.2139 +276% [+234, +318]* |
| lorenz | 0.0266 | 0.0182 | 0.521 | 0.0152 -43% [-53, -33]* | 0.0151 -15% [-49, +20] | 0.0429 +64% [-186, +314] | 0.0316 +71% [+9, +132]* | 0.0258 +46% [-22, +115] |
| quad_tank | 0.0700 | 0.0644 | 1.622 | 0.0626 -11% [-17, -4]* | 0.0623 -3% [-7, +1] | 0.0747 +7% [-8, +21] | 0.0683 +6% [+2, +10]* | 0.3344 +419% [+390, +449]* |
| vdp | 0.0191 | 0.0249 | 0.111 | 0.0184 -4% [-10, +3] | 0.0188 -24% [-47, +0] | 0.0205 +7% [-23, +37] | 0.0226 -9% [-35, +18] | 0.0754 +294% [+283, +306]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### shape_ood

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P3 |
|---|---|---|---|---|---|---|---|---|
| battery | 0.0551 | 0.0520 | 0.744 | 0.0370 -32% [-59, -5]* | 0.0388 -25% [-51, +2] | 0.0777 +43% [-26, +112] | 0.0757 +45% [+40, +51]* | 0.2838 +452% [+264, +639]* |
| bioreactor | 0.0239 | 0.0263 | 2.978 | 0.0144 -39% [-63, -15]* | 0.0138 -48% [-57, -38]* | 0.0514 +119% [-0, +239] | 0.0417 +59% [+18, +100]* | 0.0940 +299% [+142, +456]* |
| duffing | 0.0518 | 0.0537 | 0.196 | 0.0352 -32% [-43, -21]* | 0.0321 -40% [-58, -21]* | 0.0324 -37% [-47, -28]* | 0.0320 -40% [-61, -18]* | 0.1370 +164% [+145, +184]* |
| lorenz | 0.0257 | 0.0175 | 0.513 | 0.0147 -43% [-52, -34]* | 0.0150 -12% [-53, +29] | 0.0426 +70% [-201, +340] | 0.0311 +74% [+18, +130]* | 0.0254 +50% [-29, +129] |
| quad_tank | 0.0502 | 0.0500 | 1.067 | 0.0513 +2% [-10, +14] | 0.0520 +4% [-3, +11] | 0.0551 +10% [-6, +25] | 0.0521 +4% [-7, +15] | 0.1924 +285% [+278, +291]* |
| vdp | 0.0158 | 0.0225 | 0.071 | 0.0163 +3% [-1, +8] | 0.0174 -21% [-55, +13] | 0.0168 +7% [-25, +39] | 0.0196 -12% [-44, +21] | 0.0487 +209% [+194, +224]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### id

| system | P0-lora | P0-hyper | autonomous | P1-lora | P1-hyper | P2-lora | P2-hyper | P3 |
|---|---|---|---|---|---|---|---|---|
| battery | 0.0479 | 0.0487 | 0.750 | 0.0338 -29% [-45, -13]* | 0.0361 -25% [-53, +3] | 0.0672 +40% [+4, +76]* | 0.0687 +41% [+24, +58]* | 0.2993 +529% [+366, +692]* |
| bioreactor | 0.0192 | 0.0222 | 3.256 | 0.0126 -35% [-47, -22]* | 0.0115 -48% [-52, -45]* | 0.0395 +107% [+21, +193]* | 0.0318 +44% [+36, +51]* | 0.0752 +293% [+242, +345]* |
| duffing | 0.0492 | 0.0509 | 0.253 | 0.0341 -31% [-41, -20]* | 0.0317 -37% [-54, -20]* | 0.0325 -34% [-42, -26]* | 0.0321 -36% [-55, -17]* | 0.1226 +150% [+115, +184]* |
| lorenz | 0.0276 | 0.0207 | 0.513 | 0.0152 -45% [-49, -41]* | 0.0155 -24% [-44, -4]* | 0.0430 +56% [-159, +270] | 0.0328 +56% [-19, +131] | 0.0260 +27% [-11, +65] |
| quad_tank | 0.0520 | 0.0540 | 1.149 | 0.0519 -0% [-12, +11] | 0.0525 -3% [-8, +3] | 0.0562 +8% [-8, +23] | 0.0543 +1% [-10, +11] | 0.2208 +325% [+285, +365]* |
| vdp | 0.0165 | 0.0224 | 0.083 | 0.0159 -3% [-4, -2]* | 0.0176 -21% [-51, +10] | 0.0170 +3% [-19, +26] | 0.0195 -12% [-40, +16] | 0.0480 +191% [+178, +204]* |

(P1/P2/P4-k vs P0-k; P3 vs the better P0 kind.)

### Verdicts (C1-where, sequence_ood; BETTER = CI in favour on >= 4/6 and worse on none)

| arm | better on | worse on | n systems | verdict | shape_ood better/worse | id better/worse |
|---|---|---|---|---|---|---|
| P1-lora | 5 ['battery', 'bioreactor', 'duffing', 'lorenz', 'quad_tank'] | 0 [] | 6 | **BETTER** | 4/0 | 5/0 |
| P1-hyper | 3 ['battery', 'bioreactor', 'duffing'] | 0 [] | 6 | **NOT BETTER** | 2/0 | 3/0 |
| P2-lora | 1 ['duffing'] | 1 ['bioreactor'] | 6 | **NOT BETTER** | 1/0 | 1/2 |
| P2-hyper | 1 ['duffing'] | 4 ['battery', 'bioreactor', 'lorenz', 'quad_tank'] | 6 | **NOT BETTER** | 1/3 | 1/2 |
| P3 | 0 [] | 5 ['battery', 'bioreactor', 'duffing', 'quad_tank', 'vdp'] | 6 | **WORSE** | 0/5 | 0/5 |

### Controls (C1-where, sequence_ood)

| system | P3 vs autonomous | P3 vs P1-best | P1-hyper vs P1-lora | P2-hyper vs P2-lora | P0-hyper vs P0-lora | P3 vs tree latent | P3 vs tree concat |
|---|---|---|---|---|---|---|---|
| battery | -71% [-77, -65]* | +941% [+808, +1074]* | +12% [+2, +22]* | +5% [+4, +7]* | +8% [-31, +47] | +243% [+86, +400]* | +39% [+16, +62]* |
| bioreactor | -95% [-95, -95]* | +625% [+545, +704]* | -12% [-24, -0]* | -10% [-38, +17] | -17% [-51, +18] | +428% [+346, +511]* | +61% [+56, +65]* |
| duffing | -29% [-31, -27]* | +408% [+393, +423]* | -6% [-12, +0] | +4% [-15, +23] | +9% [-14, +32] | +174% [+158, +189]* | +210% [+176, +243]* |
| lorenz | -95% [-96, -94]* | +71% [+45, +97]* | -0% [-9, +9] | -21% [-58, +15] | -31% [-77, +15] | -5% [-14, +5] | -54% [-69, -39]* |
| quad_tank | -79% [-82, -77]* | +437% [+394, +481]* | -0% [-8, +7] | -8% [-29, +13] | -8% [-24, +8] | +105% [+71, +139]* | +121% [+99, +144]* |
| vdp | -32% [-39, -25]* | +310% [+289, +330]* | +2% [+2, +2]* | +11% [-4, +25] | +30% [-9, +70] | +183% [+174, +192]* | +178% [+170, +187]* |

## Zero-input collapse, parameters, compute (seed means over finished runs)

| scenario | system | arm | exact vs phase-1 base | max dev (sigma) | zero-input NRMSE | autonomous zero-input NRMSE | trainable total | conditioner | wall train (s) |
|---|---|---|---|---|---|---|---|---|---|
| C3-partial | battery | P0-lora | True | 0 | 0.2590 | 0.2590 | 147724 | 141484 | nan |
| C3-partial | battery | P0-hyper | True | 0 | 0.2590 | 0.2590 | 146292 | 140052 | nan |
| C3-partial | battery | P1-lora | False | 2.31 | 0.1166 | 0.2590 | 182417 | 141484 | 81 |
| C3-partial | battery | P1-hyper | False | 2.28 | 0.1238 | 0.2590 | 180985 | 140052 | 196 |
| C3-partial | battery | P2-lora | False | 1.8 | 0.1455 | 0.2590 | 259032 | 141484 | 211 |
| C3-partial | battery | P2-hyper | False | 1.82 | 0.1440 | 0.2590 | 257600 | 140052 | 319 |
| C3-partial | battery | P3 | False | 2.84 | 0.2455 | 0.2590 | 174525 | 139832 | 28 |
| C3-partial | battery | P4-lora | True | 0 | 0.2362 | 0.2590 | 147724 | 141484 | 65 |
| C3-partial | battery | P4-hyper | True | 0 | 0.2362 | 0.2590 | 146292 | 140052 | 188 |
| C3-partial | bioreactor | P0-lora | True | 0 | 0.0658 | 0.0658 | 146836 | 140596 | nan |
| C3-partial | bioreactor | P0-hyper | True | 0 | 0.0658 | 0.0658 | 143720 | 137480 | nan |
| C3-partial | bioreactor | P1-lora | False | 0.271 | 0.0114 | 0.0658 | 180886 | 140596 | 80 |
| C3-partial | bioreactor | P1-hyper | False | 0.275 | 0.0124 | 0.0658 | 177770 | 137480 | 190 |
| C3-partial | bioreactor | P2-lora | False | 0.18 | 0.0352 | 0.0658 | 256539 | 140596 | 202 |
| C3-partial | bioreactor | P2-hyper | False | 0.188 | 0.0311 | 0.0658 | 253423 | 137480 | 319 |
| C3-partial | bioreactor | P3 | False | 0.314 | 0.0128 | 0.0658 | 174282 | 140232 | 27 |
| C3-partial | bioreactor | P4-lora | True | 0 | 0.0269 | 0.0658 | 146836 | 140596 | 55 |
| C3-partial | bioreactor | P4-hyper | True | 0 | 0.0269 | 0.0658 | 143720 | 137480 | 153 |
| C3-partial | duffing | P0-lora | True | 0 | 0.0754 | 0.0754 | 146836 | 140596 | nan |
| C3-partial | duffing | P0-hyper | True | 0 | 0.0754 | 0.0754 | 143720 | 137480 | nan |
| C3-partial | duffing | P1-lora | False | 0.304 | 0.0323 | 0.0754 | 180886 | 140596 | 86 |
| C3-partial | duffing | P1-hyper | False | 0.357 | 0.0293 | 0.0754 | 177770 | 137480 | 204 |
| C3-partial | duffing | P2-lora | False | 0.381 | 0.0319 | 0.0754 | 256539 | 140596 | 213 |
| C3-partial | duffing | P2-hyper | False | 0.392 | 0.0306 | 0.0754 | 253423 | 137480 | 327 |
| C3-partial | duffing | P3 | False | 0.258 | 0.0795 | 0.0754 | 174282 | 140232 | 27 |
| C3-partial | duffing | P4-lora | True | 0 | 0.0766 | 0.0754 | 146836 | 140596 | 74 |
| C3-partial | duffing | P4-hyper | True | 0 | 0.0766 | 0.0754 | 143720 | 137480 | 198 |
| C3-partial | lorenz | P0-lora | True | 0 | 0.6264 | 0.6264 | 147372 | 141132 | nan |
| C3-partial | lorenz | P0-hyper | True | 0 | 0.6264 | 0.6264 | 145260 | 139020 | nan |
| C3-partial | lorenz | P1-lora | False | 3.64 | 0.0689 | 0.6264 | 181807 | 141132 | 68 |
| C3-partial | lorenz | P1-hyper | False | 3.64 | 0.0712 | 0.6264 | 179695 | 139020 | 125 |
| C3-partial | lorenz | P2-lora | False | 3.86 | 0.2011 | 0.6264 | 258038 | 141132 | 193 |
| C3-partial | lorenz | P2-hyper | False | 3.67 | 0.2025 | 0.6264 | 255926 | 139020 | 294 |
| C3-partial | lorenz | P3 | False | 3.87 | 0.0309 | 0.6264 | 174278 | 139843 | 23 |
| C3-partial | lorenz | P4-lora | True | 0 | 0.5117 | 0.6264 | 147372 | 141132 | 76 |
| C3-partial | lorenz | P4-hyper | True | 0 | 0.5117 | 0.6264 | 145260 | 139020 | 213 |
| C3-partial | quad_tank | P0-lora | True | 0 | 0.1630 | 0.1630 | 147368 | 141128 | nan |
| C3-partial | quad_tank | P0-hyper | True | 0 | 0.1630 | 0.1630 | 145264 | 139024 | nan |
| C3-partial | quad_tank | P1-lora | False | 1.34 | 0.0702 | 0.1630 | 181804 | 141128 | 71 |
| C3-partial | quad_tank | P1-hyper | False | 1.45 | 0.0705 | 0.1630 | 179700 | 139024 | 142 |
| C3-partial | quad_tank | P2-lora | False | 0.137 | 0.1587 | 0.1630 | 258034 | 141128 | 205 |
| C3-partial | quad_tank | P2-hyper | False | 0.162 | 0.1551 | 0.1630 | 255930 | 139024 | 288 |
| C3-partial | quad_tank | P3 | False | 4.27 | 0.2736 | 0.1630 | 174280 | 139844 | 22 |
| C3-partial | quad_tank | P4-lora | True | 0 | 0.1601 | 0.1630 | 147368 | 141128 | 80 |
| C3-partial | quad_tank | P4-hyper | True | 0 | 0.1601 | 0.1630 | 145264 | 139024 | 211 |
| C3-partial | vdp | P0-lora | True | 0 | 0.0329 | 0.0329 | 146836 | 140596 | nan |
| C3-partial | vdp | P0-hyper | True | 0 | 0.0329 | 0.0329 | 143720 | 137480 | nan |
| C3-partial | vdp | P1-lora | False | 0.179 | 0.0188 | 0.0329 | 180886 | 140596 | 66 |
| C3-partial | vdp | P1-hyper | False | 0.187 | 0.0179 | 0.0329 | 177770 | 137480 | 131 |
| C3-partial | vdp | P2-lora | False | 0.0552 | 0.0277 | 0.0329 | 256539 | 140596 | 192 |
| C3-partial | vdp | P2-hyper | False | 0.0678 | 0.0263 | 0.0329 | 253423 | 137480 | 270 |
| C3-partial | vdp | P3 | False | 0.122 | 0.0385 | 0.0329 | 174282 | 140232 | 22 |
| C3-partial | vdp | P4-lora | True | 0 | 0.0232 | 0.0329 | 146836 | 140596 | 58 |
| C3-partial | vdp | P4-hyper | True | 0 | 0.0232 | 0.0329 | 143720 | 137480 | 159 |
| C3-mismatch30 | battery | P0-lora | True | 0 | 0.2325 | 0.2325 | 147724 | 141484 | nan |
| C3-mismatch30 | battery | P0-hyper | True | 0 | 0.2325 | 0.2325 | 146292 | 140052 | nan |
| C3-mismatch30 | battery | P1-lora | False | 1.77 | 0.1186 | 0.2325 | 182417 | 141484 | 76 |
| C3-mismatch30 | battery | P1-hyper | False | 1.91 | 0.1336 | 0.2325 | 180985 | 140052 | 159 |
| C3-mismatch30 | battery | P2-lora | False | 1.49 | 0.1478 | 0.2325 | 259032 | 141484 | 206 |
| C3-mismatch30 | battery | P2-hyper | False | 1.28 | 0.1500 | 0.2325 | 257600 | 140052 | 305 |
| C3-mismatch30 | battery | P3 | False | 3.34 | 0.3833 | 0.2325 | 174525 | 139832 | 22 |
| C3-mismatch30 | battery | P4-lora | True | 0 | 0.3355 | 0.2325 | 147724 | 141484 | 79 |
| C3-mismatch30 | battery | P4-hyper | True | 0 | 0.3355 | 0.2325 | 146292 | 140052 | 248 |
| C3-mismatch30 | bioreactor | P0-lora | True | 0 | 0.0474 | 0.0474 | 146836 | 140596 | nan |
| C3-mismatch30 | bioreactor | P0-hyper | True | 0 | 0.0474 | 0.0474 | 143720 | 137480 | nan |
| C3-mismatch30 | bioreactor | P1-lora | False | 0.369 | 0.0134 | 0.0474 | 180886 | 140596 | 65 |
| C3-mismatch30 | bioreactor | P1-hyper | False | 0.36 | 0.0088 | 0.0474 | 177770 | 137480 | 110 |
| C3-mismatch30 | bioreactor | P2-lora | False | 0.261 | 0.0324 | 0.0474 | 256539 | 140596 | 212 |
| C3-mismatch30 | bioreactor | P2-hyper | False | 0.32 | 0.0288 | 0.0474 | 253423 | 137480 | 297 |
| C3-mismatch30 | bioreactor | P3 | False | 0.378 | 0.0133 | 0.0474 | 174282 | 140232 | 23 |
| C3-mismatch30 | bioreactor | P4-lora | True | 0 | 0.0228 | 0.0474 | 146836 | 140596 | 76 |
| C3-mismatch30 | bioreactor | P4-hyper | True | 0 | 0.0228 | 0.0474 | 143720 | 137480 | 143 |
| C3-mismatch30 | duffing | P0-lora | True | 0 | 0.0768 | 0.0768 | 146836 | 140596 | nan |
| C3-mismatch30 | duffing | P0-hyper | True | 0 | 0.0768 | 0.0768 | 143720 | 137480 | nan |
| C3-mismatch30 | duffing | P1-lora | False | 0.31 | 0.0326 | 0.0768 | 180886 | 140596 | 76 |
| C3-mismatch30 | duffing | P1-hyper | False | 0.364 | 0.0292 | 0.0768 | 177770 | 137480 | 168 |
| C3-mismatch30 | duffing | P2-lora | False | 0.365 | 0.0319 | 0.0768 | 256539 | 140596 | 222 |
| C3-mismatch30 | duffing | P2-hyper | False | 0.399 | 0.0306 | 0.0768 | 253423 | 137480 | 497 |
| C3-mismatch30 | duffing | P3 | False | 0.233 | 0.0891 | 0.0768 | 174282 | 140232 | 27 |
| C3-mismatch30 | duffing | P4-lora | True | 0 | 0.0765 | 0.0768 | 146836 | 140596 | 105 |
| C3-mismatch30 | duffing | P4-hyper | True | 0 | 0.0765 | 0.0768 | 143720 | 137480 | 214 |
| C3-mismatch30 | lorenz | P0-lora | True | 0 | 0.0260 | 0.0260 | 147372 | 141132 | nan |
| C3-mismatch30 | lorenz | P0-hyper | True | 0 | 0.0260 | 0.0260 | 145260 | 139020 | nan |
| C3-mismatch30 | lorenz | P1-lora | False | 1.13 | 0.0090 | 0.0260 | 181807 | 141132 | 67 |
| C3-mismatch30 | lorenz | P1-hyper | False | 1.21 | 0.0087 | 0.0260 | 179695 | 139020 | 151 |
| C3-mismatch30 | lorenz | P2-lora | False | 1.65 | 0.0387 | 0.0260 | 258038 | 141132 | 208 |
| C3-mismatch30 | lorenz | P2-hyper | False | 2.4 | 0.0396 | 0.0260 | 255926 | 139020 | 420 |
| C3-mismatch30 | lorenz | P3 | False | 1.24 | 0.0082 | 0.0260 | 174278 | 139843 | 26 |
| C3-mismatch30 | lorenz | P4-lora | True | 0 | 0.0237 | 0.0260 | 147372 | 141132 | 98 |
| C3-mismatch30 | lorenz | P4-hyper | True | 0 | 0.0237 | 0.0260 | 145260 | 139020 | 219 |
| C3-mismatch30 | quad_tank | P0-lora | True | 0 | 0.5982 | 0.5982 | 147368 | 141128 | nan |
| C3-mismatch30 | quad_tank | P0-hyper | True | 0 | 0.5982 | 0.5982 | 145264 | 139024 | nan |
| C3-mismatch30 | quad_tank | P1-lora | False | 2.28 | 0.0503 | 0.5982 | 181804 | 141128 | 80 |
| C3-mismatch30 | quad_tank | P1-hyper | False | 2.35 | 0.0434 | 0.5982 | 179700 | 139024 | 184 |
| C3-mismatch30 | quad_tank | P2-lora | False | 0.998 | 0.3550 | 0.5982 | 258034 | 141128 | 210 |
| C3-mismatch30 | quad_tank | P2-hyper | False | 1.06 | 0.3572 | 0.5982 | 255930 | 139024 | 317 |
| C3-mismatch30 | quad_tank | P3 | False | 3.31 | 0.1182 | 0.5982 | 174280 | 139844 | 25 |
| C3-mismatch30 | quad_tank | P4-lora | True | 0 | 0.1603 | 0.5982 | 147368 | 141128 | 91 |
| C3-mismatch30 | quad_tank | P4-hyper | True | 0 | 0.1603 | 0.5982 | 145264 | 139024 | 229 |
| C3-mismatch30 | vdp | P0-lora | True | 0 | 0.0237 | 0.0237 | 146836 | 140596 | nan |
| C3-mismatch30 | vdp | P0-hyper | True | 0 | 0.0237 | 0.0237 | 143720 | 137480 | nan |
| C3-mismatch30 | vdp | P1-lora | False | 0.0619 | 0.0184 | 0.0237 | 180886 | 140596 | 69 |
| C3-mismatch30 | vdp | P1-hyper | False | 0.0709 | 0.0176 | 0.0237 | 177770 | 137480 | 129 |
| C3-mismatch30 | vdp | P2-lora | False | 0.0351 | 0.0225 | 0.0237 | 256539 | 140596 | 207 |
| C3-mismatch30 | vdp | P2-hyper | False | 0.0377 | 0.0219 | 0.0237 | 253423 | 137480 | 265 |
| C3-mismatch30 | vdp | P3 | False | 0.123 | 0.0365 | 0.0237 | 174282 | 140232 | 22 |
| C3-mismatch30 | vdp | P4-lora | True | 0 | 0.0258 | 0.0237 | 146836 | 140596 | 72 |
| C3-mismatch30 | vdp | P4-hyper | True | 0 | 0.0258 | 0.0237 | 143720 | 137480 | 198 |
| C1-where | battery | P0-lora | True | 0 | 0.2297 | 0.2297 | 147724 | 141484 | nan |
| C1-where | battery | P0-hyper | True | 0 | 0.2297 | 0.2297 | 146292 | 140052 | nan |
| C1-where | battery | P1-lora | False | 1.68 | 0.1016 | 0.2297 | 182417 | 141484 | 96 |
| C1-where | battery | P1-hyper | False | 1.71 | 0.1278 | 0.2297 | 180985 | 140052 | 219 |
| C1-where | battery | P2-lora | False | 1.3 | 0.1323 | 0.2297 | 259032 | 141484 | 259 |
| C1-where | battery | P2-hyper | False | 1.34 | 0.1384 | 0.2297 | 257600 | 140052 | 444 |
| C1-where | battery | P3 | False | 2.59 | 0.2649 | 0.2297 | 174525 | 139832 | 28 |
| C1-where | bioreactor | P0-lora | True | 0 | 0.0342 | 0.0342 | 146836 | 140596 | nan |
| C1-where | bioreactor | P0-hyper | True | 0 | 0.0342 | 0.0342 | 143720 | 137480 | nan |
| C1-where | bioreactor | P1-lora | False | 0.161 | 0.0072 | 0.0342 | 180886 | 140596 | 104 |
| C1-where | bioreactor | P1-hyper | False | 0.156 | 0.0078 | 0.0342 | 177770 | 137480 | 235 |
| C1-where | bioreactor | P2-lora | False | 0.0982 | 0.0202 | 0.0342 | 256539 | 140596 | 261 |
| C1-where | bioreactor | P2-hyper | False | 0.0719 | 0.0245 | 0.0342 | 253423 | 137480 | 542 |
| C1-where | bioreactor | P3 | False | 0.163 | 0.0128 | 0.0342 | 174282 | 140232 | 31 |
| C1-where | duffing | P0-lora | True | 0 | 0.0770 | 0.0770 | 146836 | 140596 | nan |
| C1-where | duffing | P0-hyper | True | 0 | 0.0770 | 0.0770 | 143720 | 137480 | nan |
| C1-where | duffing | P1-lora | False | 0.313 | 0.0322 | 0.0770 | 180886 | 140596 | 96 |
| C1-where | duffing | P1-hyper | False | 0.371 | 0.0292 | 0.0770 | 177770 | 137480 | 267 |
| C1-where | duffing | P2-lora | False | 0.392 | 0.0313 | 0.0770 | 256539 | 140596 | 287 |
| C1-where | duffing | P2-hyper | False | 0.404 | 0.0306 | 0.0770 | 253423 | 137480 | 586 |
| C1-where | duffing | P3 | False | 0.248 | 0.0836 | 0.0770 | 174282 | 140232 | 39 |
| C1-where | lorenz | P0-lora | True | 0 | 0.5090 | 0.5090 | 147372 | 141132 | nan |
| C1-where | lorenz | P0-hyper | True | 0 | 0.5090 | 0.5090 | 145260 | 139020 | nan |
| C1-where | lorenz | P1-lora | False | 3.88 | 0.0556 | 0.5090 | 181807 | 141132 | 105 |
| C1-where | lorenz | P1-hyper | False | 3.85 | 0.0574 | 0.5090 | 179695 | 139020 | 266 |
| C1-where | lorenz | P2-lora | False | 3.83 | 0.1649 | 0.5090 | 258038 | 141132 | 288 |
| C1-where | lorenz | P2-hyper | False | 3.86 | 0.1579 | 0.5090 | 255926 | 139020 | 652 |
| C1-where | lorenz | P3 | False | 3.92 | 0.0255 | 0.5090 | 174278 | 139843 | 37 |
| C1-where | quad_tank | P0-lora | True | 0 | 0.1581 | 0.1581 | 147368 | 141128 | nan |
| C1-where | quad_tank | P0-hyper | True | 0 | 0.1581 | 0.1581 | 145264 | 139024 | nan |
| C1-where | quad_tank | P1-lora | False | 1.25 | 0.0723 | 0.1581 | 181804 | 141128 | 110 |
| C1-where | quad_tank | P1-hyper | False | 1.34 | 0.0730 | 0.1581 | 179700 | 139024 | 208 |
| C1-where | quad_tank | P2-lora | False | 0.158 | 0.1551 | 0.1581 | 258034 | 141128 | 274 |
| C1-where | quad_tank | P2-hyper | False | 0.186 | 0.1537 | 0.1581 | 255930 | 139024 | 535 |
| C1-where | quad_tank | P3 | False | 3.95 | 0.2818 | 0.1581 | 174280 | 139844 | 30 |
| C1-where | vdp | P0-lora | True | 0 | 0.0245 | 0.0245 | 146836 | 140596 | nan |
| C1-where | vdp | P0-hyper | True | 0 | 0.0245 | 0.0245 | 143720 | 137480 | nan |
| C1-where | vdp | P1-lora | False | 0.072 | 0.0175 | 0.0245 | 180886 | 140596 | 91 |
| C1-where | vdp | P1-hyper | False | 0.0891 | 0.0162 | 0.0245 | 177770 | 137480 | 218 |
| C1-where | vdp | P2-lora | False | 0.0384 | 0.0210 | 0.0245 | 256539 | 140596 | 268 |
| C1-where | vdp | P2-hyper | False | 0.0423 | 0.0197 | 0.0245 | 253423 | 137480 | 591 |
| C1-where | vdp | P3 | False | 0.112 | 0.0368 | 0.0245 | 174282 | 140232 | 35 |

Compute plan (loss-batch gradient evaluations per arm): P0: 15000 (15000 optimiser steps), P1: 15000 (15000 optimiser steps), P2: 15000 (5000 optimiser steps), P3: 15000 (15000 optimiser steps), P4: 15000 (15000 optimiser steps)

P0 reproduction vs the tree runs: max relative per-episode NRMSE difference 0.00e+00 over 54 runs.

P4 phase-1 diagnostics vs P0 (seed 0..2 pooled, per scenario/system means):

| scenario | system | median rel. change of f0 | eps_pde_rms P0 -> P4 | eps_dec P0 -> P4 | l_dec_q95 P0 -> P4 |
|---|---|---|---|---|---|
| C3-partial | battery | 0.097 | 0.294 -> 0.3 | 1.83 -> 1.81 | 756 -> 835 |
| C3-partial | bioreactor | 0.585 | 0.0244 -> 0.0165 | 0.0963 -> 0.0617 | 84 -> 47 |
| C3-partial | duffing | 0.248 | 0.00128 -> 0.00223 | 0.00619 -> 0.00604 | 298 -> 294 |
| C3-partial | lorenz | 0.509 | 105 -> 65.8 | 0.357 -> 0.279 | 11 -> 10 |
| C3-partial | quad_tank | 0.119 | 0.0105 -> 0.0115 | 0.17 -> 0.148 | 15 -> 13 |
| C3-partial | vdp | 0.987 | 0.0606 -> 0.171 | 0.0111 -> 0.0103 | 23 -> 16 |
| C3-mismatch30 | battery | 0.016 | 0.327 -> 0.344 | 1.83 -> 1.9 | 691 -> 1098 |
| C3-mismatch30 | bioreactor | 0.753 | 0.00994 -> 0.00586 | 0.185 -> 0.0415 | 57 -> 53 |
| C3-mismatch30 | duffing | 0.022 | 0.00249 -> 0.00254 | 0.00616 -> 0.00618 | 295 -> 293 |
| C3-mismatch30 | lorenz | 0.270 | 33.2 -> 34.8 | 0.225 -> 0.218 | 12 -> 12 |
| C3-mismatch30 | quad_tank | 0.768 | 0.0183 -> 0.018 | 0.27 -> 0.187 | 12 -> 15 |
| C3-mismatch30 | vdp | 0.178 | 0.265 -> 0.271 | 0.0137 -> 0.0162 | 26 -> 32 |

