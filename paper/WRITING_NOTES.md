# WRITING_NOTES — TMLR draft (paper/main.tex)

Build: `bash paper/build.sh` (pdflatex + bibtex + pdflatex x2; prints total pages, undefined refs,
multiply-defined labels, the page on which the main text ends, and any `[pending: MACRO]` placeholder).
State at the last build (2026-10-26 ~03:00 Dubai, after the 23:04 CET regeneration of paper/generated
and with the real-data results): 57 pages, 0 undefined references/citations, 0 multiply-defined labels,
**main text ends on page 12** (References start at the top of page 13).

Real data: `analysis/real_numbers.py` (new, run by build.sh with `python -I` by path) turns
`experiments/real/results.json` into `paper/generated_real/numbers_real.tex` (211 macros, 93 used) and
`tab_real.tex`; Section 8 (main text, ~18 lines) + Appendix G (design, model identification, observers,
statistics, Table 14, Figure 10 from experiments/real/figures/, whole discharges, 0 degC, the six
limitations of report.md, deviations). `\RealSameDataBaselines` prints "pending" until koopman/greybox
rows appear in results.json (visible red note in Section 8 and in the Table 14 caption).

Layout: `main.tex` (preamble, abstract, \input of sections, single `\appendix`), `sections/*.tex`
(intro, framework, design, c1, c3, c2, scorecard, related, limitations [+ broader impact], conclusion),
`appendix/overview.tex` (reading guide + theory status table), theory = `\input{../theory/m1..m3}`
unchanged (their internal `\appendix` is neutralised with `\let\appendix\relax` so Lemma D becomes
Appendix D), `appendix/protocol.tex`, `appendix/tables.tex` (all generated tables/figures not in the
main text), `appendix/history.tex`. Style: official `style/tmlr.sty`/`tmlr.bst` (fetched from
github.com/JmlrOrg/tmlr-style-file into the new empty `paper/style/`), anonymous mode.

Rules followed: every result number is a macro from `generated/numbers.tex` (128 distinct macros used;
all exist in the current numbers.tex, checked mechanically) or a generated table/figure. Design constants
(seeds, budgets, dims, noise levels, grid sizes) are typed from configs/node files. `\providecommand`
fallbacks exist only for the pending-node macros and print a red `[pending: NAME]` if a macro vanishes.
Generated tables with duplicate labels (`tab:c1`, `tab:c3`, `tab:c3full` in the id-tier copies) are
input through `\inputrelabel{file}{-id}`; longtables (which repeat `\label` in every page head) through
`\inputonce`, which writes the label once. Nothing under paper/generated, theory/, experiments/,
hyperkkl/, analysis/ was edited.

## Claim -> evidence map (section by section)

### Abstract / Section 1 (Introduction)
| claim | evidence |
|---|---|
| weight-space beats latent & concat on dev plants | `\ConeWeightSpaceBeatsLatConcShape` (6/6), `\ConeWeightSpaceBeatsLatConcSeq` (5/6); Fig. 2 (fig1_c1_where, panels a,b); Table 6 (tab1_c1, appendix) |
| replicated on held-out plants | `\CtestWeightSpaceBeatsLatConcShape`, `\CtestWeightSpaceBeatsLatConcSeq` (5/6, 5/6); `\OutcomeTESTCOne`; Fig. 2 (c,d) |
| single-best-location rule FAILS | `\OutcomeCOneWhere`, `\ConePreferredLocation`, `\CtestPreferredLocation`; Table 13 (tab4_decisions) |
| filters win every exact-model condition | `\CthreeFaultsRegimes` ("none of 15 conditions"); Table 2 rows exact/exact+outliers; Table 13 |
| tuned EKF several times better | `\ConeLearnedOverEkfMedianSeq` (5.9), `\ConeLearnedOverEkfRangeSeq`; `\CtestLearnedOverEkfMedianSeq`, `..RangeSeq` |
| learned wins at 10/20/30/50 % mismatch, partial, realistic | `\CthreeMismatch{Ten,Twenty,Thirty,Fifty}{Id,Seq}VsWorking`, `\CthreePartial{Id,Seq}VsWorking`, `\CthreeRealistic{Id,Seq}VsWorking`; `\OutcomeCThreeMismatch*`, `\OutcomeCThreePartial`, `\OutcomeCThreeRealistic`; Fig. 3; Table 2 |
| crossover hypothesis FAILS (first winning level 10 %) | `\OutcomeCThreeMismatchCrossover`; Table 13 |
| learned observers fail under outliers / post-training drift | Table 10 (tab2_c3_full rows exact+outliers, drift30; dagger marks); Fig. 3 red-ringed markers |
| labels required; same-data baselines pending | design/limitations text; Section 5 "Two findings" (pending note) |
| C2 rules all FAIL | `\OutcomeCTwoFloor{TwentyFive,Five,One}`, `\OutcomeCTwoWindowAOne`, `\OutcomeCTwoWindowATwo` |
| TEST-C3 status | `\OutcomeTESTCThree` (INCONCLUSIVE), `\JobsTESTCThree` (6/18) |

### Section 2 (Framework): theory pointers only (Prop. m1-where, Cor. m1-amplitude, Prop. m1-tight,
Cor. m1-ood, Rem. m1-capacity; Lemma m2-tail, Thm m2-window, Cor. m2-rule; Prop. m3-mismatch,
Prop. m3-iss) -> Appendices A-C (theory/m1-m3, unchanged). Hyper subspace k=4 at 140k -> Table 8
(tab6_capacity). Framework figure (TikZ) carries no numbers.

### Section 3 (Experimental design)
Protocol table: design constants only (configs/base.yaml, node.yaml). Plants -> Table 5 (tab5_systems,
appendix). Decision rules -> Table 13 (tab4_decisions). Pending nodes named with
`\OutcomeCOneModelfree`, `\OutcomeCThreeDataFifty`, `\OutcomeCThreeDataOneFiveZero`,
`\OutcomeCOneBasewidthSixtyFour`, `\OutcomeCOneBasewidthTwoFiveSix`. Data problem (TEST-C1 from four
code snapshots) -> RESULTS.md section 7, reproduced in Appendix G.

### Section 4 (C1)
| claim | evidence |
|---|---|
| per-location "beats every other" counts | `\Cone{Lora,Hyper}BeatsAll{Shape,Seq}` |
| both LoRA and hyper beat latent & concat | `\ConeBothWeightSpaceBeatLatConc{Shape,Seq}` |
| better/worse vs latent per location, ranges | `\Cone{Lora,Hyper,Film,Concat}{Beats,Loses}LatentShape`, `\Cone{Lora,Hyper}VsLatentRangeShape` |
| bioreactor exception (latent best on sequence OOD) | Table 6 row bioreactor; `\ConeLoraLosesLatentSeq` (1/6) |
| held-out: fhn exception; latent above climatology on cartpole/highway (shape OOD) | Table 6 TEST block (dagger marks), `\Ctest{Lora,Hyper}VsLatentRangeShape` |
| controls | `\OutcomeCOneCtx`, `\OutcomeCOneBudget`, `\OutcomeCOneBudgetS/L`, `\JobsCOneBudgetS/L`, `\OutcomeCOneRankOne/Sixteen`, `\OutcomeCOneBasewidth*`, `\JobsCOneBasewidth*`, `\OutcomeCOneModelfree`; Fig. 2 (e) of the generated figure is cropped out (`trim`) and replaced by Table 13 |
| hyper k = 1/4/16, LoRA trunk 11/44/170 | Table 8 (tab6_capacity) |
| EKF reference ratios | `\ConeLearnedOverEkf*Seq`, `\CtestLearnedOverEkf*` |

### Section 5 (C3)
| claim | evidence |
|---|---|
| exact model: 0 wins in 15 conditions, incl. robust/oracle | `\CthreeFaultsRegimes`, Table 2, Table 13 |
| coloured-t: 0/6 | `\CthreeNoiseCtSeqVsWorking`, `\OutcomeCThreeNoiseCt` |
| learned above climatology under outliers (battery, duffing, quad_tank) and drift30 (bioreactor, quad_tank) | Table 10 / Table 11 (tab2_c3_full, tab2b_c3_id_full) |
| mismatch levels, same counts vs robust/oracle | `\CthreeMismatch*{Id,Seq}Vs{Working,Robust}` |
| duffing exception, its model error | `\ModelErrDuffingMismatchThirty` (0.02), `\CthreeMismatchThirtySeqDuffingBestWorking/BestLearned` |
| battery 10 % non-win | `\CthreeMismatchTenSeqBatteryVsWorking` (+7 %) |
| partial: quad_tank exception, model errors | `\ModelErrQuadTankPartial`, `\CthreePartialSeqQuadTankBestWorking/BestLearned`, `\ModelErr{Bioreactor,Lorenz,Vdp,Duffing}Partial`; "4/5 named plants" -> Table 13 verdict text |
| realistic per plant | `\CthreeRealisticSeq*BestLearned/BestWorking/Climatology/FailedFilters/VsWorking` |
| ordering of the 36 model-wrong cells | Fig. 3, Table 10 (the 30/36 count and the 0.064/0.152 band are NOT macros; the text states only the qualitative ordering and points to Appendix C, where theory/m3 quotes them) |
| NLOX column descriptive | Table 2 last column |
| Two findings; same-data baselines pending | visible red "pending" note + LaTeX TODO; no numbers |
| partial-modelfree FAIL | `\OutcomeCThreePartialModelfree` |
| TEST-C3 | `\OutcomeTESTCThree`, `\JobsTESTCThree`, `\CthreeTESTCThreeSeq{Fhn,Cartpole}VsWorking`, `\ModelErrCartpole{Partial,MismatchThirty}` |

### Section 6 (C2, supporting)
`\CtwoFloorFive{NoLoss,TostNonInferior,ProbeAboveFloor}`, `\CtwoFloor{TwentyFive,One}NoLoss`,
`\CtwoFloorFiveProbeRatio*`, `\CtwoFloorFiveBatteryLora{Id,Seq}`, `\CtwoFloorOneBatteryLoraId`,
`\CtwoLdecQ*`, `\CtwoSfmPoleOverAlpha{Duffing,Battery}Lora`, `\CtwoWindowTheory{Id,Seq}`,
`\CtwoWindowRefined{Id,Seq}`, `\CtwoLstar{Battery,QuadTank}`, `\CtwoLrefinedBattery`,
`\CtwoWindowTwoFiveSixVsMaxBatterySeq`, `\CtwoKnee{Vdp,Duffing,Bioreactor}Seq`; figures fig2a/b/c in
Appendix F. The refined rule is labelled post hoc everywhere.

### Section 7 (C4 / scorecard)
Fig. 4 (fig4_scorecard) and Table 12 (tab3_scorecard_realistic). Rank statements in the text are read
from Fig. 4 (no macros exist for ranks): filters top on accuracy/tails/convergence/recovery under exact
model; learned better only on divergence rate and (with PF) alarm; realistic: learned top on accuracy
columns, every filter [F3/6], PF/Huber best on alarm. **Re-read after regeneration.**

### Section 8 (Real-data validation)
| claim | evidence |
|---|---|
| H1 met 3/3 (WS-best vs tuned EKF; also vs UKF) | `\RealHOneLearnedBetter`, `\RealHOneEkfBetter`, `\RealWsBestEkf{USZeroSix,HWFET,NN}{Diff,CiLo,CiHi,Seeds}`, `\RealWsBestUkf*`; Table 14 |
| H2 met 3/3 (vs latent), small margin | `\RealHTwoWsBetter`, `\RealWsBestLatent*`, `\RealWsBestLatentDiff{Min,Max}Abs` |
| ordering weight-space > latent > concat >> autonomous | Table 14 (tab_real) |
| US06 biased filter SOC | `\RealEkfUSZeroSixRmse`, `\RealWsBestUSZeroSixRmse` |
| whole discharges: gap closes except US06 | `\RealWhole{WsBest,Ekf,Ukf}{USZeroSix,HWFTa,HWFTb,NN}` |
| convergence advantage on mild cycles | `\Real{WsBest,Ekf}{HWFET,NN,USZeroSix}Tconv` (appendix) |
| 0 degC: EKF better 8/9 | `\RealZeroEkfBetter`, `\RealZeroNCycles`, `\RealZeroWs{RmseMin,RmseMax}`, `\RealZeroWsBetterCycles` |
| model fit / identification | `\RealVmFit*`, `\RealEcm*`, `\RealQAh`, `\RealArrheniusFactor` |
| limitations | report.md's six items, verbatim in substance (Appendix G) |

### Sections 9-12
Related work: citations only (see refs.bib TODO-verify list below). Limitations quote
`\ConeLearnedOverEkfMedianSeq` only.

## Single-method framing (author decision, 2026-10-26 ~01:00 CET)
HyperKKL = a hypernetwork that writes rank-r updates of the decoder weights (eq. hyperkkl in Section 2;
r = 4 default; r = 1 / 16 are ablations C1-rank1/16; the dense-subspace generator 'hyper' is an
ablation, not a competitor). Until the results layer ships 'hyperkkl' macros, the single-method
sentences use the existing rank-4 ('lora') macros: `\ConeLora*`, `\CtestLora*` (C1), real-data
`\RealLora{Ekf,Latent}CyclesABetter` and `\RealLoraLatent*` (added to analysis/real_numbers.py); the
pre-registered weight-space verdicts (`\ConeWeightSpaceBeatsLatConc*`, `\CtestWeightSpaceBeatsLatConc*`,
'better of HyperKKL and its dense ablation') are kept as the pre-registered statements. TODO comment in
sections/c1.tex marks the swap; a HyperKKL-vs-concat count macro is needed. Real data: results.json has no
'hyperkkl' label, so Section 8 reports WS-best (the pre-registered per-seed choice r=4 vs dense ablation)
AND HyperKKL r=4 alone (beats EKF \RealLoraEkfCyclesABetter/3, beats latent \RealLoraLatentCyclesABetter/3
by the strict rule, same sign everywhere) — honest: the single method passes H2 strictly on 1/3 cycles.
New text also added on request: 'Why KKL observers' (Section 1; per-step cost ratio NOT quoted, no macro
exists — TODO in intro.tex), 'Scope and assumptions' (Section 2 Setting), C3-regimes pointer
(`\OutcomeCThreeRegimes`, fallback 'pending' in main.tex until outcomes.json has the node).

## Regeneration of 23:04 CET (already absorbed in the text)
- C1-budget-s/l now PASS (18/18, no asterisk), C1-basewidth-64/256 PASS* (6/18), C1-modelfree FAIL
  (= no evidence the phase-1 model term is needed; Section 4 "Controls" says so), C3-data50/150
  REPORTED (partial; named in Section 3 only), TEST-C3 9/18 INCONCLUSIVE with 2/3 learned wins
  (fhn, highway_traffic; cartpole UKF) — Section 5 last paragraph updated.
- NEW comparison (b), best KKL vs best DATA-DRIVEN same-data baseline (nlox, koopman_kf[_bilinear],
  greybox_ekf/ukf/ekf_oracle), NOT pre-registered: macros `\OutcomeCThree*DataDriven`,
  `\Cthree<node><tier>VsDataDriven`, per-plant `...BestDataDriven[Method]`, `...VsDataDriven`.
  C3-partial (b) = FAIL* (KKL beats best data-driven 1/5 id, 2/5 seq; grey-box beats KKL on battery,
  duffing; koopman_kf on bioreactor; KKL wins lorenz, vdp). All other nodes PENDING (0/18 merged;
  realistic 3/30). Section 5 "Two findings" now quotes these macros; abstract, Section 1 item 2 (b) and
  limit (vi), Section 9 "Same-data baselines", conclusion say "three of five decided plants" (read from
  the (3L) in Table 2 / the five per-plant macros — a count macro `CthreePartialSeqDataDrivenWins`
  would remove the last hand-read number). Table 2 (tab2_c3) gained the two (b) columns and still fits.
- Data problem added by the layer: C1-budget-s runs from 2 code snapshots (Appendix H).

## Places that depend on results still pending (refresh after the 08:07 Dubai regeneration)

1. **Same-data comparison (b) on mismatch10/20/30/50, noise-ct, realistic, TEST-C3** — PENDING
   (`\OutcomeCThreeRealisticDataDriven` printed in Section 5). When merged: rewrite the (b) sentences
   in Section 5, abstract, Section 1 (b), Section 9 from the generated macros/Table 2 only; if the
   grey-box/Koopman baselines win broadly, finding (b) must say so in the abstract.
1b. **Same-data baselines on the real cycles**: `\RealSameDataBaselines` flips to "present" once
   koopman/greybox rows are in experiments/real/results.json; then add their row to the Section 8 text
   from Table 14 (tab_real.tex regenerates with the extra rows automatically).
2. **TEST-C3** (`\OutcomeTESTCThree`, `\JobsTESTCThree`, `\CthreeTESTCThree*`): Section 1 limit (v),
   Section 5 last paragraph, Appendix F caption of fig3_c3_crossover_test, Appendix G, appendix
   overview table. The prose currently says "fhn won, cartpole lost, no reading offered"; if the node
   becomes PASS/FAIL, rewrite that paragraph (the macros will update the counts automatically, but the
   sentence structure assumes 2 finished plants).
3. **C1-budget-s / C1-budget-l** (`\OutcomeCOneBudgetS/L`, `\OutcomeCOneBudget`, `\JobsCOneBudgetS/L`):
   Section 4 "Controls" says "an asterisk marks a provisional verdict"; if the asterisk disappears or a
   verdict flips to FAIL, edit the sentence.
4. **C1-basewidth-64/256** (`\OutcomeCOneBasewidth*`, `\JobsCOneBasewidth*`): Section 3, Section 4,
   Appendix G; prose is verdict-neutral.
5. **C1-modelfree** (`\OutcomeCOneModelfree`): Section 3, Section 4, Appendix G; verdict-neutral.
6. **C3-data50 / C3-data150** (`\OutcomeCThreeDataFifty/OneFiveZero`): Section 3, Appendix G;
   verdict-neutral. If they finish, consider one sentence in Section 5 (learning curve) from the
   generated table, if one is produced.
7. **C3-partial / C3-realistic / C3-mismatch30 re-evaluation** (coordinator: full re-eval running):
   every `\CthreePartial*`, `\CthreeRealistic*`, `\CthreeMismatchThirty*` macro and Tables 2/10/11/12
   and Fig. 3/4 will change. The text's qualitative statements to re-check: exceptions named (duffing
   at every level, battery at 10 %, quad_tank partial), "all eight filters above climatology on
   bioreactor and quad_tank (realistic)", "lorenz 7/8 failed, kkl_mb works", scorecard rank sentences
   in Section 7, Appendix G "verbal readings corrected".
8. **Fig. 2 (e) cropped by `trim=0 171 0 0`** and **Fig. 3 bottom panel cropped by `trim=0 140 0 0`**:
   if the regenerated figures change their panel layout, re-check the crops (render pages 6 and 8).
9. **Real-data battery experiment**: included (Section 8, Appendix G, Table 14, Figure 10); every
   number via `generated_real/numbers_real.tex`. If results.json is re-run, `bash build.sh`
   regenerates the macros; re-read the Section 8 prose (H1/H2 "3 of 3", "8 of 9 at 0 degC",
   "US06 biased SOC") against the new macros.
10. Appendix G lists pending nodes with the same macros (self-updating).

## Open issues / things a reviewer may raise that the text does not fully answer
- Two hand-read statements remain (both flagged in the text as readings of tables, no numbers typed):
  the 30/36 ordering (Section 5, pointed to Appendix C where theory/m3 quotes 0.064/0.152) and the
  "4/5 named plants" partial-node pattern (from the Table 13 verdict text). Request macros
  `CthreeModelWrongWins`, `CthreeMinModelErrWin`, `CthreeMaxModelErrNoWin`, `CthreePartialPatternHolds`
  from the results layer to remove the last hand-read numbers.
- Scorecard rank sentences (Section 7) are read from Fig. 4 (no rank macros).
- Figures 2-4 are reduced to 0.56-0.68\textwidth to fit 12 pages; fonts inside are small (readable at
  zoom). If the editor allows, restore widths (main text would grow by ~0.7 page).
- `kollmeyer2018panasonic` (dataset) added from DATA.md.
- refs.bib entries marked `% TODO verify` (typed from memory, not in any repository .bib):
  julier2004unscented, gordon1993novel, kidger2020neuralcde, rubanova2019latentode, chen2018neuralode,
  revach2022kalmannet, haarnoja2016backpropkf, kirchmeyer2022coda, williams2015edmd, lusch2018deep,
  surana2016koopman, schuirmann1987tost, huber1964robust. Archive entries flagged VERIFY in
  aaai2027.bib (peralez2021/2022/2024, janny2021, buisson2023, niazi2023, ramos2020, bernard2017,
  chauhan2024) were copied as-is.
- `shaaban2026hyperkkl` is cited as "Anonymous (2026)" in the theory and related work (anonymity);
  restore the author list at camera-ready.
- The theory appendix (A-D) is `\input` from theory/; any edit there changes the paper. theory/main.tex's
  reading-guide paragraph is not included (replaced by appendix/overview.tex).
- Protocol v1 numbers are not in the paper (Appendix G describes v1 qualitatively).
- Compute statement ("a few hundred GPU-hours", "30-120 minutes per job", "519 jobs") comes from
  RESULTS.md's job count and the reviews' timing notes, not from a macro.
- The `network` plant "weakly observable" statement comes from the lead's instruction, not from a
  generated diagnostic; if a measured observability index exists, cite it.
