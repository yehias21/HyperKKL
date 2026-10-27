# Structural model error: results

Campaign finished 2026-10-26 21:58 Dubai (18:58 CET): 42/42 jobs (7 plants x {ref, struct} x seeds 0-2),
0 failures. Rules are in `PREREG.md` (written 16:58 Dubai, before any observer run) and its Addendum 1
(cartpole-excluded variant, written before any verdict). The verdicts and every number below come
mechanically from `code/analyze.py`. The full tables (all 3 eval tiers, seed ranges, per-state residual
R^2) are in `RESULTS_auto.md`, and the machine-readable output is in `scores.json`. Per-job results are in
`runs/<plant>/<tier>/seed<k>/results.json`.

Setup in one line. **ref** is the C3-realistic / TEST-C3 setting: partial model, 30% parameter error and
coloured Student-t noise. **struct** is the same plus one hidden-dynamics error in the real plant only:
quad_tank hidden 5th tank, battery OCV hysteresis, cartpole motor lag, highway on-ramp queue, bioreactor
growth lag, vdp sensor lag, lorenz hidden regime switch. HyperKKL is the beat_greybox **A4** stack
(lora r=4 + SOAP + base decoder trainable in phase 2 + z-window context + width 256 / 280k / 6000 steps).
The grey-box comparator is the best of greybox_ekf/ukf/ekf_oracle, picked per cell on seed means. Metric:
settled NRMSE. Statistics: paired relative change with a Student-t CI over 3 seeds.

## 1. Verdicts (primary eval tier sequence_ood)

| prediction | as pre-registered | cartpole excluded (Addendum 1) |
|---|---|---|
| **P-S1** struct: A4 beats best grey-box on >= 3 window-starved plants AND ref: A4 significantly worse on >= 3 | **FAIL**: (a) 2/4 (quad_tank, battery); (b) 1/4 (highway) | **FAIL**: (a) 2/3; (b) 1/3 (need 3/3) |
| **P-S2** rich plants: A4's margin over the best grey-box does not shrink significantly | **PASS** (bioreactor -16 pp [-22, -10], vdp -40 pp [-45, -36], lorenz +1 pp [-1, +3]) | n/a (unchanged) |
| **P-S3** Spearman(grey-box degradation D, 1 - rho_struct) >= 0.6 | **FAIL**: 0.536 (n = 7, p = 0.22) | **FAIL**: 0.543 (n = 6) |

The counts on id and shape_ood are identical to sequence_ood: struct 2/4, ref 1/4.

## 2. NRMSE, sequence_ood (seed means; seed ranges in RESULTS_auto.md)

| plant | tier | HyperKKL A4 | A0 | best GB (3 std) | gbw_ekf (window control) | ekf | climatology | A4 vs best GB [95% CI] |
|---|---|---|---|---|---|---|---|---|
| quad_tank | ref | 0.0234 | 0.0438 | 0.0170 | 0.0180 | 4.68 | 0.65 | +38% [-8, +83] n.s. |
| quad_tank | struct | 0.0370 | 0.0742 | 0.0745 | **0.0269** | 4.70 | 0.67 | **-50% [-64, -36]** |
| battery | ref | 0.0135 | 0.0653 | 0.0209 | 0.0164 | 0.296 | 0.79 | **-35% [-53, -18]** |
| battery | struct | 0.0173 | 0.0588 | 0.0280 (oracle) | 2.63 (diverges; gbw_ukf 0.063) | 0.302 | 0.79 | **-38% [-63, -12]** |
| cartpole | ref | 0.562 | 0.597 | 0.0640 | 0.0620 | 0.207 | 1.26 | +779% [-531, +2089] n.s. |
| cartpole | struct | 0.509 | 0.455 | 0.166 | 0.114 | 0.259 | 1.32 | +208% [+107, +308] worse |
| highway_traffic | ref | 0.0770 | 0.130 | 0.0273 | 0.0467 | 0.139 | 0.79 | +183% [+60, +306] worse |
| highway_traffic | struct | 0.0683 | 0.136 | 0.0304 | 0.0497 | 0.149 | 0.81 | +123% [+50, +196] worse |
| bioreactor | ref | 0.0126 | 0.102 | 0.0523 | 0.0739 | 5.73 | 0.73 | **-76% [-82, -70]** |
| bioreactor | struct | 0.0341 | 0.188 | 0.412 | 1.77 | 5.87 | 1.42 | **-92% [-92, -91]** |
| vdp | ref | 0.0182 | 0.0322 | 0.0389 | 0.0377 | 0.462 | 1.15 | **-53% [-58, -49]** |
| vdp | struct | 0.0213 | 0.0378 | 0.327 | 0.305 | 0.462 | 1.15 | **-93% [-94, -93]** |
| lorenz | ref | 0.0054 | 0.0101 | 0.0967 | 0.196 | 3.17 | 0.82 | **-94% [-97, -92]** |
| lorenz | struct | 0.0100 | 0.0277 | 0.151 | 0.428 | 3.86 | 0.92 | **-93% [-94, -92]** |

Grey-box degradation from ref to struct (D) and repairability: quad_tank +337% (rho_struct 0.997),
battery +34% (0.884), cartpole +159% (0.933), highway +11% (0.995), bioreactor +687% (0.676),
vdp +741% (0.359), lorenz +56% (1.000). Over the same step HyperKKL A4 changed by +58%, +29%, -9%, -11%,
+172%, +17% and +85%.

## 3. What failed, and why (descriptive, not verdicts)

* **P-S1(b), the premise, failed first.** With the A4 recipe, HyperKKL no longer loses to the grey-box in
  the ref tier on battery (-35%, significant) or quad_tank (+38%, n.s.). The A0 recipe still loses there
  (+215%, +157%). The "loses on window-starved plants" baseline of RESULTS.md P2 (A0-era runs) therefore
  holds for A4 only on highway_traffic (and on cartpole, where both HyperKKL recipes fail, see below).
* **Structure does not always hurt the grey-box.** It hurt it a lot on quad_tank (+337%), bioreactor
  (+687%) and vdp (+741%). It hurt it little on highway (+11%: a 20 s on-ramp queue is a mild low-pass of
  a small input) and battery (+34%: a 10 mV hysteresis). So highway stayed a grey-box win.
* **cartpole:** HyperKKL (A4 0.56, A0 0.60) is far worse than every filter, and its seed spread is huge. This
  matches the coordinator's inclusion finding: the original cartpole's unbounded cart random walk breaks
  KKL's bounded-state assumption. The cells are kept and reported. cartpole_b was not run here (future work).
* **Residual-capacity control.** On quad_tank the windowed grey-box (0.0269) beats A4 (0.0370, A4 +38%
  [+23, +52] worse). There, the grey-box's struct loss was mostly "residual too small" and not
  irreparable structure. On cartpole the window also helps (0.166 to 0.114). On battery the windowed EKF
  diverges, while the UKF version is worse than the plain grey-box. On bioreactor, vdp and lorenz the
  window does not help, or makes things worse (teacher-forced window states vs. filter estimates at
  deployment). Against the best of all five grey-box variants, A4 wins on 1/4 window-starved plants
  (battery).
* **P-S3:** the output-prediction repairability score misses the quad_tank case. The hidden tank changes
  tank 4, which is not measured, so the H = t_ref output prediction barely changes (rho 0.997). Yet the
  grey-box degraded 4x. The score works for errors that show up in the measured output (vdp, bioreactor),
  but not for unmeasured-state structure.

## 4. Post hoc (not pre-registered, labelled)

* Restricting to cells where A4 lost in ref (highway, plus cartpole as pre-registered), structure did not
  flip either one.
* On every plant except highway and cartpole, A4 is better than or level with the best standard grey-box
  in **both** tiers. The 2-D "information budget x repairability" picture is not what separates the cells
  here. What separates them is the recipe (A4 vs A0) and the size of the structural error.

## 5. Plain-language insight (5 lines)

1. Under hidden structural errors HyperKKL's lead on information-rich plants holds or grows (P-S2 PASS), and on quad_tank it flips a grey-box win into a 50% HyperKKL win.
2. The pre-registered "structure flips the window-starved plants" claim fails (P-S1 FAIL). This is partly because the A4 recipe already beats or ties the grey-box on battery and quad_tank without structure, and partly because the highway structural error barely hurts the grey-box.
3. A grey-box residual that sees a short history window recovers most of the quad_tank loss. Part of what looked like "structure" there is residual capacity, so a reviewer could fairly ask for that baseline.
4. Our a-priori repairability score (output prediction) does not predict grey-box damage (P-S3 FAIL), because structure in unmeasured states is invisible to it.
5. The original cartpole is not a valid KKL testbed (unbounded state). Its cells say nothing about the hypothesis; cartpole_b should be run next.

## 6. Provenance

Code: `code/structural.py` (structural plants + run-time patches; nothing in `hyperkkl/` edited),
`code/wingb.py` (window control, climatology), `code/run_struct.py`, `code/dispatch.py`, `code/analyze.py`,
13 unit tests in `code/tests/test_structural.py`. The package copy is `code/pkg` (beat_greybox scratch
package; sha1 in `code/pkg_snapshot.sha1`). The ref-tier data were read read-only from the live cache.
Logs: `logs/`, `dispatch.log`.
