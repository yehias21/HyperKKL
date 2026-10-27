# HyperKKL — experiments and insights at 09:00 Dubai, 2026-10-26

Scope: experiments only (no paper or figure editing, per the author). Mechanical verdicts come
from `paper/generated/RESULTS.md` / `outcomes.json` (regenerated 08:07 Dubai); side-study numbers
from their own result files, named per item. Method naming follows the author's decision:
**HyperKKL** = one hypernetwork generating rank-r decoder updates (headline r = 4); the dense
generator is an ablation only. "Beats" = paired 95% CI over seeds excludes 0.

## 1. Verdicts per contribution

| Contribution | Verdict | Evidence |
|---|---|---|
| **C1** where the input enters | **holds (weak form)** | single best location: FAIL (C1-where). HyperKKL beats latent AND concat: dev shape 5/6, seq 4/6; **held-out test plants TEST-C1 PASS 5/6, 5/6**. Controls: budget PASS, ctx PASS; rank r=1/16 not equivalent to r=4 (rank matters at a 140k budget). Pipeline study P3: gains come from input conditioning, not base repair. |
| **C2** memory | **modest support; pre-registered precision FAILS** | Old rules failed (C2-floor-05, C2-window). Audit (`experiments/memory/AUDIT.md`): **no code bug**; 4 test mis-specifications (probe slope not bounded by the lemma; rules omitted ln l and the design constant; 5% bar 2.5–10x stricter than eta; window encoder capacity confound) and the flat Lorenz probe was a heavy-tail artefact. Corrected fixed-architecture truncation (C2-trunc, pre-registered): FAIL at 5% (test S 2/6, T 1/6) BUT sufficiency holds in 43/48 held-out cells at 5% and **48/48 at 10%**; with the theorem's own design constant 33/36 test cells; current-input-only context is +112% worse (median, test) → "input history is necessary" confirmed out of sample; tightness only on large-l plants. |
| **C3a** learning beats a wrong model | **holds on dev and real data; held-out replication fails** | Dev: mismatch 10/20/30/50 PASS, realistic PASS 6/6 vs best working filter; exact model: filters win every condition (C3-faults: no learned-wins regime; noise-ct FAIL). **TEST-C3 FAIL** (single method). Real battery: HyperKKL beats tuned EKF/UKF on 3/3 held-out drive cycles. |
| **C3b** KKL is the best way to learn from the data | **no** | Same-data learners (grey-box EKF/UKF = model + learned residual; Koopman-KF; NLOX): HyperKKL beats the best of them on 0/6 (mismatch 10/30/50, noise-ct), 1/6 (mismatch20, partial seq), 2/6 (realistic seq). Real battery: inconclusive at 25 °C (grey-box wins US06, tie HWFET, HyperKKL wins NN); grey-box best on 9/9 cycles at 0 °C. |
| **C3-regimes** (where KKL should help) | **no KKL-wins regime** | clean 3/6, poor_init 3/6, noise10 2/6, outliers 1/6, drift10 3/6, outage 1/6. Consistent split by PLANT: HyperKKL wins bioreactor (every condition), lorenz, vdp; grey-box wins battery, duffing, quad_tank (except drift on quad_tank). Outliers/outage: HyperKKL above climatology on battery/duffing/vdp. Poor init: grey-box forgets the prior faster (0.6 s vs 2–4 s). |
| **C4** testbed | **holds** | 12 plants, shape/sequence OOD inside actuator bounds, partial-knowledge models, 8 filters + data-driven baselines, climatology reference, deployment scorecard, pre-registered rules applied mechanically, browsable history. |

## 2. Insights learned tonight

1. **The grey-box filter is the hard baseline.** "Your model + a ~5k-parameter residual inside an EKF/UKF", trained on the same labelled data, beats the learned KKL observer on smooth plants whose missing physics a small residual can fit (battery, duffing, quad_tank) — in simulation and on real cells.
2. **The winner is predictable from the plant, not the condition.** HyperKKL wins where the residual cannot repair the model (bioreactor: missing Monod kinetics) or the plant is chaotic (lorenz), and on vdp; this split is identical across clean, poor-init, noise, drift and outage conditions. → candidate paper thesis: *predict who wins* (section 4).
3. **KKL's error floor has two causes** (`experiments/diagnostics/ERROR_FLOOR.md`): noise amplification by the steep frozen base decoder (duffing, vdp: noise is 54–89% of squared error; slope ~1.0) and under-extraction (battery, bioreactor, quad_tank, lorenz: a wider probe on the same [z, c] gets 1.5–2.8x lower error). A short z-window does not help; a long strided window (~1/alpha) closes most of the noise loss. Even the best probe stays 1.6–3x above the grey-box → closing the last gap needs model information.
4. **SOAP is the strongest lever found** (`experiments/archive_recipe/RESULTS_TEST.md`): SOAP alone beats Adam (same lr) on 8/8 cells, −31% to −67%, mainly through phase 1 (base-decoder error 5–60x lower). With SOAP the learned observer beats every baseline on vdp/partial and lorenz (both scenarios), ties the grey-box on vdp/mismatch30, and shrinks the grey-box gap on duffing/quad_tank/battery from 3.4–9.5x to 1.1–2.5x. The full archived @BEST recipe is NOT superior (fails on lorenz/partial: longer phase 1 + curvature penalty hurt); encoder half, curvature penalty, GRU switch, phase-1 x3: no gain or harmful.
5. **Unfreezing the base decoder in phase 2 helps** (`experiments/pipeline/RESULTS.md`): P1 better in 23/36 cells (−11% to −52%); end-to-end from scratch does not; grey-box KKL (repaired model in phase 1) changes nothing deployed. No pipeline change alone closes the grey-box gap (0/8 cells).
6. **The single-method re-reading cost evidence**: TEST-C3 flips PASS→FAIL; real-data HyperKKL vs latent 1/3 (strict rule), vs EKF still 3/3. Original pre-registered outcomes are kept next to the re-readings (`tab4_decisions`, `\OutcomeOriginal*`).
7. **Exact model → filters win; this is consistent with the theory** (bias-variance floor proposition only bites when the model is wrong).
8. **Data note:** the C2-trunc `nets.py` edit changed the module hash; re-evaluating old cached models from the live package now needs the runs' code snapshot (the diagnostic used `experiments/code/a1e5d75cce66.tar.gz`). New jobs retrain phase 1 once (~2 min).

## 3. Pending at 09:00 (all on GPUs 5–7 from now on)

- Equal-budget tuning study (C1-tune-*): selection running; then confirmation (≈3–4 h).
- C3-faults same-data re-evaluation (0/30; ~40 GPU-h) — **recommended to drop** (C3-regimes answers the question).
- TEST-C3 same-data merge 14/18 (network seeds slowest).

## 4. Recommended next experiments (ranked)

1. **SOAP confirmatory node** (pre-registered): SOAP lr 1e-2 in both phases for EVERY variant (latent/concat/film/HyperKKL) — shared base maps so C1 stays fair — dev then once on test systems; split arm SOAP-phase-1-only vs phase-2-only. Highest expected gain.
2. **"Who wins" law** (the strongest paper thesis): measure per plant the grey-box residual fit (validation R²), relative model error, decoder Lipschitz l, chaos (largest Lyapunov exponent), noise; fit the rule on dev; **pre-register its predictions for the 6 held-out plants and the real battery**; test.
3. **Error-floor fix**: HyperKKL decoder on a long strided z-window (~1/alpha) + unfrozen base (P1) + SOAP; then the hybrid innovation correction as the fallback for the last 1.6–3x.
4. **Outlier gating** (eval-only on existing models): gate/Huber the measurement against the predicted output before the latent filter — fixes the worst visible failure.
5. **Battery across temperatures** (operating condition as hypernetwork context) — the 0 °C failure is the clearest real-data limitation and fits C1's idea.

## 5. Where to look

`paper/generated/RESULTS.md` (all mechanical verdicts), `tab4_decisions` (rules, incl. original vs re-read), `experiments/history.html` (browsable tree history), side studies: `experiments/{diagnostics,pipeline,memory,archive_recipe,real,tuning}/`, task list `TASKS.md`, ideas `IDEAS.md`.
