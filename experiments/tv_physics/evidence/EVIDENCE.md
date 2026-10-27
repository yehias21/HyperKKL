# Physics-loss ablations: archive (AAAI/CDC) and current pipeline

Written 2026-10-26. Read-only on `archive/` and the paper. Scripts and raw outputs are in this
directory (run with `.venv/bin/python -I <script>` from here):

| script | output |
|---|---|
| `harvest_full.py <root> <out>` | `runs_HyperKKL.json` (808 runs), `runs_code.json` (482 runs): every `phase1.*`/`phase2.*`/`evaluation.*`/system scalar from `config.yaml`, per-tier per-family `nrmse_steady`, and `train_stats.*.extra` |
| `knobs.py` | `knobs_HyperKKL.txt`, `knobs_code.txt`: value counts of every physics knob, and the tags where a knob is off its default |
| `automatch.py` | `automatch.txt`: every same-(system, seed) run pair that differs in a physics knob and in at most 2 other keys |
| `compare_phys.py pairs_phys.txt` | `compare_phys.txt/.json`: matched arm-vs-control comparisons |
| `pde_replicates.py` | `pde_replicates.txt/.json`: PDE-on replicates at the PDE-off config (noise floor) |
| `tree_pde.py`, `tree_diag.py` | `tree_pde.txt/.json`, `tree_pde_logci.txt`, `tree_diag.txt`: current pipeline, PDE on vs off |

**Archive metric** (the same as `archive_recipe/STUDY.md` and the AAAI paper): per run, the median
over families of `nrmse_steady` for `dyn_gru`. Tiers: `ood` = graded OOD (constant, ramp,
sinusoid, am, prbs at 1.5-2.5x amplitude / 2-4x rate, which is beyond training); `waveform_ood` =
chirp and filtered_noise (in range, the closest analogue of v2 `shape_ood`); `id`. Pairs are matched
on (system, seed). Effect = arm/control - 1. Pooled: the geometric mean of the ratio with a paired-t
95% CI on the log ratio. Per system: the mean relative change ± the t half-width over seeds.

`hyperkkl_code/runs` is an exact subset copy: all 482 runs have results identical to
`HyperKKL/autoresearch`, so it adds no new evidence.

---

## 0. How often each knob was ever moved (808 archive runs)

| knob | values seen | non-default runs |
|---|---|---|
| `phase1.use_pde` | True 806, **False 2** | `pdefalse_fhn`, `pdefalse_bioreactor` (seed 0, legacy 25/15 epochs) |
| `phase1.lambda_pde` | 1.0 in all 808 | **never ablated** |
| `phase1.pde_u_scale` | 0.0 in all 808 | **never run in the archive artefacts** (only the cited claim below) |
| `phase1.weight_residual` | True in all 808 | **never ablated** |
| `phase1.grad_method` (label/PDE balancing) | sum; alm 7, invdir 11, relobralo 11, anneal/pcgrad/confg 4 each | §3 |
| `phase2.freeze_encoder_deltas` | True 624, False 180, **string 'False' 4** | composed `FA_*`/`FB_*`/`FC_*` (False) vs `FD_loo_frz` (True); legacy `RES2*`, `SOAPres*` |
| `phase2.residual_mode` | characteristic, jacobian 15 | `FB_resjc`, `RES2jac`, `RESjac`, `SOAPres_diff`, `SOAPres90_diff`; **`finite_difference` never run** |
| `phase2.w_label` | 1.0 in all 808 | **never ablated** |
| `train_injection(target=...)` | not a config key | always the default `input_induced` (`scripts/run_experiment.py:430`, `run_ablations.py:173,299,329,420`); **`full_residual` never run** |

**Bug found:** `RESchar` and `RESjac` (3+1 runs) passed `phase2.freeze_encoder_deltas='False'`
as a *string*. That string is truthy, so the encoder stayed frozen: `train_stats.dyn_gru.extra`
records `freeze_encoder_deltas: True`. The `res_ablation.sh:13` comment documents the fix ("the
override parser uses json.loads, so "False" stays a truthy STRING"). `RESchar` vs `SL_both` is
bit-identical (0.0% on every tier, 3/3 systems), so it is a free determinism check, **not** an
ablation. `RES2char` and `RES2jac` are the corrected reruns.

---

## 1. Phase-1 autonomous PDE (`use_pde`)

### 1a. Matched on/off (archive, seed 0, legacy schedule: Adam 1e-3, p1 25 ep / p2 15 ep, eval_rep 0)

The configs differ only in `use_pde`. The run really trained without the PDE term: the phase-1
`pde` loss is 0.0, and `extra.use_pde=False`.

| system | tier | PDE on | PDE off | off vs on |
|---|---|---|---|---|
| bioreactor | ood | 0.0240 | 0.0324 | **+34.7%** |
| bioreactor | waveform_ood | 0.0052 | 0.0133 | **+156%** |
| bioreactor | id | 0.0028 | 0.0091 | +225% |
| fhn | ood | 0.0433 | 0.0420 | -3.1% |
| fhn | waveform_ood | 0.0111 | 0.0119 | +6.6% |
| fhn | id | 0.0085 | 0.0086 | +2.0% |

This pair is n=1 per system, so no CI is possible.

**Noise floor (unpaired, same config, seed 0).** Several other tags ran at exactly the PDE-off
config, with PDE on, differing only in keys left at their defaults (`pde_replicates.txt`).
- bioreactor, PDE on, n=8: ood 0.0202 ± 0.0085 sd (min 0.0149, max 0.0396). `pj0.0`, `r0`, `pb0.0`
  and `ol0.0` give 0.0154 exactly. `pdetrue_bioreactor` gives 0.0240 although it ran 3 min after
  `pb0.0` with the same config, so single-cell values move by about ±50%.
- PDE off (0.0324) lies above 7 of the 8 PDE-on replicates on ood and waveform_ood (only `T3way`,
  a 3-method run, is worse). Against the replicate mean, PDE off is +60% on ood and +121% on
  waveform_ood.
- fhn, PDE on, n=5: ood 0.0510 ± 0.0166. PDE off at 0.0420 is better than all 5 (-18% vs the mean).
  So on fhn, removing the PDE term is neutral to slightly helpful.

**Mechanism (bioreactor, `diagnostics`).** Turning the PDE term off sends the decoder Lipschitz
proxy `l_dec_q95` from **16.6 to 1781**. `eps_dec` goes from 0.52 to 2.50 and `sigma_min_J` from
0.070 to 0.0041. The PDE term regularises the base map's conditioning.

Archive write-ups:
- `HyperKKL/autoresearch/MASTER_TABLE.txt:47-51`, "[F] DOES KKL NEED THE MODEL? phase-1 PDE term
  ablated": bioreactor 0.0211 (on) vs 0.0331 (off); fhn 0.0405 vs 0.0402. These come from a
  different aggregation and on-run.
- STUDY.md B11: "-3% / +35% (wave +7 / +156%) ... sign reverses ... keep PDE ... R, single cells".
- `code_review_gpt.md:336-349` (M2): before 07-26, "`phase1.use_pde` is ignored by the active
  trainer ... This invalidates a PDE ablation". It was fixed before the autoresearch campaign
  (`AUTORESEARCH_PLAN.md:25`, `JOURNAL.md:26`). The 07-27 pdefalse runs are post-fix, which the
  zero `pde` loss confirms.

### 1b. `pde_u_scale` (forced-input PDE on T0): cited, never in artefacts

- `JOURNAL.md:55-56`: "**PDE residual only at u=0 (#2).** `eps_pde ~ 0.019 + 0.42|u|`. Added as a
  per-system knob, default OFF: the effect reverses sign between duffing and vdp."
- `src/config.py:209-212`: "System-dependent: measured a monotone BENEFIT on duffing and a monotone
  HARM on vdp".

Both come from the *previous* autoresearch run ("Findings adopted from the previous autoresearch
run", `JOURNAL.md:44`). Every one of the 808 archived runs has `pde_u_scale = 0.0`, so there are no
numbers to recompute.

### 1c. Current pipeline: phase-1 PDE on vs off (tree, 3 seeds, paired)

`C1-modelfree` (`phase1.pde: false`) vs `C1-where`; `C3-partial-modelfree` vs `C3-partial`. Metric:
`scorecard.<method>.nrmse_mean` on the clean condition. The table gives the mean per-seed relative
change (off vs on) with a t 95% CI, n=3. **Positive = removing the PDE hurts.** These reproduce the
`paper/generated/RESULTS.md:699-714, 1029-1044` numbers exactly (they are quoted there as
"modelfree vs where").

**sequence_ood, lora (paper method)**

| system | C1: on -> off | change [95% CI] | log-ratio CI | C3-partial: on -> off | change [95% CI] | log-ratio CI |
|---|---|---|---|---|---|---|
| battery | 0.0719 -> 0.0649 | -9% [-30, +13] | -9% [-29, +17] | 0.0744 -> 0.0649 | **-13% [-25, -0]*** | -13% [-24, +0] |
| bioreactor | 0.0617 -> 0.5813 | +825% [-25, +1675] | **+786% [+272, +2013]*** | 0.0753 -> 0.5813 | +726% [-560, +2012] | **+634% [+76, +2955]*** |
| duffing | 0.0570 -> 0.0541 | -5% [-11, +2] | -5% [-11, +2] | 0.0536 -> 0.0541 | +1% [-22, +24] | +1% [-20, +28] |
| lorenz | 0.0266 -> 0.0201 | **-25% [-48, -2]*** | -25% [-46, +3] | 0.0308 -> 0.0201 | -33% [-78, +13] | -34% [-65, +25] |
| quad_tank | 0.0700 -> 0.0654 | -6% [-25, +12] | -7% [-23, +14] | 0.0717 -> 0.0654 | **-9% [-10, -7]*** | **-9% [-10, -7]*** |
| vdp | 0.0191 -> 0.0193 | +1% [-21, +23] | +1% [-18, +25] | 0.0179 -> 0.0193 | +8% [-10, +26] | +8% [-9, +27] |

Values are means over seeds 0-2. The two model-free nodes give identical runs: with no f anywhere,
partial knowledge is irrelevant. Per seed (`tree_diag.txt`), bioreactor lora seq is
s0 0.0666 -> 0.8756, s1 0.0627 -> 0.4924, s2 0.0557 -> 0.3760 (on -> off). That is **worse on 3/3
seeds, by 7-13x**. The plain-t CI on % crosses 0 only because the magnitude varies; the log-ratio CI
excludes it.

duffing has runs in the tree, but `RESULTS.md` judges 5 systems. Use `tree_pde.txt` for every
method and tier (latent, concat, film, hyper, autonomous; shape_ood, id).

**Summary of the current pipeline:**
- **bioreactor** is the only system where removing the PDE is catastrophic: every method, x5-x20
  (log-ratio CIs exclude 0). The mechanism is the same as in the archive: `l_dec_q95` goes from 44
  to 1690 and `eps_dec` from 0.020 to 0.138 (`tree_diag.txt`).
- On **lorenz, battery and quad_tank**, removing the PDE *helps* lora by 6-34%, significant in 3
  cells. lorenz is the system whose map violates the PDE (`eps_pde_rms` 77 even with the PDE on).
- **vdp and duffing** are neutral.
- Pooled over 18 (system, seed) pairs, PDE off is a geo-mean +33% worse for lora on C1 seq
  [-15, +106], entirely driven by bioreactor.
- Both pre-registered rules FAIL (0/5 systems): `RESULTS.md:127` "C1-where beats model-free phase 1
  ... shape 0/5; seq 0/5"; `RESULTS.md:176` "C3-partial ... beats model-free phase 1: id 0/5; seq
  0/5".

### 1d. Current pipeline: P4 (grey-box PDE), `experiments/pipeline/RESULTS.md`

P4 is **not** an on/off test. It replaces the partial f0 in the phase-1 PDE with
`f_known + r_theta` fitted on data (`PREREG.md:31`).

Verdicts (`RESULTS.md:19-20`): P4-lora NOT BETTER (0/6, worse 0) on C3-partial and (0/6) on
mismatch30. P4-hyper NOT BETTER (0/6) and (1/6). In `RESULTS.md:37`: "P4 (grey-box PDE) does
nothing ... it moves the PDE residual and eps_dec but not the deployed error".

C3-partial, sequence_ood, P4-lora vs P0-lora:

| system | change [95% CI] |
|---|---|
| battery | -2% [-42,+38] |
| bioreactor | -14% [-64,+37] |
| duffing | -1% [-18,+17] |
| lorenz | -7% [-51,+37] |
| quad_tank | -5% [-12,+2] |
| vdp | +10% [-23,+42] |

The only significant P4 cells are on mismatch30 bioreactor: P4-hyper -52% [-97,-8]* (seq), and
P4-lora -51% [-99,-3]* (shape).

---

## 2. Phase-2 time-varying residual (`freeze_encoder_deltas`, `residual_mode`)

In the code, `freeze_encoder_deltas=False` turns on three things together (`training2.py:958-961,
1101-1124`):
- encoder LoRA deltas;
- the encoder-side TV residual (`l_char`);
- the label term `w_label*||T(x;c)-z_char||^2`.

The deployed observer never evaluates the encoder. So the residual can act only through the shared
hypernetwork trunk and the GRU context. **The residual alone has never been isolated from encoder
modulation.** `w_label` was never varied.

### 2a. Residual ON vs OFF

| comparison | n | tier | effect of turning the residual OFF (freeze) [95% CI] | per system |
|---|---|---|---|---|
| **composed @BEST**: `FD_loo_frz` (frozen) vs `FA_wt` (unfrozen + char residual); differs only in this key; 3 sys x 3 seeds paired | 9 | **ood** | **-19.6% [-29.0, -9.0]** (median -22.3%) | bioreactor -29.9 ±18.4*, vdp -22.5 ±9.7*, duffing_mult -3.6 ±25.7 |
| same | 9 | waveform_ood | +6.3% [-2.5, +15.9] | bioreactor +22.9 ±16.7* (worse), vdp -1.1 ±5.2, duffing_mult -1.1 ±1.1 |
| same | 9 | id | +2.8% [-2.8, +8.7] | bioreactor +12.8 ±9.8* (worse), others ~-2% |
| legacy SOAP+L-BFGS 60/40 ep: `RES2char` (on) vs `SL_both` (off), seed 0; only extra diffs are inert curriculum defaults | 3 | ood | residual ON +9.8% [-30.9, +74.6] (bioreactor +22, duffing +23, vdp -12) | n=1 per system |
| same | 3 | waveform_ood | ON +26.6% [-50, +224] (bioreactor +96%) | |

Reading:
- At the composed config, removing the residual (and the encoder modulation) **improves graded OOD
  by about 20%** (8/9 pairs).
- It is neutral on the in-range waveform tier, except bioreactor, where it costs +23% on waveform and
  +13% on id, both significant.
- The legacy pair points the same way: ON is worse on 2/3 systems.

This is the AAAI supplementary result (`final paper/spine_supplement.tex:1600-1645`, tab:s-loo,
"removing encoder modulation *improves* the median by 22% ... on eight of nine pairs").
Confounds: recipe-selection systems are in-sample, and the composed recipe was selected *with* the
residual on.

### 2b. Residual form (both arms residual ON)

| comparison | n | ood | waveform_ood | id |
|---|---|---|---|---|
| composed: `FB_resjc` (jacobian) vs `FA_wt` (characteristic), paired 3x3 | 9 | +9.1% [-26.7, +62.4] (median +21.7; duffing_mult +58 ±37*) | **+21.6% [+11.1, +33.0]** | **+17.8% [+5.2, +31.7]** |
| legacy SOAP+L-BFGS: `RES2jac` vs `RES2char` | 3 | +18.9% [-13.7, +63.8] | +7.0% [-13.5, +32.2] | +2.5% [-31, +52] |
| legacy SOAP 60/40: `SOAPres_diff` vs `SOAPres_int` (duffing s0) | 1 | +33.9% | +28.4% | +26.2% |
| legacy SOAP 90/60: `SOAPres90_diff` vs `SOAPres90_int` (duffing s1) | 1 | +58.9% | +32.0% | +32.8% |

The Jacobian (differential) form is consistently worse than the characteristic (integral) form. But
both are compared against a residual that itself hurts graded OOD (2a), so this ranks forms of a
term that is better left off. `finite_difference` was never run.

Archive notes:
- `DECK_BACKGROUND.md:290-291` ("integral, characteristic form -21 / analytic Jacobian +27").
- `DECK_BACKGROUND.md:400-404` and `spine_supplement.tex:1655-1658`: an L-BFGS polish "masks the
  integral residual gain entirely".
- `QUEUE.md:44-45`: "only 4 legacy runs set freeze_encoder_deltas=false ... so the lever cannot be
  isolated". Those 4 are partly the string-bug runs above.
- `JOURNAL.md:46-48, 64-70`: the original root cause was that "The phase-2 objective was satisfied
  through `delta_enc`, which inference discards (19x more modulation to the encoder; modulation
  on/off gave identical losses)". That is why the deployment objective `||T0+(z_char;c)-x||^2`
  replaced the residual as the main loss.
- `JOURNAL.md:74-76`: "Phase 2 `char` vs `recon`: cos = **-0.475**, 100% adverse".

---

## 3. Label/PDE balancing (`grad_method`, the only other way `lambda_pde` was effectively varied)

| comparison (arm vs control) | n | ood geo-mean [95% CI] | waveform_ood |
|---|---|---|---|
| composed ReLoBRaLo (`FB_relo` vs `FA_wt`) | 9 | -4.2% [-14.9, +7.7] (vdp -16.6 ±5.6*) | +2.2% [-1.9, +6.4] |
| composed inv-Dirichlet (`FB_invd`) | 9 | +5.3% [-14.2, +29.2] | +4.0% [-1.7, +10.0] (id +6.4% [+0.4, +12.8]) |
| composed ALM (`FB_alm`, bioreactor) | 3 | +8.3% [-42.6, +104.4] | +2.3% [-11.7, +18.6] |
| legacy ALM / ReLoBRaLo / invDir vs `LAT_enc` (seed 0; also differ in optimizer-key presence) | 4 / 2 / 2 | +14% / +9% / +69% (n.s.) | +13% / +3% / +44% |
| legacy anneal / PCGrad / ConfG (both phases) vs `QL_zero` | 3 each | +352% / +92% / +307% (bioreactor anneal +1128%) | +862% / +136% / +723% |

No balancing scheme helps reliably. The magnitude-equalising methods, which demote the PDE term,
are strongly harmful.

---

## 4. Phase-2 injection heads (`train_injection` target)

**Never ablated in any archived run.** `target="full_residual"` exists only as an option
(`training2.py:452-473`). Every call site uses the default `input_induced`.

The docstring cites a one-off measurement: "on duffing after Phase 1: ||(B)|| = 2.70e-2 while
||(B)-(A)|| = 1.65e-2 -- 61% of the nominal target is base-map error ... Regressing on (B) therefore
floors the normalised loss near 1 and the head learns almost nothing, which is exactly what was
observed."

The omega/phi heads also never received the deployment objective. `FINDINGS.md:118-122` says:
"omega_gru trains to a LOWER loss than dyn_gru (8.7e-3 vs 1.24e-2) and evaluates 18x WORSE (0.363
vs 0.0205) ... NOT fixed (OmegaNet dropped by instruction)". So any omega/phi result in the archive
is invalid as physics-loss evidence. The ledger (`results.tsv`) carries no notes on any of these
knobs.

---

## 5. Bottom line

1. **Phase-1 PDE:**
   - In the archive it was ablated only twice: seed 0, legacy short schedule, n=1 per system. It
     matters on bioreactor (+35% ood, +156% waveform when removed; outside the PDE-on replicate
     spread) and not on fhn.
   - The current pipeline confirms this with 3 seeds. Removing it is catastrophic on bioreactor
     (x7-x9, CI excludes 0) through decoder conditioning (`l_dec_q95` x40), mildly *helpful* on
     lorenz, battery and quad_tank, and neutral on vdp and duffing.
   - The pre-registered "model needed" rules fail 0/5. `lambda_pde`, `weight_residual` and
     `pde_u_scale` were never ablated in any artefact.
2. **Phase-2 TV residual:**
   - The only clean test (composed, 9 paired cells) says that turning it off (with the encoder
     deltas) **improves** graded OOD by 19.6% [-29.0, -9.0]. It is neutral on the in-range tier,
     except bioreactor (+23% worse).
   - Among residual forms, characteristic beats Jacobian (+22% [11, 33] waveform), but the residual
     itself is net-negative on OOD.
3. **Injection-head target:** there is no evidence. `full_residual` was never run, and the heads
   were never moved to the deployment objective.
