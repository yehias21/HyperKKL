# What the archive (AAAI/CDC campaign) established, and what transfers to protocol v2

Written 2026-10-25/26 (Dubai), read-only on `../archive/`. Nothing in `experiments/tree`,
`queue` or `cache` was touched; nothing was trained. Every number below marked **(recomp.)**
was recomputed by me from the archived run artefacts (808 `results.json` + `config.yaml`
under `archive/HyperKKL/autoresearch/`; the 482 runs in `archive/hyperkkl_code/runs/` are a
subset copy of the same tags) with the scripts in `study_data/`:

| script (run as `.venv/bin/python -I <script> ...` from a scratch dir) | output |
|---|---|
| `harvest.py <autoresearch> runs.json` | per run: tag, system, seed, 45 config knobs, per-family NRMSE(steady) + CI per tier/method, trainable params, learning flags |
| `catalog.py runs.json` | `catalog_HyperKKL.txt`: every tag, date, systems, seeds, methods, config signature |
| `compare.py runs.json pairs.txt out.json` | `compare_out.txt/.json` (+`compare_out2.json`): paired arm-vs-control deltas |
| `diag.py`, `v2table.py`, `select_systems.py` | phase-1 diagnostics; v2 node tables; the system-selection rule of section 7 |

**Metric used in all recomputed numbers.** Per run: median over the five graded OOD families
{constant, ramp, sinusoid, am, prbs} of `nrmse_steady` (exactly the paper's aggregate,
`final paper/HyperKKL_AAAI2027_v3.tex` "Aggregation"); "wave" = the same on the withheld waveform
tier {chirp, filtered_noise}. Paired by (system, seed); per system: mean relative change over
seeds with a Student-t 95% CI over seeds when n >= 2 (`±` = half-width); "sig" = CI excludes 0.
Across systems: median of the per-system means. Negative = the arm is better.

Sources read in full or in the relevant parts: `DECK_BACKGROUND.md` §5-12; `HyperKKL/autoresearch/
{FINDINGS,JOURNAL,RESEARCH_NOTES}.md`, `drain_final.sh`; `HyperKKL/{QUEUE,report,README}.md`,
`code_review_claude.md` (headers); `paper ready/{PIVOT,REFRAME,RUN_PLAN}.md` and `tables/*.md`;
`hyperkkl_code/paper_tables/*.md`; `AAAI_INTEGRATION_PLAN.md`, `AUTORESEARCH_PLAN.md`;
`final paper/AAAI_PLAN.md`, `HyperKKL_AAAI2027_v3.tex`, `tables/*.tex`; old code
`src/{config,training2,models2,optimizers}.py`, `configs/default.yaml`. New code: `README.md`,
`configs/base.yaml`, `hyperkkl/{conditioning,train}.py`, `experiments/{JOURNAL,HISTORY}.md`,
`experiments/tuning/PROTOCOL.md`, `paper/generated/RESULTS.md`, the v2 tree runs.

---

## 1. The archived "best" recipe (@BEST), precisely

`drain_final.sh:71` (`BEST=`), `:68` (`NZ=`, added to every run), `final paper/AAAI_PLAN.md:75-78`,
realised in `autoresearch/FA_wt/*/dev_seed*/config.yaml` (24 runs, 8 systems x seeds 0-2,
`evaluation.eval_rep = seed`):

| component | archive value | evidence |
|---|---|---|
| deployed observer | exact latent filter `z' = Az + By` + decoder `T0+(z; eta + eta~(c))`, **Phi = 0**; the encoder is never evaluated online | `training2.py:890-920` docstring; deck §7 |
| conditioner | `WeightHyperNet`: bias-free context gate (2x Linear+tanh) x layer-embedding features, one head per layer emitting LoRA factors U,V (rank 4, **second factor zero-init**), rsLoRA 1/sqrt(r), learnable scalar scale; per-sample deltas for **all encoder AND decoder layers** | `training2.py:766-863` |
| context | GRU (hidden 64) -> bias-free linear to c (dim 32), window = alpha-rule memory steps (282 on vdp) | `models2.py:302-342`, `config.yaml phase2.rnn_hidden/latent_dim` |
| ctx_scale guard | rescale c only if its RMS is outside [1e-2, 1] (target 5e-2); the GRU sat in range, so ctx_scale = 1 | `training2.py:965-980`, `config.py:313-315`, `FINDINGS.md:103-117` |
| base maps | encoder SiLU, decoder tanh, 2 hidden x 350 (vdp), frozen in phase 2 | run `config.yaml` (`num_hidden 2, hidden_size 350, activation silu, decoder_activation tanh`) |
| trainable params (phase 2) | **0.98M (bioreactor) - 1.50M (vdp, lorenz)**; rivals 28-37k (!) | recomp., `train_stats.*.n_trainable` |
| phase-2 objective | deployment loss `||T0+(z_char; c_k) - x||^2` + (because `freeze_encoder_deltas=false`) characteristic (integral) residual on the modulated encoder `T(x_{k+1}; c_{k+1}) - e^{A dt} T(x_k; c_k) - (W0 y_k + W1 y_{k+1} + W2 y_{k+2})` + label term `||T(x_k;c_k) - z_char||^2` + `delta_reg 1e-3 * ||dW_dec||^2` + `delta_curv 1e-4 * ||second difference of dW_dec / dt^2||^2` | `training2.py:1091-1160`, `config.py:298-324`, `drain_final.sh:68` |
| optimiser | **SOAP** (own implementation, betas 0.95/0.95, shampoo_beta 0.95, precond every 10 steps), **lr 1e-2**, phase 1 AND phase 2, cosine LR over epochs, grad-clip 1.0 | `optimizers.py:29-99,108-126`, `training2.py:950-954,1170` |
| phase 1 | 200 epochs (batch 4096) on 667 zero-input episodes, PDE residual (collocation, `weight_residual`), `grad_method=sum`; decoder 200 epochs on fresh states | `drain_final.sh:71`, `training2.py:209-420` |
| phase 2 schedule / data | 60 epochs, batch 1024, 667 forced episodes (crossed design, ~60% retained), default 10-family pool, stride 4 | `config.yaml` (FA_wt), `QUEUE.md` lines `@BEST phase2.epochs=60` |
| sensor / plant | non-affine actuator, 3% measurement + 0.5% process noise | `drain_final.sh:68` |
| evaluation | `latent_init=encoder`, 30 episodes, settle_frac 0.5, graded OOD at 1.5-2.5x amplitude / 2-4x rate + waveform tier | `config.yaml` |
| NOT used | rebalancing (ALM/ReLoBRaLo/NTK/inv-Dirichlet), L-BFGS polish, curriculum, specialist mixtures, Sylvester init, cond penalty, SFM | deck §9, `AAAI_PLAN.md:84-90` |
| where selected | axes B and C on **bioreactor, vdp, duffing_mult (in-sample, evaluation systems)**; axis A after seeing all 8 | deck:227-232; tex "Where the recipe was selected" |

Compute mapping (vdp): archive phase 2 = 60 epochs x ~360k samples = 21.6M samples vs v2 3000 steps x
8192 = 24.6M (**equal**); archive phase 1 = 200 epochs x ~400k = ~80M samples (~19.5k steps of 4096)
vs v2 6000 steps (**archive ~3x more phase-1 compute**). Archive phase-2 capacity ~1.5M vs v2 140k.

---

## 2. Validity key (used in every table below)

| code | issue | status in the archive | consequence for a row |
|---|---|---|---|
| **L** | first-order latent labels (order 1.01) and one-step look-ahead estimates | fixed 2026-07-26 19:16 **before** every autoresearch run (`JOURNAL.md:13-22`) | all 808 runs are post-fix; any CDC/pre-autoresearch number (29% SMAPE claim, old `augmented`) is invalid |
| **S** | SMAPE | withdrawn; all runs report NRMSE | none |
| **N** | measurement noise unreachable (`add_noise` no-op) and noise scaled by `y_bar` | found and fixed during the 07-27 noise studies (`FINDINGS.md:123-126`) | runs before ~07-27 03:00 (tags `main` .. `cv*`) may be noiseless or mis-scaled; `na` is noiseless by design |
| **V** | validation split ~4% instead of 15% | fixed late (`FINDINGS.md:183-185`); phase 2 does no val selection, so it affected only val-based choices | minor for recipe rows |
| **E** | absolute EKF covariance floors | fixed before the traffic result | baseline rows only |
| **G** | legacy configuration (Adam 1e-3, p1 60 ep / 200 ep., p2 40 ep, frozen encoder deltas) vs composed (@BEST) | every axis-B/C row of the paper carries "§ legacy" (`tab_axisB_body.tex`) | sign comparable, magnitude not; interactions known to be non-additive |
| **A** | graded OOD = amplitude 1.5-2.5x and rate 2-4x **beyond** training | the paper's headline aggregate | **v2 has no amplitude extrapolation** (bounded box); the archive's waveform tier (chirp, filtered_noise, in range) is the closest analogue of v2 `shape_ood` |
| **I** | in-sample selection systems (bioreactor, vdp, duffing_mult) | disclosed | effects there are optimistic |
| **R** | eval episodes fixed across training seeds in legacy runs (`eval_rep = 0`) | composed runs couple `eval_rep = seed` | only changing the eval draw moved the median by **+13% / +26%** (EVR1/EVR2 vs LAT_enc, recomp.): single-seed legacy deltas below ~25% are within noise |
| **C** | capacity mismatch in the axis-A ladder: weight rung 0.98-1.5M trainable vs rivals 28-37k; rival `phi_gru` diverged (non-finite / increasing loss) on 5 systems in FA_riv | recomp. (`learning_flags`, `n_trainable`) | the archive ladder cannot separate location from capacity (the tex itself: "Capacity is not matched") |
| **F** | phase-1 base map failed its learning check (loss increased) on bioreactor in the 25/15-epoch outlier runs | recomp. (`rank*`, `arch_*`) | T3 (SFM vs GRU) and T4 (rank) rest partly on a broken base map |
| **O** | 2% gross sensor outliers instead of the nominal sensor | `rank*`, `arch_*`, `sd_outlier`, `R_*` | not the nominal regime |
| **M** | `omega_*`/`phi_*` heads trained on the old indirect residual (train/inference mismatch) | NOT fixed (`FINDINGS.md:118-122`) | latent-rung numbers in legacy ladders are invalid as architecture evidence |

---

## 3. Ablation register

`n` = systems / (system, seed) pairs. "deck" = number quoted in `DECK_BACKGROUND.md` / the paper.

### 3.1 Architecture (where the input enters, capacity)

| # | varied | systems x seeds | effect (recomp.) | deck | verdict | validity |
|---|---|---|---|---|---|---|
| A1 | ladder at composed config: weight (dyn_gru) vs unconditioned / input-concat / latent-GRU, best of {SOAP@200ep, SOAP@60ep, Adam@1e-3} for rivals | 8 x 3 (FA_wt, FA_riv, FA_riv60, FA_rivA) | rivals worse by +263% (uncond.), +557..+683% (concat), +545..+878% (latent GRU) median; **vdp: concat beats weight -2..-11%** (n.s.); cartpole: unconditioned -14% | weight lowest 6/8 | weight-space wins in magnitude, not isolated | **C**, A; rivals at 2-4% of the parameters, phi_gru diverged on 5/8 |
| A2 | latent-GRU vs input-concat (both rivals) | 6 x 16 | +0.7% median (n.s. everywhere except one diverged cell) | "indistinguishable 7/8" | adding history in activation space buys nothing (rivals) | C, M |
| A3 | legacy ladder NZ11: phi_current / omega_current / phi_gru / autonomous vs dyn_gru | 6 x 1 | +299..+336% | E1 "6/6" | same as A1 | C, M, G, R |
| A4 | LoRA rank 2/8/16 vs 4 (bioreactor, 2% outliers, frozen encoder deltas) | 1 x 3 | -0.6 / -1.1 / -5.4% (n.s.) | "flat" | capacity not the lever (one system) | F, O, G; breadth deficit (`AAAI_PLAN.md:407-410`) |
| A5 | latent dim nz 3/5/10 vs ny(2nx+1) | 2 x 1 each | +54 / +4 / -33% | "over-specifying collapses the immersion" (quad_tank 18->6: -31%) | design choice, system-specific | R, G |
| A6 | decoder activation tanh vs ReLU | previous run | -15.7% | adopted | ReLU makes the PDE gradient wrong | pre-autoresearch, cited only |
| A7 | Sylvester warm start | 3 x 1 | +8% | refuted | do not use | R, G |
| A8 | phase-1 cond (sigma_min) penalty | 3 x 1 | -14% | not adopted | weak | R, G |

### 3.2 Context encoder

| # | varied | systems x seeds | effect (recomp.) | deck | verdict | validity |
|---|---|---|---|---|---|---|
| X1 | SFM vs GRU (dyn_sfm vs dyn_gru) | 3 x 3 | +251% median (bioreactor +264 ±130 sig, duffing +251 ±566, duffing_mult +18); wave +8% | "1.1-4.1x worse" | GRU better **in that setting** | **F, O**, G, 25/15 epochs; SFM needed the ctx_scale fix first (7300x gradient gap) |
| X2 | context window W (CW1..CW50, W8..W305) | 3 x 2 | W=1 catastrophic; knee at ~12% of the alpha-rule window (bioreactor 20 steps, vdp 33) | "~12% suffices" | short windows suffice for the GRU | G, R |
| X3 | ctx_scale forced unit RMS vs guard | traffic 1 x 1 | forcing: ramp +26%, sinusoid +17% | guard only | keep the guard | single cell |
| X4 | GRU zero-input collapse | measured | c(u=0) != 0 (0.245, 0.141) | disclosed | GRU breaks exact collapse | - |

### 3.3 Optimiser and step size

| # | varied | systems x seeds | effect (recomp.) | deck | verdict | validity |
|---|---|---|---|---|---|---|
| O1 | **SOAP vs Adam, both lr 1e-2** (S_full vs A_lr1e2) | 4 x 4 | **-35.3% median**; duffing -51 ±11, lorenz -36 ±15, vdp -35 ±22 (all sig), bioreactor +13 (n.s.); wave -17..-34% on all 4 | -29 (3/12) | **strongest single optimiser result**; preconditioning helps at matched lr | G, R (eval fixed, CI over training seeds only) |
| O2 | SOAP vs Adam@1e-2 **at the composed config** (FB_adam vs FA_wt) | 2 x 3 | Adam: bioreactor -35 ±99, vdp +58 ±72; wave -37 / +11 | not reported | **unresolved** at composed config | in-sample (I) |
| O3 | Adam lr 1e-3 -> 1e-2 | 3 x 3 (A_lr1e2 vs QL_zero) | -10.8% median; bioreactor -68 ±9 (sig), duffing -11, vdp +11 | -23 | higher lr helps, system-dependent | G, R |
| O4 | Adam lr 1e-2 -> 1e-3 at composed config (FB_lrlo vs FB_adam) | 1 x 3 | +49% (n.s.) | - | consistent with O3 | I |
| O5 | lr 3e-2 | 2 x 1 | +552% (diverges) | +1471 | ceiling at ~1e-2 | G |
| O6 | SOAP lr 1e-3 vs Adam 1e-3 (O_soap) / SOAP lr 1e-2 (O_soaphi) | 3 x 1 | -44% / -60% | - | SOAP helps at both lrs | G, R |
| O7 | SOAP + L-BFGS polish vs SOAP | 4 x 1 | +24% median (bioreactor +49, vdp +73) | +49 | harmful; also masks the residual-form gain | G, R |
| O8 | Adam + L-BFGS | 3-6 x 1 | +3.5% (O_lbfgs); L_bfgs diverged on 4/6 | - | do not use | G, R |

### 3.4 Loss (balancing, residual form, regularisers)

| # | varied | systems x seeds | effect (recomp.) | deck | verdict | validity |
|---|---|---|---|---|---|---|
| B1 | rebalancing at **legacy**: ALM / ReLoBRaLo / inv-Dirichlet / NTK-diag (phase 1) | 2-4 x 1 | +17 / +10 / +72 / +8% | +33/+10/+72/+11 | rebalancing hurts (legacy) | G, R |
| B2 | rebalancing at **composed**: ReLoBRaLo / inv-Dirichlet / ALM | 3 x 3 (ALM 1 x 3) | -4.5 (vdp -17 ±6 sig, bioreactor +12) / +1.3 / +10.6% | "unresolved" | **null** at composed config | I |
| B3 | gradient surgery anneal / PCGrad / ConfG (both phases) | 3 x 1 | +222 / +120 / +262% | "barely moved" (older) | harmful | G, R |
| B4 | residual: characteristic vs analytic Jacobian, SOAP (no polish) | duffing 1 x 1 (x2 budgets) | -25% / -37% | -21 | integral form better | R, single system |
| B5 | same under SOAP+L-BFGS | 3 x 1 | -17% median | "tie" | polish masks it | R |
| B6 | **Jacobian residual at composed config** (FB_resjc vs FA_wt) | 3 x 3 | **+32% median** (duffing_mult +58 ±37 sig); wave +20% (12..36% on 3/3) | - | integral form confirmed at composed config | I |
| B7 | **freeze encoder deltas (drop encoder LoRA + residual) at composed config** (FD_loo_frz vs FA_wt) | 3 x 3 | **-22.5% median**: bioreactor -30 ±18 (sig), vdp -23 ±10 (sig), duffing_mult -4; wave -1% | not reported | **the encoder half of @BEST hurts the graded tier, neutral on waveform** | I; the one leave-one-out of an @BEST ingredient that landed |
| B8 | delta_curv 1e-4 / 1e-1 vs none (traffic) | 1 x 1 | -14% / -28% graded, wave +1% / +0% | "strictly dominant at 1e-4" | small; one system, one seed | R, N (07-27 01:40) |
| B9 | first-difference smoothing (traffic) | 1 x 1 | -10..-28% | Pareto-equivalent to curv | same curve | R |
| B10 | delta_reg 1e-3 (decoder delta magnitude) | earlier run | dW 65x smaller, train loss lower | adopted | cheap regulariser | cited only |
| B11 | phase-1 PDE term off vs on | fhn, bioreactor 1 x 1 | -3% / +35% (wave +7 / +156%) | "sign reverses" | keep PDE | R, single cells |
| B12 | collocation 4x / 16x | 3 x 1 | -7% / -3% | -8 / +24 | negligible | G, R |

### 3.5 Data

| # | varied | systems x seeds | effect (recomp.) | deck | verdict | validity |
|---|---|---|---|---|---|---|
| D1 | label order 1.01 -> 4.01 | label error 1.1e-2 -> 7.9e-8 | not an NRMSE experiment | precondition | **already in v2** (true-plant labels, exact discretisation) | - |
| D2 | phase-2 volume 200 -> 667 episodes (legacy) | 3 x 1 | -52% median (bio -42, duffing -53, vdp -52); wave -28% | -47 | volume matters at legacy | G, R, I |
| D3 | **phase-2 volume 667 -> 200 at composed** (FC_vol200 vs FA_wt) | 2 x 5 pairs | **-17% median** (bioreactor **-51 ±47 sig**, vdp +16 n.s.); wave +1% | not reported | **contradicts D2**: at the composed config fewer episodes was not worse | I; vdp seed 2 missing |
| D4 | phase-1 volume 667 -> 200 at composed (FC_p1vol) | 3 x 3 | +6% (n.s.); wave +2% | not reported | phase-1 volume not the lever | I |
| D5 | specialist mixtures vs default pool, legacy (MIXlow/band/high/eval) | 1-3 x 1 | +41 / -6 / -11 / +4% | +24 (total), +132 (density) | composition second-order; band-matched helps a band, hurts another | G, R, A |
| D6 | mixtures at composed (FC_mixband / FC_mixhigh vs pool@200) | 2 x 5 pairs | +168% / +7% (n.s.) | - | specialists do not beat the generalist pool | I |
| D7 | curricula: spectral / reverse / soft / self-paced / soft-reverse | 3 x 1 | -4.5 / +8.8 / +10.4 / +4.3 / +6.9% | +6 / +20 | no curriculum | G, R |

### 3.6 Training schedule

| # | varied | systems x seeds | effect (recomp.) | deck | verdict | validity |
|---|---|---|---|---|---|---|
| T1 | phase-1 epochs 60 -> 200 (DEC_ep) | 3 x 1 | -25% median (all 3 better); wave -10% | -17 | more phase-1 helps | G, R |
| T2 | phase-2 epochs 40 -> 120 (EP2x) | 3 x 1 | -16% (all 3 better) | -26 | more phase-2 helps (legacy) | G, R |
| T3 | both (180/120) | 3 x 1 | -6% | - | not additive | G, R |
| T4 | short (25/15/80-200 ep.) -> full (60/40/200) | 6 x 1 | -19% (5/6) | E4 "up to -51%" | every early table understated the method | G, R, mixed configs |
| T5 | **composed vs default** (FA_wt vs FD_def; i.e. SOAP 1e-2 + p1 200 ep/667 + p2 60 ep/667 + encoder LoRA/residual) | 8 x 3 | graded: **-21.7% median**, better 5/8, sig better 4/8 (fhn -71 ±1, rossler -49 ±27, vdp -40 ±20, duffing_mult -46 ±18), sig worse 0/8 (lorenz +453 ±974, cartpole +82 ±303 n.s.); **waveform: -42% median, better on 8/8** (-15 .. -68%) | -10..-71 / lorenz +394-435 | the composed recipe helps on the in-range (waveform) tier everywhere; the graded-tier losses are amplitude-extrapolation losses | in-sample on 3/8; one measured point; no full leave-one-out (FC_p1ep60, FC_p2ep20, FC_coll were queued and never ran) |

Mechanism hint for T5 (recomp., `diag.py`): the composed phase 1 lowers the base-decoder error
`eps_dec` on 6/8 systems (vdp 0.114 -> 0.077, duffing_mult 0.16 -> 0.033, fhn 0.44 -> 0.27,
lorenz 7.0 -> 4.1, rossler 3.3 -> 2.3, cartpole 14.7 -> 2.6) but raises the decoder Lipschitz
proxy `l_dec_q95` on 5/8 (lorenz 85 -> 496, rossler 152 -> 964, quad_tank 67 -> 298,
duffing_mult 83 -> 199, cartpole 2 -> 112). A sharper base decoder fits the training box better
(waveform tier: in range) and extrapolates worse in amplitude (graded tier), which is exactly the
split in T5, including lorenz. None of the leave-one-outs that did land (B7, D3, D4, O2) explains
the -42% waveform gain (each moves the waveform tier by <= 5%, O2 bioreactor excepted), so the
untested remaining ingredients, **phase-1 training length (x3.3) and lr 1e-2 in phase 1**, are
the likely carriers. This is an inference from the LOO pattern, not a measured result.

### 3.7 Evaluation, baselines, robustness (not recipe components; listed for completeness)

| # | varied | effect | verdict | validity |
|---|---|---|---|---|
| E1 | eval draw (eval_rep 1, 2 vs 0) | +13% / +26% median (2 systems) | eval-set variance is large; motivates coupling eval_rep = seed | R |
| E2 | latent init encoder vs zero | 0.0% on steady NRMSE (4 x 3) | no effect on the settled metric | - |
| E3 | EKF exact model, Gaussian noise | learned worse on 40/45 cells (vanilla), all 6 vs Huber EKF | model-based filter dominates when the model is exact | consistent with v2 C3-faults |
| E4 | 2% outliers vs plain EKF (sd_outlier, 5 x 5) | learned better on 3/5 (bioreactor 44x); vs Huber EKF worse 5-100x | straw-man win, withdrawn | O |
| E5 | noise kinds, process-noise distribution, parameter mismatch (EKF blind), init radius, closed loop | see `final_evidence.md`, `MASTER_TABLE.txt` | EKF robust to process-noise shape; KKL init-invariant by construction; closed loop unresolved | N for the earliest tags |
| E6 | spectrum (zeta/butterworth), alpha x2/x4, dt | 2/7 better; alpha up monotonically worse | design choices, not recipe | R |

---

## 4. Synthesis: robust, invalidated, untestable

**Robust and likely to transfer to v2** (bounded inputs, shape/sequence OOD, true-plant labels,
settled NRMSE, matched 140k):

1. *Decoder-side weight-space conditioning with frozen base maps, a zero-preserving generator and
   zero-init second factor.* Already the v2 `lora`/`hyper` design; v2 C1-where confirms the
   direction at matched parameters (weight-space beats latent and concat on shape_ood 6/6,
   sequence_ood 5/6), which the archive could not (capacity, **C**).
2. *Train the deployment objective (decode the latent the observer carries).* Already v2
   (`train._decoder_loss`). The archive's own leave-one-out (B7) says the extra encoder-LoRA +
   characteristic-residual path of @BEST **hurts** (-22.5% when removed, 2/3 significant).
3. *Preconditioning (SOAP) at a matched learning rate* (O1: 4 systems x 4 seeds, 3 significant,
   waveform tier better on all 4). Not confirmed at the composed config (O2 split) and never
   tested at a matched parameter budget. **Best candidate that v2 does not have.**
4. *Longer phase 1 / better base decoder* (T1, T5 mechanism). v2 phase 1 is ~3x shorter than the
   archive's. Pipeline-level: it changes the shared base maps, not the conditioner.
5. *Higher learning rate up to ~1e-2, not beyond* (O3, O4, O5). v2 uses 2e-3 and the tuning grid
   stops at 4e-3.
6. *Rebalancing, gradient surgery, L-BFGS polish, curricula, specialist mixtures, larger collocation
   sets, Sylvester init do not help* (B1-B3, O7-O8, D5-D7, B12, A7): robust negatives, consistent
   across legacy and composed configs; no reason to retest.
7. *Rank / capacity is not the lever* (A4; v2 C1-budget-l at 560k is within ~10% of 140k; v2
   C1-rank16 not non-inferior on 3/6). Consistent in both codebases.

**Invalidated or not transferable:**

- The **magnitude** of the axis-A ladder (rivals 3-10x worse): capacity-confounded (40x) and
  rival training diverged (**C**, **M**).
- Every **graded-tier** (amplitude-extrapolation) number as a predictor for v2 (**A**). In
  particular the lorenz/cartpole "recipe failures" of T5 are amplitude-extrapolation failures;
  on the in-range waveform tier the same runs improve.
- Pre-autoresearch numbers (29% SMAPE, old `augmented` vs `autonomous`) (**L**, **S**).
- Single-seed legacy deltas below ~25% (**R**: the eval draw alone moves them by 13-26%).
- OmegaNet / latent-rung conclusions from legacy runs (**M**).

**Untestable from the archive (never run):** a full leave-one-out of @BEST (phase-1 epochs,
phase-2 epochs, collocation lines in QUEUE.md were queued and never dispatched); SOAP vs
SOAP+L-BFGS matched at composed config (1 run); any ingredient at a matched parameter budget;
any ingredient under model error (partial knowledge, mismatch), which is where v2's claims live.

**Conflicts with tonight's v2 results, and how they resolve:**

| archive | v2 | resolution |
|---|---|---|
| SFM 1.1-4.1x worse than GRU (X1) | v2 default context = SFM | X1 is invalid as a general claim (F, O, 25/15 epochs, archive SFM had a 7300x gradient gap until ctx_scale). The new code's own v1 C1-ctx (`experiments/tree_v1/tree/C1-ctx/report.md`) found lora@gru vs lora@sfm within CI on 5/5 systems (-57% .. +22%, none significant). Not a real conflict, but untested under protocol v2: a cheap `lora@gru` arm settles it. |
| weight space wins 6/8 (A1) | C1-where: no single location passes the pre-registered rule; weight-space beats latent+concat 6/6 shape, 5/6 seq (weaker claim); TEST-C1 PASS | same direction; v2 is the valid (matched) evidence. Archive vdp exception (concat <= weight) vs v2 vdp (lora beats concat by 30-40%): v2 wins on validity (C). |
| @BEST = LoRA on encoder + decoder with SOAP | v2 lora = decoder only, Adam 2e-3 | the archive's own LOO (B7) favours decoder-only; SOAP is the open item. |
| volume 200 -> 667 = -47% (D2) | v2 uses 600 episodes | D3 contradicts D2 at the composed config; v2 is already at the archive's volume; C3-data50/150 (pending) gives v2's own curve. |
| context knee at ~12% of the alpha window (X2) | C2-window FAIL (knee not where L* predicts; some systems need >= 512) | different encoders (GRU over windows vs SFM/window in v2) and different tiers; no transfer either way. |
| EKF with the exact model wins | C3-faults: learned beats no filter on 0/6 with the exact model | agreement. |

---

## 5. Ranked candidate components for a v2 recipe

Expected effects are on v2 shape_ood / sequence_ood settled NRMSE of `lora`, from the archive
evidence discounted for validity. Cost per (system, seed) on one GPU slot, from PROTOCOL.md §6
unit costs (phase 1 ~130 s, lora phase 2 ~30-70 s, evaluation 60-540 s).

| rank | component | archive evidence | expected | cost | notes |
|---|---|---|---|---|---|
| 1 | **SOAP** (phase 1 + phase 2), lr 1e-2, with an **lr-matched Adam 1e-2 control** | O1 (-35%, 3/4 sig), O6; O2 unresolved | -10 .. -30%, system-dependent | phase-1 rebuild (~150 s) + ~1.2x step cost; vendor the archive's 100-line SOAP (`optimizers.py:29-99`; no SOAP package in the venv) | must be equal steps; it is an optimiser change, so apply it to every variant if it enters C1 |
| 2 | **phase-1 budget x3** (`steps_enc`, `steps_dec` 6000 -> 18000) | T1 (-25%, 3/3, seed 0), T5 mechanism (eps_dec down on 6/8) | -5 .. -25% where the base-decoder error dominates; risk: larger decoder Lipschitz (watch `l_dec_q95`) | +~260 s per (system, seed), cached and shared by all variants | **not equal compute**: report as a budget arm; affects all variants equally, so it cannot bias C1 |
| 3 | **lr 1e-2** (Adam) | O3/O4 (+), O5 (3e-2 diverges) | 0 .. -15% | free | above the tuning grid's 4e-3 |
| 4 | **delta_curv 1e-4 + delta_reg 1e-3** on the decoder deltas | B8 (-14%, 1 x 1), B10 | 0 .. -10%, mostly oscillatory shape_ood (chirp, triangle, sawtooth) | ~1.5x phase-2 step cost (contexts at k+1, k+2) | needs per-sample deltas at three consecutive times |
| 5 | **GRU context** (`lora@gru`) | X1 (invalid), v1 C1-ctx n.s. | ~0 | low | settles the SFM conflict; the archive GRU broke zero-input collapse (X4), the v2 `GRUEncoder` subtracts the zero-input response (`nets.py:146-170`) so c(0) = 0 exactly |
| 6 | **encoder-LoRA + characteristic-residual auxiliary** (`freeze_encoder_deltas=false`) | **B7: removing it is -22.5%** | >= 0 (harm) | ~2-3x phase-2 cost | only in the "faithful @BEST" arm |
| - | not recommended | rebalancing, grad surgery, L-BFGS, curricula, specialist mixtures, collocation x4/16, Sylvester init, rank/capacity, more data volume | ~0 or harmful | - | robust negatives (section 4) |

---

## 6. Proposed equal-budget test (to run only on the lead's go-ahead)

Everything runs in a full copy of the package in the session scratchpad
(`.../scratchpad/archive_recipe/pkg/`) with its own cache directory, outputs only under
`experiments/archive_recipe/`, `CUDA_VISIBLE_DEVICES` in {5,6,7}, < 10 GB per job. The live
package is not modified; adoption into the pipeline is decided only after the 09:00 Dubai report
and only if the rule in 6.4 holds.

### 6.1 Arms (all lora unless stated; all at `phase2.budget = 140000`, 3000 phase-2 steps, 8192 samples/step, same data, best-val checkpoint, same evaluation episodes)

| arm | what | compute vs A0 |
|---|---|---|
| A0 | current default `lora` (and `hyper` as the second reference) | = |
| A1 | + SOAP lr 1e-2 (phases 1 and 2) | = steps |
| A2 | + Adam lr 1e-2 (phases 1 and 2): the lr-matched control for A1 | = |
| A3 | + phase-1 x3 (Adam 2e-3) | phase 1 x3 (reported) |
| A4 | + GRU context | = |
| A5 | + delta_curv 1e-4 + delta_reg 1e-3 | = steps |
| A6 | + encoder LoRA + characteristic residual auxiliary | = steps |
| **A7** | **faithful @BEST** = A1 + A3 (SOAP x3) + A4 + A5 + A6 | phase 1 x3 |
| **A8** | @BEST without the encoder half (the archive's own better LOO) = A7 - A6 | phase 1 x3 |

One-at-a-time additions plus the two composites, not a factorial: 2^6 = 64 arms does not fit,
and the archive already shows the ingredients are not additive (O7, B5, D2 vs D3). An archive-size
conditioner (~1.5M) is reported once for A8 on 2 cells if time remains (capacity was flat in both
codebases, so it is low priority).

### 6.2 System selection (unbiased; rule fixed now, applied by `study_data/select_systems.py`)

Scenarios: `C3-partial` and `C3-mismatch30` (dev systems, nominal sensor, clean condition), tier
`sequence_ood`. Per cell: L = min(mean lora, mean hyper) over the node's seeds (current
defaults); B = min over **every** non-HyperKKL method in the run (EKF/UKF/PF/kkl_mb, Huber and
oracle filters, NLOX, koopman_kf*, greybox_*), using only methods present on all seeds;
r = L / B. Per scenario take the **2 cells with the largest r (current recipe loses most) and the
2 with the smallest r (wins most)**: 8 cells. The selection is frozen by running the script
**after the running data-baseline re-evaluation has finished** (greybox/koopman are still missing
on some seeds of C3-partial duffing/vdp/lorenz/quad_tank and on all of C3-mismatch30), and its
JSON output is committed to `experiments/archive_recipe/selection.json` before any arm is trained.

Provisional output (`study_data/selection_provisional.json`, computed 2026-10-26 ~01:45 Dubai):

| scenario | LOSE (largest r) | WIN (smallest r) |
|---|---|---|
| partial | battery (r 3.75, greybox_ekf 0.019 vs hyper 0.072); quad_tank (3.10, ekf_oracle) | vdp (0.58, vs NLOX; greybox on the seeds available: 0.0225 vs lora 0.0179); lorenz (0.67) |
| mismatch30 | bioreactor (2.77, vs NLOX); duffing (2.60, vs UKF) | battery (0.29); quad_tank (0.32) |

Expected change once the re-evaluation completes: duffing/partial moves to LOSE (greybox_ekf
0.0076 vs lora 0.054 on the seeds available, r ~ 7), which would displace quad_tank/partial.
On the data so far, **vdp/partial is a WIN for lora against greybox_ekf on shape_ood and
sequence_ood** (id is a tie: 0.0157 vs 0.0159); the "vdp loses to the grey-box EKF" premise is
not supported by the finished seeds; mismatch30 has no data baselines yet.

### 6.3 Seeds and cost

Seeds 0, 1, 2 for every arm; seeds 3, 4 added for A0, A7, A8 (the primary contrasts) if the
first wave finishes inside 4 h. Per (cell, seed): 5 distinct phase-1 trainings (Adam 2e-3,
Adam 1e-2, SOAP 1e-2, Adam x3, SOAP x3) ~1250 s, 9 phase-2 arms ~1200 s, evaluation of 10
learned methods ~400 s (baselines are not re-run: they are read from the C3 node runs, which use
the same episodes, `tree.EPISODE_KEYS`; this is checked by comparing the episode family lists and
the true-state arrays before pairing) = ~48 min. 8 cells x 3 seeds = 19 slot-hours; with 9
slots on GPUs 5-7 (shared with the running wave, x1.5-2 contention) = ~3-4.5 h. Seeds 3-4 for
3 arms add ~6.7 slot-hours (~1-1.5 h). Total 4-6 h. Reduction clause: if the estimate after
the first 3 jobs exceeds 6 h, drop A5 and A6 first (lowest expected value), never the controls
A0 and A2.

### 6.4 Decision rule (pre-stated)

* Contrast: per cell, per seed, relative change of the arm's mean episode NRMSE vs **the better
  of default lora / default hyper in that cell** (chosen by seed-mean on the same episodes: a
  choice that favours the default, so conservative), via `hyperkkl.metrics.paired_relative`
  (Student-t CI over seeds), for `sequence_ood` and `shape_ood` separately, clean condition.
* Two primary hypotheses (A7 faithful, A8 best-evidence), Bonferroni: 97.5% CIs.
* **"The archived recipe is superior"** iff, for that arm, the CI excludes 0 in its favour on
  **>= 5 of the 8 cells** on sequence_ood AND on shape_ood, and it is significantly worse
  (CI excludes 0 against it) on **no** cell of either tier. Inferior iff the mirror statement
  holds. Otherwise INCONCLUSIVE. The k/8 counts are reported per tier and per role (LOSE / WIN
  cells), with absolute NRMSE and the best baseline of each cell next to them.
* Component arms A1-A6: reported as paired deltas with 95% CIs and k/8 counts, no verdict;
  A1 vs A2 (SOAP vs lr-matched Adam) is the only component contrast with its own rule
  (same >= 5/8, none-worse rule) because it decides whether "SOAP" is an optimiser effect or a
  learning-rate effect.
* Also reported, no rule: id tier; the deployment-scorecard essentials (unmeasured-state NRMSE,
  p99, divergence rate) for A0/A7/A8; phase-1 `eps_dec` and `l_dec_q95` per arm (the T5 mechanism).

---

## 7. Suggestions for `experiments/tuning/PROTOCOL.md` (not edited; sections 1-6 are frozen there, so these would be section-8 deviations or a separate node)

1. **Learning-rate grid**: the archive's optimum was 1e-2 (3e-2 diverges); the grid
   {5e-4 .. 4e-3} stops below it. Suggest adding 8e-3 for every variant (equal tuning budget is
   kept if one knob value is dropped symmetrically), or note the truncation as a caveat.
2. **Optimiser as a shared factor**: if SOAP enters, it should be one factor applied to every
   variant (with the lr-matched Adam control), never a per-variant knob, so it cannot change the
   C1 ranking through unequal treatment.
3. **Phase-1 quality is orthogonal to C1** but may dominate absolute error: record `eps_dec` and
   `l_dec_q95` next to the tuned results, since the protocol already rebuilds base maps once
   (PROTOCOL "Known caveats").
4. Optional: hyper's `gen_layers` knob addresses the k = 4 subspace issue; the archive's
   generator (layer embedding x context gate, learnable output scale, rsLoRA scaling) is a
   different parameterisation of the same idea and could be a third lora/hyper knob value, but it
   is not supported by an archive ablation, so it is low priority.

---

## 8. What could not be extracted or reproduced

* No leave-one-out exists for phase-1 epochs, phase-2 epochs or collocation at the composed
  config (queued in `QUEUE.md`, never run), so the carrier of the composed recipe's waveform-tier
  gain is inferred, not measured.
* The archive's systems, input families, dt/horizons, latent spectra and amplitude tier differ
  from v2, so no archive number can be compared numerically with a v2 number; only signs and
  ranks transfer.
* The archive's ~1.5M-parameter conditioner cannot be "matched" to v2's 140k; the test uses 140k
  (v2's controlled budget), with one optional 1.5M check.
* SOAP is not installed in the venv; the test would vendor the archive's own implementation
  (which is what produced every SOAP number above), not a different reference implementation.
