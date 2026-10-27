# Archived @BEST recipe -> new code mapping (Step 1)

Evidence and effect sizes: `STUDY.md` (sections 1, 3). Archive paths are relative to
`../archive/HyperKKL/` unless stated. "v2" = this package (`hyperkkl/`), protocol v2.
Nothing is implemented yet (waiting for the lead's go-ahead); the "how mapped" column is the plan
for the scratch copy (`hyperbest` method), never the live package.

| component | archive value | evidence (file:line) | already in v2? | how mapped (plan) |
|---|---|---|---|---|
| deployed observer | exact latent filter, decoder T0+(z; eta + eta~(c)), Phi = 0, encoder unused online | `src/training2.py:890-920` | yes (`conditioning.ConditionedDecoder`, decoder-side kinds) | unchanged |
| conditioner | LoRA deltas, rank 4, second factor zero-init, rsLoRA 1/sqrt(r), bias-free context gate x layer embedding, learnable scale | `src/training2.py:766-863` | partly: `lora` (trunk + per-layer U, V heads, V zero-init, 1/sqrt(r)) | use v2 `lora` at the matched budget; the archive's gate/embedding parameterisation is not ablated in the archive, so not reproduced |
| encoder deltas + characteristic residual + label term (`freeze_encoder_deltas=false`) | on | `drain_final.sh:71`, `src/training2.py:1101-1131`, `src/config.py:298-304` | no (v2 has no phase-2 encoder path) | new `phase2.enc_aux: true`: LoRA heads for the T0 layers (counted in the 140k budget), residual T(x_{k+1};c_{k+1}) - e^{A dt} T(x_k;c_k) - (W0 y_k + W1 y_{k+1} + W2 y_{k+2}) with the archive's quadrature weights (`src/training2.py:89-117`), + label term; **archive LOO says it hurts (-22.5% when removed, STUDY B7)** |
| context | GRU hidden 64, c dim 32, window = alpha-rule | `src/models2.py:302-342`, run `config.yaml` | yes: `phase2.context.kind: gru` (`nets.GRUEncoder`, streaming, c(0) = 0 exactly) | `lora@gru`; dim 16 kept (v2 default) unless the lead wants 32 |
| ctx_scale guard | rescale only outside RMS [1e-2, 1] | `src/training2.py:965-980`, `FINDINGS.md:103-117` | no (v2 normalises u by u_ref) | implement as a no-op-unless-out-of-range constant (preserves c = 0 -> delta = 0) |
| optimiser | SOAP, lr 1e-2, betas 0.95/0.95, precond every 10, phases 1 and 2 | `src/optimizers.py:29-126`, `drain_final.sh:71` | no (Adam 2e-3); no SOAP package in `.venv` | vendor the archive's own SOAP (100 lines) into the copy; new keys `phase1.optimizer`, `phase2.optimizer`; lr-matched Adam 1e-2 control arm |
| LR schedule | cosine over epochs, no warm-up, grad clip 1.0 | `src/training2.py:954,1170` | v2: warm-up 100 + cosine, clip (`monitor.TrainMonitor`) | keep v2's schedule (difference is second-order, not ablated in the archive) |
| delta_curv | 1e-4 x mean ||(d_{k+2} - 2 d_{k+1} + d_k)/dt^2||^2 on decoder deltas | `drain_final.sh:68`, `src/training2.py:1132-1141`, `FINDINGS.md:76-77` | no | new `phase2.delta_curv`; deltas at three consecutive sample times of the same episode (contexts from the streaming encoder) |
| delta_reg | 1e-3 x mean ||dW_dec||^2 | `src/config.py:324`, `src/training2.py:1123-1124` | no | new `phase2.delta_reg` |
| phase-1 budget | 200 epochs on 667 episodes (~19.5k steps of 4096) | `drain_final.sh:71` | v2: 6000 + 6000 steps on 400 episodes | budget arm: `phase1.steps_enc/steps_dec` x3 (18000); `data.n_p1` stays 400 (archive LOO: phase-1 volume is not the lever, STUDY D4) |
| phase-2 budget | 60 epochs x ~360k samples = ~21.6M samples (vdp) | run `config.yaml` | v2 3000 x 8192 = 24.6M | already equal; unchanged |
| phase-2 data volume | 667 episodes | `drain_final.sh:71` | v2 600 | unchanged |
| capacity | 0.98-1.50M trainable | `results.json train_stats` (FA_wt) | v2 budget 140k (matched across variants) | 140k (primary); optional single 1.5M check |
| label fidelity / alignment | order-4 labels, no look-ahead | `JOURNAL.md:13-22` | yes (true-plant labels, ZOH input, no look-ahead) | unchanged (v2 correctness fixes kept) |
| metric | NRMSE(steady), settle 0.5 | run `config.yaml evaluation` | yes (settled-window NRMSE, `metrics.episode_metrics`) | unchanged |
| latent init | encoder (z0 = T0(x0)) | `src/config.py:365` | v2: centre-of-box prior | unchanged (archive: 0.0% effect on the settled metric, STUDY E2) |
| noise | 3% measurement, 0.5% process | `drain_final.sh:68` | yes (`data.sensor.noise 0.03`, `noise_proc 0.005`) | unchanged |

## Failed or buggy in the archive (do not carry over)

* first-order labels and one-step look-ahead estimates (fixed before the campaign; v2 never had them)
* SMAPE (withdrawn)
* unreachable measurement noise and y_bar-scaled noise (fixed mid-campaign; earliest tags suspect)
* 4% validation split (fixed late; phase 2 does no val selection there)
* OmegaNet / phi_* heads trained on the old indirect residual (never fixed; latent-rung results invalid)
* SOAP + L-BFGS polish (+24..+49%, masks the residual-form gain), Adam + L-BFGS (diverged)
* rebalancing (ALM, ReLoBRaLo, NTK-diag, inv-Dirichlet), gradient surgery (anneal, PCGrad, ConfG)
* curricula (forward and reverse), specialist input mixtures, Sylvester warm start
* SFM as tested there (gradient quadratic in ||c||, base map failed its learning check, 2% outliers)

## Where each @BEST component was selected

Axes B (SOAP, lr, residual form) and C (volume, epochs) on **bioreactor, vdp, duffing_mult**
(in-sample, all evaluation systems); axis A (weight space) after seeing all 8 systems
(`DECK_BACKGROUND.md:227-232`, final tex "Where the recipe was selected"). Held-out transfer of the
composed recipe: fhn -71%, rossler -51% better; lorenz +453%, cartpole +82% worse on the graded
(amplitude) tier, but **all 8 systems better on the in-range waveform tier (median -42%)**
(STUDY T5, recomputed over seeds 0-2).
