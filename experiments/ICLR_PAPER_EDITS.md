# HyperKKL (ICLR 2026 AI&PDE workshop): code changes and paper edits

Code: tmlr branch worktree, `hyperkkl/iclr.py` (Static, Dynamic, curriculum), plus small hooks in
`train.py`, `job.py`, `evaluate.py`, `nets.py`, `conditioning.py`, `data.py`, `systems.py`,
`configs/systems/iclr_*.yaml`. Experiments: `experiments/tree/ICLR*` (`python -m hyperkkl report <node>`).

## 1. What changed and why (short version)

| Paper element | Was (as written / ICLR code) | Now (code) | Evidence |
|---|---|---|---|
| Dynamic loss, Eq. 8 | decoder inverts the encoder output, `x ≈ T*(T(x;θ_u);φ_u)`, + PDE | decoder inverts the **filter latent**; encoder fitted to the exact label z = T(x(t),t) (Remark 3); PDE kept | §2.1: Eq. 8 as written is worse than autonomous on all 4 systems, whatever dT/dt |
| Static loss, Eq. 10 | stationary-PDE residual at T(x) | **rollout of the deployed observer** (Eq. 9 + frozen T*) over K = 256 steps | §2.1: residual losses 1.7–125x worse |
| Eq. 13 low rank | not implemented as stated | exactly Eq. 13: Δθ = U a(c) on every weight and bias, r = 32 / 128 | unit test vs the explicit per-sample weights (1e-8) |
| Zero-input consistency | Static: "trained to output zero"; Dynamic: not zero (future work) | **exact by construction** for both (context E(0) = 0, bias-free heads, injection M([z,c]) − M([z,0])) | fam_zero column identical to autonomous to all digits |
| Phase 1 decoder | inverts T0 on collocation + trajectory states | fitted on the labelled zero-input pairs (z, x) (Niazi et al.'s pairs) | §2.2: autonomous zero-input NRMSE Rössler 0.72 → 0.11, Lorenz 0.70 → 0.018 |
| dT/dt (Eq. 14) | fixed-x finite difference | **unchanged** (Eq. 14); JVP chain rule, BDF2, none and λ = 0 are ablations | §2.3: all within 1.2% except "none" |
| Curriculum | autoencoder target, constant → sin → square | decoder fine-tuned on the filter latent; spectral-centroid levels, cumulative replay, plateau rule | §2.2 |
| Input encoder | LSTM over a window u[t−w,t] | LSTM over the whole input history (a 100-sample FIR window is as good) | §6: ICLR-window, ICLR-encoder |
| Small gain | not measured, not enforced | measured (κ·Lip_z(Φ)/α on every run) and an enforced variant | ICLR-gain |

## 2. Evidence (seed 0, decision pilots; the seed-replicated versions are child nodes of ICLR)

### 2.1 Pilot 1 (`ICLR-pilot`, old phase-1 base): validation NRMSE of the deployed observer

| method | Duffing | VdP | Rössler | Lorenz |
|---|---|---|---|---|
| autonomous | 0.169 | 0.069 | 0.724 | 0.708 |
| Static, Eq. 10 as written (`pde`) | 0.156 | **6.12** | 0.892 | 0.610 |
| Static, input-induced target J_T[f(x,u)−f(x,0)] (`pde_input`) | 0.154 | 0.982 | 0.715 | 0.901 |
| Static, model-free finite difference (`fd`) | 0.151 | 0.048 | 0.606 | 0.147 |
| **Static, rollout** | **0.087** | **0.027** | **0.196** | **0.122** |
| Dynamic, Eq. 8 as written, dT/dt = Eq. 14 | 0.286 | 0.094 | 1.94 | 0.503 |
| Dynamic, Eq. 8, dT/dt = JVP chain rule | 0.287 | 0.093 | 1.32 | 0.501 |
| Dynamic, Eq. 8, dT/dt = BDF2 | 0.268 | 0.078 | 1.80 | 0.500 |
| Dynamic, Eq. 8, dT/dt = 0 | 0.259 | 0.085 | 1.73 | 0.509 |
| Dynamic, filter-latent target (tmlr `hyper`, id tier) | 0.132 | 0.018 | 0.095 | 0.029 |
| curriculum (filter-latent target) | 0.136 | 0.028 | 0.129 | 0.017 |

Reading: (i) every residual-trained Static fits its target to 1e-5–1e-7 yet deploys badly: the
injection is fitted at T0(x) but acts at ẑ, which differs from T0(x) by T0's own PDE error, and
the measured small-gain ratio κ·Lip_z(Φ)/α was 4–113 (violated). (ii) Eq. 8 as written is worse
than autonomous on every system and the four dT/dt choices are within ~10% of each other: the
decoder learns to invert T(x;θ_u), a latent the filter reaches only where the PDE residual is 0.

### 2.2 Phase-1 decoder (`ICLR-p1-label`, `ICLR-p1-coll_label`): autonomous observer NRMSE

| decoder data | zero input D / V / R / L | id tier D / V / R / L |
|---|---|---|
| [coll, traj] (T0 of states, previous) | 0.176 / 0.045 / 0.716 / 0.699 | 0.171 / 0.082 / 0.723 / 0.711 |
| **[label]** (labelled pairs (z, x)) | 0.224 / 0.017 / 0.109 / 0.018 | 0.209 / 0.062 / 0.180 / 0.018 |
| [coll, label] | 0.222 / 0.031 / 0.474 / 0.068 | 0.209 / 0.063 / 0.496 / 0.070 |

[label] is used from here on (Duffing is 0.04 worse; the other three improve 2.6–40x).

### 2.3 Pilot 2 (`ICLR-dt-*`, label base, filter target): dT/dt and the PDE term

Deployed NRMSE, seed 0. Validation episodes (used for the decision) / id test tier.

| method | Duffing | VdP | Rössler | Lorenz | mean (val) |
|---|---|---|---|---|---|
| autonomous | 0.210 / 0.209 | 0.061 / 0.062 | 0.158 / 0.180 | 0.0186 / 0.0182 | 0.112 |
| **Dynamic, Eq. 14 (fd)** | 0.144 / 0.136 | 0.019 / 0.020 | 0.098 / 0.105 | 0.0184 / 0.0181 | 0.0699 |
| Dynamic, JVP chain rule | 0.142 / 0.135 | 0.019 / 0.020 | 0.098 / 0.105 | 0.0184 / 0.0181 | 0.0692 |
| Dynamic, BDF2 | 0.141 / 0.134 | 0.019 / 0.020 | 0.098 / 0.105 | 0.0185 / 0.0182 | 0.0691 |
| Dynamic, dT/dt = 0 | 0.155 / 0.146 | 0.020 / 0.022 | 0.098 / 0.105 | 0.0184 / 0.0181 | 0.0729 |
| Dynamic, λ = 0 (no PDE term) | 0.142 / 0.135 | 0.018 / 0.018 | 0.098 / 0.105 | 0.0184 / 0.0181 | 0.0690 |
| Dynamic, Eq. 8 as written (decoder inverts T(x;θ)) | 0.273 / 0.270 | 0.133 / 0.146 | 1.97 / 2.21 | 0.411 / 0.445 | 0.70 |
| Static (rollout) | 0.108 / 0.087 | 0.017 / 0.018 | 0.103 / 0.109 | 0.0159 / 0.0153 | 0.061 |
| curriculum | 0.141 / 0.144 | 0.030 / 0.033 | 0.129 / 0.143 | 0.0166 / 0.0164 | 0.079 |
| fine-tuning without ordering (K = 1) | 0.144 / 0.147 | 0.031 / 0.033 | 0.128 / 0.145 | 0.0166 / 0.0164 | 0.080 |

Zero-input column (fam_zero): Static and every Dynamic variant equal autonomous to all printed
digits (0.224 / 0.017 / 0.109 / 0.018).

Reading:
- dT/dt: Eq. 14 is within 1.2% of the best alternative (tie rule: keep the paper's Eq. 14).
  Dropping dT/dt costs 4% on average (10% on Duffing).
- The PDE term does not improve the deployed error (λ = 0 is as good). It is what makes the
  encoder an approximate solution of Eq. 7 and what Proposition 1 bounds; say this explicitly
  rather than claiming an accuracy benefit.
- Eq. 8 as written stays worse than autonomous on the new base too: the decoder target is the fix.
- Static and Dynamic: neither uniformly better (Static best on Duffing and Lorenz, Dynamic on
  Rössler, tie on VdP). The paper's "neither variant uniformly better" is likely to survive.
- Curriculum improves on autonomous under forcing but is behind both HyperKKL variants on three
  of four systems; its ordering adds nothing (K = 1 is equal).
- Static's measured small-gain ratio κ·Lip_z(Φ)/α: 5.4 / 4.3 / 0.13 / 29.6. Prop. 2's condition is
  violated on three systems without any instability: it is sufficient, not necessary (Remark 4).

## 3. Paper edits, section by section

Notation used below: ẑ = latent of filter (3) on the measured output (what the deployed decoder
sees); z = T(x(t),t), the latent of the same filter on the noise-free output from z(0) = 0, which
by Remark 3 is the exact solution of Eq. 5 along the trajectory.

**Abstract, Contributions (ii)–(iii).** Replacement text in §5 (from the 5-seed runs).

**§2.3, last paragraph.** "we use curriculum learning to train this inverse map in an
autoencoder-like fashion" → "on the latent states the filter of equation 3 produces under forcing".

**§3.1, Eq. 8.** Replace by

```latex
\min_{\psi}\ \mathbb{E}\Big[
 \underbrace{\big\|\Sigma_x^{-1}\big(x-\hat T^{*}(\hat z;\phi_u)\big)\big\|^{2}}_{\mathcal L_{\rm dec}}
+\underbrace{\big\|\Sigma_z^{-1}\big(z-\hat T(x;\theta_u)\big)\big\|^{2}}_{\mathcal L_{\rm lab}}
+\lambda\,\underbrace{s_z^{-2}\big\|W\big(\tfrac{\partial \hat T}{\partial x}f(x,u)+\tfrac{\Delta \hat T}{\Delta t}-A\hat T-Bh(x)\big)\big\|^{2}}_{\mathcal L_{\rm PDE}}
\Big]
```

with Σ_x, Σ_z the per-coordinate standard deviations, W = −A⁻¹ (maps a residual to the steady
latent error it causes), s_z the latent scale, λ = 1; L_dec and L_lab on trajectory samples
(after the washout), L_PDE also on Sobol collocation states paired with trajectory contexts. Add:
"Training the decoder on the encoder output T̂(x;θ_u), as an autoencoder, makes it invert a latent
that the deployed filter reaches only where L_PDE vanishes; in our pilot this made the Dynamic
variant worse than the autonomous observer on all four systems (Appendix D)." The sentence after
the equation becomes "... the Static variant keeps the maps fixed and minimizes the rollout loss of
equation 10 instead."

**§3.2.1 (Static).**
- Eq. 9: φ̄(ẑ, c) with c the LSTM context; φ̄(ẑ,c) = σ_z ⊙ [M([ẑ/σ_z, c]) − M([ẑ/σ_z, 0])], M an
  MLP (2 hidden layers, SiLU, width 360, 140k parameters).
- Replace "processes a sliding window u[t−w,t]" by "processes the input history".
- Eq. 10: replace by
```latex
\mathcal L_{\rm static}(\xi)=\mathbb{E}\sum_{j=1}^{K}\big\|\Sigma_x^{-1}\big(x_{k+j}-\hat T^{*}(\hat z_{k+j})\big)\big\|^{2},
\qquad \hat z_{k}=\hat T(x_k),\quad \hat z_{i+1}=\mathcal S_{\Delta t}(\hat z_i,y_i,y_{i+1};\bar\varphi),
```
  S = one step of equation 9 (exact for the linear part with a first-order hold on y, midpoint rule
  for φ̄; the same step is used at deployment), K = 256. Replace "that is, the injection is fitted
  to the part of the latent dynamics that the autonomous map leaves unexplained" by: "The
  stationary residual ∂T̂/∂x f − AT̂ − Bh − φ̄(T̂(x)) would fit φ̄ at T̂(x), but φ̄ acts at ẑ, which
  differs from T̂(x) by T̂'s own PDE error; fitting it directly was 1.7–125x worse than the rollout
  in our pilot (Appendix D). For an exact T, the Bernard–Andrieu injection φ̄(T(x)) = ∂T/∂x g(x) u
  makes ẑ = T(x) and zeroes this loss."
- Replace "The injection network is trained to output zero when u ≡ 0" by "The injection is zero at
  u ≡ 0 by construction (the context of a zero input is exactly zero and φ̄(ẑ,0) = 0), so the
  autonomous observer is recovered exactly on unforced dynamics."

**§3.2.2 (Dynamic).**
- "context variable is the input history c = u[t−w,t]" → "c = E(u[0,t]), an LSTM over the input
  history with its zero-input response subtracted, so E(0) = 0".
- After Eq. 13: "a_θ, a_φ are MLPs (2 hidden layers of 128 SiLU units) without biases whose last
  layer is initialised at zero; U_θ, U_φ have one column per rank and span every weight and bias of
  the encoder and decoder." Keep r = 32 / 128.
- Delete "It does not, however, force the perturbation to vanish ... leave for future work."
  Replace with: "Because E(0) = 0 and the heads are bias-free, a(0) = 0: the Dynamic variant
  coincides exactly with the autonomous observer at u ≡ 0."
- Figure 2: input "u[t−w,t]" → "u[0,t]"; loss label "L_rec + λL_PDE" → "L_dec + L_lab + λL_PDE";
  draw the decoder's training input as the filter latent ẑ (not T̂(x)).

**§3.3.1.**
- Phase 1: add "the decoder T̂*_base is fitted on the same labelled pairs (z, x), so it inverts the
  latent the filter carries rather than the encoder's approximation of it" and "the PDE weight is
  ramped linearly over the first quarter of training".
- Phase 2: "the encoder and decoder perturbations are trained together through the single loss of
  equation 8" stays true. Eq. 14 stays. Add one sentence pointing to the dT/dt ablation
  (ICLR-dtdt): the alternatives (JVP chain rule, BDF2) give the same error, and the PDE term
  itself (λ = 0) does not change it; its role is consistency with Eq. 7, which Proposition 1 uses.

**§3.3.2 (curriculum), rewrite.** "The baseline keeps the autonomous architecture (no input
pathway). T̂ is frozen and T̂* is initialised from Phase 1 and fine-tuned on forced trajectories,
with the filter latent ẑ as input and x as target. Difficulty is the spectral centroid of an
episode's input after the washout (in cycles per reference time; 0 for zero and constant inputs).
Level 1 contains the zero and constant inputs, levels 2..K the quantile bins of the centroid of the
others (K = 4). Stages are cumulative: level k trains on levels 1..k, so the autonomous regime is
replayed rather than overwritten. Every 100 steps the validation loss of the current stage is
checked; the scheduler advances when it improved by less than ε = 1% (relative) for p = 3
consecutive checks (at least 200 steps per stage), or after 25% of the budget. The last stage uses
all data until the budget of 3000 steps and selects the model on the full validation set. The
control K = 1 is the same fine-tuning without ordering."

**§3.4.** "We do not enforce the small-gain condition during training; doing so ... lacks" →
"We measure κ·Lip_z(φ̄)/α for every trained injection (Appendix D) and evaluate a variant that
enforces Lip_z(φ̄) ≤ ρα/κ by construction: each weight of M is projected to spectral norm ≤ 1
after every step and the output is scaled by the fixed γ = ρα / (2κ (max σ_z/min σ_z) 1.0998^d),
d the number of hidden layers (1.0998 = max |SiLU'|)."

**§4 intro / §4.1 Datasets and Tasks.**
- ∆t: per system from max|λ(A)|∆t = 0.1: Duffing 0.0103 s, Van der Pol 0.0206 s, Rössler
  0.0223 s, Lorenz 0.0111 s (not 0.05 s). Horizon T = 50 s (Rössler 160 s). RK4 for (x, z) jointly.
- Noise: process noise with std rate 0.5% of each state's reference std; sensor noise 3% of the
  output's std, in training and test (not σ = 0.01).
- Inputs: bounded in [−1, 1]; 11 training families (zero, constant, step, staircase, PRBS, square,
  sinusoid, multisine, ramp, AM, drive); 600 training and 64 validation episodes; 400 zero-input
  episodes for Phase 1. Test sets (64 episodes each, seeds independent of training): the paper's
  four regimes as per-family columns (zero, constant, sinusoid, square); held-out waveforms
  (chirp, filtered noise, triangle, sawtooth); stop-and-go sequences; and out-of-range amplitude
  2x and 4x (reviewer request).
- Metric: replace SMAPE by NRMSE (per-state RMSE divided by the state's reference std, averaged
  over states) on the second half of each episode; keep RMSE in physical units for continuity.
  5 training seeds; mean ± std, and paired 95% bootstrap CIs vs autonomous.

**§4.1 Baselines.** Add (iii) EKF with the true model (process and sensor covariances from the
noise model) and (iv) the model-based stationary KKL: the Phase-1 maps with the exact
Bernard–Andrieu injection ∂T̂/∂x [f(x̂,u) − f(x̂,0)] evaluated at x̂ = T̂*(ẑ). Delete "Non-KKL
estimators are outside the scope". Curriculum description: as §3.3.2.

**§4.1 Implementation Details.** "All encoders are 3-layer MLPs (150 units ..., 350 ...)" →
"encoder 3×192 SiLU, decoder 3×128 tanh (all systems)". "LSTM hypernetwork (64 units, window
w = 100 samples)" → "LSTM (64 units, context dimension 16) over the input history". Training:
Phase 1, 6000 Adam steps each for encoder and decoder, batch 4096; Phase 2, 3000 Adam steps; lr
2·10⁻³ with 100 warm-up steps and cosine decay; gradient-norm clipping at 1; Dynamic batch = 32
episodes × 96 times + 1024 collocation states; Static batch = 32 rollouts of 256 steps; model
selection on validation loss every 250 steps. Delete "The window length was fixed a priori and
not tuned" (it is ablated now).

**§4.2 Results.** Replace Tables 1–2 and the four finding paragraphs: replacement text in §5.

**§4.2 Limitations.** Delete the items now covered: EKF comparison, variability, window/encoder/
finite-difference ablations, noisy inputs, out-of-range amplitude, overhead, higher-dimensional
system. Keep: synthetic systems, scalar input/output, full model knowledge for the PDE term.

**§5.** Remove the future-work items now done (gating at zero input, small-gain by construction,
scaling, ablations, non-KKL comparison, noisy inputs, repeated runs, higher-dimensional benchmark).

**Appendix A.** Per system: observer design (n_z, eigenvalues of A between α and max|λ|, B), IC
box, horizon, ∆t, input entry, measured state. GPU: NVIDIA A100-SXM4-80GB; PyTorch 2.10.

**Appendix C (theory: notation only).**
- C.1 after the constants: "The first two are the trajectory-wise suprema of the terms of the loss
  in equation 8" → "ε_PDE is the supremum of the residual in L_PDE; ε_rec is controlled by L_dec and
  L_lab together". Optional remark (one line, follows from the same proof with T̂_t(x(t)) replaced
  by the noise-free filter latent z(t), for which ζ̇ = Aζ − Bv): lim sup ‖x − x̂‖ ≤ ε_dec +
  κL*‖B‖v̄/α, with ε_dec = sup ‖x − T̂*_t(z)‖, the term L_dec penalises.
- C.2: "which is the residual minimized in equation 10" → "the stationary residual (the rollout
  loss of equation 10 penalises its effect on the decoded state rather than R_s itself)".
- Remark 2, last sentence: "which we do not do in this work" → "Section 4 evaluates a variant that
  does".
- Remark 3, last sentence: "and do not ablate it" → "and ablate it in Section 4 (causal FIR windows
  of 10–200 samples against the LSTM over the full history)". Its first part is now the
  justification of the encoder label: z = T(x(t),t) is exactly the filter latent from z(0) = 0.
- Remark 4: rewrite from the measured κ·Lip_z(φ̄)/α (the hypothesis can now be checked).

**Appendix D (new).** The pilot tables of §2 above, plus ICLR-static-loss, ICLR-dtdt.

## 4. Experiments (node → reviewer row)

| node | content | reviewer rows |
|---|---|---|
| ICLR | 4 systems × 5 seeds: autonomous, Static, Dynamic, curriculum, curriculum (K = 1), EKF, model-based KKL; id/OOD/per-family/amp2/amp4; noisy y and noisy u | 4, 5, 7, 8, 10 |
| ICLR-dtdt | dT/dt: Eq. 14, JVP, BDF2, none; λ = 0; Eq. 8 as written | 2 |
| ICLR-static-loss | rollout vs Eq. 10 vs input-induced vs finite difference | (Eq. 10 change) |
| ICLR-window | FIR window 10–200 vs LSTM | 1 |
| ICLR-encoder | current / window / GRU / LSTM / SFM | 3 |
| ICLR-finetune | frozen vs jointly fine-tuned base | 6 |
| ICLR-gain | Static with the small-gain condition enforced | 9 |
| ICLR-lorenz-u10 | Lorenz with u ∈ [−10, 10] | 9 |
| ICLR-large | highway_traffic | 11 |

## 5. Main results (node ICLR, 5 seeds, complete) and replacement text

Tables: `experiments/iclr_tables_rmse.tex` (RMSE, the paper's columns Zero/Const/Sin/Sqr, drop-in
for Tables 1–2) and `experiments/iclr_tables_nrmse.tex` (NRMSE, plus id/shape/sequence/2x/4x
columns for an appendix table). Mean ± std over 5 seeds; ↓ = below autonomous on every seed.
Full report with per-family paired CIs: `experiments/tree/ICLR/report.md`.

Note: the new numbers are not comparable with the submitted Tables 1–2 (different ∆t, noise,
input set and the fixed methods); replace the tables wholesale.

Paired NRMSE change vs autonomous, id tier (all 10 input families; * = 95% CI excludes 0):

| | Duffing | VdP | Rössler | Lorenz |
|---|---|---|---|---|
| Static HyperKKL | −55%* | −72%* | −39%* | −21%* |
| Dynamic HyperKKL | −35%* | −70%* | −42%* | −1%* |
| Curriculum | −29%* | −48%* | −22%* | −12%* |
| Fine-tuning without ordering | −30%* | −47%* | −21%* | −12%* |
| Model-based KKL | −8%* | −21%* | −38%* | +474%* |
| EKF (exact model) | −97%* | −86%* | −73%* | −61%* |

### Replacement text

**Abstract, results sentences.** "Over five training seeds, both variants lower the estimation
error under forcing relative to the autonomous observer on all four systems, by 35–72% on the
Duffing, Van der Pol and Rössler systems and by 21% (Static) on Lorenz, whose state the tested
inputs barely move; neither variant is uniformly better, and both coincide exactly with the
autonomous observer at zero input. Fine-tuning the inverse map on the latent states of the
filter recovers part of this gain (12–48%) and stays behind both variants on three of the four
systems; ordering the data as a curriculum adds nothing."

**Contribution (ii).** "We provide a controlled comparison against a training-only baseline:
fine-tuning the autonomous inverse map on the latent states the filter produces under forcing
recovers part of the gain of input conditioning, but not all of it on three of four systems,
and a difficulty curriculum adds nothing over plain fine-tuning."

**Contribution (iii).** Add "over five training seeds, with an extended Kalman filter and a
model-based KKL observer as references, on held-out waveforms and amplitudes and with noisy
outputs and inputs", and replace "where it does not (Lorenz)" by "where the gain is small
(Lorenz)".

**§4.2, paragraph 1 (replaces "Input conditioning helps on the two oscillators and on Rössler").**
"Input conditioning helps on all four systems, with neither variant uniformly better. Relative to
the Autonomous observer, the Static variant lowers the NRMSE by 55%, 72%, 39% and 21% on Duffing,
Van der Pol, Rössler and Lorenz, and the Dynamic variant by 35%, 70%, 42% and 1%; every change
is significant across seeds (paired 95% CI) and holds for each of the ten input families. The
Static variant is best on Duffing and Lorenz, the Dynamic variant on Rössler, and the two are
tied on Van der Pol. The Static variant varies more across training seeds on Duffing (NRMSE
0.091 ± 0.034 vs 0.133 ± 0.001)."

**§4.2, paragraph 2 (replaces "The curriculum baseline does not transfer...").** "Fine-tuning
helps, ordering does not. Trained on the latent states the filter produces under forcing, the
curriculum baseline improves on the Autonomous observer on every system (12–48%), which confirms
that the failure of an autoencoding objective, which never sees those states, was a training
problem rather than a representational one. Its gain is smaller than that of both HyperKKL
variants on Duffing, Van der Pol and Rössler, and the same fine-tuning without a curriculum
(K = 1) is within 2% of it on every system: the difficulty ordering contributes nothing. Since
the curriculum retrains the inverse map, it also changes the zero-input error (better on Duffing
and Lorenz, worse on Van der Pol), whereas both HyperKKL variants leave it unchanged by
construction."

**§4.2, paragraph 3 (replaces "Lorenz: no gain from input conditioning, and two different
failure patterns").** "Lorenz. Inputs in [−1, 1] barely move the Lorenz state: the Autonomous
observer has the same error with and without forcing (RMSE 0.144 at zero input, 0.146–0.155
under forcing). The Dynamic variant accordingly gains nothing (−1%), while the Static variant
still lowers the error by 21%. Neither collapses: the collapse reported in the submitted version
came from fitting the injection to the residual of the stationary PDE, which is not what the
deployed filter needs (Appendix D)." Then add the ×10-input sentence of §6.

**§4.2, new paragraph "References and robustness".** "The EKF, which has the exact model and noise
statistics, is far more accurate than every learned observer (61–97% below Autonomous); the
learned observers use no model at deployment. The model-based KKL observer, which adds the exact
input correction ∂T̂/∂x [f(x̂,u) − f(x̂,0)] to the autonomous maps, improves on Autonomous by
8–38% on the first three systems and is 5.7 times worse on Lorenz, where the correction amplifies
the decoder's error; the Static variant matches or beats it without using the model online.
Noise of 5–20% of the input's standard deviation on the input fed to the observers changes no
error measurably. With sensor noise raised from 3% to 10%, all observers degrade; the Static
variant keeps its advantage on every system, the Dynamic variant on all but Duffing. On inputs
of twice the training amplitude, all gains persist; at four times, they shrink, the Dynamic
variant becomes sensitive to the training seed (Duffing 0.35 ± 0.23, Rössler 0.35 ± 0.06 NRMSE),
and the model-based KKL observer is the best KKL observer on Duffing and Rössler. No observer
diverged in any run."

**§4.2, "Zero-input consistency" paragraph → one sentence.** "At u ≡ 0 both variants reproduce
the Autonomous observer exactly on every system and seed, by construction (Section 3.2)."

**§4.2, cost (reviewer row 7), new sentence.** "On one CPU core (batch 1), a step costs 87 µs for
the Autonomous observer, 0.7 ms for the Static variant, 1.8 ms (r = 32) and 6.5 ms (r = 128) for
the Dynamic variant, 1.0 ms for the EKF and 4.4 ms for the model-based KKL observer. Phase 2
trains 158k parameters for the Static variant and 3.6M (r = 32) or 14.3M (r = 128) for the
Dynamic variant, whose basis U spans every weight; the Static variant is the more accurate on two
systems with 23–90 times fewer parameters."

**Figure 3.** Regenerate from `experiments/tree/ICLR/runs/iclr_duffing/seed0/traj.npz` (fam_square
tier).

**§5 Conclusion.** Replace the results summary with the abstract's sentences; replace "A curriculum
baseline ... performs worse than the autonomous observer in every condition" by the fine-tuning
finding; drop "the Static variant collapses under any nonzero input" and "Naive input conditioning
can thus be worse than ignoring the input" (no longer observed); keep the stability paragraph.

## 6. Ablations (all complete, 3 seeds) and replacement text

Tables for a new appendix: `experiments/iclr_tables_ablations.tex` (NRMSE, id tier, mean ± std
over 3 seeds; rows carry the code's method names, relabel in LaTeX). Reports with paired CIs:
`experiments/tree/ICLR-<name>/report.md`.

NRMSE (id tier), Duffing / Van der Pol / Rössler / Lorenz; autonomous is 0.208 / 0.064 / 0.183 / 0.017.

| row | ablation | result |
|---|---|---|
| 2 | dT/dt: Eq. 14 / JVP / BDF2 / none | 0.134–0.141 / 0.019–0.020 / 0.104 / 0.017, all within seed noise |
| 2 | no PDE term (λ = 0) | 0.147 / 0.017 / 0.104 / 0.017, same |
| 2 | Eq. 8 as written (decoder inverts T(x;θ)) | 0.278 / 0.118 / 1.85 / 0.42, worse than autonomous |
| Eq. 10 | Static: rollout / Eq. 10 / input part / finite difference | 0.100 / 0.018 / 0.108 / 0.014 vs Eq. 10 0.18 / 0.94 / 1.91 / 0.58 (seed spread up to ±1.4) |
| 1 | FIR window w = 10 / 25 / 50 / 100 / 200 vs LSTM | w ≥ 50 matches or beats the LSTM everywhere; Rössler keeps improving to w = 200 (0.092) |
| 3 | encoder: current sample / window / GRU / LSTM / SFM | window 0.132 / 0.018 / 0.095 / 0.016 best or tied; current sample worst (0.165 / 0.024 / 0.103 / 0.017) |
| 6 | base maps fine-tuned jointly | a further 8–25% lower (0.123 / 0.015 / 0.089 / 0.015), but zero-input error changes (no longer = autonomous) |
| 9 | Static with Lip_z(φ̄) ≤ ρα/κ enforced (ρ = 0.5, 0.9) | certified (κ·Lip/α ≤ 0.05) but 0.197 / 0.028 / 0.108 / 0.017: loses most of the gain on Duffing and Lorenz, half on VdP, none on Rössler; no help at 2x amplitude |
| 9 | Lorenz, inputs ×10 | autonomous 0.031 (0.018 at zero input); Static 0.017 (−45%), Dynamic 0.018 (−41%), curriculum 0.024, model-based KKL 0.30 |
| 11 | highway_traffic (5 states, 3 outputs, n_z = 12) | autonomous 0.309, Static 0.270, Dynamic 0.085 (−73%), curriculum 0.060, EKF 0.012; the Phase-1 base is weak here even at zero input (0.263) |

### Replacement / new text

**§3.3.1, after Eq. 14 (row 2).** "The choice of the time-derivative approximation does not
matter in practice: a two-step backward difference, the chain rule through the hypernetwork
(one Jacobian-vector product in (x, c)) and omitting ∂T̂/∂t altogether give the same error within
seed variation, and so does dropping L_PDE (λ = 0; Appendix D). The deployed error is set by
L_dec; the PDE term keeps the conditioned encoder consistent with equation 7, which is what
Proposition 1 requires, rather than improving accuracy."

**§3.2.2 / §4.1, input encoder and window (rows 1 and 3).** "We use an LSTM over the input
history. A causal finite window of w samples performs as well once w ≥ 50 (0.5–1.1 s, about one
to one and a half time constants 1/α of the latent filter), in line with Remark 3, and a window
of 100 samples matches or slightly improves on the LSTM on every system. Conditioning on the
current input sample alone is the weakest choice but still improves on the autonomous observer.
A GRU and a spectral fading-memory encoder perform similarly (Appendix D)." Remark 3, last
sentence: "... and Appendix D confirms that w ≥ 50 samples suffices."

**§3.3.1 Phase 2 (row 6).** After "No joint fine-tuning of the base maps takes place": "Unfreezing
θ_base and ϕ_base in Phase 2 lowers the error by a further 8–25%, but the zero-input observer then
differs from the autonomous one, so the exact zero-input consistency of Section 3.2 is lost; we
keep the base frozen to retain it (Appendix D)."

**§3.4 and Remark 2 (row 9).** "Enforcing the small-gain condition by construction (spectral
projection of every weight of the injection network and a fixed output gain, so that
κ Lip_z(φ̄) ≤ ρα) certifies Proposition 2 but costs most of the Static variant's gain on Duffing
and Lorenz and half of it on Van der Pol, and does not help at twice the training amplitude
(Appendix D). The unconstrained injection violates the condition (κ Lip_z(φ̄)/α up to 42, measured
on the training states) without any divergence in our runs: the condition is sufficient and, with
a product-of-norms bound, conservative." Remark 4: replace the Lorenz hypothesis by this.

**§4.2, Lorenz paragraph, add (row 9).** "With inputs ten times larger, the input does move the
Lorenz state (the Autonomous NRMSE nearly doubles, 0.018 → 0.031), and both variants recover most
of the loss (Static −45%, Dynamic −41%, curriculum −22%). The small gain on Lorenz in the main
table therefore reflects the small inputs, not a limitation of the approach."

**§4.2, new paragraph (row 11).** "On a larger system, a highway traffic model with five states,
three outputs and a 12-dimensional latent space, the Dynamic variant lowers the NRMSE by 73%
(0.309 → 0.085) and the Static variant by 12%. The curriculum baseline is lower still (0.060)
because the autonomous base maps are weak on this system even at zero input (0.263): the
zero-input trajectories used in Phase 1 cover little of the forced state space, and a method that
refits the inverse map on forced data repairs this, whereas both HyperKKL variants keep the base
frozen by design."

**§4.2 Limitations.** Replace by: "All systems are synthetic, with full model knowledge used in
training (the PDE term), and inputs are scalar. The HyperKKL variants inherit the accuracy of the
frozen Phase-1 maps at zero input; when the zero-input data cover the forced state space poorly
(highway traffic), refitting the inverse map is more effective. Gains shrink at four times the
training amplitude, where the Dynamic variant becomes sensitive to the training seed, and the EKF,
which has the exact model, remains far more accurate."

**Appendix D (new).** The pilot tables (§2) and `iclr_tables_ablations.tex`, one table per row
above, plus the measured κ Lip_z(φ̄)/α for every Static variant.
