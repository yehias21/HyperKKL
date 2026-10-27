# Audit: HyperKKL ('lora' path) optimisation vs the concat decoder

2026-10-26. Scope: `hyperkkl/conditioning.py` (lora/concat), `nets.py` (SFM context), `train.py`
phase 2, `monitor.py`, the training logs of C1-where, C1-tuned, C1-tune-*. The live package, tree,
queue and cache were not modified. All experiments ran from a scratch copy of the package. Its
knobs are env-gated, and with no knob set it reproduces the cached C1-tuned model bit for bit
(lora vdp seed 0: val 2.8546e-4 = cached value). Runs: phase 2 only, the full 3000 steps, on the
cached C1-tuned phase-1 maps and data, GPUs 5-7.

**Bottom line.** No bug was found. HyperKKL is not losing to concat because of a gradient or
initialisation pathology: its gradient path is linear in ||c|| (not quadratic), its generated
delta is large and fully used, it is clipped less than concat, and its learning rate is
interior-optimal at 4e-3. None of the five proposed fixes improves HyperKKL by more than seed
noise. The asymmetry runs the other way: **concat is the under-tuned variant.** Its optimum lies
at or above the edge of the lr grid, and calibrating the context scale helps concat (-21% to -32%
val MSE) while it hurts HyperKKL. With those two changes, concat beats HyperKKL on vdp and duffing
(3/3 seeds each). HyperKKL keeps a clear lead on bioreactor and quad_tank.

## Findings

| # | Item | Class | Short verdict |
|---|---|---|---|
| 1 | dL/dU = 0 at init | OK (by design) | Exactly 0, as expected with V = 0. The V path opens at once, linearly in c. |
| 2 | Gradient vs \|\|c\|\| | OK | Linear for both lora and concat, with the same order of magnitude. The archive's quadratic law no longer applies: U is affine (bias), so U(c) is non-zero at c = 0. |
| 3 | Context scale | OK for lora | c RMS is 0.07-0.23 at init and 0.2-0.5 trained, never ~1e-2. |
| 4 | Gradient clipping | OK / mild PATHOLOGY for concat | It bites only on bioreactor and quad_tank in the first ~1000 steps, and it hits concat **more** than lora. |
| 5 | Delta magnitude, rank | OK | ‖ΔW‖/‖W‖ is 0.2-4.8 in the hidden layers, and every rank is used. Not under-used. |
| 6 | Generator saturation | OK (note) | The SiLU trunk runs near its linear regime (pre-activation RMS 0.12-0.39, \|max\| ≤ 2). No dead units. |
| 7 | lr grid edge | PATHOLOGY of the **tuning protocol, for concat** | lora's optimum is interior (2e-3 < 4e-3 > 8e-3). concat improves at 8e-3 (vdp) and 1.6e-2 (bioreactor). |
| 8 | Bilinear early instability | PATHOLOGY (lora, mild) | Gradient spikes up to 4.5e3 on bioreactor. One seed ended 2.2x worse. A small non-zero V init removed it, but the effect is not general. |
| 9 | 1/sqrt(r), separate head lr, U-zero init | OK | None helps. Some hurt. |

## Evidence

### 1-2. Gradient flow at initialisation
Fresh modules, C1-tuned widths (lora width 44, rank 4; concat width 258, depth 3). Batch of
32x256 training samples, seed 0. Global norm of dL/dθ per group, with the context c multiplied by
0.1 / 1 / 10:

| system | lora V.w (c×0.1 / ×1 / ×10) | lora U, trunk, ctx | concat R.last (×0.1 / ×1 / ×10) | concat R.hidden, ctx |
|---|---|---|---|---|
| vdp | 4.9e-4 / 4.8e-3 / 1.3e-1 | 0 | 4.6e-4 / 4.6e-3 / 4.7e-2 | 0 |
| bioreactor | 1.1e-1 / 1.1 / 28.8 | 0 | 7.0e-2 / 0.70 / 7.9 | 0 |
| duffing | 1.3e-3 / 1.3e-2 / 0.37 | 0 | 7.4e-4 / 7.4e-3 / 7.6e-2 | 0 |
| quad_tank | 3.3e-2 / 0.33 / 13.5 | 0 | 9.2e-3 / 9.2e-2 / 1.04 | 0 |

- At c×0.1 → ×1 the gradient grows 10x for both variants. That is linear, not the quadratic 100x
  of the archive bug (FINDINGS.md §3).
- At ×10, lora grows 26-40x because the trunk's SiLU curvature makes it super-linear.
- The lora gradient is 1-4x **larger** than concat's at every scale.
- Exactly as the code implies, dL/dU, dL/dtrunk and dL/dctx are 0 at step 0 for lora (V = 0),
  and dL/dR.hidden and dL/dctx are 0 for concat (zero last layer). Both open after one Adam step.

### 3. Context-encoder output scale (SFM, dim 16)
RMS of c over post-washout training samples:

| | battery | bioreactor | duffing | lorenz | quad_tank | vdp |
|---|---|---|---|---|---|---|
| fresh encoder (init) | 0.071 | 0.163 | 0.067 | 0.067 | 0.234 | 0.104 |
| trained, lora run | 0.36 | 0.53 | 0.31 | 0.29 | 0.32 | 0.21 |
| trained, concat run | 0.40 | 0.34 | 0.17 | 6.35 | 0.39 | 0.48 |

- This is inside the archive's measured working range (1e-2..1). Nothing is near the 1e-16
  dead zone.
- The psi tanh layers are mostly unsaturated (fraction of pre-activations with |pre| > 2 ≤ 2%).
  The exception is bioreactor: 31% / 22%.

### 4. Clipping (global norm 1.0), from the logs
Fraction of steps clipped, steps ≤ 1000 / > 1000, mean over the C1-tuned seeds:

| | lora | concat | latent | film | hyper |
|---|---|---|---|---|---|
| bioreactor | 0.53 / 0.00 | **0.92 / 0.11** | 0.75 / 0.02 | 0.12 / 0.00 | 0.24 / 0.00 |
| quad_tank | 0.41 / 0.01 | 0.54 / 0.01 | 0.93 / 0.24 | 0.58 / 0.01 | 0.68 / 0.04 |
| battery | 0.02 / 0.00 | 0.08 / 0.21 | 0.72 / 0.25 | 0.17 / 0.00 | 0.10 / 0.00 |
| vdp, duffing, lorenz | ≤ 0.01 | 0.00 | 0.00 | 0.00 | ≤ 0.01 |

C1-where (5 seeds, default recipe) shows the same pattern: concat 0.98 vs lora 0.49 early on
bioreactor.

- Clipping only matters where the frozen base decoder is far off. Base error RMS is 3.06σ on
  bioreactor and 1.09σ on quad_tank, so the initial loss is large.
- On vdp, duffing and lorenz (where concat catches up) nothing is ever clipped, and
  |g| ~ 3e-3-5e-2.
- With Adam, a clip mostly rescales the step only when the gradient scale changes abruptly. So
  HyperKKL's effective step is **not** smaller than concat's.
- If anything, clipping handicaps concat on bioreactor.

### 5. Generated delta: magnitude, rank, use
Trained C1-tuned lora, seed 0. Per decoder layer 0..3 (5→128→128→128→nx):

| system | ‖ΔW‖F/‖W‖F | ‖Δh‖/‖Wh+b‖ | effect on output, lora vs concat (σ units, RMS ‖T(z;c)−T(z;0)‖) |
|---|---|---|---|
| vdp | 0.21, 0.57, 0.58, 0.01 | 0.15, 0.14, 0.09, 0.004 | 0.084 vs 0.083 |
| bioreactor | 0.24, 4.83, 1.28, 0.05 | 0.11, 11.2, 1.20, 0.60 | 3.06 vs 3.06 |
| duffing | 0.15, 1.95, 0.33, 0.08 | 0.06, 0.13, 0.06, 0.05 | 0.261 vs 0.261 |
| quad_tank | 0.05, 0.22, 0.65, 0.05 | 0.02, 0.09, 0.69, 0.06 | 1.09 vs 1.09 |
| battery | 0.15, 1.12, 4.03, 0.45 | 0.06, 2.08, 3.56, 0.80 | 0.72 vs 0.71 |

- The delta is large, comparable to or larger than the base weights. It is not stuck near zero.
- Per-sample singular values of Δ use all r = 4 directions. Example, vdp layer 1:
  3.7 / 1.5 / 0.65 / 0.38.
- Across samples, vec(Δ(c)) lies in a ≤ 4-dimensional subspace: 96-100% of its variance falls
  in the top 4 principal directions. The input-history manifold of these single-input plants is
  effectively low-dimensional. So rank is not the limit. This agrees with C1-rank16 ≈ rank 4.
- Both variants remove the same input-driven part of the error. Their output effects match to
  3 digits, so the difference lies only in the residual.
- Late in training, dθ/θ per 250 steps is similar for lora and concat (≈1e-3 at the end, under
  the cosine schedule).
- Validation loss is still falling in the last 1000 steps for every variant (v3000/v2000 =
  0.75-0.96). This is shared, and partly the cosine anneal. It is not lora-specific.

### 6. Generator trunk activations
SiLU pre-activations in the lora trunk (trained):

| | vdp | bioreactor | duffing | quad_tank | battery | lorenz |
|---|---|---|---|---|---|---|
| RMS | 0.12 | 0.39 | 0.23 | 0.18 | 0.26 | 0.21 |
| \|max\| | 0.59 | 2.02 | 1.12 | 1.07 | 1.33 | 0.47 |

- No unit is in SiLU's flat region (pre < −2: 0%).
- The trunk is therefore nearly affine-plus-quadratic in c. Δ(c) = U(t)V(t) is then a
  low-degree polynomial in c. This is a capacity property, not a bug.
- A wider or deeper trunk did not help:

| system | trunk 128 | trunk 128x2 | base |
|---|---|---|---|
| vdp | 3.03 | 2.94 (3 seeds) | 2.88 |
| bioreactor | 2.83 | 3.07 / 3.26 / **12.7** (one instability) | 2.94 |

  Values are val MSE ×1e-4.

### 7-9. Learning-rate sweep and fixes
Phase-2 best validation MSE (σ² units, the deployed decoder error on data.val). The C1-tuned
recipe uses lora[lr=4e-3] and concat[lr=4e-3, depth 3]. "ctx1" = per-system constant context
calibration: the last psi layer is multiplied once at init so that RMS(c) = 1 on training
data. This is bias-free, so E(0) = 0 is kept. The factor was 4-14.5. "ctx0.3" sets RMS(c) = 0.3.

**lr sweep, seed 0 (×1e-4 for vdp/bioreactor, ×1e-4 for duffing):**

| | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 | 3.2e-2 |
|---|---|---|---|---|---|
| vdp lora | 3.18 | **2.85** | 4.53 | 5.30 | |
| vdp concat | | 3.40 | **2.72** | 5.94 | 6.85 |
| bioreactor lora | 3.75 | **2.94** | 3.29 | 7.18 | |
| bioreactor concat | | 9.89 | 7.28 | **5.55** | 8.74 |
| duffing lora | | **24.5** | 29.3 | 36.8 | |
| duffing concat | | 24.9 | **23.6** | 27.0 | |

- For lora, 4e-3 is a genuine interior optimum. The lr grid edge was **binding for concat only**.
- lora is sharply hurt above 4e-3. This is consistent with the bilinear U·V product: the step
  in Δ scales with lr² early on.

**HyperKKL fixes at lr 4e-3 (vdp / bioreactor, seed 0; base 2.85 / 2.94):**

| fix | vdp | bioreactor | verdict |
|---|---|---|---|
| (a) V small non-zero init (default × 1e-2) | 2.88 | 2.86 | ≈ noise |
| (a') U zero-init, V default init | 2.96 | 3.21 | worse |
| (b) ctx RMS → 1 | 3.35 | 3.15 | worse |
| (b) ctx RMS → 1, lr 2e-3 | 3.41 | 4.96 | worse |
| (c) U/V heads at 4x lr | 6.25 | 3.20 | worse |
| (d) no 1/sqrt(r) (×1) | 3.07 | 3.00 | slightly worse |
| (d) ×2 (alpha = 2r) | 3.01 | 3.02 | slightly worse |

**Paired over seeds 0/1/2:**

| system | lora 4e-3 (C1-tuned) | lora V-small | concat 8e-3 | **concat ctx1 + lr 8e-3** | concat lr 1.6e-2 |
|---|---|---|---|---|---|
| vdp (×1e-4) | 2.86 / 2.81 / 2.99 → **2.88** | 2.88 / 2.79 / 2.97 → 2.88 | 2.72 / 2.88 / 2.61 → 2.73 | **2.21 / 2.45 / 2.14 → 2.27 (−21% vs lora, 3/3)** | 5.88 |
| duffing (×1e-4) | 24.5 / 24.9 / 27.9 → **25.8** | 36.2 / 27.6 / 30.0 → 31.3 | 23.6 / 29.5 / 24.3 → 25.8 | **18.0 / 18.1 / 16.6 → 17.6 (−32%, 3/3)** | — |
| bioreactor (×1e-4) | 2.94 / **6.47** / 2.93 → 4.11 | **2.86 / 3.04 / 2.71 → 2.87** | 7.28 / 8.17 / 6.09 → 7.18 | 5.93 / 6.80 / 4.84 → 5.86 (lora −30..−50%) | 6.03 |
| quad_tank (×1e-4) | 57.8 / 61.3 / 62.8 → **60.6** | 60.3 / 62.9 / 63.6 → 62.3 | 80.9 / 112 / 104 → 99.2 | 73.6 / 80.6 / 83.3 → 79.2 (lora −23%) | — |

Context calibration acts in opposite directions on the two variants (duffing, 3 seeds):

| variant | ctx 0.3 | ctx 1 | base |
|---|---|---|---|
| lora | 28.3 | 31.6 | 25.8 |
| concat (lr 8e-3) | 19.7 | 17.6 | 25.8 |

On vdp, lora with ctx 0.3 gives 3.08 vs base 2.88.

- **HyperKKL prefers a small context, and concat prefers a large one.** With U(t) and V(t) both
  linear in the trunk output, Δ is quadratic in t. Inflating c inflates Δ's curvature in c and
  the lr-sensitivity of the product.
- The bioreactor seed-1 outlier (6.47e-4, reproduced exactly from the cache) is the bilinear
  early instability: train loss 1.47 at step 250, |g| max 4.5e3 with 100% clipping at steps
  250-500, a 2.2x worse final val.
- V-small removed it on that seed (3.04). It was neutral on vdp and quad_tank, but **hurt
  duffing** (+21%). It is not a general fix.
- The same spike pattern appears in C1-tune-lora-r4 bioreactor seed 0 (|g| max 2.2e3) and on
  quad_tank (85-350), without a final penalty there.

## Recommendations

1. **No fix to HyperKKL is recommended for the C1 claim.** None of (a)-(d), a wider trunk, or
   lr ∈ {2e-3, 8e-3, 1.6e-2} beats the current recipe outside seed noise. The current recipe is
   at its tuned optimum.
2. **The C1-tuned comparison favours HyperKKL for an avoidable reason: concat is under-tuned.**
   - Its best lr is at or above the grid edge (8e-3 on vdp and duffing, 1.6e-2 on bioreactor).
   - A per-system constant context calibration that HyperKKL does not want gives concat −21%
     (vdp) and −32% (duffing) val MSE.
   - Concat then beats HyperKKL on vdp and duffing (3/3 seeds each), and narrows the gap on
     bioreactor and quad_tank, where HyperKKL still wins by 30-50% and 23%.

   Before the paper claims "HyperKKL beats concat", extend the grid. Suggested:
   - lr grid {…, 4e-3, 8e-3, 1.6e-2} for every variant.
   - Add `ctx_scale` ∈ {none, RMS 0.3, RMS 1} as a per-variant tuning knob, applied to all
     variants and selected on validation like the other knobs (PROTOCOL.md).
   - Re-run C1-tuned on the dev and test systems.

   Expected outcome from this audit: HyperKKL keeps bioreactor, quad_tank and battery (the
   systems with large base-decoder error, where weight-space conditioning helps most), and
   probably loses vdp and duffing. Battery was not tested here.
3. **Optional robustness patch for HyperKKL:** V init = default × 1e-2 instead of 0. Δ is still
   exactly 0 at c = 0, because V is bias-free. It fixes the one observed bilinear blow-up but is
   +21% on duffing. Only adopt it as a selected knob, not a default. A cheaper guard against the
   lr² sensitivity is a longer warmup for lora (not tested).
4. **Caveat:**
   - All numbers are phase-2 validation MSE (the in-distribution deployed decoder error), not
     shape_ood / sequence_ood NRMSE, which the C1 rule votes on.
   - Seeds are 0-2 only. Seeds 3-4 have no cached phase-1 maps.
   - The ranking change for concat should be confirmed with a proper tree node before it enters
     the paper.

## Reproduction
- Scratch package and scripts:
  `/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/`
  - `pkg/hyperkkl` with env knobs `AUDIT_LORA_INIT`, `AUDIT_LORA_MULT`, `AUDIT_CTX_SCALE`,
    `AUDIT_HEAD_LR_MULT`, `AUDIT_LORA_TRUNK_HIDDEN/DEPTH`.
  - `run_p2.py`: phase 2 on cached C1-tuned artefacts, read-only.
  - `diag.py`: gradient, delta and saturation diagnostics.
  - `parse_logs.py`, `parse2.py`: log statistics.
  - `results/*.json`: per-run val, history and monitor windows.
