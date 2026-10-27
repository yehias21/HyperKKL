# Why the learned KKL observer has an error floor (eval-only diagnostic)

Written 2026-10-26, about 02:15 CET. No model was trained. The only fits are the in-memory grey-box residuals, which are refitted exactly as `reeval` does, and the small probes of Exp 3. Nothing was written to `experiments/tree`, `queue` or `cache`: the loader's cache function was replaced by a read-only one that raises on a miss, and the stored scales were used.

Scripts and raw per-run output are in `diagnostics/error_floor/`: `common.py` (loader), `exp12.py`, `exp34.py`, `faults_noise10.py`, `aggregate.py`, `out12/*.json`, `out34/*.json` and `tables.md`. The aggregate is in `ERROR_FLOOR.json`.

**Coverage.** Exp 1 covers C1-where, C3-partial and C3-mismatch30, for the six dev systems and seeds 0-2. All **54/54** runs loaded from cache, with methods latent, lora, hyper, ekf, greybox_ekf and greybox_ukf, so no run was skipped. Exp 2 covers C1-where (18 runs) plus the stored C3-faults noise10 block. Exp 3 and Exp 4 cover C1-where and C3-partial (36 runs).

**Reproduction check.** Each run re-evaluated the stored `clean` condition alongside the new ones. All 54 runs reproduce the stored per-method NRMSE (relative difference < 1e-3) for latent, lora, hyper, ekf and greybox. The one exception is greybox_ekf/ukf on C3-partial lorenz, which diverges there anyway (NRMSE about 1.2-1.3) and is nondeterministic at the 5% level.

**Code-drift incident.** At 01:14 CET another process edited the live `hyperkkl/nets.py`. That changed the MODEL_MODULES hash, so the cache keys no longer matched (26 runs failed with a cache miss). Those runs were re-done from a frozen copy of the package in the scratchpad: the live files, with `nets.py` restored from the runs' own snapshot `experiments/code/a1e5d75cce66.tar.gz`. The hash came back to the runs' value and the stored numbers reproduce. Everything that ran before 01:14 used identical code.

**Method.**
- **Noise-free evaluation (Exp 1).** Same evaluation episodes, with `y = y_clean` (`EpisodeSet.with_sensor`, noise 0). Every observer recomputes its latent online from that y.
- **Filters.** EKF and grey-box EKF/UKF keep their nominal R, built for 3% noise. The learned observers likewise were trained at 3%. So "noise-free" means *same estimators, cleaner data*.
- **Noise part.** `e_n(sigma) = sqrt(e(sigma)^2 - e(0)^2)`. The noise-free error `e(0)` is the floor.
- **NRMSE definitions.** Tables 1-2 use the paper's NRMSE (mean over episodes of the settled-window NRMSE). Exp 3 and Exp 4 compare decoders and probes on one pooled NRMSE over all settled samples (k >= settle) of the same evaluation episodes. That is why the lora numbers differ slightly between tables.

## Verdict

**The error floor has two separable causes. Which one dominates splits the six systems into two groups. The information limit of z is a real but secondary third term.**

1. **H-noise dominates on duffing and vdp. It is also large on battery (id tier) and a moderate share on lorenz and quad_tank.**
   - On duffing (C1-where, sequence OOD), lora's error is 0.057. Removing the sensor noise cuts it to 0.030; on the id tier, from 0.049 to 0.016. The noise accounts for 71% of the MSE on sequence OOD and 89% on id.
   - On vdp the error goes from 0.019 to 0.013 (54% / 76%). The latent variant is the most noise-driven: 95% of its MSE on duffing.
   - The noise part grows **exactly linearly** with sigma. The fitted slope of `e_n` over 1-10% is 0.97-1.07 for every KKL variant on every system. The one exception is bioreactor lora, whose noise part is about 0.
   - It is fully explained by the decoder Jacobian: `RMS(J (z_obs - z_clean))` matches the actual `T+(z_obs,c) - T+(z_clean,c)` to within 2% wherever the noise part is non-negligible.
   - At 3%, KKL's noise part is **2-15x the EKF's**: duffing 15x, battery 10x, quad_tank 8x, vdp 3.6x, lorenz 2.1x. Its gain follows the decoder Jacobian (lora ||J|| median: duffing 445, battery 149, vdp 22, quad_tank 14, lorenz 5, bioreactor 1).
   - That gain is **inherited from the frozen base map** (autonomous ||J||: duffing 366, vdp 21). The conditioner does not reduce it (lora duffing: 445).
   - Bioreactor is the counter-example. Its lora decoder learned to ignore z (||J|| about 1), so it has zero noise sensitivity. Its whole error is approximation error.

2. **H-approx dominates on battery (sequence OOD), bioreactor, quad_tank and lorenz.**
   - On these systems the noise-free error is almost the noisy error. Lora at 3% vs 0%, sequence OOD: battery 0.072 vs 0.068, bioreactor 0.062 vs 0.062, quad_tank 0.070 vs 0.066, lorenz 0.027 vs 0.024. The grey-box gap does not shrink without noise (battery 13.8x -> 14.5x).
   - A wide MLP probe on the **same inputs** `[z_obs, c]`, trained on the same training episodes, beats the trained decoder by 1.5-2.8x on held-out evaluation episodes (sequence OOD): battery 0.074 -> 0.029, bioreactor 0.070 -> 0.025, lorenz 0.028 -> 0.014, quad_tank 0.071 -> 0.048.
   - The probe fits id and sequence OOD alike, so this is not a generalisation gap. The information is in `(z_obs, c)`; **the frozen-base + 140k conditioner trained for 3000 steps cannot extract it.**
   - The same holds on duffing given its noisy z: the probe reaches 0.037 vs the decoder's 0.057. The decoder is not the best regressor even of its own noisy input, because it keeps the base's high gain.

3. **H-info: under noise z_obs is not a sufficient statistic. It matters where the noise does, plus on quad_tank.**
   - A probe on a strided z window (20 samples, stride 10, about 200 samples of history) closes most of the noise loss. Duffing: 0.037 -> 0.023 on sequence OOD and 0.031 -> 0.013 on id, versus 0.012-0.015 for the probe on noise-free z. Also vdp 0.0156 -> 0.0118 and battery (id) 0.028 -> 0.020.
   - Short unstrided windows (k = 5 or 20 consecutive samples) barely help: duffing 0.037 -> 0.035. Consecutive z samples at max|lambda| dt = 0.1 are nearly identical, so the window has to span about 1/alpha or more.
   - **Quad_tank is the information-limited case.** Even the probe on *noise-free* z plus c stalls at 0.046 (sequence OOD). Only a 600-sample window (stride 30, 30 s) helps, down to 0.038. Its nz = 6 latent with these poles does not capture the slow tank states (consistent with L* = 599 samples in C2).
   - Lorenz and bioreactor gain 10-25% from windows.

4. **None of these fixes alone reaches the grey-box filter.**
   - The best probe per system is a lower bound on what a better decoder of z (plus a window) could reach. It is still 1.6-3x above the grey-box EKF/UKF of C3-partial: duffing 0.021-0.023 vs 0.0078, battery 0.029 vs 0.019, quad_tank 0.037 vs 0.014.
   - Noise-free, the clean-z probe floor is 0.006-0.046, against the exact EKF's 0.003-0.014.
   - Closing the remaining factor needs information the latent filter does not carry (the model, through an innovation correction), not a better regressor.

**Which fix is indicated where:**

| system | dominant cause | evidence | indicated fix |
|---|---|---|---|
| duffing | H-noise (71-89% of MSE), base-map gain 370-445 | noise-free 0.057 -> 0.030; e_n slope 1.0; noise gain 15x EKF; z-window probe 0.023 (id 0.012) | **decoder on a strided z-window** (span >= 1/alpha) is the larger lever. A **Jacobian/Lipschitz penalty** is bounded by the [z_obs,c] probe (0.037 at best, -35%). Then hybrid correction for the last 3x |
| vdp | H-noise (54-76%) | 0.019 -> 0.013; z-window probe 0.012 | z-window, then Jacobian penalty. Already within 2x of grey-box; it wins in C3-partial |
| battery | H-approx (seq OOD; 10% noise share), H-noise on id (50%) | decoder 0.074 vs probe 0.029 with the same inputs; noise-free gap unchanged | **richer decoder (unfreeze or retrain the base, longer phase 2)**: 2.5x available. Window helps on id. Hybrid for the last 1.6x vs grey-box |
| bioreactor | H-approx (0% noise share; decoder ignores z) | probe 0.025 vs 0.070 (seq); ||J|| about 1 | richer decoder / unfrozen base. Not a noise problem |
| quad_tank | H-approx + H-info (latent) | probe 0.048 vs 0.071; noise-free-z probe 0.046; only a 600-sample window helps (0.038) | **richer latent / slower pole design** (or a long window), plus a richer decoder |
| lorenz | H-approx (19-25% noise) | probe 0.014 vs 0.027 (lora) / 0.019 (hyper) | richer decoder. KKL already beats the data-driven baselines here |

**Hypothesis by hypothesis:**
- **H-noise** is confirmed as the mechanism of the "smooth-plant" losses on duffing and vdp. It enters exactly linearly through the decoder Jacobian, and that Jacobian comes from the frozen base.
- **H-approx** is confirmed on battery, bioreactor, quad_tank and lorenz, and also partly on duffing. EVIDENCE.md calls it a "fit floor": it is a decoder bottleneck, because a probe with the same inputs does 1.5-2.8x better.
- **H-info** is real but secondary. It matters mostly as "z_obs is not sufficient under noise" (fixed by a long z-window) and, on quad_tank, as a latent that misses slow states.

Caveats:
- 3 seeds.
- The probes (4x512 SiLU MLP, 6000 Adam steps, 600k training samples, selected on the validation episodes) are lower bounds only for this probe class. A bigger probe could go lower, which would strengthen the H-approx / H-info conclusions, not weaken them.
- The kNN probe (k = 10, standardised [z, c]) is useless in this dimension (0.04-0.58, worse than the MLP everywhere). It is reported but carries no evidence.
- "Noise-free" decoders are evaluated off their training distribution (trained on noisy z), so e(0) for KKL is an upper bound on its noise-free approximation error.
- For C1-where (exact model), greybox_ekf = ekf numerically, since the learned residual is about 0.
- The EKF at 10% noise diverges on one battery seed (sequence OOD 1.57).

## Summary decomposition (C1-where, exact model; lora; seed mean)

The noise-gain column is `e_n(lora) / e_n(EKF)` at 3%. The probe columns are pooled held-out NRMSE. For comparison, the lora decoder's own pooled numbers are in Exp 3, column 1.

**sequence_ood**

| system | lora 3% | lora noise-free e0 | lora noise part e_n | EKF 3% / e0 / e_n | noise gain vs EKF | gap lora/EKF 3% -> 0% | MLP probe [z_obs,c] | best z-window probe | probe [z_clean,c] | lora ||J|| median (q95) |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0719 | 0.0683 | 0.0225 | 0.0052 / 0.0047 / 0.0022 | 10.1x | 13.8x -> 14.5x | 0.0291 | 0.0289 (zwin20s10_c) | 0.0280 | 149 (236) |
| bioreactor | 0.0617 | 0.0617 | 0.0018 | 0.0062 / 0.0062 / 0.0005 | 3.4x | 10.0x -> 10.0x | 0.0245 | 0.0203 (zwin5_c) | 0.0245 | 1 (2) |
| duffing | 0.0570 | 0.0302 | 0.0483 | 0.0049 / 0.0037 / 0.0032 | 15.2x | 11.7x -> 8.2x | 0.0371 | 0.0230 (zwin20s10_c) | 0.0150 | 445 (603) |
| lorenz | 0.0266 | 0.0239 | 0.0117 | 0.0065 / 0.0031 / 0.0057 | 2.1x | 4.1x -> 7.6x | 0.0139 | 0.0110 (zwin20s10_c) | 0.0084 | 5 (11) |
| quad_tank | 0.0700 | 0.0658 | 0.0240 | 0.0141 / 0.0137 / 0.0031 | 7.9x | 5.0x -> 4.8x | 0.0477 | 0.0384 (zwin20s30_c) | 0.0462 | 14 (17) |
| vdp | 0.0191 | 0.0130 | 0.0140 | 0.0058 / 0.0043 / 0.0038 | 3.6x | 3.3x -> 3.0x | 0.0156 | 0.0118 (zwin20s10_c) | 0.0072 | 22 (35) |

**id**

| system | lora 3% | lora noise-free e0 | lora noise part e_n | EKF 3% / e0 / e_n | noise gain vs EKF | gap lora/EKF 3% -> 0% | MLP probe [z_obs,c] | best z-window probe | probe [z_clean,c] | lora ||J|| median (q95) |
|---|---|---|---|---|---|---|---|---|---|---|
| battery | 0.0479 | 0.0340 | 0.0337 | 0.0059 / 0.0051 / 0.0030 | 11.1x | 8.1x -> 6.7x | 0.0281 | 0.0197 (zwin20s10_c) | 0.0230 | 149 (236) |
| bioreactor | 0.0192 | 0.0189 | 0.0033 | 0.0062 / 0.0062 / 0.0006 | 5.9x | 3.1x -> 3.0x | 0.0116 | 0.0107 (zwin5_c) | 0.0116 | 1 (2) |
| duffing | 0.0492 | 0.0160 | 0.0465 | 0.0046 / 0.0035 / 0.0030 | 15.6x | 10.7x -> 4.5x | 0.0305 | 0.0119 (zwin20s30_c) | 0.0123 | 445 (603) |
| lorenz | 0.0276 | 0.0240 | 0.0136 | 0.0067 / 0.0032 / 0.0058 | 2.3x | 4.1x -> 7.4x | 0.0138 | 0.0110 (zwin20s10_c) | 0.0083 | 5 (11) |
| quad_tank | 0.0520 | 0.0473 | 0.0217 | 0.0144 / 0.0141 / 0.0030 | 7.2x | 3.6x -> 3.4x | 0.0342 | 0.0278 (zwin20s30_c) | 0.0332 | 14 (17) |
| vdp | 0.0165 | 0.0081 | 0.0144 | 0.0056 / 0.0043 / 0.0036 | 4.0x | 2.9x -> 1.9x | 0.0135 | 0.0087 (zwin20s10_c) | 0.0058 | 22 (35) |

## Detailed tables

**Hypothesis map.**
- **H-noise:** Exp 1 (noise-free), Exp 2 (scaling), and Exp 4 (Jacobian and noise-part decomposition).
- **H-approx:** Exp 3, decoder vs the MLP probe on the same `[z_obs, c]`.
- **H-info:** Exp 3, z-window probes and the noise-free-z probe.

**Probe set-up.**
- Probe c = the trained lora context (`ctx.seq(u)`).
- The z-window features are `[z_t, z_{t-s}, ..., z_{t-(k-1)s}]` scaled by the training std of z.
- Probes are fitted on the training episodes (k >= burn), selected on the validation episodes, and scored on the evaluation episodes (k >= settle).
- Probe names: "z_win5" / "z_win20" use k consecutive samples; "stride10" / "stride30" use 20 samples spaced 10 or 30 apart.

### Exp 1 (id): NRMSE at 3% sensor noise (stored clean episodes) -> noise-free (same episodes), seed mean

| node | system | latent | lora | hyper | ekf | greybox_ekf | greybox_ukf | best KKL / best grey-box: 3% -> 0% |
|---|---|---|---|---|---|---|---|---|
| C1-where | battery | 0.123 -> 0.0792 | 0.0479 -> 0.0340 | 0.0487 -> 0.0347 | 0.0059 -> 0.0051 | 0.0059 -> 0.0051 | 0.0073 -> 0.0071 | 8.1x -> 6.7x |
| C1-where | bioreactor | 0.0344 -> 0.0169 | 0.0192 -> 0.0189 | 0.0222 -> 0.0170 | 0.0062 -> 0.0062 | 0.0062 -> 0.0062 | 0.0062 -> 0.0062 | 3.1x -> 2.7x |
| C1-where | duffing | 0.0771 -> 0.0147 | 0.0492 -> 0.0160 | 0.0509 -> 0.0172 | 0.0046 -> 0.0035 | 0.0046 -> 0.0035 | 0.0046 -> 0.0035 | 10.7x -> 4.2x |
| C1-where | lorenz | 0.0300 -> 0.0267 | 0.0276 -> 0.0240 | 0.0207 -> 0.0156 | 0.0067 -> 0.0032 | 0.0067 -> 0.0032 | 0.0066 -> 0.0032 | 3.1x -> 4.9x |
| C1-where | quad_tank | 0.161 -> 0.144 | 0.0520 -> 0.0473 | 0.0540 -> 0.0491 | 0.0144 -> 0.0141 | 0.0146 -> 0.0143 | 0.0146 -> 0.0143 | 3.6x -> 3.4x |
| C1-where | vdp | 0.0253 -> 0.0085 | 0.0165 -> 0.0081 | 0.0224 -> 0.0096 | 0.0056 -> 0.0043 | 0.0056 -> 0.0043 | 0.0056 -> 0.0043 | 2.9x -> 1.9x |
| C3-partial | battery | 0.164 -> 0.126 | 0.0476 -> 0.0345 | 0.0436 -> 0.0322 | 0.402 -> 0.402 | 0.0275 -> 0.0274 | 0.0270 -> 0.0269 | 1.6x -> 1.2x |
| C3-partial | bioreactor | 0.0454 -> 0.0325 | 0.0239 -> 0.0236 | 0.0291 -> 0.0249 | 1.623 -> 1.623 | 0.294 -> 0.294 | 0.281 -> 0.281 | 0.1x -> 0.1x |
| C3-partial | duffing | 0.0736 -> 0.0146 | 0.0490 -> 0.0157 | 0.0503 -> 0.0175 | 0.356 -> 0.356 | 0.0073 -> 0.0066 | 0.0073 -> 0.0066 | 6.7x -> 2.2x |
| C3-partial | lorenz | 0.0390 -> 0.0355 | 0.0298 -> 0.0267 | 0.0221 -> 0.0178 | 3.635 -> 3.634 | 1.255 -> 1.289 | 1.247 -> 1.230 | 0.0x -> 0.0x |
| C3-partial | quad_tank | 0.164 -> 0.147 | 0.0517 -> 0.0479 | 0.0533 -> 0.0494 | 0.0287 -> 0.0285 | 0.0150 -> 0.0147 | 0.0150 -> 0.0147 | 3.4x -> 3.3x |
| C3-partial | vdp | 0.0315 -> 0.0102 | 0.0157 -> 0.0081 | 0.0180 -> 0.0093 | 0.322 -> 0.322 | 0.0196 -> 0.0193 | 0.0196 -> 0.0193 | 0.8x -> 0.4x |
| C3-mismatch30 | battery | 0.105 -> 0.0589 | 0.0382 -> 0.0238 | 0.0481 -> 0.0310 | 0.180 -> 0.180 | 0.0127 -> 0.0124 | 0.0131 -> 0.0127 | 3.0x -> 1.9x |
| C3-mismatch30 | bioreactor | 0.0911 -> 0.0877 | 0.0329 -> 0.0328 | 0.0326 -> 0.0307 | 4.572 -> 4.572 | 0.0722 -> 0.0722 | 0.0691 -> 0.0691 | 0.5x -> 0.4x |
| C3-mismatch30 | duffing | 0.0765 -> 0.0146 | 0.0493 -> 0.0161 | 0.0507 -> 0.0179 | 0.0478 -> 0.0475 | 0.0048 -> 0.0038 | 0.0048 -> 0.0038 | 10.3x -> 3.9x |
| C3-mismatch30 | lorenz | 0.0107 -> 0.0034 | 0.0066 -> 0.0029 | 0.0099 -> 0.0033 | 0.495 -> 0.495 | 0.0202 -> 0.0202 | 0.0202 -> 0.0202 | 0.3x -> 0.1x |
| C3-mismatch30 | quad_tank | 0.171 -> 0.159 | 0.0221 -> 0.0215 | 0.0225 -> 0.0219 | 4.749 -> 4.749 | 0.0134 -> 0.0131 | 0.0134 -> 0.0131 | 1.6x -> 1.6x |
| C3-mismatch30 | vdp | 0.0247 -> 0.0090 | 0.0179 -> 0.0087 | 0.0211 -> 0.0098 | 0.109 -> 0.109 | 0.0089 -> 0.0079 | 0.0089 -> 0.0079 | 2.0x -> 1.1x |

### Exp 1 (sequence_ood): NRMSE at 3% sensor noise (stored clean episodes) -> noise-free (same episodes), seed mean

| node | system | latent | lora | hyper | ekf | greybox_ekf | greybox_ukf | best KKL / best grey-box: 3% -> 0% |
|---|---|---|---|---|---|---|---|---|
| C1-where | battery | 0.129 -> 0.104 | 0.0719 -> 0.0683 | 0.0761 -> 0.0700 | 0.0052 -> 0.0047 | 0.0053 -> 0.0047 | 0.0071 -> 0.0064 | 13.8x -> 14.5x |
| C1-where | bioreactor | 0.0438 -> 0.0312 | 0.0617 -> 0.0617 | 0.0510 -> 0.0501 | 0.0062 -> 0.0062 | 0.0062 -> 0.0062 | 0.0062 -> 0.0062 | 7.1x -> 5.1x |
| C1-where | duffing | 0.0782 -> 0.0181 | 0.0570 -> 0.0302 | 0.0620 -> 0.0341 | 0.0049 -> 0.0037 | 0.0049 -> 0.0037 | 0.0049 -> 0.0037 | 11.7x -> 4.9x |
| C1-where | lorenz | 0.0271 -> 0.0238 | 0.0266 -> 0.0239 | 0.0182 -> 0.0136 | 0.0065 -> 0.0031 | 0.0065 -> 0.0031 | 0.0065 -> 0.0031 | 2.8x -> 4.4x |
| C1-where | quad_tank | 0.163 -> 0.148 | 0.0700 -> 0.0658 | 0.0644 -> 0.0610 | 0.0141 -> 0.0137 | 0.0142 -> 0.0139 | 0.0142 -> 0.0139 | 4.6x -> 4.4x |
| C1-where | vdp | 0.0266 -> 0.0121 | 0.0191 -> 0.0130 | 0.0249 -> 0.0143 | 0.0058 -> 0.0043 | 0.0058 -> 0.0043 | 0.0058 -> 0.0043 | 3.3x -> 2.8x |
| C3-partial | battery | 0.140 -> 0.115 | 0.0744 -> 0.0706 | 0.0717 -> 0.0657 | 0.239 -> 0.239 | 0.0191 -> 0.0191 | 0.0191 -> 0.0194 | 3.8x -> 3.4x |
| C3-partial | bioreactor | 0.0867 -> 0.0811 | 0.0753 -> 0.0757 | 0.0703 -> 0.0700 | 1.656 -> 1.656 | 0.293 -> 0.293 | 0.281 -> 0.281 | 0.2x -> 0.2x |
| C3-partial | duffing | 0.0766 -> 0.0212 | 0.0536 -> 0.0274 | 0.0642 -> 0.0393 | 0.236 -> 0.237 | 0.0078 -> 0.0072 | 0.0078 -> 0.0071 | 6.9x -> 3.0x |
| C3-partial | lorenz | 0.0354 -> 0.0318 | 0.0308 -> 0.0286 | 0.0192 -> 0.0157 | 3.637 -> 3.637 | 1.220 -> 1.220 | 1.225 -> 1.254 | 0.0x -> 0.0x |
| C3-partial | quad_tank | 0.168 -> 0.153 | 0.0717 -> 0.0682 | 0.0625 -> 0.0598 | 0.0211 -> 0.0209 | 0.0142 -> 0.0138 | 0.0142 -> 0.0138 | 4.4x -> 4.3x |
| C3-partial | vdp | 0.0343 -> 0.0165 | 0.0179 -> 0.0124 | 0.0199 -> 0.0143 | 0.352 -> 0.352 | 0.0257 -> 0.0254 | 0.0257 -> 0.0254 | 0.7x -> 0.5x |
| C3-mismatch30 | battery | 0.106 -> 0.0778 | 0.0515 -> 0.0433 | 0.0696 -> 0.0596 | 0.177 -> 0.177 | 0.0151 -> 0.0149 | 0.0153 -> 0.0151 | 3.4x -> 2.9x |
| C3-mismatch30 | bioreactor | 0.168 -> 0.167 | 0.0977 -> 0.0980 | 0.123 -> 0.122 | 4.576 -> 4.576 | 0.0705 -> 0.0705 | 0.0674 -> 0.0674 | 1.4x -> 1.5x |
| C3-mismatch30 | duffing | 0.0779 -> 0.0183 | 0.0561 -> 0.0313 | 0.0628 -> 0.0373 | 0.0216 -> 0.0214 | 0.0059 -> 0.0050 | 0.0059 -> 0.0050 | 9.5x -> 3.7x |
| C3-mismatch30 | lorenz | 0.0108 -> 0.0037 | 0.0070 -> 0.0036 | 0.0097 -> 0.0036 | 0.495 -> 0.495 | 0.0201 -> 0.0201 | 0.0201 -> 0.0201 | 0.3x -> 0.2x |
| C3-mismatch30 | quad_tank | 0.185 -> 0.176 | 0.0422 -> 0.0418 | 0.0436 -> 0.0433 | 4.691 -> 4.691 | 0.0129 -> 0.0126 | 0.0129 -> 0.0126 | 3.3x -> 3.3x |
| C3-mismatch30 | vdp | 0.0257 -> 0.0116 | 0.0205 -> 0.0128 | 0.0232 -> 0.0126 | 0.115 -> 0.115 | 0.0098 -> 0.0088 | 0.0098 -> 0.0088 | 2.1x -> 1.3x |

### Exp 2: C1-where, NRMSE vs sensor noise 0 / 1 / 3 / 10 % (re-drawn noise, same episodes), sequence_ood; slope b = d log e / d log sigma fitted on 1-10 %; e(0) = noise-free floor

| system | method | 0% | 1% | 3% | 10% | slope b (1-10%) | noise part e_n(3%) = sqrt(e3^2 - e0^2) | slope of e_n (1-10%) | noise share of MSE at 3% |
|---|---|---|---|---|---|---|---|---|---|
| battery | latent | 0.104 | 0.107 | 0.130 | 0.273 | 0.41 | 0.0774 | 0.98 | 36% |
| battery | lora | 0.0683 | 0.0687 | 0.0718 | 0.102 | 0.17 | 0.0222 | 1.03 | 10% |
| battery | hyper | 0.0700 | 0.0705 | 0.0757 | 0.123 | 0.24 | 0.0289 | 1.07 | 15% |
| battery | ekf | 0.0047 | 0.0048 | 0.0055 | 1.572 | 2.55 | 0.0029 | 3.33 | 28% |
| battery | greybox_ekf | 0.0047 | 0.0048 | 0.0055 | 1.572 | 2.55 | 0.0029 | 3.33 | 28% |
| battery | greybox_ukf | 0.0064 | 0.0065 | 0.0070 | 0.0143 | 0.34 | 0.0029 | 1.02 | 17% |
| bioreactor | latent | 0.0312 | 0.0326 | 0.0430 | 0.104 | 0.51 | 0.0296 | 1.01 | 47% |
| bioreactor | lora | 0.0617 | 0.0617 | 0.0617 | 0.0617 | 0.00 | 0.0000 | 3.48 | -0% |
| bioreactor | hyper | 0.0501 | 0.0502 | 0.0508 | 0.0576 | 0.06 | 0.0087 | 0.97 | 3% |
| bioreactor | ekf | 0.0062 | 0.0062 | 0.0062 | 0.0064 | 0.02 | 0.0004 | 1.85 | 1% |
| bioreactor | greybox_ekf | 0.0062 | 0.0062 | 0.0062 | 0.0064 | 0.02 | 0.0004 | 1.40 | 1% |
| bioreactor | greybox_ukf | 0.0062 | 0.0062 | 0.0062 | 0.0065 | 0.02 | 0.0005 | 3.26 | 1% |
| duffing | latent | 0.0181 | 0.0311 | 0.0780 | 0.254 | 0.91 | 0.0758 | 1.00 | 95% |
| duffing | lora | 0.0302 | 0.0341 | 0.0564 | 0.167 | 0.69 | 0.0476 | 1.02 | 71% |
| duffing | hyper | 0.0341 | 0.0381 | 0.0616 | 0.174 | 0.66 | 0.0514 | 1.00 | 69% |
| duffing | ekf | 0.0037 | 0.0038 | 0.0047 | 0.0104 | 0.44 | 0.0028 | 1.05 | 37% |
| duffing | greybox_ekf | 0.0037 | 0.0038 | 0.0047 | 0.0104 | 0.44 | 0.0028 | 1.06 | 36% |
| duffing | greybox_ukf | 0.0037 | 0.0038 | 0.0047 | 0.0105 | 0.44 | 0.0028 | 1.06 | 37% |
| lorenz | latent | 0.0238 | 0.0242 | 0.0271 | 0.0494 | 0.31 | 0.0131 | 1.00 | 23% |
| lorenz | lora | 0.0239 | 0.0242 | 0.0266 | 0.0448 | 0.27 | 0.0116 | 0.99 | 19% |
| lorenz | hyper | 0.0136 | 0.0142 | 0.0181 | 0.0407 | 0.46 | 0.0119 | 0.98 | 43% |
| lorenz | ekf | 0.0031 | 0.0037 | 0.0067 | 0.0199 | 0.73 | 0.0059 | 1.00 | 78% |
| lorenz | greybox_ekf | 0.0031 | 0.0037 | 0.0067 | 0.0225 | 0.79 | 0.0059 | 1.06 | 78% |
| lorenz | greybox_ukf | 0.0031 | 0.0037 | 0.0066 | 0.0197 | 0.73 | 0.0058 | 1.00 | 78% |
| quad_tank | latent | 0.148 | 0.150 | 0.162 | 0.269 | 0.26 | 0.0652 | 1.06 | 16% |
| quad_tank | lora | 0.0658 | 0.0662 | 0.0698 | 0.103 | 0.19 | 0.0232 | 1.05 | 11% |
| quad_tank | hyper | 0.0610 | 0.0613 | 0.0644 | 0.0925 | 0.18 | 0.0205 | 1.04 | 10% |
| quad_tank | ekf | 0.0137 | 0.0138 | 0.0141 | 0.0171 | 0.09 | 0.0031 | 0.99 | 5% |
| quad_tank | greybox_ekf | 0.0139 | 0.0139 | 0.0142 | 0.278 | 1.32 | 0.0031 | 2.45 | 5% |
| quad_tank | greybox_ukf | 0.0139 | 0.0139 | 0.0142 | 0.0172 | 0.09 | 0.0031 | 1.00 | 5% |
| vdp | latent | 0.0121 | 0.0146 | 0.0269 | 0.0807 | 0.75 | 0.0240 | 0.99 | 80% |
| vdp | lora | 0.0130 | 0.0138 | 0.0192 | 0.0488 | 0.55 | 0.0141 | 1.00 | 54% |
| vdp | hyper | 0.0143 | 0.0159 | 0.0251 | 0.0697 | 0.65 | 0.0206 | 0.99 | 68% |
| vdp | ekf | 0.0043 | 0.0045 | 0.0057 | 0.0130 | 0.46 | 0.0037 | 0.98 | 42% |
| vdp | greybox_ekf | 0.0043 | 0.0045 | 0.0057 | 0.0130 | 0.46 | 0.0037 | 0.99 | 42% |
| vdp | greybox_ukf | 0.0043 | 0.0045 | 0.0057 | 0.0130 | 0.46 | 0.0037 | 0.99 | 42% |

### Exp 3 (C1-where, id): held-out pooled NRMSE (k >= settle) of the trained decoders vs probes fit on the training episodes, seed mean

| system | lora dec (z_obs) | hyper dec | lora dec on noise-free z | MLP [z_obs,c] | kNN [z_obs,c] | MLP [z_win5,c] | MLP [z_win20,c] | MLP [z_win20 stride10,c] | MLP [z_win20 stride30,c] | MLP [z_clean,c] | MLP [z_clean win20,c] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| battery (n=3) | 0.0551 | 0.0551 | 0.0388 | 0.0281 | 0.264 | 0.0273 | 0.0250 | 0.0197 | 0.0211 | 0.0230 | 0.0204 |
| bioreactor (n=3) | 0.0255 | 0.0248 | 0.0252 | 0.0116 | 0.0378 | 0.0107 | 0.0111 | 0.0131 | 0.0145 | 0.0116 | 0.0106 |
| duffing (n=3) | 0.0511 | 0.0528 | 0.0177 | 0.0305 | 0.218 | 0.0281 | 0.0232 | 0.0134 | 0.0119 | 0.0123 | 0.0109 |
| lorenz (n=3) | 0.0423 | 0.0362 | 0.0407 | 0.0138 | 0.0709 | 0.0139 | 0.0121 | 0.0110 | 0.0118 | 0.0083 | 0.0053 |
| quad_tank (n=3) | 0.0537 | 0.0565 | 0.0491 | 0.0342 | 0.219 | 0.0340 | 0.0338 | 0.0322 | 0.0278 | 0.0332 | 0.0331 |
| vdp (n=3) | 0.0168 | 0.0226 | 0.0085 | 0.0135 | 0.102 | 0.0126 | 0.0114 | 0.0087 | 0.0092 | 0.0058 | 0.0055 |

### Exp 3 (C1-where, sequence_ood): held-out pooled NRMSE (k >= settle) of the trained decoders vs probes fit on the training episodes, seed mean

| system | lora dec (z_obs) | hyper dec | lora dec on noise-free z | MLP [z_obs,c] | kNN [z_obs,c] | MLP [z_win5,c] | MLP [z_win20,c] | MLP [z_win20 stride10,c] | MLP [z_win20 stride30,c] | MLP [z_clean,c] | MLP [z_clean win20,c] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| battery (n=3) | 0.0740 | 0.0773 | 0.0704 | 0.0291 | 0.383 | 0.0300 | 0.0298 | 0.0289 | 0.0322 | 0.0280 | 0.0275 |
| bioreactor (n=3) | 0.0698 | 0.0515 | 0.0697 | 0.0245 | 0.0917 | 0.0203 | 0.0208 | 0.0310 | 0.0342 | 0.0245 | 0.0200 |
| duffing (n=3) | 0.0571 | 0.0621 | 0.0304 | 0.0371 | 0.528 | 0.0348 | 0.0345 | 0.0230 | 0.0254 | 0.0150 | 0.0158 |
| lorenz (n=3) | 0.0275 | 0.0186 | 0.0251 | 0.0139 | 0.0936 | 0.0136 | 0.0119 | 0.0110 | 0.0118 | 0.0084 | 0.0050 |
| quad_tank (n=3) | 0.0707 | 0.0649 | 0.0665 | 0.0477 | 0.252 | 0.0476 | 0.0480 | 0.0484 | 0.0384 | 0.0462 | 0.0470 |
| vdp (n=3) | 0.0192 | 0.0250 | 0.0131 | 0.0156 | 0.177 | 0.0144 | 0.0136 | 0.0118 | 0.0128 | 0.0072 | 0.0068 |

### Exp 3 (C3-partial, id): held-out pooled NRMSE (k >= settle) of the trained decoders vs probes fit on the training episodes, seed mean

| system | lora dec (z_obs) | hyper dec | lora dec on noise-free z | MLP [z_obs,c] | kNN [z_obs,c] | MLP [z_win5,c] | MLP [z_win20,c] | MLP [z_win20 stride10,c] | MLP [z_win20 stride30,c] | MLP [z_clean,c] | MLP [z_clean win20,c] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| battery (n=3) | 0.0556 | 0.0490 | 0.0393 | 0.0272 | 0.265 | 0.0265 | 0.0242 | 0.0199 | 0.0215 | 0.0231 | 0.0197 |
| bioreactor (n=3) | 0.0381 | 0.0350 | 0.0378 | 0.0141 | 0.0416 | 0.0126 | 0.0131 | 0.0153 | 0.0158 | 0.0140 | 0.0123 |
| duffing (n=3) | 0.0508 | 0.0523 | 0.0173 | 0.0306 | 0.227 | 0.0281 | 0.0227 | 0.0126 | 0.0111 | 0.0122 | 0.0109 |
| lorenz (n=3) | 0.0480 | 0.0426 | 0.0467 | 0.0138 | 0.0624 | 0.0139 | 0.0121 | 0.0110 | 0.0118 | 0.0084 | 0.0054 |
| quad_tank (n=3) | 0.0536 | 0.0558 | 0.0498 | 0.0335 | 0.215 | 0.0339 | 0.0336 | 0.0322 | 0.0276 | 0.0327 | 0.0333 |
| vdp (n=3) | 0.0161 | 0.0185 | 0.0085 | 0.0132 | 0.0972 | 0.0122 | 0.0108 | 0.0087 | 0.0091 | 0.0059 | 0.0055 |

### Exp 3 (C3-partial, sequence_ood): held-out pooled NRMSE (k >= settle) of the trained decoders vs probes fit on the training episodes, seed mean

| system | lora dec (z_obs) | hyper dec | lora dec on noise-free z | MLP [z_obs,c] | kNN [z_obs,c] | MLP [z_win5,c] | MLP [z_win20,c] | MLP [z_win20 stride10,c] | MLP [z_win20 stride30,c] | MLP [z_clean,c] | MLP [z_clean win20,c] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| battery (n=3) | 0.0766 | 0.0730 | 0.0726 | 0.0312 | 0.382 | 0.0324 | 0.0321 | 0.0299 | 0.0333 | 0.0303 | 0.0295 |
| bioreactor (n=3) | 0.0825 | 0.0717 | 0.0830 | 0.0288 | 0.110 | 0.0218 | 0.0221 | 0.0351 | 0.0379 | 0.0289 | 0.0206 |
| duffing (n=3) | 0.0537 | 0.0644 | 0.0276 | 0.0382 | 0.580 | 0.0357 | 0.0350 | 0.0214 | 0.0235 | 0.0144 | 0.0162 |
| lorenz (n=3) | 0.0319 | 0.0196 | 0.0299 | 0.0140 | 0.0765 | 0.0137 | 0.0119 | 0.0110 | 0.0118 | 0.0085 | 0.0051 |
| quad_tank (n=3) | 0.0724 | 0.0630 | 0.0689 | 0.0462 | 0.247 | 0.0460 | 0.0467 | 0.0482 | 0.0372 | 0.0452 | 0.0466 |
| vdp (n=3) | 0.0180 | 0.0200 | 0.0125 | 0.0152 | 0.163 | 0.0137 | 0.0124 | 0.0113 | 0.0125 | 0.0072 | 0.0067 |

### Exp 4 (C1-where): decoder Jacobian ||diag(1/sigma) dT+/dz||_2 along evaluation trajectories (sequence_ood, settled), seed mean; noise part = RMS of T+(z_obs,c) - T+(z_clean,c); linearised = RMS of J (z_obs - z_clean)

| system | method | ||J|| median | ||J|| q95 | ||J diag(z_std)|| median | RMS z noise | noise part (actual) | noise part (linearised J e_z) | total error | noise-free error |
|---|---|---|---|---|---|---|---|---|---|
| battery | lora | 149 | 236 | 6.22 | 0.0023 | 0.0234 | 0.0234 | 0.0740 | 0.0704 |
| battery | hyper | 157 | 306 | 6.63 | 0.0023 | 0.0317 | 0.0304 | 0.0773 | 0.0711 |
| battery | autonomous_base | 1259 | 1617 | 74.85 | 0.0023 | 0.228 | 0.227 | 1.504 | 1.484 |
| bioreactor | lora | 1 | 2 | 0.08 | 0.0017 | 0.0013 | 0.0013 | 0.0698 | 0.0697 |
| bioreactor | hyper | 7 | 24 | 0.55 | 0.0017 | 0.0087 | 0.0089 | 0.0515 | 0.0505 |
| bioreactor | autonomous_base | 44 | 70 | 4.45 | 0.0017 | 0.0367 | 0.0367 | 4.721 | 4.721 |
| duffing | lora | 445 | 603 | 8.27 | 0.00088 | 0.0482 | 0.0492 | 0.0571 | 0.0304 |
| duffing | hyper | 413 | 552 | 6.47 | 0.00087 | 0.0511 | 0.0516 | 0.0621 | 0.0343 |
| duffing | autonomous_base | 366 | 378 | 4.50 | 0.00087 | 0.0761 | 0.0756 | 0.303 | 0.293 |
| lorenz | lora | 5 | 11 | 3.52 | 0.026 | 0.0112 | 0.0112 | 0.0275 | 0.0251 |
| lorenz | hyper | 4 | 11 | 3.50 | 0.026 | 0.0113 | 0.0114 | 0.0186 | 0.0143 |
| lorenz | autonomous_base | 13 | 28 | 10.14 | 0.026 | 0.0159 | 0.0162 | 0.522 | 0.522 |
| quad_tank | lora | 14 | 17 | 11.68 | 0.027 | 0.0236 | 0.0237 | 0.0707 | 0.0665 |
| quad_tank | hyper | 13 | 15 | 11.17 | 0.027 | 0.0207 | 0.0209 | 0.0649 | 0.0615 |
| quad_tank | autonomous_base | 19 | 21 | 37.51 | 0.027 | 0.0690 | 0.0686 | 1.632 | 1.630 |
| vdp | lora | 22 | 35 | 2.27 | 0.005 | 0.0141 | 0.0143 | 0.0192 | 0.0131 |
| vdp | hyper | 21 | 29 | 1.79 | 0.0049 | 0.0203 | 0.0199 | 0.0250 | 0.0145 |
| vdp | autonomous_base | 21 | 29 | 1.77 | 0.005 | 0.0239 | 0.0244 | 0.111 | 0.109 |

### Exp 4 (C3-partial): decoder Jacobian ||diag(1/sigma) dT+/dz||_2 along evaluation trajectories (sequence_ood, settled), seed mean; noise part = RMS of T+(z_obs,c) - T+(z_clean,c); linearised = RMS of J (z_obs - z_clean)

| system | method | ||J|| median | ||J|| q95 | ||J diag(z_std)|| median | RMS z noise | noise part (actual) | noise part (linearised J e_z) | total error | noise-free error |
|---|---|---|---|---|---|---|---|---|---|
| battery | lora | 152 | 221 | 6.50 | 0.0023 | 0.0245 | 0.0245 | 0.0766 | 0.0726 |
| battery | hyper | 153 | 211 | 6.48 | 0.0023 | 0.0291 | 0.0277 | 0.0730 | 0.0666 |
| battery | autonomous_base | 1185 | 1548 | 80.11 | 0.0023 | 0.247 | 0.246 | 1.677 | 1.657 |
| bioreactor | lora | 5 | 10 | 0.22 | 0.0017 | 0.0089 | 0.0073 | 0.0825 | 0.0830 |
| bioreactor | hyper | 7 | 22 | 0.50 | 0.0017 | 0.0059 | 0.0061 | 0.0717 | 0.0714 |
| bioreactor | autonomous_base | 132 | 251 | 7.89 | 0.0017 | 0.0357 | 0.0357 | 4.563 | 4.563 |
| duffing | lora | 448 | 575 | 8.10 | 0.00088 | 0.0461 | 0.0463 | 0.0537 | 0.0276 |
| duffing | hyper | 408 | 548 | 6.12 | 0.00087 | 0.0506 | 0.0508 | 0.0644 | 0.0397 |
| duffing | autonomous_base | 370 | 382 | 4.45 | 0.00087 | 0.0746 | 0.0742 | 0.314 | 0.305 |
| lorenz | lora | 4 | 12 | 3.41 | 0.026 | 0.0108 | 0.0108 | 0.0319 | 0.0299 |
| lorenz | hyper | 4 | 10 | 3.22 | 0.026 | 0.0107 | 0.0107 | 0.0196 | 0.0163 |
| lorenz | autonomous_base | 19 | 37 | 15.53 | 0.026 | 0.0189 | 0.0190 | 0.642 | 0.642 |
| quad_tank | lora | 13 | 17 | 9.60 | 0.027 | 0.0217 | 0.0220 | 0.0724 | 0.0689 |
| quad_tank | hyper | 13 | 14 | 9.67 | 0.027 | 0.0181 | 0.0183 | 0.0630 | 0.0603 |
| quad_tank | autonomous_base | 20 | 21 | 37.77 | 0.027 | 0.0688 | 0.0685 | 1.610 | 1.609 |
| vdp | lora | 21 | 32 | 2.49 | 0.005 | 0.0129 | 0.0129 | 0.0180 | 0.0125 |
| vdp | hyper | 20 | 29 | 2.24 | 0.0049 | 0.0137 | 0.0136 | 0.0200 | 0.0144 |
| vdp | autonomous_base | 30 | 37 | 2.76 | 0.005 | 0.0309 | 0.0315 | 0.103 | 0.0981 |

### Exp 2b: C3-faults (exact model), noise10 / clean ratio and implied slope b = log(ratio)/log(10/3), seed mean (3 seeds)

| system | tier | latent | lora | hyper | nlox | ekf | ukf | pf | ekf_oracle |
|---|---|---|---|---|---|---|---|---|---|
| battery | id | x2.43 (b 0.74) | x2.28 (b 0.69) | x2.23 (b 0.66) | x1.07 (b 0.05) | x1.71 (b 0.45) | x1.88 (b 0.52) | x1.06 (b 0.05) | x1.60 (b 0.39) |
| battery | sequence_ood | x2.14 (b 0.63) | x1.43 (b 0.30) | x1.61 (b 0.40) | x1.02 (b 0.02) | x301.94 (b 4.74) | x1.93 (b 0.55) | x1.14 (b 0.11) | x1.61 (b 0.39) |
| bioreactor | id | x2.94 (b 0.90) | x1.10 (b 0.08) | x2.08 (b 0.61) | x1.21 (b 0.16) | x1.03 (b 0.03) | x1.05 (b 0.04) | x1.04 (b 0.03) | x1.00 (b 0.00) |
| bioreactor | sequence_ood | x2.37 (b 0.72) | x1.00 (b 0.00) | x1.13 (b 0.10) | x1.03 (b 0.02) | x1.03 (b 0.03) | x1.04 (b 0.04) | x1.04 (b 0.03) | x1.00 (b 0.00) |
| duffing | id | x3.28 (b 0.99) | x3.36 (b 1.01) | x3.14 (b 0.95) | x2.71 (b 0.83) | x2.31 (b 0.70) | x2.33 (b 0.70) | x2.08 (b 0.61) | x1.42 (b 0.29) |
| duffing | sequence_ood | x3.24 (b 0.98) | x2.94 (b 0.90) | x2.80 (b 0.86) | x1.84 (b 0.51) | x2.14 (b 0.63) | x2.15 (b 0.64) | x1.91 (b 0.54) | x1.40 (b 0.28) |
| lorenz | id | x1.73 (b 0.46) | x1.71 (b 0.45) | x2.12 (b 0.62) | x1.52 (b 0.35) | x2.95 (b 0.90) | x2.97 (b 0.90) | x4.45 (b 1.24) | x2.85 (b 0.87) |
| lorenz | sequence_ood | x1.82 (b 0.50) | x1.68 (b 0.43) | x2.27 (b 0.68) | x1.51 (b 0.34) | x3.05 (b 0.93) | x3.05 (b 0.93) | x4.65 (b 1.28) | x2.95 (b 0.90) |
| quad_tank | id | x1.68 (b 0.43) | x1.70 (b 0.44) | x1.68 (b 0.43) | x1.33 (b 0.23) | x1.20 (b 0.15) | x1.20 (b 0.15) | x1.24 (b 0.18) | x1.09 (b 0.07) |
| quad_tank | sequence_ood | x1.64 (b 0.41) | x1.47 (b 0.32) | x1.44 (b 0.30) | x1.17 (b 0.13) | x1.21 (b 0.16) | x1.21 (b 0.16) | x1.27 (b 0.20) | x1.10 (b 0.08) |
| vdp | id | x3.12 (b 0.94) | x2.86 (b 0.87) | x2.97 (b 0.90) | x2.65 (b 0.81) | x2.28 (b 0.68) | x2.28 (b 0.68) | x5.76 (b 1.45) | x1.73 (b 0.46) |
| vdp | sequence_ood | x3.03 (b 0.92) | x2.55 (b 0.78) | x2.78 (b 0.85) | x2.18 (b 0.65) | x2.24 (b 0.67) | x2.24 (b 0.67) | x13.79 (b 2.18) | x1.76 (b 0.47) |