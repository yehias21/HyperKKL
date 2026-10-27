# Pre-registration: C2-trunc (truncated-history SFM context), nodes C2-trunc-test and C2-trunc-dev

Written 2026-10-25 23:50 CET (2026-10-26 02:50 Dubai) BEFORE any training run of these nodes (the nodes are written
to `experiments/tree/` by `experiments/make_tree_memory.py` and queued only after the code change below has passed its
tests; no job of either node exists at the time of writing). Author: memory-audit agent. Follows `AUDIT.md`.

## Question

How much input history does the decoder of a learned KKL observer need? Corollary m2-rule (sufficient half of
Theorem m2-window(a)) says a window of
`L*(eta) = [ln(1/eta) + ln l] / alpha` seconds (plus the design constant `ln(kappa |B| ybar/alpha)/alpha`)
suffices for a worst-case error eta, with l the Lipschitz constant of the inverse of the windowed map.

## Design (fixes the four mismatches M1-M4 of AUDIT.md)

* **Fixed architecture, truncated history.** The SFM context (`phase2.context.kind: sfm`, 8 blocks, dim 16, hidden 64,
  the C1-where encoder) gets one new option `phase2.context.history: H`: its filter state `xi_k` is computed from the
  last H inputs only, `xi_k = sum_{l=1}^{min(H,k)} Ad^{l-1} Bd u_{k-l}/u_ref` (inputs older than H samples replaced by
  zero), identically in training (`seq`) and deployment (`step`). `H = null` (default) is the current SFM, bit-for-bit.
  The trainable parameter count is IDENTICAL for every H (no FIR bank grows with H: no capacity confound). H counts
  past inputs that have acted on the plant by t_k (u_k enters only through the direct path, Prop. m2-current(a)), so a
  context with history H covers exactly H dt seconds of acting history.
* **Sweep** H in {1, 8, 32, 64, 128, 256, 512, 1024, 2048, inf} samples (inf = plain `lora` / `hyper`), methods lora and
  hyper (20 phase-2 trainings per job), 3 seeds (0, 1, 2), same frozen phase-1 base maps for all arms of a job
  (phase 1 is retrained once per (system, seed) because the module hash changes; every arm of a job shares it).
* **Systems.** `C2-trunc-test`: the six HELD-OUT systems fhn, rossler, cartpole, two_link_arm, highway_traffic, network
  (never used for any C2 decision; the decision is taken here). `C2-trunc-dev`: the six dev systems, reported separately
  as in-sample.
* **Evaluation.** tiers id and sequence_ood (clean condition), 64 paired episodes per tier, settled window k >= 0.5 N;
  paired relative change of every arm against the inf arm of the same method and seed (`metrics.paired_relative`,
  Student-t over seeds, 95% CI; 90% CI for TOST).

## Pre-registered constants

eta = 0.05. l_i = median over the EXISTING phase-1 runs' `l_dec_q95` (TEST-C1 for the test systems, C1-where for the dev
systems; the new phase 1 is a re-training of the same recipe and its `l_dec_q95` will be recorded next to these as a
check, but the L*_i below are the ones the rule uses). dt_i from the design (`0.1 sin(pi/2 n_z)/alpha`; quad_tank fixed
0.05). Primary L*_i = (ln(1/eta) + ln l_i)/(alpha_i dt_i) in samples (the task's rule, l restored, design constant
omitted). Secondary L*_full,i adds ln(|B| ybar/alpha_i) (the constant the theorem carries; AUDIT T2). settle_i = floor(0.5 N_i):
an arm H is informative only if H <= settle_i (otherwise the truncation is inactive on part of the scored window).

| system | role | n_z | alpha | dt | alpha dt | l_i | L*_i [samples] | L*_full,i | L*_i/4 | settle_i | grid arms >= L*_i and <= settle_i | largest grid arm <= L*_i/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fhn | test | 5 | 0.5 | 0.0618 | 0.0309 | 8 | 163 | 241 | 41 | 809 | 256, 512 | 32 |
| rossler | test | 7 | 1.0 | 0.0223 | 0.0223 | 138 | 356 | 540 | 89 | 3595 | 512, 1024, 2048 | 64 |
| cartpole | test | 6 | 1.5 | 0.0173 | 0.0259 | 76 | 283 | 539 | 71 | 869 | 512 | 64 |
| two_link_arm | test | 6 | 2.0 | 0.0129 | 0.0259 | 224 | 325 | 485 | 81 | 3864 | 512, 1024, 2048 | 64 |
| highway_traffic | test | 12 | 0.5 | 0.0261 | 0.0131 | 49719 | 1058 | 1115 | 264 | 3831 | 2048 | 256 |
| network | test | 12 | 0.3 | 0.0435 | 0.0131 | 40 | 512 | 849 | 128 | 1149 | 512, 1024 | 128 |
| battery | dev | 7 | 1.5 | 0.0148 | 0.0223 | 815 | 436 | 542 | 109 | 1011 | 512 | 64 |
| bioreactor | dev | 5 | 0.5 | 0.0618 | 0.0309 | 43 | 219 | 289 | 55 | 809 | 256, 512 | 32 |
| duffing | dev | 5 | 3.0 | 0.0103 | 0.0309 | 299 | 281 | 276 | 70 | 2427 | 512, 1024, 2048 | 64 |
| lorenz | dev | 7 | 2.0 | 0.0111 | 0.0223 | 9 | 234 | 412 | 58 | 2247 | 256, 512, 1024, 2048 | 32 |
| quad_tank | dev | 6 | 0.1 | 0.0500 | 0.0050 | 14 | 1123 | 2483 | 281 | 3000 | 2048 | 256 |
| vdp | dev | 5 | 1.5 | 0.0206 | 0.0309 | 23 | 199 | 249 | 50 | 1214 | 256, 512, 1024 | 32 |

(l_i per seed: fhn 8/8/7; rossler 149/122/138; cartpole 76/93/71; two_link_arm 236/224/218; highway_traffic
50818/49719/48174; network 37/42/40; battery 847/756/815/816/843; bioreactor 43/46/43/45/42; duffing 299/302/284/316/294;
lorenz 12/9/9/12/9; quad_tank 14/14/13/14/13; vdp 24/23/23/27/23. highway_traffic's l is 3 orders of magnitude above the
others: its L*_i is dominated by ln l and its secondary L*_full adds little.)

## Predictions and decision rule (mechanical; `analysis/evidence.py: analyse_trunc`)

Per system i, per method m in {lora, hyper}, per tier in {id, sequence_ood}, e(H) = mean settled NRMSE of arm H,
e(inf) = the method's untruncated arm, rel(H) = paired relative change (e(H) - e(inf))/e(inf) over seeds.

* **(S) Sufficiency (the proven half).** For EVERY grid arm H with L*_i <= H <= settle_i: point estimate
  rel(H) <= +0.05, for both methods and both tiers. Secondary: TOST non-inferiority at margin 5% (90% CI upper bound
  < +0.05), reported with the underpowered flag (3 seeds).
* **(T) Tightness (empirical; the theorem does not prove it, Theorem (b) being linear-only).** At the LARGEST grid arm H
  <= L*_i/4: rel(H) significantly > 0 (paired 95% CI lower bound > 0), for both methods and both tiers. All smaller
  arms are reported.
* **Verdict.** `C2-trunc-test` SUPPORTS the refined rule if (S) holds on >= 4/6 test systems AND (T) holds on >= 4/6 test
  systems. (S) alone on >= 4/6 is "sufficient bound supported, tightness not"; neither is FAIL. Systems with no grid arm
  in [L*_i, settle_i] are "not testable" for (S) and count as missing (the need stays 4 of 6). `C2-trunc-dev` applies the
  same rule and is reported as in-sample.
* **Also reported, not part of the verdict:** the same two counts with L*_full,i; the knee (smallest H with rel(H') <=
  +0.05 for all H' >= H, log-interpolated between neighbours, as a descriptive); H = 1 vs inf (current input plus one
  sample; Prop. m2-current); monotonicity (no arm worse than a shorter arm by > 10%); and the new phase-1 l_dec_q95.

## What would falsify what

* (S) failing on a system means the learned observer still improves past the sufficient window: either l_dec_q95
  underestimates l (the proxy is a stationary-decoder quotient), or the decoder's own floor moves with H for a reason
  outside the truncation bound (optimisation). The design-constant variant L*_full discriminates the first case.
* (T) failing means the bound is loose on that plant (as duffing's knee at 32 samples in C2-window-a1 suggests); this
  does not contradict the theorem.

## Cost estimate (before queuing)

Per job (one system, one seed): phase 1 ~130 s; 20 decoder-side phase-2 trainings at ~50-100 s (SFM lora/hyper on a
loaded GPU: 46-75 s measured; the long-episode plants rossler, two_link_arm, highway_traffic up to ~2x); evaluation 20
observers x 2 tiers x 64 episodes at ~7-50 s each. ~35 min (small plants) to ~60 min (long-episode plants) per job;
36 jobs (12 systems x 3 seeds) ~ 25 GPU-job-hours; on GPUs 5-7 with 4 slots each (12 concurrent jobs) ~2.5-3.5 h
wall. Test node queued first (priority 18), dev node second (17). Memory probe off (the truncation is verified by unit
tests, not by the probe).

## Implementation (applied only when the job queue is empty; hashed module `nets.py` only)

* `SFMEncoder(..., history=None)`: `filter_seq` multiplies the impulse response by `1[l < history]` before the FFT
  convolution (so lags >= history+1 of u relative to k are zero); `init_state`/`step` keep a buffer of the last
  `history` normalised inputs and form `xi_k` by the exact finite sum with the truncated impulse response (no recursion
  error); `history=None` leaves every existing code path untouched (bit-for-bit).
* `build_context` passes `ccfg.get("history")`; `method_cfg('lora@sfm:history=64')` sets it per method (existing
  mechanism; the phase-2 cache key includes it).
* Tests `tests/test_memory_history.py`: truncation exact (equals the full SFM on the input with samples older than H
  zeroed, per k); seq == step (float64 1e-12, float32 1e-4); history None and history >= N both bit-identical to the
  plain SFM; parameter count independent of H; Jacobian support exactly the last H+1 samples; method string parsing.
