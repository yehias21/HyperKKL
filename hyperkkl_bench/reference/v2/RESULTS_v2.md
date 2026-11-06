# bench_v2 RESULTS: all baselines on hyperkkl-bench 2.0.0rc1 (equal data, seeds 0-2)

Written 2026-11-05 (Dubai) after every pre-registered job had a result. Rules: PREREG.md (+ Addenda 1-3); design:
DESIGN.md (+ Addendum A). Numbers: results.json, results_extras.json; full tables: logs/tables.md (every cell, every
method, rank, seed sd, lr per seed), logs/extras.md, logs/summary_cells.md. Package copy and exact reproduction:
hyperkkl_bench/reference/v2 (`python reference/v2/reproduce_v2.py --check` -> ALL MATCH, 20 cells, 419 method means).

**What was run.** 935 GPU jobs (every job of PREREG section 6, items 1-5) + 76 CPU real-track jobs + 12 real-track
HyperKKL jobs. Every learned method used 3 seeds (0, 1, 2); filters are deterministic (seed 0). 3 seeds make these
reference results, not headline claims in the sense of PROTOCOL 5 (5 seeds). Every method was trained on the same
mixed episodes of its cell: train + train_event (1200), val + val_event (128). **Equal-data deviation (PREREG
Addendum 4a):** the HyperKKL family (hyperkkl, hk_base, auto, latent, concat, film) and the UDE / NODE filters also
used the 400 zero-input phase1 episodes; GRU, NCDE, NLOX-M and DKKL did not. Selection used validation splits only
(val + val_ood for HyperKKL, peers, conditioning variants and tuned filters; val alone inside the job for UDE / NODE /
GRU / NCDE, Addendum 4c). Every method was scored by the bench scorer: settled UNMEASURED NRMSE, primary block B
`test_op_ood/clean`, A `test_id/clean`. Real tracks use the track scorer (battery: settled SOC RMSE, % SOC;
mechanics: velocity NRMSE).

## Family B: non-autonomous (13 cells, realistic level: partial model, 30 % mismatch, coloured Student-t noise)

| cell | best (value) | hyperkkl (rank) | best grey-box (UDE) | best KKL peer | best filter | gru |
|---|---|---|---|---|---|---|
| battery | ude1_ekf_gated (0.00821) | 0.00839 (3/24) | ude1_ekf_gated 0.00821 | nlox_m 0.0846 | ukf 0.176 | 0.0258 |
| bioreactor | **hyperkkl** (0.00755) | 0.00755 (1/22) | ude32_ekf_gated 0.00781 | nlox_m 0.00755 | ekf_gated_tuned 7.1 | 0.0127 |
| cartpole_b | ude32_ekf_gated (0.0127) | 0.0134 (3/24) | ude32_ekf_gated 0.0127 | nlox_m 0.0211 | ekf_gated_tuned 0.407 | 0.0177 |
| duffing | ude32_ekf_gated (0.0132) | 0.0132 (5/22) | ude32_ekf_gated 0.0132 | nlox_m 0.0442 | ukf_gated 0.122 | 0.0172 |
| fhn | hk_base (0.0118) | 0.0121 (2/22) | ude32_ekf_gated 0.0132 | dkkl_m 0.0316 | ukf_gated_tuned 0.134 | 0.0151 |
| highway_v2 | node32_ekf_gated (0.027) | 0.0681 (6/22) | ude32_ukf_gated 0.0343 | nlox_m 0.124 | ekf 0.177 | 0.134 |
| lorenz | hk_base (0.00515) | 0.00516 (2/22) | ude32_ukf_gated 0.0197 | nlox_m 0.00606 | ekf_gated_tuned 1.78 | 0.0102 |
| lorenz96 | ude32_ekf_gated (0.0122) | 0.431 (12/23) | ude32_ekf_gated 0.0122 | dkkl_m 1.26 | ukf_gated_tuned 0.15 | 0.33 |
| network | ude1_ekf_gated (0.0136) | 0.0158 (7/22) | ude1_ekf_gated 0.0136 | dkkl_m 0.0375 | ukf_gated_tuned 0.234 | 0.0251 |
| quad_tank | ude32_ekf_gated (0.0169) | 0.0181 (6/22) | ude32_ekf_gated 0.0169 | nlox_m 0.174 | ukf_gated_tuned 5.47 | 0.0517 |
| rossler | hk_base (0.137) | 0.141 (2/22) | ude32_ukf_gated 0.164 | dkkl_m 0.198 | ekf_gated_tuned 1.31 | 0.18 |
| two_link_arm_v2 | gru (0.144) | 0.872 (7/22) | ude32_ukf_gated 1.54 | dkkl_m 0.366 | ukf 0.71 | 0.144 |
| vdp | hk_base (0.016) | 0.0165 (2/24) | ude1_ekf_gated 0.0174 | dkkl_inj_m 0.0217 | ukf_gated_tuned 0.472 | 0.0223 |

Best method per cell: corrected grey-box / NODE 7 of 13; HyperKKL (default or base) 5 of 13 (hyperkkl itself 1,
hk_base 4); GRU 1. HyperKKL default ranks in the top 3 in 7 of 13 cells (median rank 3, tied with ude32_ekf_gated).

**Paired vs hyperkkl** (per-seed log ratio, 95 % t-interval, Holm over the 13 cells; better / tie / worse = the
method against hyperkkl; geo-mean ratio method / hyperkkl):

| method | better | tie | worse | geo-mean ratio |
|---|---|---|---|---|
| ude32_ukf_gated (multi-step UDE + gated UKF) | 3 | 7 | 3 | 0.83 |
| ude32_ekf_gated | 3 | 8 | 2 | 0.90 |
| ude1_ekf_gated (corrected grey-box, ude5k1 style) | 2 | 10 | 1 | 0.92 |
| ude1_ukf_gated | 2 | 9 | 2 | 0.91 |
| node32_ekf_gated | 0 | 12 | 1 | 1.02 |
| hk_base (HyperKKL without C5) | 1 | 12 | 0 | 0.99 |
| gru | 2 | 2 | 9 | 1.33 |
| nlox_m (NLOX at matched budget) | 0 | 5 | 8 | 2.53 |
| film / dkkl_inj_m (3 cells) / dkkl_m | 0 / 0 / 1 | 0 | 13 / 3 / 12 | 3.04 / 3.56 / 4.84 |
| latent / ncde (4 cells) / concat | 0 | 0 | 13 / 4 / 13 | 8.0 / 8.8 / 16.1 |
| ekf_gated_tuned / ukf_gated_tuned | 1 / 1 | 1 / 0 | 11 / 12 | 18.2 / 18.9 |
| ekf / ukf / ekf_gated / ukf_gated | 1 / 2 / 0 / 0 | 0 | 12 / 11 / 13 / 13 | 22.8 / 21.9 / 28.2 / 27.7 |
| auto (autonomous KKL, phase-1 maps) | 0 | 0 | 13 | 39.9 |

**Reading.** On realistic operating profiles, the corrected grey-box (UDE + gated EKF/UKF, K = 1 or 32) and HyperKKL
are statistically tied in most cells (7-10 of 13 ties). Neither dominates: the grey-box wins battery, cartpole_b,
duffing, lorenz96, network and quad_tank, and NODE wins highway. HyperKKL wins bioreactor, fhn, lorenz, rossler and
vdp. Every published learned KKL peer (NLOX-M, DKKL, DKKL-inj) and every other conditioning variant is worse than
HyperKKL in most cells. Model-based filters on the partial model are an order of magnitude worse. C5 (hyperkkl vs
hk_base) is a tie in 12 of 13 cells.

## Family A: autonomous (7 cells, partial level: partial model, white noise)

| cell | best (value) | hyperkkl (rank) | best grey-box (UDE) | best KKL peer | best filter | gru |
|---|---|---|---|---|---|---|
| fhn | ude32_ekf_gated (0.00942) | 0.0128 (5/18) | ude32_ekf_gated 0.00942 | nlox_m 0.0163 | ekf_gated_tuned 0.308 | 0.016 |
| lorenz | ude32_ukf_gated (0.00795) | 0.00934 (3/18) | ude32_ukf_gated 0.00795 | nlox_m 0.00911 | ekf_gated_tuned 2.36 | 0.00981 |
| lorenz96 | ude32_ukf_gated (0.00395) | 0.263 (13/18) | ude32_ukf_gated 0.00395 | dkkl_m 0.992 | ukf_gated_tuned 0.0935 | 0.207 |
| lorenz96_n10 | ude32_ekf_gated (0.00461) | 0.0649 (5/18) | ude32_ekf_gated 0.00461 | nlox_m 0.821 | ukf_gated_tuned 0.0937 | 0.0799 |
| network | ude32_ekf_gated (0.009) | 0.0102 (5/18) | ude32_ekf_gated 0.009 | dkkl_m 0.0144 | ukf_gated_tuned 0.1 | 0.0133 |
| rossler | **hyperkkl** (0.0452) | 0.0452 (1/18) | ude32_ukf_gated 0.0513 | dkkl_m 0.0528 | ekf_gated_tuned 0.65 | 0.0528 |
| vdp | **hyperkkl** (0.00716) | 0.00716 (1/18) | ude32_ekf_gated 0.00794 | nlox_m 0.00911 | ukf_gated_tuned 0.311 | 0.0111 |

Paired vs hyperkkl (Holm over 7 cells): ude32_ukf_gated better 4 / tie 3 / worse 0 (geo-mean 0.36); ude32_ekf_gated
4 / 3 / 0 (0.40); ude1_ekf_gated 3 / 4 / 0 (0.45); hk_base 0 / 5 / 2 (1.05); nlox_m 0 / 4 / 3 (2.23); dkkl_m 0 / 2 / 5
(2.10); gru 1 / 0 / 6 (1.17); auto 0 / 0 / 7 (13.3); tuned filters 1 / 0 / 6 (11).
Best per cell: multi-step UDE 5 of 7, HyperKKL 2 of 7 (rossler, vdp); hyperkkl median rank 5.

**Autonomous HyperKKL vs autonomous KKL.** With u = 0 the input context is constant
(`tests/test_v2.py::test_autonomous_context_constant`), so input conditioning is inactive. HyperKKL is then an
autonomous KKL observer whose decoder also sees a strided window of past latents (a delay embedding; DESIGN Addendum B),
and it is compared as such. hyperkkl / nlox_m (end-to-end autonomous KKL at the same budget; its
injection is 0 at u = 0): fhn 0.80, lorenz 1.03, rossler 0.84, vdp 0.79 (inside [0.67, 1.5]); network 0.57,
lorenz96 0.14, lorenz96_n10 0.08 (outside, hyperkkl better). hyperkkl / dkkl_m 0.07-0.99; hyperkkl / auto 0.02-0.27.
The gains over the other autonomous KKL observers come from training (phase-2 fine-tuning, the z-window, C5, budget; and
phase1, which nlox_m / dkkl_m did not get), not from input conditioning, which is inactive. No conditioning benefit
is claimed in family A.

**Secondary family-A tiers** (logs/extras.md, section "Family A secondary tiers"). The test_ic_ood settled error equals
test_id for most methods: the median ratio over all cells and methods is 1.00 (range 0.92-2.6). After the transient,
the attractor no longer remembers the IC. On the transient window (all states), HyperKKL is worse than the multi-step
UDE + UKF and the GRU in all 7 cells, e.g. fhn 1.47 vs 0.11 and 0.41, lorenz96 0.41 vs 0.09 and 0.21.
test_param_shift (parameters a further exp(+-0.15) off after training) raises the error by a median 7.9x over cells
and methods (range 0.6-63x); for fhn, hyperkkl goes 0.013 -> 0.133. Caveat (PREREG Addendum 4b, PITFALLS P30): the methods marked * there used phase1, whose ICs cover
the IC shell, so their test_ic_ood numbers are not out-of-distribution.

## Real-data tracks (3 seeds; filters deterministic)

| track (headline) | hyperkkl_a3 | gru | ude32_ekf_gated | ude1_ekf_gated | ekf_gated_tuned | ekf_gated |
|---|---|---|---|---|---|---|
| battery_pf (% SOC) | 1.16 | **1.12** | 1.28 | 2.72 | 4.00 | 4.00 |
| battery_lg (% SOC) | 0.995 | **0.968** | 0.969 | 1.46 | 1.82 | 1.85 |
| emps (velocity NRMSE, outages) | 0.907 | 0.170 | **0.0116** | 0.0120 | 0.0132 | 0.0188 |
| silverbox (velocity NRMSE) | 0.0582 | **0.0203** | 0.144 | 0.123 | 0.200 | 0.201 |

HyperKKL A3 is close to the best on the two batteries, second on silverbox, and fails during the EMPS outages (it
never saw a frozen encoder in training, as in v1.1). On real data the HyperKKL trainer is A3 without C5 (the
real-data trainer has no C5), as stated. Peers (NLOX-M, DKKL) were not run on the real tracks: their training code
has no masked-label support (CUT, below).

## Pre-registered predictions (PREREG section 5) against the outcome

| prediction | outcome |
|---|---|
| B: corrected grey-box best in >= 5 of 13 cells | **held**: grey-box / NODE best in 7 of 13 (UDE 6, NODE 1) |
| B: hyperkkl top 3 in >= 6 of 13, best in 3-5 (likely bioreactor, lorenz, vdp, lorenz96, battery) | **partly**: top 3 in 7 of 13; best in 1 (bioreactor), 5 counting hk_base; lorenz96 and battery were wrong (rank 12 and 3) |
| B: plain ungated filters last or near last | held (ranks 17-22 of 22-24, only `auto` lower) |
| B: nlox_m beats dkkl_m in >= 10 of 13 | **failed**: 8 of 13 |
| B: hyperkkl beats nlox_m in >= 10 of 13 | held: 12 of 13 by seed mean with bioreactor equal to 3 digits (8 significant, 5 ties) |
| auto worse than every trained KKL variant | **mostly held**: 17 of 20 cells |
| op_ood / op_id close to 1 for model-based, larger for learned | **partly**: close to 1 for nearly every method (medians 0.98-1.23, ncde 2.0); the different-operator tier is NOT harder than op_id for most methods (limitation below) |
| Lorenz-96 (B): ude32 >= 10x worse than hyperkkl (earlier stopped runs: 42-46x) | **failed, reversed**: ude32 is 35x BETTER (ude32_ekf / hyperkkl 0.028 [0.027, 0.030] on op_ood; 0.027-0.045 on every tier). The 42-46x of the stopped runs does not hold on equal data |
| network: C5 costs ~one extra phase 2; accuracy within +-10 %, better under outliers | cost: C5 training wall time 20x (driven) / 18x (autonomous) a base phase 2, far more than predicted; accuracy: driven 1.07 [1.02, 1.11] (7 % worse), outliers 1.11 [0.64, 1.92] (tie); autonomous 0.81 [0.78, 0.84] clean and 0.48 [0.40, 0.58] under outliers (better) |
| A: hyperkkl / nlox_m in [0.67, 1.5] in >= 5 of 7 | **failed**: 4 of 7 (the 3 others favour hyperkkl: network, lorenz96, lorenz96_n10) |
| A: hyperkkl < auto | held in 7 of 7 |
| A: filters / UDE rank high on vdp / fhn; lorenz96 hard for every filter | UDE yes (fhn best), tuned filters no (rank >= 11); lorenz96 hard for filters (0.09) but easy for UDE (0.004) |

## Equal-data comparisons the paper needs

* **Lorenz-96, UDE vs HyperKKL (equal mixed data, 3 seeds)**: multi-step UDE + EKF 0.0122 vs HyperKKL default
  0.431 on op_ood (ratio 0.028 [0.027, 0.030]). Same direction in family A (0.0040 vs 0.263) and on the legacy tiers.
  The unmodelled term of L96 is a constant spatial forcing per state, which the residual learns almost exactly.
* **network C5 cost**: C5 multiplies the phase-2 training time by 18-20 (range 9.9-26.3 over 6 runs per family;
  `meta.hyperkkl.c5_extra_wall_s / base_wall_s`). Accuracy: -7 % on driven/network (significant), +19 % clean and
  +52 % under outliers on autonomous/network.
* **Drone (quad_planar) under outliers, equal mixed data**: compiled from experiments/equaldata_mix (5 seeds,
  not rerun in bench_v2; the drone needs the case-study sensor pipeline, which v2 does not rebuild). UDE + UKF 0.0805
  vs HyperKKL default 0.211 on id/outliers, ratio 0.40 [0.26, 0.61] (grey-box 2.5x better). Clean: 0.042 vs 0.064.

## Legacy tiers vs operating-profile tiers
Method rankings agree strongly between test_op_ood and test_legacy_sequence_ood (Kendall tau 0.80-0.98 per cell,
logs/extras.md). The realistic profiles change absolute numbers, but not who wins.

## Cuts, exclusions, deviations (nothing silent)
* Equal data: phase1 used by the HyperKKL family and UDE / NODE only (Addendum 4a); IC-shell overlap of phase1 in
  family A (4b); selection on val only for UDE / NODE / GRU / NCDE (4c); 3 seeds (4d).
* Seeds 3-4 (optional item 6) removed by author instruction before any started (PREREG Addendum 2): all results are 3 seeds.
* NCDE only on driven vdp, battery, cartpole_b, lorenz96; DKKL-inj only on driven vdp, battery, cartpole_b (cost;
  PREREG section 6 items 4-5, run in full).
* Family A: node32, ncde, latent, concat, film, dkkl_inj_m not run (with u = 0 the conditioning / injection variants
  reduce to an autonomous decoder; PREREG section 3).
* Family A level: partial only (white data); B level: realistic only. The other levels are built and shipped, not run.
* Real tracks: peers (NLOX-M, DKKL) and the conditioning variants not run (no masked-label training code); HyperKKL
  without C5 there.
* Deployment (quad_planar, cstr_exo) and scaling (fhnring) tracks: not rebuilt in v2 (v1.x files unchanged); the
  drone-outlier comparison is compiled from equaldata_mix.
* Autonomous rossler at +0.3 excluded (DESIGN.md Addendum A); autonomous lorenz nominal only.
* Harness fixes during the run (PREREG Addendum 3): numerically failed filter episodes scored as diverged; S-singular
  jitter / pseudo-inverse fallback in the filter copy (network UKF crash); cuFFT transient retries (5 jobs rerun, all
  passed). All 935 + 76 + 12 jobs have a result; no job is missing.

## Limitations
* The op_ood tier (different operator, same actuator and box) is not harder than op_id for nearly every method. It
  tests a different operator, not a harder one. Most v2 plants are driven far from their actuator limits, so the
  state envelope barely changes between operators.
* One frozen data set per cell: seeds vary the training only. Data-sampling variance is not in the intervals.
* 3 seeds: the t-intervals have 2 degrees of freedom, so many paired comparisons are ties by construction.
* Profiles were synthesised from the input review, not fitted to logs. Five plants' profile families are uncited
  (design/LITERATURE.md).
* Learned methods' tuning budgets differ (2 configurations for HyperKKL and peers, 4 for UDE / GRU); every comparator
  got at least HyperKKL's budget.
