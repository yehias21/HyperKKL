## Reading (auditor, 2026-10-26 04:30 CET; appended by verdict.py; everything below the verdicts is interpretation)

**Mechanical outcome.** Both nodes FAIL the pre-registered rule of `PREREG.md`: `C2-trunc-test` (S) 2/6, (T) 1/6;
`C2-trunc-dev` (S) 3/6, (T) 3/6 (need 4/6 each); TOST at 5% is 0/6 on both (3 seeds: underpowered at that margin
everywhere). All 36 jobs completed (02:03-04:08 CET, median 80 min per job at 32 concurrent jobs), no errors; every arm
of a job has identical trainable parameters (checked on all 36 results: conditioner 140.6-141.5k, context 6240); the
retrained phase 1 reproduces the pre-registered `l_dec_q95` (e.g. battery 806 vs 815, cartpole 80 vs 76, fhn 7.7 vs 8).

**What the cells say (descriptive).**

* Sufficiency, cell level (method x tier x arm with L* <= H <= settle): test 43/48 within +5% and 48/48 within +10%;
  dev 47/56 within +5%, 49/56 within +10%. The per-system rule ("every cell within +5%") fails on four test systems
  through single hyper cells at +6% to +9% with CIs of +-15 to +-40% (fhn hyper/id H256 +9% [-33, +52]; cartpole hyper/id
  H512 +7% [-9, +23]; highway_traffic hyper H2048 +5/+6%; rossler lora/seq H512 +7% [+2, +12]). On dev the misses are
  real: battery H512 +8..16% (knee at 1024) and lorenz H256 +28/+29% (knee at 512), plus bioreactor's non-monotone
  lora/seq curve (+49% at 256, +20% at 512, CIs crossing 0).
* With the design constant (L*_full, the theorem's own constant, pre-registered as secondary): test (S) 3/5 systems
  (33/36 cells within +5%), dev 3/4 (39/40 cells). Battery (L*_full 542: knee in (512, 1024]) and lorenz (412: knee in
  (256, 512]) are exactly the two dev systems where the primary L* (without the constant) fails and L*_full does not.
  Post hoc, at grid resolution: on 9/12 systems every one of the four knees lies at or below the first grid arm >= L*_full;
  the three exceptions are the single noisy cells above (fhn hyper/id, highway_traffic hyper, bioreactor lora/seq).
* Tightness: 13/24 test cells and 15/24 dev cells are significantly worse at the largest arm <= L*/4 (median point change
  +11% test, +22% dev). The plants with small l and small design constant (fhn, two_link_arm, vdp, duffing) are flat at
  L*/4 (knees 32-64 samples, i.e. 1-2 s): the sufficient bound is loose there by 4-8x. The plants with large l or large
  constant (battery, lorenz, quad_tank, highway_traffic, cartpole, rossler) are 20-200% worse at L*/4 and their knees sit
  within one grid step of L*_full.
* Current input only plus one sample (H = 1): +9% to +588% worse than the full context on the test systems (median +112%),
  +26% to +2548% on dev (median +204%). Proposition m2-current (the current input is not enough) is confirmed on six
  held-out plants.
* Monotonicity (no arm worse than a shorter one by > 10%): 20/24 test curves, 20/24 dev curves; the exceptions are fhn
  (all four, flat curves with +-5% noise) and bioreactor lora (as in C2-window-a1).

**Reading.** The sufficient half of Corollary m2-rule survives on the held-out systems in the sense a sufficient bound can:
with its own constants the learned observer's error has plateaued by L*_full to within the 5% precision on 33/36 test cells
(the rest within 10%), and the plateau is reached earlier where l and the design constant are small. The rule as
pre-registered (5% point estimate on every cell, without the design constant) is too strict for 3-seed curves whose
cell CIs are +-10-40% wide, and dropping the design constant costs exactly the two dev systems (battery, lorenz) where it
is largest in samples. Tightness (error at L*/4 significantly worse) holds on half the cells only: the bound is a bound,
not a prediction of the knee, on the plants with small l. None of this is a FAIL of the theorem; it is a FAIL of the
pre-registered precision, reported as such.

**Not done / caveats.** 3 seeds (TOST at 5% underpowered); the 5% criterion compares against a learned floor (effective
eta 2.5-10x below 0.05, AUDIT M3 ii); l is a proxy (`l_dec_q95`); no second bandwidth (the 1/alpha scaling is still
untested); highway_traffic's l_dec_q95 of 5e4 makes its L* dominated by ln l and should be read with care (its id-tier
curves are the shallowest of the set).
