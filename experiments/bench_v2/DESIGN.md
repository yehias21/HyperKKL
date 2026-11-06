# hyperkkl-bench v2 (2.0.0rc1): design

Written 2026-11-03 (Dubai), before any v2 data or result existed. Changes after this date go to the end as dated
addenda. Scope: state estimation for AUTONOMOUS (family A) and NON-AUTONOMOUS (family B) nonlinear systems.
Open-loop inputs only.

## 1. Two families

| | A. autonomous | B. non-autonomous (input-driven) |
|---|---|---|
| question | reconstruct the unmeasured states of a system with no exogenous input | the same, while a known input drives the system, under realistic operating profiles |
| input | u = 0 always (the plant's own actuator law at rest stays in f) | the plant's physical operating profiles (INPUT_REVIEW.md, `code/profiles_v2.py`) through the actuator model |
| plants | vdp, fhn, rossler, lorenz (nominal data only), lorenz96 (n 20), network, lorenz96_n10 (size variant) | battery, bioreactor, duffing, lorenz, quad_tank, vdp, cartpole_b, fhn, highway_v2, network, rossler, two_link_arm_v2, lorenz96 (core); quad_planar, cstr_exo (deployment); fhnring_n4..64 (scaling); real tracks |
| test tiers | `id` (IC box core), `ic_ood` (outer shell of the IC box, held out from training), `param_shift` (plant parameters drifted by a further +-15 % after training) | `op_id` (training operators, new seeds), `op_ood` (different operator / route / schedule, same actuator and box); `legacy_id`, `legacy_shape_ood`, `legacy_sequence_ood` (the v1 tiers) |
| conditions | sensor: clean, noise10, outliers, dropout, burst_outage, poor_init | sensor: clean, outliers, burst_outage, poor_init; input record: input_realistic, input_cmd_vs_applied, input_quant |
| headline | settled unmeasured NRMSE, `id`, clean; `ic_ood` is scored on the TRANSIENT window too (the IC only matters before the attractor is reached) | settled unmeasured NRMSE, `op_ood`, clean (`op_id` next to it) |

Not input-OOD in family A: there is no input. Tiers there are initial-condition OOD, parameter shift and sensor
conditions only.

## 2. Inclusion rule "non-trivial unforced dynamics" (family A)

New check (`check-plant --autonomous`, here `code/scan_autonomy.py`): simulate the plant at u = 0 from 64 ICs of
its box for 5 T. In the last 20 % of the run, the median over ICs of the time-std of the most active state must be at
least 5 % of its reference std (`frac_active` >= 0.9). It must also not be decaying (late / early ratio >= 0.5).
Both must hold for the nominal plant and, for the mismatch / realistic cells, for the +0.3 perturbed plant.
Results (`design/autonomy_scan/*.json`, 2026-11-03):

| plant | nominal | +0.3 | family A? |
|---|---|---|---|
| vdp, fhn, rossler, lorenz96, lorenz96_n10, network | oscillating / chaotic (late std 1.5-3.1 x_std, ratio ~1) | same | yes, every knowledge level |
| lorenz | chaotic (1.71) | collapses to a fixed point (0.0) | yes, at the nominal-data levels (exact, partial) only |
| duffing | decays to an equilibrium (0.0) | 0.0 | no: B only (damped double well) |
| battery, bioreactor, cartpole_b, highway_traffic | equilibrium (0.0) | 0.0 | no: B only (forced plants, rest at u = 0) |
| quad_tank | equilibrium (0.0) | not run | no: B only |
| two_link_arm_b | equilibrium (late std 0.0008) | 64 % of ICs still swinging at 5 T (0.059) | no: B only (fails at nominal) |

A forced plant that comes to rest at u = 0 would give the trivial task "estimate an equilibrium". It is excluded by
this rule, not by choice.

**HyperKKL with no input.** With u = 0 the input context c(u_0..u_k) of the hypernetwork is a fixed vector c0, so the
generated decoder T+(z; c0) is ONE static map of z: an autonomous KKL observer with a (fixed) LoRA-modified base
decoder. The default (A3) also feeds a strided z-window to the hypernetwork. That is a delay embedding of the
latent, not input conditioning. So family A measures HyperKKL as an autonomous KKL observer. No conditioning benefit
can exist there, and none is claimed. Verified in code: `tests/test_v2.py::test_autonomous_context_constant` checks
that the context is identical for every episode and time.

## 3. Family B inputs: realistic operating profiles

`code/profiles_v2.py` = experiments/input_review/profiles.py (119 tests) with these changes:
* two_link_arm_v2: torque box +-12 N m (was +-1 N m; the shoulder gravity torque of the 1 m / 1 kg links is
  14.7 N m, so +-1 N m could not move the arm). Rate limit 120 N m/s, lag 5 ms. Same plant as two_link_arm_b
  otherwise, actuator u_bar = 24 N m. For scale: a Franka Panda shoulder is rated 87 N m, a UR5 shoulder 150 N m,
  both for heavier links (literature check). The legacy +-1 N m tiers are regenerated on the v2 plant as `legacy_*`.
* highway_v2: ramp-inflow box 0.05-0.5 veh/s (180-1800 veh/h; u in [-0.25, 0.2] around 0.3 veh/s), u_scale 0.25,
  u_bar 0.5 (the legacy box +-0.15 clipped 23-36 % of realistic minute counts). T stays 200 s (cut: the review's
  optional T >= 1800 s is not done; stated).
* Removed: the A-stat acceleration range 0.004-0.012 1/h^2 attributed to Paalme et al. (not verifiable). The A-stat
  profile now ramps between two levels of the box over 10-40 h (benchmark choice, uncited).
* Uncited (flagged in the card, no paper attached): stepped-sine / swept-sine / band-random shaker tests (vdp,
  duffing) and minimum-jerk moves (cart repositioning, pick-and-place). They were to rest on Kerschen et al. 2006
  and Flash & Hogan 1985, which were not read.
* US06 is 600 s and the UDDS trace 1369 s (EPA). The stiction range "0.5-5 %" (Choudhury et al.) is not used as a
  cited figure; cstr_exo keeps a 0.5 % deadband as a benchmark choice.

OOD = different operator on the same device, same actuator, same box (checked by the profile tests).

## 4. Training data recipes (both families)

| recipe | family A | family B |
|---|---|---|
| standard (`train`, 600; `val`, 64) | u = 0, ICs Sobol in the CORE of the IC box (centre +- 0.6 half-width) | operating-profile episodes of the TRAINING operators |
| event-structured (`train_event`, 600; `val_event`, 64) | u = 0 with impulsive state kicks: segments log-U[3, 12] t_ref; at each segment start, with probability 0.45, the state receives a kick of size U[0.2, 1] x_std in a random direction (clipped to the envelope) | eventmix D1: concatenated segments of the identification families (p 0.55) and rest / pulse / ramp events (p 0.45), same box |
| **mixed (default)** | standard + event (1200 / 128) | standard + event (1200 / 128) |
| also shipped | `phase1` (400 zero-input episodes over the envelope), `val_ood` (64; A: a second held-out IC region; B: the v1 val-OOD generators) | |

Every method in a comparison is trained on the same mixed episodes (PROTOCOL section 2). One frozen data set per
cell: seeds vary the training (initialisation, batches), not the data.

## 5. Knowledge levels and cells

Same four levels as v1 (exact, partial, mismatch, realistic). This study runs the **realistic** level for family B
(partial model, 30 % mismatch, coloured Student-t noise) and the **partial** level for family A (partial model, white
noise; lorenz has no realistic autonomous cell). Model-free methods do not depend on the level within one data variant.

## 6. API (planned for the port)

`hkb.load("autonomous/vdp/partial", "train", recipe="mixed")`, `hkb.load("driven/battery/realistic", "test_op_ood")`;
the v1 call `hkb.load("vdp", "realistic", "test_id")` keeps working and means `driven/vdp/realistic` with the
`legacy_id` tier.

## Addendum A (2026-11-03 17:00 Dubai, before any family-A result was read)
Autonomous rossler at the +0.3 data levels (mismatch, realistic) is excluded like lorenz. From the outer IC shell
(val_ood 0.6-0.8, test_ic_ood 0.8-1.0) about half of the episodes leave 1.5x the envelope even after 6 redraws, and
the research generator redraws the IC uniformly over the whole box. 20 of 56 test_ic_ood episodes would then sit
inside the training core, and test_param_shift does not exist (the +0.45 plant escapes from every IC). Family A
rossler is offered at exact / partial only. The two cells were moved to data_excluded/ and are never released.
Every other family-A cell has 0 shell violations in val_ood, test_ic_ood and hidden_ic_ood (checked; tests/test_v2.py).

## Addendum B (2026-11-05, review round)
* Section 2, "HyperKKL with no input": the precise statement is that input conditioning is inactive (constant
  context). The A3 decoder still sees a strided window of past latents (a delay embedding), so in family A HyperKKL
  is an autonomous KKL observer with a delay-embedded decoder, not the plain map z -> x of a standard KKL. C5's
  slope term also acts (it perturbs the context directly). Gains in family A come from training and that window.
* The inclusion check is exposed as `hyperkkl-bench check-autonomous [--perturb 0,0.3]` (not `check-plant --autonomous`).
* The shoulder gravity torque of the 1 m / 1 kg arm reaches 19.6 N m (arm stretched horizontally); 14.7 N m is the
  link-1 term only. The +-12 N m box cannot hold the arm horizontal statically; it drives the arm around the hanging pose.
* Section 4, family A: test_ic_ood is disjoint from train / train_event / val, not from phase1 (PREREG Addendum 4b).
