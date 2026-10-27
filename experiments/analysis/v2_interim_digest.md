# Protocol-v2 interim results digest (2026-10-25, ~22:30 Abu Dhabi; ~70% of the v2 tree done)

Source: experiments/tree/<node>/runs/<system>/seed<k>/results.json (protocol v2). Reports:
experiments/tree/<node>/report.md. Numbers: NRMSE over the settled window, normalised per state;
paired per-seed relative change, Student-t 95% CI over seeds, `*` = CI excludes 0.
Dev systems: battery, bioreactor, duffing, lorenz, quad_tank, vdp. Test systems (frozen-recipe
pass, not yet run): fhn, rossler, cartpole, two_link_arm, highway_traffic, network.

## Protocol v2 (what OOD means)
All inputs stay in one amplitude box [-1, 1] x u_scale in every tier; time parameters are in
per-plant units t_ref. Tiers: id (training families/ranges, new seeds); shape_ood (waveform
families never trained on: chirp, filtered noise, triangle, sawtooth; same box and band);
sequence_ood (pedal-like 'drive' events on the same 9-level grid: training = smooth driver,
|dlevel| 0.36, dwell 3.98 t_ref; test = aggressive stop-and-go, |dlevel| 1.33, dwell 1.25 t_ref).
State coverage: 0% of test states outside the training-state box on every dev system and tier.

## C1 - where the input enters (C1-where, 5 seeds, 140k matched conditioner, SFM context)
Paired change vs latent injection:

| tier | concat | film | lora | hyper |
|---|---|---|---|---|
| id | worse on 3/6 | better 4/6 | better 6/6 (-10..-68%) | better 6/6 (-13..-67%) |
| shape_ood | worse 3/6 | better 4/6 | better 5/6 | better 5/6 (-24..-69%), vdp n.s. |
| sequence_ood | worse 3/6 (up to +230%) | mixed | better 4/6, worse bioreactor (+50%) | better 4/6 (-19..-62%), vdp n.s., worse bioreactor (+25%) |

Best learned per system on sequence_ood: battery hyper, bioreactor latent, duffing lora,
lorenz hyper, quad_tank hyper, vdp lora. hyper vs lora (sequence_ood): hyper better on
lorenz (-31%*), bioreactor (-16%*), quad_tank (-9%*); lora better on vdp (+34%* for hyper).

## C2 - memory (C2-floor-*, 3 seeds; forgetting floor on the SFM context, rate >= c*alpha)
Paired change vs no floor (C1-where seeds 0-2), lora and hyper, id and sequence_ood:
- c = 0.25: no significant loss on any system.
- c = 0.5: no significant loss on 5/6; battery +23..+36% (lora id significant).
- c = 1: battery about 2x worse; others mostly within +-15%.
Memory probe (keep last L inputs, swap older history, measure |dxhat|): without a floor the
learned SFM forgets at its slowest learned pole (0.14-0.2 x alpha on vdp/duffing/battery), i.e.
keeps ~5-7x more memory than Lemma 1 requires; with a floor the probe rate is ~0.85-0.9 x the
floor (vdp: floor 0.75/s -> probe 0.63/s; floor 1.5/s -> 1.36/s).
Battery = integrator state (SOC, coulomb counting): the Lemma-1' worst case.
Window sweep (C2-window-a1): pending analysis.

## C3 - when learned beats filters (best of latent/lora/hyper vs best of 8 model-based filters:
ekf, ukf, pf, kkl_mb, ekf_huber, ukf_huber, ekf_oracle, ukf_oracle; oracle Q/R tuned per job on
validation). sequence_ood, clean unless stated:

| node | learned wins (CI) | filters win |
|---|---|---|
| C3-faults: exact model, 13 sensor/input/plant conditions + burst outage | 0/6 on every condition except input_bias 2/5, input_delay 2/5 | 5-6/6 |
| C3-noise-ct: exact model, AR(1) Student-t sensor noise in train+deploy | 0/6 | 6/6 |
| C3-mismatch10: real plant parameters 10% off | 4/6 (-25..-91%) | duffing; battery tie |
| C3-mismatch30 (partial) | 5/6 (-51..-95%) | duffing |
| C3-partial: one physical term missing from the model | 5/6 (-37..-96%) | quad_tank (model error 15%) |
| C3-realistic: partial + 30% + colored Student-t | 6/6 (-35..-97%) | none |

vs NLOX (learned neural Luenberger baseline): ours better on 5/6 under partial knowledge;
NLOX better on bioreactor. Model-free phase 1 under partial knowledge: bioreactor 8x worse,
vdp +20%, others neutral -> a partial model in the PDE term helps.
Relative model error median |f - f_known|/|f|: vdp 0.65, duffing 0.13, bioreactor 1.69,
lorenz 0.73, quad_tank 0.15, battery 0.22.

## Pending
C3-mismatch20/50, C3-data50/150 (learning curve), C1 budget/base-width controls, C2 window
analysis, frozen test-system pass, C4 testbed export.
