# quad_tank: persistent excitation or observability? Stage-1 diagnostics

Written 2026-10-26, about 11:30 CET. CPU only (CUDA_VISIBLE_DEVICES="", 8 threads). No model was
trained. Inputs: the C3 training / validation / evaluation episodes, read read-only from the live
cache; the stored C3 per-episode results (`experiments/tree/C3-*/runs/quad_tank/seed*/results.json`);
the beat_greybox campaign's arm A2 (SOAP + unfrozen base), seed 0. Scripts and raw output in the
scratchpad: `quadtank/diag_s1.py` -> `s1.json`, `quadtank/diag_s1b.py` -> `s1b.json`
(`/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/quadtank/`).
Copies of both JSON files are next to this file (`s1.json`, `s1b.json`).

Windows used throughout: training = the labelled samples (k >= burn = 1735, t >= 86.7 s);
evaluation = the scored samples (t >= 150 s, settle = 0.5 N). All "x_std" units are the stored
`scales.x_std` = (0.47, 0.67, 0.29, 0.45) cm, the NRMSE normaliser.

## Verdict

**Observability-limited, not PE-limited (Stage 1). The training set covers and excites what the
evaluation needs; what it cannot deliver is information about h3/h4 that the measured levels
carry only weakly and that the latent stores in near-degenerate directions.** One soft PE-leaning
signal (the slow-band share of the inputs is lower than on vdp/battery relative to each plant's
own slow corner) is real but is not where the error comes from. Stage 2 (PREREG.md) tests this
directly by changing only the training data.

| question | finding | points to |
|---|---|---|
| (a) do h3, h4 vary less in training than in evaluation? | Within an episode yes, in cm: training 0.15 / 0.15 cm vs h1, h2 0.33 / 0.31 cm. But the pooled training spread of h3/h4 (0.29 / 0.44 cm = 1.0 x_std) is >= every evaluation tier (0.22-0.30 / 0.35-0.48), the evaluation ranges lie inside the training ranges for all four states, and only 1.8% of scored sequence_ood states (0.6% for the (h3, h4) pair) lie beyond the 99th percentile of held-out validation nearest-neighbour distances (1% by construction). The one gap: sequence_ood moves the upper tanks MORE within an episode (0.20 / 0.30 cm, 1.3x / 1.9x training) | coverage fine; dynamic range of seq_ood slightly richer |
| (b) slow-band energy of the inputs | 13.6% of the training input's fluctuation energy lies below the slow corner 1/49 s, 62% below alpha = 0.1 rad/s; the primary tier sequence_ood has LESS (2.7% / 32%). Relative to each plant's own slow corner, vdp's training inputs carry 24.5% and battery's 43%. A 300-s episode resolves 0.0209 rad/s, i.e. the corner itself: below it an episode holds only its mean and trend | the soft PE-leaning signal; but evaluation is not slower than training |
| (c) empirical observability Gramian | Over the latent's memory (1/alpha = 10 s), the Cramer-Rao std of (h1, h2, h3, h4) is (0.004, 0.005, **0.080, 0.045**) x_std, condition number 1.5e3, weakest direction 0.95 h3 + 0.31 h4 (vdp: 0.011 / 0.036, cond 47). It takes ~30 s (3/alpha) to reach (0.017, 0.013) and ~1 slow time constant (49 s) to reach 0.009 | **h3 is weakly observable over the window the latent remembers** |
| (d) persistent excitation of the regressors | input Hankel (15 lags spanning 3 slow time constants, episode mean removed): lambda_min / lambda_max 0.088 (vdp 0.052, battery 0.0073). [z, c] covariance: cond 3.4e7 (vdp 7.9e6, battery 1.3e7), not unusual. Only 7.8% / 0.4% of the training h3 / h4 variance is independent of (h1, h2, input history), but the evaluation tiers need even less (0.7-1.0% / 0.2-0.4%) | PE adequate |
| errors | HyperKKL's error is on the upper tanks: C3-partial sequence_ood per-state NRMSE (h1..h4) lora 0.020 / 0.014 / **0.125 / 0.065** vs grey-box EKF 0.0057 / 0.0056 / 0.022 / 0.015. The per-episode error of A2 is uncorrelated with how far the episode's states are from the training data (Spearman |rho| <= 0.13 on every tier and scenario; A2 / grey-box ratio |rho| <= 0.12) | error not driven by coverage |

## (a) Spread of the states: training vs evaluation (seed 0; the evaluation episodes do not depend on the seed)

C3-partial (true plant = nominal). Pooled std over the window / mean within-episode std (cm):

| set | h1 | h2 | h3 | h4 |
|---|---|---|---|---|
| train (t >= 86.7 s) | 0.47 / 0.33 | 0.69 / 0.31 | 0.29 / 0.15 | 0.44 / 0.15 |
| id (t >= 150 s) | 0.27 / 0.19 | 0.53 / 0.17 | 0.30 / 0.11 | 0.48 / 0.15 |
| shape_ood | 0.26 / 0.20 | 0.47 / 0.14 | 0.22 / 0.10 | 0.36 / 0.16 |
| sequence_ood | 0.32 / 0.29 | 0.40 / 0.17 | 0.23 / 0.20 | 0.35 / 0.30 |

1-99% ranges (cm), train vs sequence_ood: h1 9.77-12.11 vs 10.65-12.09; h2 9.45-12.82 vs
10.06-11.93; h3 4.63-5.89 vs 4.84-5.77; h4 2.42-4.31 vs 2.64-4.10. Every evaluation range is
inside the training range. (The IC box is not a useful yardstick here: after the washout h3 sits
at 4.6-5.9 cm, entirely above the IC box [0.5, 4]; every state has left its IC box region by
the labelled window.)

C3-mismatch30 (perturbed true plant): the same picture; the perturbed h3 barely moves (pooled std
0.09 cm in training and 0.08-0.10 in evaluation, within-episode 0.15 / 0.31 x_std train / seq).

Nearest-neighbour coverage (standardised x, scored evaluation states vs labelled training states,
stride 20): fraction beyond the 99th percentile of held-out VALIDATION distances (1% expected
in-distribution):

| | id | shape_ood | sequence_ood |
|---|---|---|---|
| C3-partial, all 4 states | 3.6% | 1.1% | 1.8% |
| C3-partial, (h3, h4) | 4.8% | 0.2% | 0.6% |
| C3-mismatch30, all 4 | 2.9% | 1.3% | 1.8% |
| C3-mismatch30, (h3, h4) | 4.0% | 0.7% | 4.2% |

**Answer:** within an episode the upper tanks move about half as much as the lower ones (in cm;
0.34-0.51 vs 0.47-0.71 x_std), in training and evaluation alike; the training set does not vary
h3/h4 less than the evaluation sets do, and it covers the states they visit.

## (b) Spectral content of the inputs

Slow corner w_c = 1/49 s = 0.0204 rad/s (config's median slow time constant; the linearisation at
the nominal equilibrium has time constants 19, 41, 61, 84 s). Latent bandwidth alpha = 0.1 rad/s.
Zero-padded periodogram per episode, pooled; "fluctuation" = per-episode mean removed.

| set | DC share of energy | fluct. energy < w_c | fluct. energy < alpha | var of u low-passed at 49 s / fluct. var |
|---|---|---|---|---|
| train | 48% | 13.6% | 62% | 0.14 |
| id | 53% | 12.8% | 71% | 0.17 |
| shape_ood | 41% | 2.7% | 56% | 0.08 |
| sequence_ood | 4% | 2.7% | 32% | 0.07 |

Per training family (fluct. energy < w_c / < alpha): ramp 61% / 94%, step 49 / 92, drive 45 / 94,
staircase 29 / 88, prbs 3.5 / 60, multisine 2.7 / 46, square 0.5 / 45, sinusoid 0.4 / 46,
am 0.1 / 1.1 (constant and zero have no fluctuation).

References at their own corners (vdp 1/3 s, battery 1/1 min): fluct. energy below the corner
24.5% (vdp), 43% (battery); low-passed variance ratio 0.26, 0.44.

**Answer:** relative to its slow corner, quad_tank's training inputs carry about half vdp's and a
third of battery's slow-band share; relative to the latent bandwidth they carry the same as both
(62-65%). The evaluation tiers, and above all the primary sequence_ood, are FASTER than training,
not slower. A 300-s episode cannot hold a sub-corner oscillation (its frequency resolution is
0.0209 rad/s = the corner); "slow-band content" inside one episode is its level and trend.

## (c) Empirical observability Gramian along training trajectories

48 training episodes x 2 start times (0.35 T, 0.6 T, start moved back if the window does not
fit), central finite differences of the true plant (C3-partial: nominal; C3-mismatch30: the
perturbed plant) driven by the episode's own input. Coordinates: x in x_std, y in units of the
sensor-noise std (3% y_std), so I0 = sum_k S_k' S_k is the Fisher information of x(t0) from the
samples in [t0, t0 + H] (no process noise, no prior); CRB = sqrt(diag(I0^-1)) in x_std. Medians
over the 96 windows.

| plant (scenario) | window H | CRB h1 | CRB h2 | CRB h3 | CRB h4 | cond(I0) | weakest direction (abs loadings h1..h4) |
|---|---|---|---|---|---|---|---|
| quad_tank (partial) | 10 s = 1/alpha | 0.004 | 0.005 | **0.080** | **0.045** | 1.5e3 | (0.03, 0.04, 0.95, 0.31) at the window end |
| | 30 s = 3/alpha | 0.003 | 0.003 | 0.017 | 0.013 | 203 | (0.12, 0.11, 0.97, 0.20) |
| | 49 s = tau_slow | 0.003 | 0.003 | 0.009 | 0.009 | 91 | (0.25, 0.18, 0.94, 0.13) |
| | 98 s | 0.002 | 0.002 | 0.005 | 0.006 | 44 | |
| | 147 s | 0.002 | 0.002 | 0.004 | 0.006 | 39 | |
| quad_tank (mismatch30) | 10 s | 0.005 | 0.005 | 0.071 | 0.056 | 1.5e3 | (0.02, 0.05, 0.87, 0.49) |
| | 49 s | 0.003 | 0.003 | 0.010 | 0.012 | 147 | |
| vdp (partial) | 0.67 s = 1/alpha | 0.011 | 0.036 | | | 47 | |
| | 3 s = tau_slow | 0.006 | 0.007 | | | 8.4 | |
| battery (partial) | 0.67 min = 1/alpha | 0.26 | 0.31 | 1.33 | 0.009 | 0.024 (h5) | 8.8e4 | SOC / V1 / V2 |
| | 3 min = 3 tau_slow | 0.018 | 0.088 | 0.128 | 0.004 | 0.024 (h5) | 3.2e3 | |

(The "weakest direction" column is from the information on the window-end state; the end-state
CRB itself is not tabulated because without process noise it is dominated by the plant's
contraction.) Linearised at the nominal equilibrium, the observability matrix [C; CA; CA^2; CA^3]
has singular values (1.0, 1.0, 0.045, 0.017).

**Answer:** the upper tanks are weakly observable on the time scale the KKL latent remembers:
over 1/alpha = 10 s, the information on h3 is (0.080/0.004)^2 ~ 400x smaller than on h1, and the
Gramian is 30x worse conditioned than vdp's. Enough information accumulates only over ~30-50 s
(3/alpha to one slow time constant). HyperKKL's per-state error on h3/h4 (C3 lora, sequence_ood:
0.125 / 0.065 partial, 0.036 / 0.071 mismatch30) is of the order of the 10-s CRB, and the grey-box
filter's (0.022 / 0.015) of the ~30-s CRB: the grey-box integrates the innovation through a model
with unbounded memory, the learned observer reads a 6-D latent whose memory is ~1/alpha. Battery
is even worse conditioned at 1/alpha (cond 8.8e4) and is the other system where HyperKKL loses
widely to the grey-box; vdp (cond 47) is one where it wins.

## (d) Persistent excitation of the training set

1. **Input, classical PE.** Hankel matrix of u with 15 lags spaced to span 3 slow time constants
   (147 s; vdp / battery: 3 of theirs), episode mean removed, labelled window:
   lambda_min / lambda_max = 0.088 (quad_tank train), 0.047 / 0.030 / 0.078 (id / shape / seq);
   vdp 0.052, battery 0.0073. The training input is persistently exciting of order 15 over three
   slow time constants, better conditioned than on the reference plants.
2. **Regressor of the learned map (z, c) -> x.** z = z_obs (6), c = a model-free proxy of the SFM
   context (u through first-order low-passes at tau = 10 x 1.5^j s, j = 0..5, the SFM's initial
   pole range), standardised with training statistics. Covariance condition number: [z, c] 3.4e7,
   z alone 3.0e5 (vdp 7.9e6 / 1.0e6, battery 1.3e7 / 6.2e6). Evaluation relative to training, along
   the training eigendirections: sequence_ood has 2.7-3.6x the training second moment in the four
   weakest z directions (id / shape_ood 0.75-0.95x); Mahalanobis radius beyond the training 99th
   percentile: 1.8% (z) and 0.9% ([z, c]) of scored sequence_ood samples. The references show the
   same pattern or worse (vdp 5.3% / 2.2%, battery 3.4% / 3.0%), and vdp wins against the grey-box.
3. **Independent excitation of the unmeasured states.** Fraction of the variance of h3 / h4 not
   linearly explained by (h1, h2, the six input filters): training 7.8% / 0.4% (C3-partial),
   evaluation 0.7-1.0% / 0.2-0.4%. With a single scalar input both upper tanks are driven along a
   one-parameter steady-state curve; their independent motion is the decaying IC transient and
   dynamic lag, and the training set holds more of it than the evaluation episodes.
4. **Where the information sits.** In the training latent, h1 and h2 are carried by the two
   strongest z directions (R^2 >= 0.998); h3 and h4 correlate 40% (R^2) with the three WEAKEST
   directions, whose variances are 3e-6 .. 6e-5 of the strongest (mismatch30: 23-29%). The linear
   R^2 of z -> h3 is only 0.78 (h4 0.94). The decoder must amplify near-degenerate latent
   directions to read the upper tanks: an observability/conditioning property of the KKL map at
   alpha = 0.1, not of the input data.

## (e) Verdict

**Observability-limited.** The training data are persistently exciting (input Hankel well
conditioned over three slow time constants, evaluation states inside the training support, more
independent upper-tank variation in training than in evaluation, no error-coverage correlation);
the error sits on h3/h4, which the measured levels reveal only slowly (Gramian condition 1.5e3 and
CRB 0.08 x_std over the latent's 10-s memory) and which the latent stores in directions 1e-5 as
strong as those carrying h1/h2. This matches the ERROR_FLOOR finding that a probe on NOISE-FREE
(z, c) stalls at 0.046 and only a 600-sample (30 s) z-window helps (0.038): the bottleneck is the
information the 6-D latent holds about h3, not the number or richness of training examples.

Caveats, stated before Stage 2: (i) the slow-band share of quad_tank's training inputs relative to
its own slow corner is lower than vdp's and battery's (13.6% vs 24-43%), so a modest data effect is
possible; (ii) sequence_ood excites the weakest latent directions ~3x more than training, which is
an extrapolation channel that slower or longer training inputs might or might not fill; (iii) the
CRB is a no-prior, no-process-noise bound on the window-start state, indicative only; (iv) one
seed for the per-episode correlations (A2 only has seed 0 finished in the campaign).
Prediction registered in PREREG.md: < 10% change from slow-band or longer training data.
