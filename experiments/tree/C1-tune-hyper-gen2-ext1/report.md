# C1-tune-hyper-gen2-ext1 — Tuning grid extension ext1: hyper, gen_layers = 2, lr in [0.008, 0.016]

*parent* `C1-tune-hyper-gen2` · *contribution* C1 · *status* done · jobs 6/6

**Question.** Validation NRMSE of hyper with gen_layers = 2 at the extended learning rates (seed 0, dev).

**Hypothesis.** Part of the section-9 extension.

**Decision rule.** See PROTOCOL.md section 9 (analysis/select_tuned.py --stages ...).

**WARNING: these runs come from 2 code versions (b1a814aa4292, d228453526ab). Re-run (queue --force) before drawing a verdict.**

# Metric `nrmse`

Definitions in hyperkkl/metrics.py. Mean over training seeds with a Student-t 95% CI over seeds (single-seed nodes: episode bootstrap, which ignores training variability); lower is better. Paired cells: mean per-seed relative change vs the reference, negative = better, bold = CI excludes 0. µs/step: CPU, batch 1.

> **Single seed**: intervals are episode-bootstrap only and do not support a verdict about methods (training variability is not measured).

## Deployment scorecard

One row per method. nrmse columns: clean condition, mean over episodes and seeds; unmeasured = mean NRMSE over the unmeasured states; worst sensor fault = max over the sensor conditions present (named); p99 over episodes; div. rate = max over every tier/condition block (named); forget prior = median time until the poor_init estimate merges with the same observer's nominal-prior estimate (paired); conv. = median time after which the poor_init error stays below 2x its settled median; recovery = median time to return within 1.5x the pre-outage error after burst_outage; drift = NRMSE(last 20%) / NRMSE(20% after settling) on `long` episodes; alarm AUROC = output-residual alarm detecting state error > 3x median (metrics.py). Tier: sequence_ood unless named in the cell.

### battery

| method |  |
|---|

### bioreactor

| method |  |
|---|

### duffing

| method |  |
|---|

### lorenz

| method |  |
|---|

### quad_tank

| method |  |
|---|

### vdp

| method |  |
|---|
