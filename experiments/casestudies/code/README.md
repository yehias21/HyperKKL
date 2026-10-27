# Case-study code

* `pkg/` — package copy = experiments/cartpole_bounded/code/pkg (live package at that time + beat_greybox
  SOAP / beat patch) + the live `systems_bounded.py` and `systems_casestudy.py` + the casestudies patch:
  `data.measure` (per-output `noise_mult`, multi-rate `hold`, burst `channels`), `data.sparse_labels /
  interp_labels` (assay labels), `metrics.episode_metrics` (NaN-masked targets), `train.masked_sq_mean`
  (phase-1 labelled rows, phase-2 / NLOX masked losses), `beat._decoder_loss_bg` (masked),
  `observers.HOLD_MISSING_DEFAULT` (filters skip stale held samples), per-output R in `evaluate` /
  `oracle_noise`. Every patch reduces to the original computation on dense data (tests). Full diff vs the
  live package: `casestudies_vs_live.patch`. The live package only received `hyperkkl/systems_casestudy.py`
  and a lazy registry hook in `hyperkkl/__init__.py` (no hashed module changed).
* `b2cs/` — copy of experiments/baselines2/code/b2 (models, train) with masked sparse-label losses; its job
  runner is replaced by `run_cs.py`.
* `cs_common.py` (cases, levels, configs), `run_cs.py` (one job), `dispatch.py` (GPU launcher),
  `inclusion_cs.py`, `infolib_cs.py` + `budget_cs.py` (information budgets), `latency_cs.py`,
  `analyze_cs.py`, `tests/`.
