"""C3-regimes (2026-10-26 ~03:50 Dubai): KKL vs grey-box filters in the conditions where KKL's
intended advantages should show (poor initial guess, outliers, noise, plant drift, sensor outage).
Same config as C3-partial (knowledge=partial) except eval.conditions, so trained models are reused
from cache (model digests do not include eval). Writes only the new node directory."""
from pathlib import Path
import datetime as _dt, yaml
TREE = Path(__file__).resolve().parent / "tree"
d = {"id": "C3-regimes", "parent": "C3-partial",
     "title": "KKL vs grey-box filters where KKL's advantages should show (partial knowledge)",
     "contribution": "C3",
     "question": "Under partial knowledge, do learned KKL observers beat the grey-box EKF/UKF (partial model + learned residual, same data) under a poor initial guess, outliers, higher sensor noise, post-training plant drift or a sensor outage — the conditions where global convergence and no linearisation should matter?",
     "hypothesis": "Grey-box filters keep winning on clean data, but KKL wins under poor_init (global convergence) and loses less under outliers/outage; plant drift hurts both.",
     "decision_rule": "Per condition, best KKL (lora/hyper) vs best grey-box filter (greybox_ekf/ukf/ekf_oracle): paired 95% CI excluding 0 on >= 4/6 dev systems on sequence_ood = 'KKL-wins regime'; report every condition with k/6 and the absolute NRMSE; also report divergence rate and forget-prior time.",
     "set": {"system.knowledge": "partial",
             "eval.conditions": ["clean", "poor_init", "noise10", "outliers", "plant_drift10", "burst_outage"]},
     "methods": ["lora", "hyper"],
     "baselines": ["ekf", "ukf", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle", "koopman_kf"],
     "seeds": [0, 1, 2],
     "compare": {"reference": ["greybox_ekf", "hyper"], "tier": "sequence_ood", "condition": "poor_init"},
     "priority": 25, "status": "planned", "created": f"{_dt.date.today():%Y-%m-%d}"}
p = TREE / d["id"]
if (p / "node.yaml").exists():
    print("exists"); raise SystemExit
p.mkdir(parents=True); (p / "node.yaml").write_text(yaml.safe_dump(d, sort_keys=False, width=100)); print("wrote", d["id"])
