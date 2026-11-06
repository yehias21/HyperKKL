"""Rescore HyperKKLTrain runs from their saved models, without training, through the benchmark API (after a scorer change).

    python tools/rescore_hyperkkl.py neurobem --gpu 2 --seed 0

The report keeps the training meta of the original smoke run (budget, fit_info, wall time).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--gpu", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
import hyperkkl_bench.real as hr  # noqa: E402
from hyperkkl_bench.baselines.hyperkkl_train import HyperKKLTrain  # noqa: E402
est = HyperKKLTrain(seed=a.seed, device="cuda", gpu=a.gpu)
est.spec_path = HERE / "logs" / "hkkl_train" / f"{a.dataset}_A3_s{a.seed}_x1" / "spec.json"
out = HERE / "reference" / "real" / "smoke"
out = (out, f"{a.dataset}__hyperkkl") if a.seed == 0 else (out / "seeds", f"{a.dataset}__hyperkkl_s{a.seed}")
old = json.loads((out[0] / f"{out[1]}.json").read_text())
rep = hr.evaluate(est, a.dataset, fit=False, verbose=True, threads=4)
rep.meta["fit_info"], rep.meta["fit_s"], rep.meta["estimator"] = old["meta"]["fit_info"], old["meta"]["fit_s"], old["meta"]["estimator"]
rep.meta["smoke"] = dict(old["meta"]["smoke"], rescored_from_saved_model=True)
rep.save(out[0] / f"{out[1]}.json")
(out[0] / f"{out[1]}.md").write_text(rep.table() + "\n")
print(rep.table())
