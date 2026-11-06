"""Copy the stored reference numbers of the not-yet-frozen tracks (deployment, scaling) into
src/hyperkkl_bench/resources/tracks_reference.json, each with its source path. Read-only on experiments/."""
import csv
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[2] / "experiments"
OUT = Path(__file__).resolve().parents[1] / "src" / "hyperkkl_bench" / "resources" / "tracks_reference.json"
cs = json.loads((EXP / "casestudies" / "results.json").read_text())
lat = json.loads((EXP / "casestudies" / "latency_realistic.json").read_text())
dep = {"about": "Case A = quad_planar drone (budget 150 us / 512 KB per step on an MCU); Case B = cstr_exo with sparse "
                "C_A lab labels (every 45 +- 15 samples). Metric: settled unmeasured NRMSE, mean over 5 seeds.",
       "source_accuracy": "experiments/casestudies/results.json:cells.<case>.<tier>.<level>.<method>.mean",
       "source_cost": "experiments/casestudies/latency_realistic.json:cases.<case>.<method>",
       "budget": {"latency_us": 150.0, "memory_bytes": 524288}, "cases": {}}
for case, name in (("A", "quad_planar"), ("B", "cstr_exo")):
    acc = {t: {lv: {m: round(v["mean"], 5) for m, v in ms.items()} for lv, ms in lvs.items()}
           for t, lvs in cs["cells"][case].items()}
    cost = {m: {k: v[k] for k in ("flops", "bytes_f32", "us_cpu_median", "mcu_us_est", "meets_budget_mcu_est") if k in v}
            for m, v in lat["cases"][case].items()}
    dep["cases"][name] = {"case": case, "accuracy": acc, "cost": cost}
sc = {"about": "fhnring n_x in 4..64 (n_y = n_x/4); mean unmeasured NRMSE over seeds per tier x level x method",
      "source": "experiments/scaling/figure_data.csv", "rows": []}
with open(EXP / "scaling" / "figure_data.csv") as f:
    for r in csv.DictReader(f):
        sc["rows"].append({k: (float(v) if k in ("mean_unmeasured_nrmse", "sd_over_seeds") else (int(v) if k in ("nx", "n_seeds") else v))
                           for k, v in r.items()})
OUT.write_text(json.dumps({"deployment": dep, "scaling": sc}, indent=1))
print("wrote", OUT, len(sc["rows"]), "scaling rows")
