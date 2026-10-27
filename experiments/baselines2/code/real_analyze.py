"""Real battery: GRU / NCDE vs stored grey-box UKF/EKF, HyperKKL (lora) and A4+D3 (seeds 0-2, settled SOC
RMSE %, per test family; paired t-CI over seeds of the difference). -> ../real/summary.json, printed table."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments")
FAM25 = {"US06": ["test25/US06"], "HWFET": ["test25/HWFTa", "test25/HWFTb"], "NN": ["test25/NN"]}
REFS = ("greybox_ukf", "greybox_ekf", "lora", "nlox")


def main():
    stored = json.load(open(LIVE / "real/results.json"))["tables"]
    a4 = json.load(open(LIVE / "beat_greybox/real/results_A4+D3.json"))["settled_rmse"]
    out = {}
    for kind in ("gru", "ncde"):
        runs = [json.load(open(ROOT / "real" / f"{kind}_seed{s}.json")) for s in (0, 1, 2)
                if (ROOT / "real" / f"{kind}_seed{s}.json").exists()]
        if len(runs) < 3:
            continue
        out[kind] = {"n_params": runs[0]["n_params"], "lr": [r["lr"] for r in runs], "fam": {}}
        for fam, sets in FAM25.items():
            per = [float(np.mean(np.concatenate([r["results"][s]["rmse_settled"] for s in sets]))) for r in runs]
            row = {"mean": float(np.mean(per)), "per_seed": per, "vs": {}}
            refs = {m: stored["test25"][fam][m]["per_seed"] for m in REFS}
            refs["A4+D3"] = a4[fam]["A4+D3"]
            for m, ref in refs.items():
                d = np.asarray(per) - np.asarray(ref)
                se = d.std(ddof=1) / np.sqrt(3)
                tq = stats.t.ppf(0.975, 2)
                row["vs"][m] = {"ref_mean": float(np.mean(ref)), "diff": float(d.mean()),
                                "ci": [float(d.mean() - tq * se), float(d.mean() + tq * se)],
                                "all_seeds_same_sign": bool((d < 0).all() or (d > 0).all())}
            out[kind]["fam"][fam] = row
        out[kind]["test0_mean"] = {k.split("/")[1]: float(np.mean([np.mean(r["results"][k]["rmse_settled"]) for r in runs]))
                                   for k in runs[0]["results"] if k.startswith("test0/")}
    (ROOT / "real" / "summary.json").write_text(json.dumps(out, indent=1))
    for kind, o in out.items():
        print(f"{kind} ({o['n_params']} params, lr {o['lr']})")
        for fam, r in o["fam"].items():
            print(f"  {fam}: {r['mean']:.2f}  " + "  ".join(
                f"vs {m} {v['ref_mean']:.2f}: {v['diff']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]"
                for m, v in r["vs"].items()))
        print("  0 degC (OOD temperature):", {k: round(v, 2) for k, v in o["test0_mean"].items()})


if __name__ == "__main__":
    main()
