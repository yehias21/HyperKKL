"""Stage-2 analysis (PREREG.md section 5). CPU only. Writes analysis.json and prints tables."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_qt import c3_results, np  # noqa: E402
from hyperkkl.metrics import paired_relative  # noqa: E402

NODES = ["C3-partial", "C3-mismatch30"]
TIERS = ["sequence_ood", "shape_ood", "id"]
ARMS = ["X0", "X1", "X2", "X3"]
GB = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
SEEDS = [0, 1, 2]
BGR = HERE.parent / "beat_greybox" / "runs"


def load(node, seed, arm):
    """HyperKKL result + grey-box refit (in-job if present, else the CPU gb-only job)."""
    f = HERE / "runs" / node / f"seed{seed}" / f"{arm}.json"
    g = HERE / "runs" / node / f"seed{seed}" / f"{arm}_gb.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    if "greybox_ekf" not in d["results"]["sequence_ood"]:
        if not g.exists():
            return None
        gb = json.load(open(g))
        for t in d["results"]:
            d["results"][t].update(gb["results"][t])
        d["gb_source"] = "gb-only job"
    else:
        d["gb_source"] = "in-job"
        if g.exists():       # duplicate CPU refit (same code, same seed): consistency check
            gb = json.load(open(g))
            d["gb_dup_seq"] = {m: [float(np.mean(d["results"]["sequence_ood"][m]["nrmse"])),
                                   float(np.mean(gb["results"]["sequence_ood"][m]["nrmse"]))] for m in GB}
    return d


def fmt(r):
    return f"{100 * r['rel']:+.1f}% [{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}]" + \
        ("*" if (r["better"] or r["worse"]) else "")


A = {}
for node in NODES:
    R = {a: [load(node, s, a) for s in SEEDS] for a in ARMS}
    have = [a for a in ARMS if all(R[a])]
    print(f"\n######## {node}: complete arms {have}")
    A[node] = {"complete": have}
    for tier in TIERS:
        T = {}
        h = {a: [np.asarray(R[a][i]["results"][tier]["hyperkkl"]["nrmse"]) for i in range(3)] for a in have}
        stored = {g: [np.asarray(c3_results(node, "quad_tank", s)[tier]["clean"]["methods"][g]["nrmse"]) for s in SEEDS] for g in GB}
        gbs = min(GB, key=lambda g: np.mean([x.mean() for x in stored[g]]))
        e_gbs = float(np.mean([x.mean() for x in stored[gbs]]))
        T["best_stored_gb"] = {"name": gbs, "nrmse": e_gbs}
        rt = {}
        for a in have:
            gb_a = {g: [np.asarray(R[a][i]["results"][tier][g]["nrmse"]) for i in range(3)] for g in GB}
            gb_best = min(GB, key=lambda g: np.mean([x.mean() for x in gb_a[g]]))
            rt[a] = (gb_best, gb_a[gb_best])
        print(f"\n== {node} / {tier}: best stored grey-box {gbs} {e_gbs:.4f}")
        for a in have:
            e = float(np.mean([x.mean() for x in h[a]]))
            row = {"hyperkkl": e, "hyperkkl_per_seed": [float(x.mean()) for x in h[a]],
                   "gb_retrained_best": rt[a][0], "gb_retrained": float(np.mean([x.mean() for x in rt[a][1]])),
                   "vs_stored_gb": paired_relative(h[a], stored[gbs]),
                   "vs_retrained_gb": paired_relative(h[a], rt[a][1])}
            per = {}
            for j in range(1, 5):
                per[f"x{j}"] = float(np.mean([np.mean(R[a][i]["results"][tier]["hyperkkl"][f"nrmse_x{j}"]) for i in range(3)]))
                per[f"gb_x{j}"] = float(np.mean([np.mean(R[a][i]["results"][tier][rt[a][0]][f"nrmse_x{j}"]) for i in range(3)]))
            row["per_state"] = per
            if a != "X0" and "X0" in have:
                row["vs_X0"] = paired_relative(h[a], h["X0"])
                e0 = float(np.mean([x.mean() for x in h["X0"]]))
                row["gap_closure_stored"] = (e0 - e) / (e0 - e_gbs)
                e0r = float(np.mean([x.mean() for x in rt["X0"][1]]))
                row["gap_closure_retrained"] = (e0 - e) / (e0 - row["gb_retrained"])
                row["gb_retrained_vs_gbX0"] = paired_relative(rt[a][1], rt["X0"][1])
            if a == "X0":
                row["gbX0_vs_stored"] = paired_relative(rt["X0"][1], stored[rt["X0"][0]])
                a2 = []
                for s in SEEDS:
                    f = BGR / node / "quad_tank" / f"seed{s}" / "A2.json"
                    if f.exists():
                        a2.append((s, float(np.mean(json.load(open(f))["results"][tier]["nrmse"]))))
                row["campaign_A2"] = a2
            T[a] = row
            msg = f"  {a}: hkkl {e:.4f} (seeds {', '.join(f'{v:.4f}' for v in row['hyperkkl_per_seed'])}) | gb-retr {rt[a][0]} {row['gb_retrained']:.4f}"
            msg += f" | vs stored gb {fmt(row['vs_stored_gb'])} | vs retr gb {fmt(row['vs_retrained_gb'])}"
            if "vs_X0" in row:
                msg += f"\n        vs X0 {fmt(row['vs_X0'])}  gap closure {row['gap_closure_stored']:+.2f} (retr {row['gap_closure_retrained']:+.2f})  gb vs gbX0 {fmt(row['gb_retrained_vs_gbX0'])}"
            if a == "X0":
                msg += f"\n        gbX0 vs stored {fmt(row['gbX0_vs_stored'])}; campaign A2 {row['campaign_A2']}"
            msg += "\n        per-state hkkl " + " ".join(f"{per[f'x{j}']:.4f}" for j in range(1, 5)) + \
                   " | gb " + " ".join(f"{per[f'gb_x{j}']:.4f}" for j in range(1, 5))
            print(msg)
        A[node][tier] = T

# pre-registered verdict (sequence_ood)
ver = {}
for node in NODES:
    T = A[node].get("sequence_ood", {})
    if not all(k in T for k in ("X2", "X3")):
        continue
    best = min(("X2", "X3"), key=lambda k: T[k]["vs_X0"]["rel"])
    r = T[best]["vs_X0"]
    ver[node] = {"best": best, "rel": r["rel"], "ci": r["ci"], "gap": T[best]["gap_closure_stored"],
                 "pe": r["rel"] <= -0.30 and r["ci"][1] < 0 and T[best]["gap_closure_stored"] >= 0.5,
                 "obs": abs(T["X2"]["vs_X0"]["rel"]) < 0.10 and abs(T["X3"]["vs_X0"]["rel"]) < 0.10}
if len(ver) == 2:
    v = "PE-limited" if all(x["pe"] for x in ver.values()) else (
        "observability-limited" if all(x["obs"] for x in ver.values()) else "mixed / partially PE-limited")
    print(f"\nVERDICT (pre-registered rule): {v}  {ver}")
    A["verdict"] = {"verdict": v, "per_node": ver}
json.dump(A, open(HERE / "analysis.json", "w"), indent=1)
