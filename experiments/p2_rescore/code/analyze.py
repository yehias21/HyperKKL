"""P2 re-score: fold ratios HyperKKL / best grey-box, old recipe vs SOAP (A2, A4). -> results.json, tables.md

    .venv/bin/python experiments/p2_rescore/code/analyze.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
P = HERE.parent
ROOT = P.parent.parent
TREE = ROOT / "experiments" / "tree"
CAMP = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/beat_greybox/runs")
sys.path.insert(0, str(ROOT / "experiments" / "information"))
import infolib as L  # noqa: E402

GREYBOX = ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
TIERS = ("id", "shape_ood", "sequence_ood")
SEEDS = (0, 1, 2)
NEW_SRC = {"battery": CAMP / "C3-partial" / "battery", "quad_tank": CAMP / "C3-partial" / "quad_tank",
           "cartpole": P / "runs" / "TEST-C3" / "cartpole", "fhn": P / "runs" / "TEST-C3" / "fhn",
           # rich dev plants: campaign C3-partial cells (C3-partial == C3-regimes config; reused for the P2 re-score)
           "bioreactor": CAMP / "C3-partial" / "bioreactor", "lorenz": CAMP / "C3-partial" / "lorenz",
           "vdp": CAMP / "C3-partial" / "vdp"}
ARMS = ("A2", "A4")


def stored(system, seed):
    node = L.RUN_PARTIAL[system]
    return json.loads((TREE / node / "runs" / system / f"seed{seed}" / "results.json").read_text())["results"]


def vals_stored(res, tier, m):
    sc = res[tier]["clean"]["scorecard"][m]
    return {"tot": float(np.mean(res[tier]["clean"]["methods"][m]["nrmse"])), "unm": float(sc["nrmse_unmeasured"])}


def vals_new(system, seed, arm):
    f = NEW_SRC[system] / f"seed{seed}" / f"{arm}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())["results"]
    return {t: {"tot": float(np.mean(r[t]["nrmse"])), "unm": float(r[t]["scorecard"]["nrmse_unmeasured"])} for t in TIERS}


def tci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
    return float(math.exp(d.mean() - h)), float(math.exp(d.mean() + h))


def plant_table(system):
    st = {s: stored(system, s) for s in SEEDS}
    out = {}
    for tier in TIERS:
        for key in ("tot", "unm"):
            gb = {m: [vals_stored(st[s], tier, m)[key] for s in SEEDS] for m in GREYBOX if m in st[0][tier]["clean"]["scorecard"]}
            gbm = min(gb, key=lambda m: np.mean(gb[m]))
            g = np.array(gb[gbm])
            row = {"greybox": gbm, "greybox_seeds": g.tolist(), "greybox_mean": float(g.mean())}
            old = np.array([vals_stored(st[s], tier, "lora")[key] for s in SEEDS])
            row["old"] = {"seeds": old.tolist(), "mean": float(old.mean()), "fold": float(old.mean() / g.mean()),
                          "fold_ci": tci(np.log(old / g))}
            for arm in ARMS:
                v = [vals_new(system, s, arm) for s in SEEDS]
                ok = [i for i, x in enumerate(v) if x is not None]
                if not ok:
                    continue
                a = np.array([v[i][tier][key] for i in ok]); gg = g[ok]; oo = old[ok]
                row[arm] = {"seeds_present": ok, "seeds": a.tolist(), "mean": float(a.mean()),
                            "fold": float(a.mean() / gg.mean()), "fold_ci": tci(np.log(a / gg)),
                            "vs_old": float(a.mean() / oo.mean()), "vs_old_ci": tci(np.log(a / oo))}
            out[f"{tier}/{key}"] = row
    return out


def p2(ach, folds_new, arm, tier, key="unmeasured", groups=None):
    """fold per plant (unmeasured, tier) from achieved.json (old), replaced by SOAP where available."""
    A = ach
    rows = {}
    for s in L.ALL:
        if s == "duffing":
            continue
        ms = A[s]["partial"]["methods"]
        if "lora" not in ms or tier not in ms["lora"]:
            continue
        c = {m: ms[m][tier][key] for m in GREYBOX if m in ms and tier in ms[m]}
        if not c:
            continue
        gbv = min(c.values())
        old = ms["lora"][tier][key] / gbv
        new = folds_new.get(s, {}).get(arm)
        rows[s] = {"old": old, "used": new if new is not None else old, "source": "SOAP " + arm if new is not None else "old (flagged)"}
        rows[s]["ws_only"] = rows[s]["used"] if groups[s] == "window-starved" else old
    return rows


def main():
    sc = json.loads((ROOT / "experiments" / "information" / "scores.json").read_text())
    groups = sc["groups"]
    ach = json.loads((ROOT / "experiments" / "information" / "achieved.json").read_text())
    tabs = {s: plant_table(s) for s in NEW_SRC}
    lines = []
    f3 = lambda v: f"{v:.4f}"
    ci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]" if not math.isnan(c[0]) else "[n<2]"
    for key, lab in (("unm", "unmeasured-state NRMSE"), ("tot", "total NRMSE")):
        lines += [f"### Fold ratio HyperKKL / best grey-box, {lab}", "",
                  "| plant | tier | best grey-box | old lora | old fold [95% CI] | A2 | A2 fold [95% CI] | A2/old [95% CI] | A4 | A4 fold [95% CI] | A4/old [95% CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, t in tabs.items():
            for tier in TIERS:
                r = t[f"{tier}/{key}"]
                cells = [s, tier, f"{f3(r['greybox_mean'])} ({r['greybox'].replace('greybox_', '')})", f3(r["old"]["mean"]),
                         f"{r['old']['fold']:.2f}x {ci(r['old']['fold_ci'])}"]
                for arm in ARMS:
                    if arm in r:
                        a = r[arm]; n = "" if len(a["seeds_present"]) == 3 else f" (seeds {a['seeds_present']})"
                        cells += [f3(a["mean"]) + n, f"{a['fold']:.2f}x {ci(a['fold_ci'])}", f"{a['vs_old']:.2f} {ci(a['vs_old_ci'])}"]
                    else:
                        cells += ["-", "-", "-"]
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    # P2 rescore
    res = {"plants": tabs, "P2": {}}
    for arm, tier, key in [(a, t, k) for k in ("unm", "tot") for a in ARMS for t in ("sequence_ood", "id")]:
        if True:
            fn = {s: {a: tabs[s][f"{tier}/{key}"][a]["fold"] for a in ARMS if a in tabs[s][f"{tier}/{key}"]
                      and len(tabs[s][f"{tier}/{key}"][a]["seeds_present"]) == 3} for s in tabs}
            rows = p2(ach, fn, arm, tier, {"unm": "unmeasured", "tot": "nrmse"}[key], groups)
            out = {}
            for which in ("old", "used", "ws_only"):
                ws = {s: rows[s][which] for s in rows if groups[s] == "window-starved"}
                rich = {s: rows[s][which] for s in rows if groups[s] == "rich"}
                mr = float(np.median(list(rich.values()))); mw = float(np.median(list(ws.values())))
                n_above = sum(v > mr for v in ws.values())
                out[which] = {"median_ws": mw, "median_rich": mr, "gmean_ws": float(np.exp(np.mean(np.log(list(ws.values()))))),
                              "gmean_rich": float(np.exp(np.mean(np.log(list(rich.values()))))), "n_above": int(n_above),
                              "n_ws": len(ws), "pass_strict": bool(mw > mr and n_above >= len(ws)),
                              "pass_3of4": bool(mw > mr and n_above >= 3)}
            res["P2"][f"{arm}/{tier}/{key}"] = {"rows": rows, "summary": out}
            lines += [f"### P2 re-score ({tier}, {key} fold, duffing excluded; SOAP = {arm} where available)", "",
                      "| plant | group | old fold | fold used | source |", "|---|---|---|---|---|"]
            for s, r in rows.items():
                lines.append(f"| {s} | {groups[s]} | {r['old']:.2f}x | {r['used']:.2f}x | {r['source']} |")
            for which, lab in (("old", "old runs only"), ("used", f"mixed ({arm} where available, both groups)"),
                               ("ws_only", f"{arm} on window-starved only, rich kept old")):
                o = out[which]
                lines.append(f"\n**{lab}**: median fold window-starved {o['median_ws']:.2f}x vs rich {o['median_rich']:.2f}x "
                             f"(geometric means {o['gmean_ws']:.2f}x vs {o['gmean_rich']:.2f}x); {o['n_above']}/{o['n_ws']} window-starved above the "
                             f"rich median -> strict (4/4): **{'PASS' if o['pass_strict'] else 'FAIL'}**, >=3/4: **{'PASS' if o['pass_3of4'] else 'FAIL'}**")
            lines.append("")
    (P / "results.json").write_text(json.dumps(res, indent=1))
    (P / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
