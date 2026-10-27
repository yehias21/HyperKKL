"""Mechanical analysis (PREREG.md section 3-4) -> ../results.json, ../tables.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from dispatch import CELLS, DEV, TEST

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("sequence_ood", "shape_ood", "id")
GB = ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
HK = ("A0", "A3", "A4+D3")
BASE = ["gru140", "gru500", "ncde140", "node5k_ekf", "node5k_ukf", "node140k_ekf", "node140k_ukf",
        "ude5k_ekf", "ude5k_ukf", "ude5k1_ekf", "ude5k1_ukf", "ude140k_ekf", "ude140k_ukf", "nlox140", "nlox500"]
WSTARVED = {"C3-partial/battery", "C3-mismatch30/battery", "C3-partial/quad_tank", "C3-mismatch30/quad_tank",
            "TEST-C3/highway_traffic"}


def cell(n, s):
    return f"{n}/{s}"


def load():
    data, meta = {}, {}
    for f in (ROOT / "runs").glob("*/*/seed*/*.json"):
        if f.parts[-5] == "smoke":
            continue
        d = json.load(open(f))
        c = cell(d["node"], d["system"])
        for t, r in d["results"].items():
            for m, v in r.items():
                data.setdefault(c, {}).setdefault(m, {}).setdefault(t, {})[d["seed"]] = v
        for m, v in d["methods"].items():
            meta.setdefault(c, {}).setdefault(m, {})[d["seed"]] = {
                "lr": v.get("lr"), "n_params": v.get("n_params"), "arch": v.get("arch"),
                "wall_s": d["wall_s"]}
    refs = json.load(open(ROOT / "references.json"))["refs"]
    for c, d in refs.items():
        for m, v in d.items():
            for t, sv in v.items():
                for sd, x in sv.items():
                    data.setdefault(c, {}).setdefault(m, {}).setdefault(t, {})[int(sd)] = x
    return data, meta


def smean(d, seeds):
    return {s: float(np.mean(d[s]["nrmse"])) for s in seeds}


def compare(a, b):
    """a, b: {seed: per-episode dict}. Paired over common seeds."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    r = np.array([(np.mean(a[s]["nrmse"]) - np.mean(b[s]["nrmse"])) / max(np.mean(b[s]["nrmse"]), 1e-12)
                  for s in seeds])
    m, se = float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))
    tq = stats.t.ppf(0.975, len(r) - 1)
    p = float(stats.ttest_1samp(r, 0.0).pvalue) if se > 0 else (0.0 if m != 0 else 1.0)
    return {"rel": m, "ci": [m - tq * se, m + tq * se], "p": p, "seeds": seeds}


def holm(ps):
    idx = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(idx):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj


GBFIX = GB + ("ude5k1_ekf", "ude5k1_ukf")      # post-hoc: consistency-corrected grey-box (AUDIT_UDE.md 6)


def best_gb(data, c, t, pool=GB):
    cands = []
    for g in pool:
        d = data.get(c, {}).get(g, {}).get(t, {})
        if len(d) >= 3:
            cands.append((np.mean(list(smean(d, d).values())), g))
    return min(cands)[1] if cands else None


def families(data, cells, tier, method, ref):
    """Comparisons of `method` vs `ref` ('bestGB' or a HyperKKL arm) over cells, Holm-adjusted."""
    rows = {}
    for c in cells:
        a = data.get(c, {}).get(method, {}).get(tier)
        r = best_gb(data, c, tier) if ref == "bestGB" else (best_gb(data, c, tier, GBFIX) if ref == "bestGBfix" else ref)
        b = data.get(c, {}).get(r, {}).get(tier) if r else None
        if a and b:
            x = compare(a, b)
            if x:
                rows[c] = {**x, "ref": r}
    if rows:
        adj = holm(np.array([v["p"] for v in rows.values()]))
        for (c, v), pa in zip(rows.items(), adj):
            v["p_holm"] = float(pa)
            v["verdict"] = "beats" if pa < 0.05 and v["rel"] < 0 else ("worse" if pa < 0.05 and v["rel"] > 0 else "tie")
    return rows


def main():
    data, meta = load()
    out = {"comparisons": {}, "means": {}, "params": {}, "lr": {}, "repro": {}}
    dev_c = [cell(*x) for x in DEV]
    dev_nd = [c for c in dev_c if "duffing" not in c]
    test_c = [cell(*x) for x in TEST]
    methods = BASE + list(GB) + ["nlox40_stored"]
    # seed means
    for c in dev_c + test_c:
        for m in methods + list(HK):
            for t in TIERS:
                d = data.get(c, {}).get(m, {}).get(t)
                if d:
                    sm = smean(d, d)
                    out["means"].setdefault(c, {}).setdefault(m, {})[t] = {
                        "mean": float(np.mean(list(sm.values()))), "n_seeds": len(sm),
                        "per_seed": sm, "div": float(np.mean([np.mean(d[s]["diverged"]) for s in d]))}
        for m, v in meta.get(c, {}).items():
            out["params"].setdefault(c, {})[m] = sorted({x["n_params"] for x in v.values()})
            out["lr"].setdefault(c, {})[m] = {s: x["lr"] for s, x in v.items()}
    # reproduction: re-fit grey-box vs stored (seeds 0-2)
    for c in dev_c + test_c:
        for g in GB:
            a, b = data.get(c, {}).get(g, {}).get("sequence_ood"), data.get(c, {}).get(g + "_stored", {}).get("sequence_ood")
            if a and b:
                ss = sorted(set(a) & set(b))
                out["repro"].setdefault(c, {})[g] = max(abs(np.mean(a[s]["nrmse"]) - np.mean(b[s]["nrmse"])) /
                                                       np.mean(b[s]["nrmse"]) for s in ss)
    # comparisons
    fams = {"dev": dev_c, "dev_noduffing": dev_nd, "test": test_c}
    for m in BASE + ["nlox40_stored"] + list(HK):
        for ref in ["bestGB", "bestGBfix"] + list(HK) + (["nlox40_stored"] if m.startswith("nlox") and m != "nlox40_stored" else []):
            for t in TIERS:
                for fam, cs in fams.items():
                    for scen in (["C3-partial", "C3-mismatch30"] if fam.startswith("dev") else ["test"]):
                        cc = [c for c in cs if fam == "test" or c.startswith(scen + "/")]
                        rows = families(data, cc, t, m, ref)
                        if rows:
                            out["comparisons"].setdefault(m, {}).setdefault(ref, {}).setdefault(t, {})[f"{fam}|{scen}"] = rows
    out["predictions"] = predictions(out, dev_c, test_c)
    (ROOT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    tables(out, dev_c, test_c)


def _v(out, m, ref, t, c):
    for k, rows in out["comparisons"].get(m, {}).get(ref, {}).get(t, {}).items():
        if k.startswith("dev_noduffing"):
            continue
        if c in rows:
            return rows[c]
    return None


def predictions(out, dev_c, test_c):
    t = "sequence_ood"
    P = {}
    # P1: lowest median rank over 12 dev cells among the 13 learned variants
    have = [m for m in BASE if all(out["means"].get(c, {}).get(m, {}).get(t) for c in dev_c)]
    ranks = {m: [] for m in have}
    for c in dev_c:
        order = sorted(have, key=lambda m: out["means"][c][m][t]["mean"])
        for i, m in enumerate(order):
            ranks[m].append(i + 1)
    med = {m: float(np.median(r)) for m, r in ranks.items()}
    best = min(med, key=lambda m: (med[m], np.mean(ranks[m]))) if med else None
    P["P1"] = {"median_rank": med, "strongest": best, "holds": bool(best and best.startswith("ude"))}
    # P2: GRU140 vs NCDE140 cells
    g = n = 0
    for c in dev_c:
        a, b = out["means"].get(c, {}).get("gru140", {}).get(t), out["means"].get(c, {}).get("ncde140", {}).get(t)
        if a and b:
            g += a["mean"] < b["mean"]; n += a["mean"] > b["mean"]
    P["P2"] = {"gru_better": g, "ncde_better": n, "holds": g > n}
    # P3: no learned baseline beats best GB on window-starved cells
    beats = []
    for m in BASE:
        for c in WSTARVED:
            v = _v(out, m, "bestGB", t, c)
            if v and v["verdict"] == "beats":
                beats.append((m, c, v["rel"]))
    P["P3"] = {"beats": beats, "holds": not beats}
    # P4: NODE + filter worse than best GB on >= 8/12 dev cells (best NODE variant per cell)
    worse = 0
    for c in dev_c:
        vs = [_v(out, m, "bestGB", t, c) for m in BASE if m.startswith("node")]
        vs = [v for v in vs if v]
        if vs and all(v["verdict"] == "worse" for v in vs):
            worse += 1
    P["P4"] = {"cells_all_node_variants_worse": worse, "holds": worse >= 8}
    # P5: NLOX-140k better than stored 40k on at most half of dev cells
    b = sum(1 for c in dev_c if out["means"].get(c, {}).get("nlox140", {}).get(t) and
            out["means"][c]["nlox140"][t]["mean"] < out["means"][c]["nlox40_stored"][t]["mean"])
    P["P5"] = {"nlox140_better_cells": b, "holds": b <= 6}
    # P6: A4+D3 beats GRU140 on >= 6/12 dev cells (Holm): GRU140 vs A4+D3 'worse'
    w = sum(1 for c in dev_c if (_v(out, "gru140", "A4+D3", t, c) or {}).get("verdict") == "worse")
    P["P6"] = {"cells": w, "holds": w >= 6}
    return P


def _fmt(v):
    if not v:
        return "-"
    mark = {"beats": "**", "worse": "", "tie": ""}[v["verdict"]]
    s = f"{100 * v['rel']:+.0f}%"
    return f"{mark}{s}{mark}" + ("*" if v["verdict"] == "worse" else "")


def tables(out, dev_c, test_c):
    L = ["# baselines2 tables (generated by code/analyze.py)", "",
         "Seed-mean settled NRMSE; rel = per-seed relative change vs reference, mean over seeds; "
         "**bold** = baseline significantly better (Holm over the plants of the family), * = significantly worse.",
         ""]
    cols = BASE
    for t in TIERS:
        L += [f"## {t}: seed-mean NRMSE", "",
              "| cell | best GB | A0 | A3 | A4+D3 | NLOX40 | " + " | ".join(cols) + " |",
              "|" + "---|" * (6 + len(cols))]
        for c in dev_c + test_c:
            mm = out["means"].get(c, {})
            gbn = min(((mm[g][t]["mean"], g) for g in GB if mm.get(g, {}).get(t)), default=None)
            f = lambda m: f"{mm[m][t]['mean']:.4f}" if mm.get(m, {}).get(t) else "-"
            L.append(f"| {c} | " + (f"{gbn[0]:.4f} ({gbn[1].replace('greybox_', '')})" if gbn else "-") + " | "
                     + " | ".join(f(m) for m in ("A0", "A3", "A4+D3", "nlox40_stored")) + " | "
                     + " | ".join(f(m) for m in cols) + " |")
        L.append("")
        for ref in ("bestGB", "bestGBfix", "A0", "A4+D3"):
            L += [f"### {t}: vs {ref} (rel. change; Holm)", "",
                  "| cell | " + " | ".join(cols) + " |", "|" + "---|" * (1 + len(cols))]
            for c in dev_c + test_c:
                L.append(f"| {c} | " + " | ".join(_fmt(_v(out, m, ref, t, c)) for m in cols) + " |")
            L.append("")
    # counts
    L += ["## Counts vs best grey-box (beats / tie / worse), Holm", "",
          "| baseline | tier | dev (12) | dev no duffing (10) | test (6) |", "|---|---|---|---|---|"]
    for m in cols:
        for t in TIERS:
            cnt = {}
            for fam in ("dev", "dev_noduffing", "test"):
                k = [0, 0, 0]
                for key, rows in out["comparisons"].get(m, {}).get("bestGB", {}).get(t, {}).items():
                    if key.split("|")[0] != fam:
                        continue
                    for v in rows.values():
                        k[("beats", "tie", "worse").index(v["verdict"])] += 1
                cnt[fam] = "/".join(map(str, k))
            L.append(f"| {m} | {t} | {cnt['dev']} | {cnt['dev_noduffing']} | {cnt['test']} |")
    L += ["", "## Parameters (trained) and selected lr", "", "| cell | " + " | ".join(cols) + " |",
          "|" + "---|" * (1 + len(cols))]
    for c in dev_c + test_c:
        p = out["params"].get(c, {})
        L.append(f"| {c} | " + " | ".join(",".join(map(str, p.get(m, []))) or "-" for m in cols) + " |")
    L += ["", "| cell | " + " | ".join(cols) + " |", "|" + "---|" * (1 + len(cols))]
    for c in dev_c + test_c:
        p = out["lr"].get(c, {})
        L.append(f"| {c} | " + " | ".join(
            ",".join(f"{p[m][s]:g}" for s in sorted(p[m])) if m in p else "-" for m in cols) + " |")
    L += ["", "## Grey-box re-fit reproduction (max over seeds 0-2 of |refit - stored| / stored, seq OOD)", ""]
    for c, v in out["repro"].items():
        L.append(f"- {c}: " + ", ".join(f"{g} {x:.2e}" for g, x in v.items()))
    L += ["", "## Predictions", "", "```", json.dumps(out["predictions"], indent=1, default=float), "```"]
    (ROOT / "tables.md").write_text("\n".join(L))
    print("\n".join(L[:80]))


if __name__ == "__main__":
    main()
