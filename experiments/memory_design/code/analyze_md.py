"""Mechanical analysis of the memory-design study (rules: ../PREREG.md section 3).

    python analyze_md.py            -> ../analysis.json, ../tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "pkg"))
from hyperkkl.metrics import _t_interval, paired_relative  # noqa: E402

LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
CAMP = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/beat_greybox/runs")
QT = LIVE / "experiments/quadtank/runs"
NODES = ["C3-partial", "C3-mismatch30"]
STARVED = ["quad_tank", "battery"]
RICH = ["vdp", "bioreactor"]
SYSTEMS = STARVED + RICH
TIERS = ["sequence_ood", "shape_ood", "id"]
SEEDS = [0, 1, 2]
GB = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
MEM = ["M1", "M2", "M3", "M4", "M5"]
FIXED = ["M1", "M2", "M3", "M5"]
REF = ["A3", "A4", "A4+D3"]
SHORT = {"C3-partial": "partial", "C3-mismatch30": "mm30"}
CRB30 = {"C3-partial": (0.0172, 0.0135), "C3-mismatch30": (0.0174, 0.0170)}   # DIAG (c), Gramian x0 median
CRB10 = {"C3-partial": (0.0801, 0.0449), "C3-mismatch30": (0.0710, 0.0558)}


def cname(n, s):
    return f"{s}/{SHORT[n]}"


def load(n, s, seed, arm):
    f = ROOT / "runs" / n / s / f"seed{seed}" / f"{arm}.json"
    if arm in REF:
        f = CAMP / n / s / f"seed{seed}" / f"{arm}.json"
    return json.load(open(f)) if f.exists() else None


def per_seed(n, s, arm, tier, key="nrmse"):
    out = []
    for sd in SEEDS:
        r = load(n, s, sd, arm)
        if r is None or key not in r["results"][tier]:
            return None
        out.append(np.asarray(r["results"][tier][key], float))
    return out


_tree = {}


def tree(n, s, sd):
    k = (n, s, sd)
    if k not in _tree:
        r = json.load(open(LIVE / "experiments/tree" / n / "runs" / s / f"seed{sd}/results.json"))
        _tree[k] = {t: r["results"][t]["clean"]["methods"] for t in TIERS}
    return _tree[k]


def gb_best(n, s, tier):
    best = None
    for g in GB:
        a = [np.asarray(tree(n, s, sd)[tier][g]["nrmse"], float) for sd in SEEDS]
        m = np.mean([x.mean() for x in a])
        if best is None or m < best[1]:
            best = (g, m, a)
    return best


def smean(a):
    return float(np.mean([x.mean() for x in a])) if a is not None else None


def cmp(a, b):
    if a is None or b is None:
        return None
    r = paired_relative(a, b)
    return {"rel": r["rel"], "ci": r["ci"], "better": r["better"], "worse": r["worse"]}


def val(n, s, arm):
    v = []
    for sd in SEEDS:
        r = load(n, s, sd, arm)
        if r is None:
            return None
        v.append(r["info"]["val"]["val_nrmse"])
    return float(np.mean(v))


def pick(n, s, arms):
    vs = {a: val(n, s, a) for a in arms}
    vs = {a: v for a, v in vs.items() if v is not None}
    return (min(vs, key=vs.get) if len(vs) == len(arms) else None), vs


def upper_err(n, s, arm, tier="sequence_ood"):
    """per-seed (n3, n4, e_up); M0 per-state from this study's M0 re-evaluation."""
    x3, x4 = per_seed(n, s, arm, tier, "nrmse_x3"), per_seed(n, s, arm, tier, "nrmse_x4")
    if x3 is None:
        return None
    n3 = np.array([a.mean() for a in x3]); n4 = np.array([a.mean() for a in x4])
    return n3, n4, np.sqrt((n3 ** 2 + n4 ** 2) / 2)


def gb_states(n, s="quad_tank", t="sequence_ood"):
    g = gb_best(n, s, t)[0]
    m = tree(n, s, 0)[t][g]
    return {k: float(np.mean([np.mean(tree(n, s, sd)[t][g][k]) for sd in SEEDS])) for k in ("nrmse_x3", "nrmse_x4") if k in m}


def main():
    A = {"cells": {}, "repro": {}, "predictions": {}}
    # ------------------------------------------------------------- reproduction: M0 == stored A2
    for n in NODES:
        for s in SYSTEMS:
            m0, a2 = per_seed(n, s, "M0", "sequence_ood"), None
            st = []
            for sd in SEEDS:
                f = CAMP / n / s / f"seed{sd}" / "A2.json"
                st.append(np.asarray(json.load(open(f))["results"]["sequence_ood"]["nrmse"], float) if f.exists() else None)
            if m0 is not None and all(x is not None for x in st):
                A["repro"][cname(n, s)] = max(abs(a.mean() / b.mean() - 1) for a, b in zip(m0, st))
    # ------------------------------------------------------------- per cell tables
    for n in NODES:
        for s in SYSTEMS:
            c = {"tiers": {}}
            for t in TIERS:
                gname, gm, ga = gb_best(n, s, t)
                m0 = per_seed(n, s, "M0", t)
                row = {"gb": {"name": gname, "mean": gm}, "arms": {}}
                for arm in ["M0"] + MEM + REF:
                    a = per_seed(n, s, arm, t)
                    if a is None:
                        continue
                    row["arms"][arm] = {"mean": smean(a), "vs_M0": cmp(a, m0) if arm != "M0" else None,
                                        "vs_gb": cmp(a, ga), "val": val(n, s, arm) if arm not in REF else None}
                c["tiers"][t] = row
            c["best_mem"], c["val_mem"] = pick(n, s, MEM)
            c["best_fixed"], _ = pick(n, s, FIXED)
            pars, gpuh = {}, {}
            for arm in ["M0"] + MEM:
                r = load(n, s, 0, arm)
                if r:
                    pars[arm] = r["info"]["params"]
                    w = [load(n, s, sd, arm) for sd in SEEDS]
                    if all(w):
                        gpuh[arm] = {"p1_s": float(np.mean([x["info"]["phase1_wall_s"] or 0 for x in w])),
                                     "p2_s": float(np.mean([x["info"]["phase2_wall_s"] or 0 for x in w]))}
                    if arm == "M4":
                        c["warp_s"] = [load(n, s, sd, "M4")["info"].get("warp_s") for sd in SEEDS if load(n, s, sd, "M4")]
            c["params"], c["wall"] = pars, gpuh
            A["cells"][cname(n, s)] = c
    # ------------------------------------------------------------- P-M1
    pm1 = {}
    for n in NODES:
        c = A["cells"][cname(n, "quad_tank")]
        b = c["best_mem"]
        e0, eb = upper_err(n, "quad_tank", "M0"), upper_err(n, "quad_tank", b) if b else None
        if e0 is None or eb is None:
            pm1[n] = None; continue
        r = eb[2] / e0[2] - 1
        pt, lo, hi = _t_interval(r)
        n3, n4 = float(eb[0].mean()), float(eb[1].mean())
        ok_a = bool(pt <= -0.30 and hi < 0)
        ok_b = bool(n3 <= 1.5 * CRB30[n][0] and n4 <= 1.5 * CRB30[n][1])
        allarms = {}
        for arm in ["M0"] + MEM:
            e = upper_err(n, "quad_tank", arm)
            if e is not None:
                allarms[arm] = {"n3": float(e[0].mean()), "n4": float(e[1].mean()), "e_up": float(e[2].mean())}
        pm1[n] = {"best": b, "rel_e_up": pt, "ci": [lo, hi], "n3": n3, "n4": n4, "crb30": CRB30[n],
                  "ratio3": n3 / CRB30[n][0], "ratio4": n4 / CRB30[n][1], "a": ok_a, "b": ok_b, "pass": ok_a and ok_b,
                  "m0": {"n3": float(e0[0].mean()), "n4": float(e0[1].mean())}, "all": allarms,
                  "gb": gb_states(n)}
    A["predictions"]["P-M1"] = {"cells": pm1, "PASS": all(v and v["pass"] for v in pm1.values())}
    # ------------------------------------------------------------- P-M2
    pm2, k = {}, 0
    for n in NODES:
        for s in STARVED:
            c = A["cells"][cname(n, s)]
            b = c["best_mem"]
            row = c["tiers"]["sequence_ood"]
            if not b or b not in row["arms"]:
                pm2[cname(n, s)] = None; continue
            e0, eb, eg = row["arms"]["M0"]["mean"], row["arms"][b]["mean"], row["gb"]["mean"]
            sig = row["arms"][b]["vs_M0"]["better"]
            if e0 > eg:
                g = (e0 - eb) / (e0 - eg); ok = bool(sig and g >= 0.5); kind = "M0 loses"
            else:
                g = None; ok = bool(sig and eb < eg); kind = "M0 already wins"
            pm2[cname(n, s)] = {"best": b, "e_M0": e0, "e_best": eb, "e_gb": eg, "gap_closure": g,
                                "vs_M0": row["arms"][b]["vs_M0"], "vs_gb": row["arms"][b]["vs_gb"], "kind": kind, "success": ok}
            k += ok
    A["predictions"]["P-M2"] = {"cells": pm2, "n_success": k, "PASS": k >= 3}
    # ------------------------------------------------------------- P-M3
    pm3, held, tot = {}, 0, 0
    for n in NODES:
        for s in RICH:
            row = A["cells"][cname(n, s)]["tiers"]["sequence_ood"]["arms"]
            for arm in MEM:
                if arm in row:
                    v = row[arm]["vs_M0"]
                    ok = bool(v["ci"][1] <= 0.10)
                    pm3[f"{cname(n, s)}:{arm}"] = {"rel": v["rel"], "ci": v["ci"], "ok": ok}
                    held += ok; tot += 1
    A["predictions"]["P-M3"] = {"comparisons": pm3, "held": held, "total": tot, "PASS": tot == 20 and held == tot}
    # ------------------------------------------------------------- P-M4
    pm4, ks, kr = {}, 0, 0
    for n in NODES:
        for s in SYSTEMS:
            c = A["cells"][cname(n, s)]
            bf = c["best_fixed"]
            if bf is None or per_seed(n, s, "M4", "sequence_ood") is None:
                pm4[cname(n, s)] = None; continue
            v = cmp(per_seed(n, s, "M4", "sequence_ood"), per_seed(n, s, bf, "sequence_ood"))
            ok = bool((not v["worse"]) and v["rel"] <= 0.05)
            pm4[cname(n, s)] = {"best_fixed": bf, "M4_vs_best_fixed": v, "ok": ok}
            if s in STARVED:
                ks += ok
            else:
                kr += ok
    A["predictions"]["P-M4"] = {"cells": pm4, "starved_ok": ks, "rich_ok": kr, "PASS": ks >= 3 and kr == 4}
    (ROOT / "analysis.json").write_text(json.dumps(A, indent=1, default=float))
    md = tables(A)
    (ROOT / "tables.md").write_text(md)
    print(md)


def f(x, p=4):
    return "-" if x is None else f"{x:.{p}f}"


def fr(v):
    if v is None:
        return "-"
    s = f"{100 * v['rel']:+.0f}% [{100 * v['ci'][0]:+.0f}, {100 * v['ci'][1]:+.0f}]"
    return f"**{s}**" if v["better"] else (f"*{s}*" if v["worse"] else s)


def tables(A):
    L = []
    L.append("Reproduction (max |M0 / stored A2 - 1| over seeds, sequence_ood): " +
             ", ".join(f"{k} {100 * v:.3f}%" for k, v in A["repro"].items()))
    for t in TIERS:
        L.append(f"\n### {t}: seed-mean NRMSE; vs M0 and vs best stored grey-box (paired t-CI over 3 seeds; bold = better, italic = worse)\n")
        L.append("| cell | grey-box | arm | NRMSE | val NRMSE | vs M0 | vs grey-box |")
        L.append("|---|---|---|---|---|---|---|")
        for cn, c in A["cells"].items():
            row = c["tiers"][t]
            for i, (arm, r) in enumerate(row["arms"].items()):
                tagb = " (best mem)" if arm == c["best_mem"] else ""
                L.append(f"| {cn if i == 0 else ''} | {row['gb']['name'].replace('greybox_', '') + ' ' + f(row['gb']['mean']) if i == 0 else ''} | "
                         f"{arm}{tagb} | {f(r['mean'])} | {f(r['val'])} | {fr(r['vs_M0'])} | {fr(r['vs_gb'])} |")
    P = A["predictions"]
    L.append("\n### P-M1 (quad_tank upper tanks, sequence_ood)\n")
    L.append("| scenario | best mem (val) | e_up change vs M0 | n3 / 1.5 CRB30 | n4 / 1.5 CRB30 | pass |")
    L.append("|---|---|---|---|---|---|")
    for n, v in P["P-M1"]["cells"].items():
        if v:
            L.append(f"| {n} | {v['best']} | {100 * v['rel_e_up']:+.1f}% [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] | "
                     f"{v['n3']:.4f} / {1.5 * v['crb30'][0]:.4f} | {v['n4']:.4f} / {1.5 * v['crb30'][1]:.4f} | {v['pass']} |")
    L.append(f"P-M1 PASS: {P['P-M1']['PASS']}")
    L.append("\n| scenario | arm | n3 | n4 | e_up |\n|---|---|---|---|---|")
    for n, v in P["P-M1"]["cells"].items():
        if v:
            for arm, q in v["all"].items():
                L.append(f"| {n} | {arm} | {q['n3']:.4f} | {q['n4']:.4f} | {q['e_up']:.4f} |")
            L.append(f"| {n} | grey-box | {v['gb'].get('nrmse_x3', float('nan')):.4f} | {v['gb'].get('nrmse_x4', float('nan')):.4f} | |")
            L.append(f"| {n} | CRB 10 s / 30 s | {CRB10[n][0]:.4f} / {CRB30[n][0]:.4f} | {CRB10[n][1]:.4f} / {CRB30[n][1]:.4f} | |")
    L.append("\n### P-M2 (window-starved cells)\n")
    L.append("| cell | best mem | e_M0 | e_best | e_gb | gap closure | best vs M0 | best vs gb | success |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cn, v in P["P-M2"]["cells"].items():
        if v:
            L.append(f"| {cn} | {v['best']} | {f(v['e_M0'])} | {f(v['e_best'])} | {f(v['e_gb'])} | "
                     f"{f(v['gap_closure'], 2) if v['gap_closure'] is not None else v['kind']} | {fr(v['vs_M0'])} | {fr(v['vs_gb'])} | {v['success']} |")
    L.append(f"P-M2: {P['P-M2']['n_success']}/4 -> PASS {P['P-M2']['PASS']}")
    L.append(f"\n### P-M3 (rich cells): {P['P-M3']['held']}/{P['P-M3']['total']} comparisons with CI upper <= +10% -> PASS {P['P-M3']['PASS']}\n")
    L.append("| comparison | rel vs M0 | CI | ok |\n|---|---|---|---|")
    for k, v in P["P-M3"]["comparisons"].items():
        L.append(f"| {k} | {100 * v['rel']:+.1f}% | [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] | {v['ok']} |")
    L.append(f"\n### P-M4: starved {P['P-M4']['starved_ok']}/4, rich {P['P-M4']['rich_ok']}/4 -> PASS {P['P-M4']['PASS']}\n")
    L.append("| cell | best fixed (val) | M4 vs best fixed | ok |\n|---|---|---|---|")
    for cn, v in P["P-M4"]["cells"].items():
        if v:
            L.append(f"| {cn} | {v['best_fixed']} | {fr(v['M4_vs_best_fixed'])} | {v['ok']} |")
    L.append("\n### Parameters (phase-2 trainable total, seed 0) and mean wall-clock per seed (s)\n")
    L.append("| cell | " + " | ".join(["M0"] + MEM) + " |\n|---|" + "---|" * 6)
    for cn, c in A["cells"].items():
        L.append(f"| {cn} | " + " | ".join(
            (f"{c['params'][a]['phase2_trainable_total']}" if a in c["params"] else "-") +
            (f" ({c['wall'][a]['p1_s']:.0f}+{c['wall'][a]['p2_s']:.0f})" if a in c["wall"] else "") for a in ["M0"] + MEM) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
