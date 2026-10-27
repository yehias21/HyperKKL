"""Analysis per PREREG.md. Writes analysis.json and tables.md next to RESULTS.md.

    python analyze.py [--select-only]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
NODES = ["C3-partial", "C3-mismatch30"]
PLANTS = ["battery", "bioreactor", "lorenz", "quad_tank", "vdp"]
SEEDS = [0, 1, 2]
GRID = ["0.1", "1", "10"]
TIERS = ["sequence_ood", "shape_ood", "id"]
NICK = {"C3-partial": "partial", "C3-mismatch30": "mm30"}


def load(node, plant, seed, arm):
    f = RUNS / node / plant / f"seed{seed}" / f"{arm}.json"
    return json.load(open(f)) if f.exists() else None


def nrmse(d, tier):
    return float(np.mean(d["results"][tier]["nrmse"]))


def select():
    out = {}
    for n in NODES:
        for p in PLANTS:
            rows = []
            for g in GRID:
                d = load(n, p, 0, f"TVr{g}")
                if d is None:
                    rows = None
                    break
                rows.append((d["info"]["val"]["val_nrmse"], float(g), g))
            if rows:
                best = min(rows, key=lambda r: (round(r[0], 12), r[1]))
                out[f"{n}/{p}"] = {"chosen": best[2], "val": {r[2]: r[0] for r in rows}}
    return out


def paired(on, off):
    """log-ratio paired t over seeds -> (mean %, lo %, hi %, p, n)."""
    d = np.log(np.asarray(on) / np.asarray(off))
    n = len(d)
    if n < 2:
        return None
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return (math.expm1(m) * 100, math.expm1(m) * 100, math.expm1(m) * 100, 0.0 if m != 0 else 1.0, n)
    h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    p = 2 * stats.t.sf(abs(m) / (s / math.sqrt(n)), n - 1)
    return (math.expm1(m) * 100, math.expm1(m - h) * 100, math.expm1(m + h) * 100, float(p), n)


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def compare(arm_on, arm_off, sel=None):
    res = {}
    for n in NODES:
        for p in PLANTS:
            a_on = arm_on if sel is None else (f"TVr{sel[f'{n}/{p}']['chosen']}" if f"{n}/{p}" in sel else None)
            if a_on is None:
                continue
            row = {"arm_on": a_on, "arm_off": arm_off}
            for t in TIERS:
                on, off = [], []
                for s in SEEDS:
                    x, y = load(n, p, s, a_on), load(n, p, s, arm_off)
                    if x is not None and y is not None:
                        on.append(nrmse(x, t)); off.append(nrmse(y, t))
                row[t] = {"on": on, "off": off, "ci": paired(on, off) if len(on) >= 2 else None}
            res[f"{n}/{p}"] = row
    ps = [r["sequence_ood"]["ci"][3] for r in res.values() if r["sequence_ood"]["ci"]]
    keys = [k for k, r in res.items() if r["sequence_ood"]["ci"]]
    for k, a in zip(keys, holm(ps) if ps else []):
        res[k]["holm_p_seq"] = a
    return res


def verdict(res):
    out = {}
    for n in NODES:
        better = worse = 0
        cells = [res.get(f"{n}/{p}") for p in PLANTS]
        complete = all(c and c["sequence_ood"]["ci"] and c["sequence_ood"]["ci"][4] == 3 for c in cells)
        for c in cells:
            if c and c["sequence_ood"]["ci"]:
                m, lo, hi = c["sequence_ood"]["ci"][:3]
                better += hi < 0
                worse += lo > 0
        holm_b = sum(1 for p in PLANTS if res.get(f"{n}/{p}", {}).get("holm_p_seq", 1) < 0.05
                     and res[f"{n}/{p}"]["sequence_ood"]["ci"][0] < 0)
        out[n] = {"better": int(better), "worse": int(worse), "helps": bool(better >= 3 and worse == 0),
                  "complete": complete, "holm_better": holm_b}
    hn = [out[n]["helps"] for n in NODES]
    out["overall"] = "helps" if all(hn) else ("helps in one regime" if any(hn) else "does not help")
    return out


def fmt(ci):
    if ci is None:
        return "n/a"
    m, lo, hi, p, n = ci
    s = f"{m:+.0f}% [{lo:+.0f}, {hi:+.0f}]"
    if hi < 0:
        return f"**{s}**"
    if lo > 0:
        return f"*{s}*"
    return s


def table(res, title):
    L = [f"### {title}", "", "| tier | " + " | ".join(f"{p}/{NICK[n]}" for n in NODES for p in PLANTS) + " |",
         "|---|" + "---|" * (len(NODES) * len(PLANTS))]
    for t in TIERS:
        L.append(f"| {t} | " + " | ".join(fmt(res.get(f"{n}/{p}", {}).get(t, {}).get("ci")) for n in NODES
                                          for p in PLANTS) + " |")
    L.append("| Holm p (seq) | " + " | ".join(
        (f"{res[f'{n}/{p}']['holm_p_seq']:.2f}" if 'holm_p_seq' in res.get(f'{n}/{p}', {}) else "n/a")
        for n in NODES for p in PLANTS) + " |")
    L.append("| arm on (seq NRMSE on/off, seed means) | " + " | ".join(
        (f"{res[f'{n}/{p}']['arm_on']} {np.mean(res[f'{n}/{p}']['sequence_ood']['on']):.4f}/"
         f"{np.mean(res[f'{n}/{p}']['sequence_ood']['off']):.4f}" if f"{n}/{p}" in res and res[f'{n}/{p}']['sequence_ood']['on'] else "n/a")
        for n in NODES for p in PLANTS) + " |")
    L.append("")
    L.append("Effect = NRMSE_on / NRMSE_off - 1, paired t 95% CI on the log ratio over seeds (df = 2); "
             "**bold** = on significantly better, *italic* = significantly worse.")
    L.append("")
    return "\n".join(L)


def diag_table():
    L = ["### TV residual diagnostic (validation, seed means; units sigma_x * alpha)", "",
         "| cell | floor (perfect decoder, modeller f) | OFF residual | TV chosen residual | TOr1 residual (vs true f) |",
         "|---|---|---|---|---|"]
    sel = select()
    for n in NODES:
        for p in PLANTS:
            def g(arm, key):
                v = [load(n, p, s, arm) for s in SEEDS]
                v = [x["info"]["tv_diag"].get(key) for x in v if x is not None and "tv_diag" in x["info"]]
                v = [x for x in v if x is not None]
                return f"{np.mean(v):.3g}" if v else "n/a"
            ch = f"TVr{sel[f'{n}/{p}']['chosen']}" if f"{n}/{p}" in sel else None
            L.append(f"| {p}/{NICK[n]} | {g('OFF', 'tv_floor_rms')} | {g('OFF', 'tv_rms')} | "
                     f"{g(ch, 'tv_rms') if ch else 'n/a'} | {g('TOr1', 'tv_rms')} |")
    return "\n".join(L) + "\n"


def main():
    sel = select()
    if "--select-only" in sys.argv:
        print(json.dumps(sel, indent=1))
        return
    tv = compare(None, "OFF", sel)
    p1 = compare("OFF", "P1OFF")
    to = compare("TOr1", "OFF")
    to_tv = compare("TOr1", "TVr1")
    tv1 = compare("TVr1", "OFF")
    an = {"selection": sel, "tv_vs_off": tv, "pde_on_vs_off": p1, "oracle_vs_off": to,
          "oracle_vs_tvr1": to_tv, "tvr1_vs_off": tv1,
          "verdict_tv": verdict(tv), "verdict_pde": verdict(p1), "verdict_oracle": verdict(to)}
    (ROOT / "analysis.json").write_text(json.dumps(an, indent=1))
    md = [table(tv, "Test 1 (confirmatory): TV loss ON (chosen lambda~) vs OFF"),
          "verdict: " + json.dumps(an["verdict_tv"]), "",
          table(p1, "Test 2 (confirmatory): phase-1 PDE ON (A4+D3) vs OFF (P1OFF)"),
          "verdict: " + json.dumps(an["verdict_pde"]), "",
          table(to, "Exploratory: TV with the ORACLE (true) f, lambda~ = 1, vs OFF"),
          "verdict (same rule, exploratory): " + json.dumps(an["verdict_oracle"]), "",
          table(to_tv, "Exploratory: oracle-f TV vs modeller-f TV at lambda~ = 1"), "",
          table(tv1, "Supplementary: TVr1 (fixed lambda~ = 1) vs OFF"), "",
          "### lambda~ selection (seed-0 validation NRMSE)", "",
          "| cell | " + " | ".join(f"lambda~={g}" for g in GRID) + " | chosen |", "|---|---|---|---|---|"]
    for k, v in sel.items():
        md.append(f"| {k} | " + " | ".join(f"{v['val'][g]:.5f}" for g in GRID) + f" | {v['chosen']} |")
    md += ["", diag_table()]
    (ROOT / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
