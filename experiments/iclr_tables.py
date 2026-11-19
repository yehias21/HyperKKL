"""Tables 1-2 of the ICLR workshop paper from an experiment node (default: ICLR).

    python experiments/iclr_tables.py [NODE] [--metric nrmse|rmse] [--condition clean]

Columns: the paper's four input regimes (per-family tiers fam_zero, fam_constant, fam_sinusoid,
fam_square), plus the protocol tiers if --all-tiers. Entry: mean +- std over training seeds of
the per-seed mean over test episodes. rmse is the state-vector RMSE of an episode,
sqrt(mean_j rmse_xj^2), in physical units (the paper's original metric); nrmse divides each
state's error by its reference std. Bold: lowest mean among the learned KKL observers; a down
arrow marks a mean below the autonomous observer's whose seed-paired difference is negative for
every seed. Model-based references (EKF, kkl_mb) are listed below a rule and never bolded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TREE = Path(__file__).resolve().parent / "tree"
SYSTEMS = [("iclr_duffing", "Duffing"), ("iclr_vdp", "Van der Pol"),
           ("iclr_rossler", "R\\\"ossler"), ("iclr_lorenz", "Lorenz")]
COLS = [("fam_zero", "Zero"), ("fam_constant", "Const"), ("fam_sinusoid", "Sin"), ("fam_square", "Sqr")]
EXTRA = [("id", "ID"), ("shape_ood", "Shape"), ("sequence_ood", "Seq"), ("amp2", "2x"), ("amp4", "4x")]
LEARNED = [("autonomous", "Autonomous"), ("curriculum", "Curriculum"),
           ("curriculum[cur_levels=1]", "Fine-tuning (no curriculum)"),
           ("static", "Static HyperKKL"), ("dynamic", "Dynamic HyperKKL")]
REFS = [("kkl_mb", "Model-based KKL"), ("ekf", "EKF")]


def episode_metric(m: dict, metric: str) -> np.ndarray:
    if metric == "nrmse":
        return np.asarray(m["nrmse"], float)
    keys = sorted(k for k in m if k.startswith("rmse_x"))
    return np.sqrt(np.mean([np.asarray(m[k], float) ** 2 for k in keys], 0))


def collect(node: str, system: str, metric: str, condition: str) -> dict:
    """{(tier, method): array over seeds of the episode-mean metric}"""
    out = {}
    for f in sorted((TREE / node / "runs" / system).glob("seed*/results.json")):
        res = json.loads(f.read_text())["results"]
        for tier, conds in res.items():
            if condition not in conds:
                continue
            for meth, m in conds[condition]["methods"].items():
                out.setdefault((tier, meth), []).append(float(np.mean(episode_metric(m, metric))))
    return {k: np.asarray(v) for k, v in out.items()}


def fmt(v: np.ndarray, bold: bool, arrow: bool) -> str:
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    s = f"{v.mean():#.3g}" + (f"\\,$\\pm$\\,{sd:.1g}" if sd > 1e-6 * abs(v.mean()) else "")  # EKF: no seed spread
    s = f"\\textbf{{{s}}}" if bold else s
    return s + ("$\\downarrow$" if arrow else "")


def table(node: str, systems, cols, metric: str, condition: str) -> str:
    data = {s: collect(node, s, metric, condition) for s, _ in systems}
    n_seeds = max((len(v) for d in data.values() for v in d.values()), default=0)
    head = " & ".join(f"\\multicolumn{{{len(cols)}}}{{c}}{{{lab}}}" for _, lab in systems)
    sub = " & ".join(lab for _ in systems for _, lab in cols)
    lines = ["\\begin{tabular}{l" + "c" * (len(cols) * len(systems)) + "}", "\\toprule",
             f"Method & {head} \\\\", f" & {sub} \\\\", "\\midrule"]
    for group, rule in ((LEARNED, True), (REFS, False)):
        for key, lab in group:
            cells = []
            for s, _ in systems:
                d = data[s]
                for tier, _ in cols:
                    v = d.get((tier, key))
                    if v is None:
                        cells.append("--")
                        continue
                    learned = [d[(tier, k)].mean() for k, _ in LEARNED if (tier, k) in d]
                    bold = rule and np.isclose(v.mean(), min(learned))
                    a = d.get((tier, "autonomous"))
                    arrow = (rule and key != "autonomous" and a is not None and len(a) == len(v)
                             and bool(np.all(v - a < 0)))
                    cells.append(fmt(v, bold, arrow))
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if rule:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"% {metric.upper()}, condition {condition}, {n_seeds} seeds, node {node}"]
    return "\n".join(lines)


def ablation(node: str, tier: str, metric: str, condition: str) -> str:
    """One row per method of an ablation node, one column per system (mean +- std over seeds)."""
    systems = [(s, lab) for s, lab in SYSTEMS + [("highway_traffic", "Highway")]
               if (TREE / node / "runs" / s).exists()]
    data = {s: collect(node, s, metric, condition) for s, _ in systems}
    methods = list(dict.fromkeys(m for d in data.values() for (t, m) in d if t == tier))
    lines = ["\\begin{tabular}{l" + "c" * len(systems) + "}", "\\toprule",
             "Method & " + " & ".join(lab for _, lab in systems) + " \\\\", "\\midrule"]
    for m in methods:
        cells = [fmt(data[s][(tier, m)], False, False) if (tier, m) in data[s] else "--" for s, _ in systems]
        lines.append(m.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"% {metric.upper()}, tier {tier}, node {node}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("node", nargs="?", default="ICLR")
    ap.add_argument("--metric", default="nrmse", choices=["nrmse", "rmse"])
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--all-tiers", action="store_true")
    ap.add_argument("--ablation", metavar="TIER", help="ablation table of NODE on TIER (e.g. id)")
    a = ap.parse_args()
    if a.ablation:
        print(ablation(a.node, a.ablation, a.metric, a.condition))
        return
    cols = COLS + (EXTRA if a.all_tiers else [])
    for pair in (SYSTEMS[:2], SYSTEMS[2:]):
        print(table(a.node, pair, cols, a.metric, a.condition), end="\n\n")


if __name__ == "__main__":
    main()
