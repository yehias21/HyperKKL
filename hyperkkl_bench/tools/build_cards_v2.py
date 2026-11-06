"""Plant cards of the v2 families from the export metadata (+ the v2 reference results when present).

    python build_cards_v2.py [--data DIR] [--out DIR]     -> <out>/<family>__<plant>.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "experiments" / "bench_v2"   # the v2 study


def card(m, res):
    fam, p = m["family"], m["plant"]
    L = [f"# {fam}/{p}", "", f"system `{m['system']}` · n_x {m['n_x']} · measured {m['sensors']} · unmeasured {m['unmeasured']}"
         f" · dt {m['dt']:.4g} · N {m['N']} · T {m['T']:.4g} · t_ref {m['t_ref']}", "",
         f"Unmodelled term (partial / realistic levels): {m.get('unmodelled_term')}", ""]
    if fam == "driven":
        i = m["input"]
        L += ["## Input", "", f"* unit: {i['unit']}; physical = (offset + u) x scale with (offset, scale, name) = {i['physical']}",
              f"* applied box [{i['box'][0]}, {i['box'][1]}]; actuator {i['actuator']}",
              f"* training operators: {', '.join(i['train_operators'])}",
              f"* OOD operators (test_op_ood): {', '.join(i['ood_operators'])}",
              f"* input-record preset (input_realistic): {i['record_preset']}"]
        if i.get("uncited"):
            L.append(f"* **uncited**: {i['uncited']} (benchmark choice; no paper attached)")
        L.append("")
    else:
        L += ["## Initial-condition regions", ""] + [f"* {k}: {v}" for k, v in m["ic_regions"].items()] + [""]
    L += ["## Splits per data variant (episodes)", "", "| variant | " + " | ".join(next(iter(m["variants"].values()))["files"]) + " |",
          "|---|" + "---|" * len(next(iter(m["variants"].values()))["files"])]
    for v, d in m["variants"].items():
        L.append(f"| {v} | " + " | ".join(str(f["n"]) for f in d["files"].values()) + " |")
    L.append("")
    key = f"{fam}/{p}/realistic" if fam == "driven" else f"{fam}/{p}/white"
    c = (res or {}).get("cells", {}).get(key)
    if c:
        L += [f"## v2 reference results ({key}, primary block, settled unmeasured NRMSE, seed mean)", "",
              "| rank | method | mean |", "|---|---|---|"]
        for k, o in enumerate(c["order"][:10]):
            L.append(f"| {k + 1} | {o} | {c['summary'][o]['mean']:.4g} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data_v2"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "cards" / "v2"))
    a = ap.parse_args()
    res = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.data, "metadata").glob("*/*.json")):
        m = json.loads(f.read_text())
        (out / f"{m['family']}__{m['plant']}.md").write_text(card(m, res))
        print("card", m["family"], m["plant"])


if __name__ == "__main__":
    main()
