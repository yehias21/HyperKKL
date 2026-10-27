"""Mechanical verdict for the truncated-history nodes (experiments/memory/PREREG.md).

    .venv/bin/python experiments/memory/verdict.py            # -> experiments/memory/RESULTS.md

Read-only on experiments/; applies analysis.evidence.analyse_trunc (the pre-registered rule) to
C2-trunc-test (decision) and C2-trunc-dev (in-sample) and writes the arm-by-arm tables.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import rules as R                      # noqa: E402
from analysis.evidence import TRUNC_NODES, analyse_trunc, fmt_p   # noqa: E402
from analysis.store import Store                     # noqa: E402

OUT = Path(__file__).resolve().parent / "RESULTS.md"


def arm_table(cur: dict) -> list:
    """One markdown table per system: rows = arms H, columns = (method, tier) relative change vs inf."""
    cells = cur["cells"]
    keys = sorted(cells)
    Hs = sorted({H for c in cells.values() for H in c["H"]})
    L = ["| H [samples] | " + " | ".join(f"{m}/{t[:3]}" for m, t in keys) + " |",
         "|---|" + "---|" * len(keys)]
    for H in Hs:
        row = []
        for k in keys:
            c = cells[k]
            row.append(fmt_p(c["rel"].get(H)) + (f" ({c['mean'][H]:.4f})" if H in c["mean"] else ""))
        L.append(f"| {H} | " + " | ".join(row) + " |")
    L.append("| inf | " + " | ".join(f"NRMSE {cells[k]['mean_inf']:.4f}" for k in keys) + " |")
    return L


def main():
    store = Store(use_cache=False)
    L = [f"# C2-trunc results (generated {_dt.datetime.now():%Y-%m-%d %H:%M} local)", "",
         "Rule and constants: `PREREG.md` (2026-10-25 23:50 CET). Mechanical: `analysis.evidence.analyse_trunc`.",
         "Cells: paired relative change of the arm against the same method's untruncated (inf) arm, Student-t over "
         "seeds, 95% CI; `*` = CI excludes 0; the arm's mean NRMSE in parentheses.", ""]
    for nid in TRUNC_NODES:
        r = analyse_trunc(store, nid)
        L += [f"## {nid} ({r['split']}) - {r['label']}", "", f"- rule: {r['rule']}", f"- verdict: {r['verdict']}",
              f"- progress: {r['progress']}"]
        L += [f"  - {e}" for e in r["evidence"]]
        L.append("")
        for s, cur in sorted(r["data"].get("curves", {}).items()):
            L += [f"### {s}: L* = {cur['Lstar']} (full {cur['Lfull']}), settle {cur['settle']}, "
                  f"new phase-1 l_dec_q95 = {cur['l_dec_new']:.3g} (pre-registered {cur['l_prereg']})", ""]
            L += arm_table(cur) + [""]
    if store.problems:
        L += ["## Data problems", ""] + [f"- {n}: {m}" for n, m in store.problems]
    reading = OUT.parent / "READING.md"            # hand-written interpretation, kept across regenerations
    if reading.exists():
        L += ["", reading.read_text().rstrip()]
    OUT.write_text("\n".join(L) + "\n")
    print(OUT)
    for nid in TRUNC_NODES:
        r = analyse_trunc(store, nid)
        print(f"{nid}: {r['label']} - {r['verdict']}")


if __name__ == "__main__":
    main()
