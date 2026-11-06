"""Re-reading of the bench_v2 runs under the author decisions of 2026-11-06 (no new runs):

* default = hk_base (A3 + mixed); C5 (`hyperkkl`) is reported as an ablation;
* duffing is excluded (12 driven cells);
* the conditioning variants latent / concat / film and the untrained phase-1 maps (auto) are not reported.

Same selection, scorer and statistics as analyze.py (PREREG section 4), with the reference switched to hk_base.
Writes results_default.json and logs/tables_default.md; results.json and logs/tables.md are not touched.
python code/analyze_default.py
"""
from __future__ import annotations

import json

import analyze as A

EXCLUDE_CELLS = {("driven", "duffing")}
EXCLUDE_METHODS = {"latent", "concat", "film", "auto"}


def main():
    load0 = A.load

    def load():
        return {k: v for k, v in load0().items()
                if (k[0], k[1]) not in EXCLUDE_CELLS and k[3] not in EXCLUDE_METHODS}

    A.load = load
    A.REF = "hk_base"
    out_json, out_md = A.HERE / "results.json", A.HERE / "logs" / "tables.md"
    keep_json, keep_md = out_json.read_bytes(), out_md.read_bytes()
    try:
        A.analyse()
        (A.HERE / "results_default.json").write_bytes(out_json.read_bytes())
        md = out_md.read_text().replace("hyperkkl rank", "hk_base rank").replace("paired vs hyperkkl", "paired vs hk_base")
        (A.HERE / "logs" / "tables_default.md").write_text(
            "<!-- analyze_default.py: default = hk_base; duffing, latent, concat, film, auto excluded -->\n" + md)
    finally:
        out_json.write_bytes(keep_json)
        out_md.write_bytes(keep_md)
    r = json.loads((A.HERE / "results_default.json").read_text())
    for fam, f in r["families"].items():
        print(fam, "cells", len(f["best_per_cell"]))


if __name__ == "__main__":
    main()
