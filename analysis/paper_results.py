"""The paper's TRANSPARENT results layer: experiment tree -> figures, tables, numbers, report.

    .venv/bin/python -m analysis.paper_results            # everything -> paper/generated/
    .venv/bin/python -m analysis.paper_results --no-pdf   # skip the pdflatex previews
    .venv/bin/python -m analysis.paper_results --no-cache # re-parse every results.json

Read-only on experiments/ (runs, queue, cache are never written). Safe to run while the GPU
wave is going: unfinished nodes come out PENDING / INCONCLUSIVE, never silently dropped.
Outputs (paper/generated/):
    fig1_c1_where, fig2a_c2_floor, fig2b_c2_probe, fig2c_c2_window, fig3_c3_crossover[_id|_test],
    fig4_scorecard                                 (.pdf vector + .png)
    tab1_c1, tab1b_c1_id, tab2_c3, tab2_c3_full, tab2b_c3_id, tab3_scorecard_realistic,
    tab4_decisions, tab5_systems, tab6_capacity    (.tex booktabs + .pdf preview)
    numbers.tex, RESULTS.md, outcomes.json, run_log.txt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import figures as F
from . import tables as T
from .capacity import capacity
from .evidence import analyse_all_both
from .report import collect_numbers, write_results_md
from .store import DEV, ROOT, TEST, Store

OUT = ROOT / "paper" / "generated"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o if isinstance(o, (str, int, bool, type(None))) else str(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def step(name, fn):
        t0 = time.time()
        try:
            v = fn()
            log.append(f"[ok]   {name} ({time.time() - t0:.0f}s)")
            print(log[-1], flush=True)
            return v
        except Exception:
            log.append(f"[FAIL] {name}\n{traceback.format_exc()}")
            print(log[-1], flush=True)
            return None

    store = Store(use_cache=not a.no_cache)
    res = step("evidence (all nodes)", lambda: analyse_all_both(store))
    cap = step("capacity", capacity) or {}
    scen, _ = F.scenario_order(DEV)
    figs = {}
    figs["F1"] = step("F1", lambda: F.fig1(res, cap, out, store))
    figs["F1main"] = step("F1 main", lambda: F.fig1(res, cap, out, store, panel_e=False))   # paper, panels (a)-(d)
    figs["F2a"] = step("F2a", lambda: F.fig2a(res, out))
    figs["F2b"] = step("F2b", lambda: F.fig2b(store, res, out))
    figs["F2c"] = step("F2c", lambda: F.fig2c(res, out))
    figs["F3"] = step("F3", lambda: F.fig3(store, res, out, "sequence_ood", DEV, "fig3_c3_crossover", scen))
    figs["F3main"] = step("F3 main", lambda: F.fig3(store, res, out, "sequence_ood", DEV, "fig3_c3_crossover_main",
                                                   scen, show_model_error=False))      # paper, 2x3, no model-error row
    figs["F3id"] = step("F3 id", lambda: F.fig3(store, res, out, "id", DEV, "fig3_c3_crossover_id", scen))
    test_scen = [("realistic (TEST-C3)", "TEST-C3", "clean", "partial", 0.3)]
    figs["F3test"] = step("F3 test", lambda: F.fig3(store, res, out, "sequence_ood", TEST, "fig3_c3_crossover_test",
                                                   test_scen, show_model_error=False, only_data=True))
    figs["F4"] = step("F4", lambda: F.fig4(store, res, out))
    tabs = {}
    tabs["T1"] = step("T1", lambda: T.t1(res, out, store=store))
    tabs["T1b"] = step("T1b", lambda: T.t1(res, out, ("id",), "tab1b_c1_id", store=store))
    tabs["T1c"] = step("T1c dense ablation", lambda: T.t1_ablation(res, out))
    tabs["Trank"] = step("T rank ablation", lambda: T.t_rank(res, store, out))
    tabs["Texcl"] = step("T excluded (appendix)", lambda: T.t_excluded(store, res, out, scen))
    t2 = step("T2", lambda: T.t2(store, res, out, scen)) or []
    t2b = step("T2b", lambda: T.t2(store, res, out, scen, "id", "tab2b_c3_id")) or []
    tabs["T3"] = step("T3", lambda: T.t3(store, out))
    treg = step("T-regimes", lambda: T.t_regimes(res, out)) or []
    tabs["T4"] = step("T4", lambda: T.t4(res, out))
    tabs["T5"] = step("T5", lambda: T.t5(out))
    tabs["T6"] = step("T6", lambda: T.t6(cap, store, out))
    nums = step("numbers.tex", lambda: collect_numbers(res, store, scen))
    if nums:
        nums.write(out / "numbers.tex")
    tex = [p for p in list(tabs.values()) + t2 + t2b + treg if p]
    pdf = {} if a.no_pdf else (step("pdflatex previews", lambda: T.compile_pdfs(tex, out, out / "numbers.tex")) or {})
    step("RESULTS.md", lambda: write_results_md(res, store, out / "RESULTS.md", figs, tabs, pdf, scen))
    outcomes = {k: {kk: v.get(kk) for kk in ("node", "contribution", "split", "label", "shown", "provisional", "reread", "original_label", "original_verdict", "single_label", "single_verdict", "excluded_reading", "exclusion_note", "verdict", "rule", "question",
                                              "progress", "evidence")} for k, v in (res or {}).items()}
    (out / "outcomes.json").write_text(json.dumps(_jsonable({"generated": f"{_dt.datetime.now():%Y-%m-%d %H:%M}",
                                                             "outcomes": outcomes, "problems": store.problems,
                                                             "pdf": pdf}), indent=1))
    try:
        from .make_history import main as history
        step("history", lambda: history([]))
    except ImportError:
        pass
    (out / "run_log.txt").write_text("\n".join(log) + "\n\nPDF previews:\n" + "\n".join(f"{k}: {v}" for k, v in pdf.items()))
    print("\n".join(f"{k}: {v}" for k, v in pdf.items()))
    print(f"outputs in {out}")
    return 0 if not any(l.startswith("[FAIL]") for l in log) else 1


if __name__ == "__main__":
    raise SystemExit(main())
