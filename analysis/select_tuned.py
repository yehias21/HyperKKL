"""C1 per-variant tuning: the MECHANICAL selection (experiments/tuning/PROTOCOL.md, section 3).

    .venv/bin/python -m analysis.select_tuned            # -> experiments/tuning/selection.json
    .venv/bin/python -m analysis.select_tuned --check    # print the table, write nothing

Reads only the selection nodes' results.json (seed 0, the 6 dev systems, eval.tiers = []):
the score of a configuration on a system is `train.<method>.val_nrmse`, the deployed observer's
settled NRMSE on the phase-2 validation episodes (job.validation_nrmse). Per variant:
    score(config)  = arithmetic mean over the 6 dev systems (non-finite = +inf)
    selected       = argmin; if it is not the default and score_best > 0.995 x score_default,
                     the default is selected (ties go to the default)
    refuses        if any (config, system) result of that variant is missing, or a run did
                     evaluate tiers (the selection must not see evaluation data)
Reported, not used: geometric mean, mean rank over systems, the phase-2 validation loss.

DESIGN below is the single definition of the design spaces; experiments/make_tree_tuning.py
builds the nodes from it, and the method strings of C1-tuned come from selection.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyperkkl import tree as Tr          # noqa: E402

OUT = ROOT / "experiments" / "tuning" / "selection.json"
LRS = (0.0005, 0.001, 0.002, 0.004)
# PROTOCOL.md section 9: equal extensions of the lr grid for every variant (pre = the pre-registered grid)
GRIDS = {"pre": LRS, "ext1": (0.008, 0.016), "ext1b": (0.032,)}
# section 10: ext2 = ctx_rms {0.3, 1} x lr {lr*/2, lr*, 2 lr*} at the ext1(b)-selected knob, per variant
CTX_RMS = (0.3, 1.0)
EXT2_DESIGN = ROOT / "experiments" / "tuning" / "ext2_design.json"
DEFAULT_LR = 0.002
TIE = 0.005                              # 0.5% relative: within it, the default wins
# the 6 dev systems of the pre-registration (PROTOCOL.md section 3). FROZEN: independent of later
# post-hoc exclusions in analysis/store.py (store.DEV dropped duffing on 2026-10-26 for the headline reading)
TUNE_SYSTEMS = ("battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp")
SEED = 0
# variant -> (knob key, [(knob value or None = default/absent, node suffix)], default value label)
DESIGN = {
    "latent": ("latent_depth", [(None, "d2"), (3, "d3")]),
    "concat": ("concat_depth", [(None, "d2"), (3, "d3")]),
    "film": ("film_depth", [(None, "d1"), (2, "d2")]),
    "lora": ("rank", [(None, "r4"), (16, "r16")]),
    "hyper": ("gen_layers", [(None, "all"), (2, "gen2")]),
}
DEFAULT_KNOB = {"latent": 2, "concat": 2, "film": 1, "lora": 4, "hyper": "all"}


def _lr_str(lr: float) -> str:
    return f"{lr:.4f}".rstrip("0")


def method_string(kind: str, lr: float, knob_value=None, ctx_rms=None) -> str:
    """'hyper[lr=0.001,gen_layers=2]'. The default knob is left out, so the default config
    has the cache key of the plain C1-where method."""
    key = DESIGN[kind][0]
    kv = [f"lr={_lr_str(lr)}"] + ([f"{key}={knob_value}"] if knob_value is not None else []) + \
        ([f"ctx_rms={ctx_rms:g}"] if ctx_rms is not None else [])
    return f"{kind}[{','.join(kv)}]"


def _rms_suf(r):
    return "rms" + f"{r:g}".replace(".", "")


def _ext2_design() -> dict:
    return json.loads(EXT2_DESIGN.read_text())["design"] if EXT2_DESIGN.exists() else {}


def write_ext2_design(sel_file) -> dict:
    """Section 10: mechanical ext2 design from the ext1(b) selection file."""
    sel = json.loads(Path(sel_file).read_text())
    des = {}
    for kind, v in sel["variants"].items():
        lr = float(v["lr"])
        kv = None if v["knob_value"] == DEFAULT_KNOB[kind] else v["knob_value"]
        des[kind] = {"lrs": [lr / 2, lr, 2 * lr], "knob_value": kv, "from": v["selected"]}
    EXT2_DESIGN.write_text(json.dumps({"generated": f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                                       "from_selection": str(Path(sel_file).name), "ctx_rms": list(CTX_RMS),
                                       "design": des}, indent=1))
    return des


def _node_id(kind, suf, stage):
    return f"C1-tune-{kind}-{suf}" + ("" if stage == "pre" else f"-{stage}")


def select_nodes(stage: str = "pre") -> dict:
    """{node id: {'kind', 'knob', 'value', 'methods': [method strings]}} of one grid stage"""
    out = {}
    if stage == "ext2":
        for kind, d in _ext2_design().items():
            for r in CTX_RMS:
                out[f"C1-tune-{kind}-{_rms_suf(r)}-ext2"] = {
                    "kind": kind, "knob": DESIGN[kind][0], "value": d["knob_value"], "ctx_rms": r,
                    "methods": [method_string(kind, lr, d["knob_value"], r) for lr in d["lrs"]]}
        return out
    for kind, (key, vals) in DESIGN.items():
        for v, suf in vals:
            out[_node_id(kind, suf, stage)] = {"kind": kind, "knob": key, "value": v,
                                               "methods": [method_string(kind, lr, v) for lr in GRIDS[stage]]}
    return out


def configs(kind: str, stages=("pre",)) -> list:
    key, vals = DESIGN[kind]
    out = [{"node": _node_id(kind, suf, st), "method": method_string(kind, lr, v), "lr": lr, "stage": st,
            "knob": key, "knob_value": DEFAULT_KNOB[kind] if v is None else v, "ctx_rms": None,
            "default": v is None and lr == DEFAULT_LR}
           for st in stages if st != "ext2" for v, suf in vals for lr in GRIDS[st]]
    if "ext2" in stages:
        d = _ext2_design()[kind]
        v = d["knob_value"]
        out += [{"node": f"C1-tune-{kind}-{_rms_suf(r)}-ext2", "method": method_string(kind, lr, v, r), "lr": lr,
                 "stage": "ext2", "knob": key, "knob_value": DEFAULT_KNOB[kind] if v is None else v, "ctx_rms": r,
                 "default": False} for r in CTX_RMS for lr in d["lrs"]]
    return out


def _finite(x) -> float:
    return float(x) if x is not None and math.isfinite(float(x)) else math.inf


def collect(dev=None, stages=("pre",)) -> tuple:
    """-> ({kind: [config rows with per-system data]}, problems)"""
    dev = list(dev or TUNE_SYSTEMS)
    problems = []
    table = {}
    for kind in DESIGN:
        rows = []
        for c in configs(kind, stages):
            per = {}
            for s in dev:
                p = Tr.run_dir(c["node"], s, SEED) / "results.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                tiers = (r.get("config") or {}).get("eval", {}).get("tiers")
                if tiers or r.get("results"):
                    problems.append(f"{c['node']} {s}: the run evaluated tiers {tiers}; refused")
                    continue
                t = (r.get("train") or {}).get(c["method"])
                if not t or "val_nrmse" not in t:
                    problems.append(f"{c['node']} {s}: no val_nrmse for {c['method']}")
                    continue
                per[s] = {"val_nrmse": t["val_nrmse"], "val_diverged": t.get("val_diverged"),
                          "val_loss": t.get("val"), "n_params": t.get("n_params"), "width": t.get("width"),
                          "lr": t.get("lr"), "rank": t.get("rank"), "arch": t.get("arch"),
                          "wall_s": t.get("wall_s"), "code": r.get("code"), "job_wall_s": r.get("wall_s")}
            rows.append({**c, "per_system": per, "complete": all(s in per for s in dev)})
        table[kind] = rows
    return table, problems


def _rank_means(rows, dev, key):
    ranks = np.zeros(len(rows))
    for s in dev:
        v = np.array([_finite(r["per_system"][s][key]) for r in rows])
        ranks += np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1
    return ranks / len(dev)


def select(table: dict, dev=None) -> dict:
    dev = list(dev or TUNE_SYSTEMS)
    out = {}
    for kind, rows in table.items():
        if not rows or not all(r["complete"] for r in rows):
            miss = [f"{r['method']}:{s}" for r in rows for s in dev if s not in r["per_system"]]
            out[kind] = {"status": "incomplete", "missing": miss}
            continue
        for r in rows:
            v = [_finite(r["per_system"][s]["val_nrmse"]) for s in dev]
            r["score_mean"] = float(np.mean(v))
            r["score_gmean"] = float(np.exp(np.mean(np.log(v)))) if all(math.isfinite(x) and x > 0 for x in v) else math.inf
            lv = [_finite(r["per_system"][s]["val_loss"]) for s in dev]
            r["val_loss_mean"] = float(np.mean(lv))
        mr = _rank_means(rows, dev, "val_nrmse")
        for r, x in zip(rows, mr):
            r["mean_rank"] = float(x)
        d = next(r for r in rows if r["default"])
        best = min(rows, key=lambda r: (r["score_mean"], not r["default"]))
        tie = (not best["default"]) and not (best["score_mean"] <= (1 - TIE) * d["score_mean"])
        chosen = d if tie else best
        alt = {"gmean": min(rows, key=lambda r: r["score_gmean"])["method"],
               "mean_rank": min(rows, key=lambda r: r["mean_rank"])["method"],
               "val_loss": min(rows, key=lambda r: r["val_loss_mean"])["method"]}
        top = max(r["lr"] for r in rows)
        out[kind] = {"status": "selected", "selected": chosen["method"], "selected_node": chosen["node"],
                     "ctx_rms": chosen.get("ctx_rms"),
                     "at_grid_edge": chosen["lr"] == top, "grid_top_lr": top,
                     "lr": chosen["lr"], "knob": chosen["knob"], "knob_value": chosen["knob_value"],
                     "is_default": chosen["default"], "default": d["method"], "argmin": best["method"],
                     "tie_to_default": tie,
                     "score_selected": chosen["score_mean"], "score_default": d["score_mean"],
                     "rel_vs_default": chosen["score_mean"] / d["score_mean"] - 1 if d["score_mean"] > 0 else None,
                     "robustness": {k: {"choice": v, "agrees": v == chosen["method"]} for k, v in alt.items()},
                     "configs": rows}
    return out


def _clean(o):
    """Strict JSON: non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print, write nothing")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", default="pre", help="comma list of stages: pre, ext1, ext1b, ext2 (PROTOCOL.md 9-10)")
    ap.add_argument("--write-ext2-design", default=None, metavar="SELECTION_FILE")
    a = ap.parse_args(argv)
    if a.write_ext2_design:
        if EXT2_DESIGN.exists():
            print(f"{EXT2_DESIGN} exists: written once (PROTOCOL.md section 10)")
            return 1
        print(json.dumps(write_ext2_design(a.write_ext2_design), indent=1))
        return 0
    DEV = TUNE_SYSTEMS
    stages = tuple(a.stages.split(","))
    if a.out is None:
        a.out = str(OUT if stages == ("pre",) else OUT.with_name(f"selection_{stages[-1]}.json"))
    table, problems = collect(stages=stages)
    sel = select(table)
    for kind, s in sel.items():
        if s["status"] != "selected":
            print(f"{kind:7s} INCOMPLETE: {len(s['missing'])} missing ({', '.join(s['missing'][:4])} ...)")
            continue
        print(f"{kind:7s} selected {s['selected']:34s} score {s['score_selected']:.4f}  default {s['default']} "
              f"{s['score_default']:.4f}  ({100 * s['rel_vs_default']:+.1f}%)" + ("  [tie -> default]" if s["tie_to_default"] else "")
              + ("  [AT GRID EDGE]" if s["at_grid_edge"] else ""))
        for r in sorted(s["configs"], key=lambda r: r["score_mean"]):
            print(f"          {r['method']:34s} mean {r['score_mean']:.4f}  gmean {r['score_gmean']:.4f}  "
                  f"rank {r['mean_rank']:.2f}  " + "  ".join(f"{k[:4]} {v['val_nrmse']:.3f}/{v['width']}"
                                                             for k, v in r["per_system"].items()))
    for p in problems:
        print("PROBLEM:", p)
    if a.check:
        return 0
    if any(s["status"] != "selected" for s in sel.values()) or problems:
        print("selection refused: incomplete or problematic data (nothing written)")
        return 1
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists: the selection is made once (PROTOCOL.md section 3); delete it by hand to redo")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated": f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}", "protocol": "experiments/tuning/PROTOCOL.md",
               "rule": {"score": "mean over dev systems of train.<method>.val_nrmse (seed 0)", "tie": TIE,
                        "tie_rule": "default kept unless argmin < (1 - tie) x default", "seed": SEED, "systems": list(DEV),
                        "stages": list(stages), "lr_grid": [lr for st in stages if st in GRIDS for lr in GRIDS[st]], "design": {k: {"knob": v[0], "values": [DEFAULT_KNOB[k] if x is None else x
                                                                                       for x, _ in v[1]]}
                                                         for k, v in DESIGN.items()}},
               "selected": {k: s["selected"] for k, s in sel.items()},
               "variants": sel}
    out.write_text(json.dumps(_clean(payload), indent=1, allow_nan=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
