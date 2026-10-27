"""C1 per-variant tuning study (experiments/tuning/PROTOCOL.md): tree nodes.

    .venv/bin/python experiments/make_tree_tuning.py select   # C1-tune-select + its 10 children
    .venv/bin/python experiments/make_tree_tuning.py tuned    # C1-tuned, TEST-C1-tuned from selection.json

Writes ONLY new node directories (existing nodes are never touched).

select  C1-tune-select (umbrella: the shared settings and the rule; no methods of its own) and
        one child per (variant, knob value), each with the 4 learning rates as methods
        'kind[lr=...,knob=...]', seed 0, the 6 dev systems, eval.tiers = [] (no evaluation
        data), eval.val_nrmse = true (the selection score). 10 nodes x 6 jobs = 60 jobs,
        40 configurations x 6 systems = 240 phase-2 trainings.
tuned   refuses unless experiments/tuning/selection.json exists (written by
        analysis/select_tuned.py); the method strings are copied from it verbatim.
ext1, ext1b      PROTOCOL.md section 9: the equal lr-grid extension nodes C1-tune-<v>-<knob>-<stage>.
ext2             PROTOCOL.md section 10: ctx_rms x lr-neighbourhood nodes C1-tune-<v>-rms<r>-ext2
                 (from experiments/tuning/ext2_design.json).
tuned-ext FILE LABEL   C1-tuned-LABEL / TEST-C1-tuned-LABEL with the selected methods of FILE
                 that no earlier confirmation node has run (LABEL: ext1 or ext2).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.select_tuned import DESIGN, GRIDS, LRS, OUT as SELECTION, select_nodes   # noqa: E402

TREE = ROOT / "experiments" / "tree"
TODAY = f"{_dt.date.today():%Y-%m-%d}"
NODES = []
PROTOCOL = "experiments/tuning/PROTOCOL.md"
RULE = ("Tuned HyperKKL (the low-rank hypernetwork, 'lora' code path) beats tuned latent AND tuned concat with a paired "
        "95% CI on >= 4/6 dev (C1-tuned) and >= 4/6 test systems (TEST-C1-tuned), shape_ood and sequence_ood (each "
        "separately; PASS iff all four votes pass). Dense 'hyper' = appendix ablation, not in the rule "
        "(PROTOCOL.md section 8, 2026-10-26 00:52).")


def node(id, parent, title, question, hypothesis, rule, *, set=None, methods, baselines=(), seeds=None,
         reference=(), tier="shape_ood", priority=0, systems=None, notes=None):
    d = {"id": id, "parent": parent, "title": title, "contribution": "C1",
         "question": question, "hypothesis": hypothesis, "decision_rule": rule}
    if set:
        d["set"] = set
    if systems:
        d["systems"] = list(systems)
    d.update({"methods": list(methods), "baselines": list(baselines)})
    if seeds is not None:
        d["seeds"] = list(seeds)
    d.update({"compare": {"reference": list(reference), "tier": tier, "condition": "clean"},
              "priority": priority, "status": "planned", "created": TODAY})
    if notes:
        d["notes"] = notes
    NODES.append(d)


def stage_select():
    node("C1-tune-select", "C1-where", "Per-variant tuning: selection on validation (umbrella)",
         "With an equal tuning budget per variant (8 configurations: 4 learning rates x 1 architecture knob, "
         "140k parameters, 3000 steps), which configuration of each conditioning variant has the lowest "
         "validation NRMSE on the dev systems?",
         "Tuning changes the per-variant error but not the C1 ordering (tested in C1-tuned).",
         "REPORTED: per variant the argmin of the mean (over the 6 dev systems, seed 0) validation NRMSE "
         "(train.<method>.val_nrmse) of its 8 configurations, ties (< 0.5%) to the default; computed by "
         "analysis/select_tuned.py -> experiments/tuning/selection.json. No evaluation tier is run.",
         set={"eval.tiers": [], "eval.val_nrmse": True, "eval.memory_probe": False, "eval.save_traj": False},
         methods=[], baselines=[], seeds=[0], priority=30,
         notes=f"umbrella of the 10 C1-tune-<variant>-<knob> nodes; queue those, not this one ({PROTOCOL})")
    for nid, d in select_nodes().items():
        kind, knob, v = d["kind"], d["knob"], d["value"]
        vs = "default" if v is None else v
        tag = {"lora": " (HyperKKL)", "hyper": " (dense hypernetwork: appendix ablation)"}.get(kind, "")
        node(nid, "C1-tune-select", f"Tuning grid: {kind}{tag}, {knob} = {vs}, lr in {list(LRS)}",
             f"Validation NRMSE of {kind} with {knob} = {vs} at 4 learning rates (seed 0, dev).",
             "Part of C1-tune-select.", "See C1-tune-select (selection by analysis/select_tuned.py).",
             methods=d["methods"], seeds=[0],
             priority={"latent": 32, "hyper": 30}.get(kind, 31))   # latent first (longest); the ablation last


DEV_CONFIRM = ("C1-tuned", "C1-tuned-ext1", "C1-tuned-ext2")
TEST_CONFIRM = ("TEST-C1-tuned", "TEST-C1-tuned-ext1", "TEST-C1-tuned-ext2")


def stage_ext(stage):
    """PROTOCOL.md sections 9-10: the equal extension nodes, same for every variant."""
    for nid, d in select_nodes(stage).items():
        kind, knob, v = d["kind"], d["knob"], d["value"]
        vs = "default" if v is None else v
        parent = "C1-tune-select" if stage == "ext2" else nid[: -len(stage) - 1]
        extra = f", ctx_rms {d['ctx_rms']}" if stage == "ext2" else ""
        node(nid, parent, f"Tuning extension {stage}: {kind}, {knob} = {vs}{extra}, lr in "
                          f"{[float(f'{x:.6g}') for x in (GRIDS.get(stage) or ())] or [m.split('lr=')[1].split(',')[0] for m in d['methods']]}",
             f"Validation NRMSE of {kind} in the {stage} extension (seed 0, dev).",
             "Part of the equal extension.", "See PROTOCOL.md sections 9-10 (analysis/select_tuned.py --stages ...).",
             methods=d["methods"], seeds=[0], priority={"latent": 23, "hyper": 21}.get(kind, 22),
             notes=f"{PROTOCOL} sections 9-10 (equal extension after the pre-registered selection hit the lr-grid edge)")


def _ran(nodes) -> set:
    out = set()
    for n in nodes:
        p = TREE / n / "node.yaml"
        if p.exists():
            out |= set(yaml.safe_load(p.read_text()).get("methods") or [])
    return out


def stage_tuned_ext(sel_file, label):
    """Confirmation (3 seeds, dev + test) of the selected methods that no confirmation node has run yet."""
    ext = json.loads(Path(sel_file).read_text())
    done = _ran(DEV_CONFIRM) & _ran(TEST_CONFIRM)
    meth = [ext["selected"][k] for k in ("latent", "concat", "film", "lora", "hyper") if ext["selected"][k] not in done]
    if not meth:
        print("every selected method is already confirmed: nothing to queue")
        return
    note = (f"selected methods of {Path(sel_file).name} ({ext['generated']}) not yet confirmed: {meth}; the others "
            f"reuse earlier confirmation nodes. {PROTOCOL} sections 9-10")
    node(f"C1-tuned-{label}", "C1-tuned", f"Tuned recipe after extension {label} (new choices only), dev systems",
         f"Does the C1-tuned verdict change after the equal {label} extension?", "As C1-tuned.",
         f"Verdict '{label}': the C1-tuned rule on {Path(sel_file).name} (PROTOCOL.md sections 9-10).",
         methods=meth, baselines=[], seeds=[0, 1, 2], priority=20, notes=note)
    node(f"TEST-C1-tuned-{label}", "TEST-C1-tuned", f"Tuned recipe after extension {label} (new choices only), test",
         f"As C1-tuned-{label}, held-out test plants.", "As C1-tuned.",
         f"Verdict '{label}' (PROTOCOL.md sections 9-10). Run once.", methods=meth, baselines=[], seeds=[0, 1, 2],
         priority=19, notes=note)


def stage_tuned():
    if not SELECTION.exists():
        raise SystemExit(f"{SELECTION} missing: run analysis/select_tuned.py first (PROTOCOL.md section 3)")
    sel = json.loads(SELECTION.read_text())
    meth = [sel["selected"][k] for k in ("latent", "concat", "film", "lora", "hyper")]
    note = f"methods copied verbatim from experiments/tuning/selection.json ({sel['generated']}); {PROTOCOL}"
    node("C1-tuned", "C1-where", "Tuned recipe per variant (equal tuning budget), dev systems",
         "After every variant gets the same tuning budget (8 configurations, selected on validation), does "
         "HyperKKL (low-rank hypernetwork) still beat latent injection and the additive residual?",
         "Tuned HyperKKL still beats tuned latent and tuned concat on shape_ood and sequence_ood.",
         RULE, set={"eval.memory_probe": False}, methods=meth, baselines=[], seeds=[0, 1, 2],
         reference=[sel["selected"]["latent"], sel["selected"]["concat"]], priority=25, notes=note)
    node("TEST-C1-tuned", "TEST-C1", "Tuned recipe per variant, held-out test systems (run once)",
         "Does the tuned-recipe C1 conclusion hold on the six test plants?",
         "As C1-tuned.", RULE + " Run once.", set={"eval.memory_probe": False}, methods=meth, baselines=[], seeds=[0, 1, 2],
         reference=[sel["selected"]["latent"], sel["selected"]["concat"]], priority=24, notes=note)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"select": stage_select, "tuned": stage_tuned, "ext1": lambda: stage_ext("ext1"),
     "ext1b": lambda: stage_ext("ext1b"), "ext2": lambda: stage_ext("ext2"),
     "tuned-ext": lambda: stage_tuned_ext(sys.argv[2], sys.argv[3])}.get(stage, lambda: sys.exit(__doc__))()
    for d in NODES:
        p = TREE / d["id"]
        if (p / "node.yaml").exists():
            print("exists, skipped:", d["id"])
            continue
        p.mkdir(parents=True)
        (p / "node.yaml").write_text(yaml.safe_dump(d, sort_keys=False, width=100))
        print("wrote", d["id"])
