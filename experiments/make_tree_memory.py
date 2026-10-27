"""C2 memory re-test with a fixed-architecture truncated-history context (experiments/memory/PREREG.md).

Writes ONLY new node directories (existing nodes are never touched):
  C2-trunc-test   six held-out test systems, SFM context with history H in {1,...,2048, inf}, lora & hyper, 3 seeds
  C2-trunc-dev    the six dev systems, same arms, reported as in-sample

    .venv/bin/python experiments/make_tree_memory.py          # write the node dirs
    .venv/bin/python -m hyperkkl queue C2-trunc-test          # queue (only after the code change + tests)
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml

TREE = Path(__file__).resolve().parent / "tree"
TODAY = f"{_dt.date.today():%Y-%m-%d}"
CFG = Path(__file__).resolve().parent.parent / "configs" / "systems"
ROLE = {f.stem: yaml.safe_load(f.read_text()).get("role") for f in CFG.glob("*.yaml")}
TEST = sorted(s for s, r in ROLE.items() if r == "test")
DEV = sorted(s for s, r in ROLE.items() if r == "dev")
HIST = (1, 8, 32, 64, 128, 256, 512, 1024, 2048)
METHODS = [f"{m}@sfm:history={H}" for m in ("lora", "hyper") for H in HIST] + ["lora", "hyper"]

QUESTION = ("How much input history does the decoder need? Corollary m2-rule: a window of L* = (ln(1/eta) + ln l)/alpha "
            "seconds suffices (eta = 0.05, l = Lipschitz constant of the inverse windowed map, proxy l_dec_q95). Tested "
            "with the SFM context whose filter state uses only the last H input samples (inputs older than H -> 0), in "
            "training and deployment alike, so every arm has the same trainable parameters (no capacity confound).")
HYPOTHESIS = ("(S) for every arm H with L*_i <= H <= 0.5 N_i the error is within +5% of the untruncated context (lora and "
              "hyper, id and sequence_ood); (T) at the largest arm H <= L*_i/4 the error is significantly worse. L*_i per "
              "system is fixed in experiments/memory/PREREG.md from the existing phase-1 l_dec_q95.")
RULE = ("Supported if (S) holds on >= 4/6 systems AND (T) holds on >= 4/6 systems (analysis.evidence.analyse_trunc; "
        "paired relative change vs the same method's inf arm, Student-t over 3 seeds; (S) point estimate <= +5% on every "
        "informative arm, TOST at 5% reported; (T) 95% CI lower bound > 0). (S) alone on >= 4/6: 'sufficient bound "
        "supported, tightness not'. Systems with no arm in [L*_i, 0.5 N_i] are not testable for (S).")

NODES = []


def node(id, parent, title, systems, priority, notes):
    NODES.append({"id": id, "parent": parent, "title": title, "contribution": "C2", "question": QUESTION,
                  "hypothesis": HYPOTHESIS, "decision_rule": RULE,
                  "set": {"eval.tiers": ["id", "sequence_ood"], "eval.memory_probe": False},
                  "systems": list(systems), "methods": METHODS, "baselines": [], "seeds": [0, 1, 2],
                  "compare": {"reference": ["lora", "hyper"], "tier": "sequence_ood", "condition": "clean"},
                  "priority": priority, "status": "planned", "created": TODAY, "notes": notes})


node("C2-trunc-test", "TEST-C1", "Truncated-history SFM context on the six HELD-OUT test systems (window rule, pre-registered)",
     TEST, 18, "Decision node for the refined window rule; pre-registered in experiments/memory/PREREG.md "
               "(2026-10-25 23:50 CET) before any run. Phase 1 is the TEST-C1 recipe retrained (module hash changed).")
node("C2-trunc-dev", "C1-where", "Truncated-history SFM context on the six dev systems (in-sample report)",
     DEV, 17, "Same arms and rule as C2-trunc-test; reported as in-sample (the dev systems informed C2-floor/C2-window).")

if __name__ == "__main__":
    for d in NODES:
        p = TREE / d["id"]
        if (p / "node.yaml").exists():
            print("exists, skipped:", d["id"])
            continue
        p.mkdir(parents=True)
        (p / "node.yaml").write_text(yaml.safe_dump(d, sort_keys=False, width=100))
        print("wrote", d["id"], len(d["methods"]), "methods x", len(d["systems"]), "systems x", len(d["seeds"]), "seeds")
