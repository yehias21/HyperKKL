"""v2 tree additions after the simulated editor reviews (2026-10-25 22:40 AD).

Writes ONLY new node directories (existing nodes are never touched):
  TEST-C1, TEST-C3   frozen recipe on the 6 held-out test systems (Park MUST-FIX 5)
  C1-rank1, C1-rank16  LoRA rank sweep: is LoRA ~ hyper an artefact of rank 4 vs trunk width 4?
                       (Park MUST-FIX 1; hyper's trunk width at each budget is reported separately)
  C2-window-a2       second bandwidth for the window rule on 3 dev systems (Li MUST-FIX 4b)
  C1-budget-s/l, C1-basewidth-*: seed 2 added in place would rewrite those nodes, so the third
                       seed is queued as separate nodes '<id>-s2' (Park/Reviewer-D statistics).

    .venv/bin/python experiments/make_tree_v2b.py
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml

TREE = Path(__file__).resolve().parent / "tree"
TODAY = f"{_dt.date.today():%Y-%m-%d}"
CFG = Path(__file__).resolve().parent.parent / "configs" / "systems"
TEST = sorted(f.stem for f in CFG.glob("*.yaml") if yaml.safe_load(f.read_text()).get("role") == "test")
FILTERS = ["ekf", "ukf", "pf", "kkl_mb", "nlox"]
ROBUST = ["ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle"]
CT = {"kind": "colored_t", "rho": 0.9, "nu": 3.0}
NODES = []


def node(id, parent, title, contribution, question, hypothesis, rule, *, set=None, methods,
         baselines=(), seeds=(0, 1, 2), reference, tier="sequence_ood", priority=0,
         systems=None, notes=None):
    d = {"id": id, "parent": parent, "title": title, "contribution": contribution,
         "question": question, "hypothesis": hypothesis, "decision_rule": rule}
    if set:
        d["set"] = set
    if systems:
        d["systems"] = list(systems)
    d.update({"methods": list(methods), "baselines": list(baselines), "seeds": list(seeds),
              "compare": {"reference": list(reference), "tier": tier, "condition": "clean"},
              "priority": priority, "status": "planned", "created": TODAY})
    if notes:
        d["notes"] = notes
    NODES.append(d)


node("TEST-C1", "C1-where", "FROZEN recipe, held-out test systems: where the input enters", "C1",
     "Does the dev-system C1 conclusion (weight-space conditioning beats latent injection and the "
     "additive 'concat' variant) hold on six plants never used for any design decision?",
     "Weight-space (lora or hyper) beats latent and concat on >= 4/6 test systems.",
     "Supported if the better of lora/hyper beats both latent and concat with a paired CI "
     "excluding 0 on >= 4/6 test systems on shape_ood and on sequence_ood. Run once; no tuning.",
     methods=["autonomous", "latent", "concat", "film", "lora", "hyper"], baselines=["ekf"],
     reference=["latent", "concat"], priority=20, systems=TEST)

node("TEST-C3", "C3-realistic", "FROZEN recipe, held-out test systems: realistic deployment", "C3",
     "Partial model + 30% parameter error + colored Student-t noise on six held-out plants: does "
     "the best learned observer beat every filter variant?",
     "Learned wins on >= 4/6 test systems; it loses where the unmodelled term is small.",
     "Best learned vs best WORKING filter (NRMSE below climatology), paired CI, id and "
     "sequence_ood, >= 4/6; report against the relative model error of each plant. Run once.",
     set={"system.knowledge": "partial", "system.mismatch": 0.3,
          "data.sensor": {"noise": 0.03, **CT}},
     methods=["latent", "lora", "hyper"], baselines=FILTERS + ROBUST,
     reference=["ekf", "hyper"], priority=19, systems=TEST)

for r in (1, 16):
    node(f"C1-rank{r}", "C1-where", f"LoRA rank {r} at the matched 140k budget (default 4)", "C1",
         "Is the LoRA ~ hypernetwork tie an artefact of the low-rank choice?",
         "The weight-space advantage over latent/concat is stable across rank 1-16.",
         "Stable if lora at this rank stays within an equivalence margin of +-10% of lora "
         "rank 4 (C1-where) or better, on >= 4/6 dev systems (sequence_ood).",
         set={"phase2.rank": r}, methods=["lora"], reference=["lora"], priority=15)

node("C2-window-a2", "C2-window-a1", "Window sweep at alpha x 2 (second bandwidth)", "C2",
     "Does the window knee move as 1/alpha (Theorem: L* ~ (ln(1/eta) + ln l)/alpha)?",
     "With dt proportional to 1/alpha the knee in samples is unchanged; in seconds it halves.",
     "Compare knees with C2-window-a1 on vdp, duffing, battery; supported if the knee in samples "
     "changes by <= 2x on >= 2/3 systems.",
     set={"system.alpha_scale": 2.0},
     methods=[f"lora@window:window={L}" for L in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)],
     reference=["lora@window:window=512"], priority=14, systems=["battery", "duffing", "vdp"])

for base, setkv in (("C1-budget-s", {"phase2.budget": 35000}),
                    ("C1-budget-l", {"phase2.budget": 560000}),
                    ("C1-basewidth-64", {"phase1.dec_width": 64}),
                    ("C1-basewidth-256", {"phase1.dec_width": 256})):
    node(f"{base}-s2", base, f"{base}: third seed (statistics)", "C1",
         "Third seed for the control node, so its equivalence test has >= 2 degrees of freedom.",
         "As the parent node.", "Pooled with the parent node (seeds 0-2).",
         set=setkv, methods=["autonomous", "latent", "concat", "film", "lora", "hyper"],
         seeds=[2], reference=["latent"], priority=12)

if __name__ == "__main__":
    for d in NODES:
        p = TREE / d["id"]
        if (p / "node.yaml").exists():
            print("exists, skipped:", d["id"])
            continue
        p.mkdir(parents=True)
        (p / "node.yaml").write_text(yaml.safe_dump(d, sort_keys=False, width=100))
        print("wrote", d["id"])
