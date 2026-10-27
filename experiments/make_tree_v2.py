"""Experiment tree for input protocol v2 (2026-10-25).

OOD = bounded input whose SHAPE (shape_ood) or SEQUENCE (sequence_ood) differs from
training; the amplitude tier of v1 is dropped. v1 nodes live in experiments/tree_v1/.

    .venv/bin/python experiments/make_tree_v2.py          # writes experiments/tree/<id>/node.yaml
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml

TREE = Path(__file__).resolve().parent / "tree"
TODAY = f"{_dt.date.today():%Y-%m-%d}"
LEARNED = ["autonomous", "latent", "concat", "film", "lora", "hyper"]
FILTERS = ["ekf", "ukf", "pf", "kkl_mb", "nlox"]
ROBUST = ["ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle"]   # wired after the filter agent
OOD = "shape_ood and sequence_ood (clean), each separately"
DEV = sorted(yaml.safe_load(f.read_text()).get("role") == "dev" and f.stem or ""
             for f in (Path(__file__).resolve().parent.parent / "configs" / "systems").glob("*.yaml"))
DEV = [s for s in DEV if s]
CONDITIONS = ["clean", "noise10", "outliers", "dropout", "colored", "colored_t", "quantized", "sensor_drift",
              "poor_init", "input_noise", "input_bias", "input_delay", "plant_drift10",
              "plant_drift30", "burst_outage", "long"]

NODES = []


def node(id, parent, title, contribution, question, hypothesis, rule, *, set=None,
         methods=None, baselines=(), seeds=(0, 1, 2), reference=None, tier="shape_ood",
         condition="clean", metrics=None, priority=0, notes=None, systems=None):
    d = {"id": id, "parent": parent, "title": title, "contribution": contribution,
         "question": question, "hypothesis": hypothesis, "decision_rule": rule}
    if set:
        d["set"] = set
    if systems:
        d["systems"] = list(systems)
    d.update({"methods": list(methods or LEARNED), "baselines": list(baselines),
              "seeds": list(seeds),
              "compare": {"reference": reference or ["latent"], "tier": tier,
                          "condition": condition, **({"metrics": metrics} if metrics else {})},
              "priority": priority, "status": "planned", "created": TODAY})
    if notes:
        d["notes"] = notes
    NODES.append(d)


node("R0", None, "Protocol-v2 sanity run: every method on every dev system, one seed",
     "all", "Does the v2 input protocol (one amplitude box, per-plant t_ref, id / shape_ood / "
     "sequence_ood tiers) train and evaluate cleanly, with no mass rejection and plausible errors?",
     "Every learned variant beats the autonomous observer on id; no NaN; rejection < 5%.",
     "Pass if no job fails, rejection < 5% per tier, and every conditioned variant beats "
     "autonomous on id/clean on every dev system.",
     baselines=["ekf"], seeds=[0], reference=["autonomous"], tier="id", priority=10, systems=DEV)

node("C1-where", "R0", "Where should the input enter? Five locations at matched parameters and history",
     "C1", "With identical frozen base maps, context encoder (SFM) and a matched 140k-parameter "
     "conditioner, which location generalises best to unseen input SHAPES and SEQUENCES within the "
     "training amplitude box: latent dynamics, decoder input (concat), features (FiLM), low-rank "
     "weights (LoRA), full weights (hypernetwork)?",
     "Weight-space conditioning (hyper, lora) beats latent injection and concat on shape_ood and "
     "sequence_ood (v1 waveform tier: hyper beat latent on 5/5 dev systems).",
     f"A location is preferred if it beats every other location with a paired 95% CI excluding 0 "
     f"on the all-families column on >= 4/6 dev systems, on {OOD}, and is significantly worse on "
     f"none. Report id alongside (in-distribution accuracy).",
     set={"eval.memory_probe": True},
     baselines=["ekf"], seeds=[0, 1, 2, 3, 4], reference=["latent", "concat"], priority=9)

node("C1-ctx", "C1-where", "Matched budget INCLUDING the context encoder", "C1",
     "Does the C1 ranking survive when the context encoder's parameters count toward the budget?",
     "Ranking unchanged (the SFM encoder is small relative to 140k).",
     "Ranking preserved if the C1-where winner stays within CI of the best here on >= 4/6 systems.",
     set={"phase2.budget_includes_context": True}, priority=5)

node("C1-budget-s", "C1-where", "Budget 35k (x0.25)", "C1",
     "Is the C1 ranking an artefact of one parameter budget?", "Ranking stable across 35k-560k.",
     "Stable if the C1-where winner is best-or-within-CI at both budgets on >= 4/6 systems.",
     set={"phase2.budget": 35000}, priority=4, seeds=[0, 1])
node("C1-budget-l", "C1-where", "Budget 560k (x4)", "C1",
     "Is the C1 ranking an artefact of one parameter budget?", "Ranking stable across 35k-560k.",
     "Stable if the C1-where winner is best-or-within-CI at both budgets on >= 4/6 systems.",
     set={"phase2.budget": 560000}, priority=4, seeds=[0, 1])

node("C1-modelfree", "C1-where", "Model-free phase 1 (no PDE term, no f anywhere)", "C1",
     "How much of the accuracy needs the model in phase 1?",
     "Model-free phase 1 costs accuracy mainly on shape/sequence OOD (PDE term regularises T0 "
     "off the data).", "Model needed if C1-where beats this node with CI excluding 0 on >= 4/6.",
     set={"phase1.pde": False}, priority=4)

for w in (64, 256):
    node(f"C1-basewidth-{w}", "C1-where", f"Base decoder width {w} (default 128)", "C1",
         "Does a stronger or weaker base decoder change where the input should enter?",
         "Decoder-side gains shrink with a stronger base decoder (less to correct).",
         "Report the C1 ranking per width; 'base-dependent' if the winner changes on >= 3/6.",
         set={"phase1.dec_width": w}, priority=3, seeds=[0, 1])

# C2. The memory probe runs on the C1-where models (eval.memory_probe: free). Smoke run:
# the learned SFM context forgets at its slowest learned pole (~0.14/s on vdp, alpha 1.5),
# i.e. it KEEPS memory far longer than Lemma 1 says the map needs. The intervention below
# tests whether that extra memory is used: a floor on the SFM forgetting rate.
for fl, tag in ((0.25, "025"), (0.5, "05"), (1.0, "1")):
    node(f"C2-floor-{tag}", "C1-where",
         f"Forgetting floor: SFM context forgets at >= {fl} x alpha", "C2",
         "Corollary 1: input history older than ~ln(1/eta)/alpha barely affects the map. If the "
         "context is forced to forget at rate >= c alpha (Lemma 2 then bounds its memory), does "
         "accuracy suffer?",
         "No loss for c <= 0.5 (memory ln(20)/(0.5 alpha) covers the theory window); a small "
         "loss is possible at c = 1 (no margin).",
         "Memory beyond the theory window is unnecessary if C2-floor-05 is within CI of "
         "C1-where (no floor) for lora and hyper on >= 5/6 dev systems, on id and sequence_ood. "
         "The probe (on) must decay at >= the floor rate (Lemma 2 check).",
         set={"phase2.context.sfm_floor": fl, "eval.memory_probe": True},
         methods=["lora", "hyper"], reference=["lora"], tier="sequence_ood", priority=6)

node("C2-window-a1", "R0", "Window length L sweep at the default bandwidth (secondary to the probe)",
     "C2", "With a raw input window of L samples instead of the full-history SFM context, how "
     "long must the window be?",
     "A window >= L* = ln(1/eta)/alpha (eta = 0.05) is within 5% of the longest window "
     "(sufficient direction of Corollary 1); shorter windows lose accuracy.",
     "Supported if e(L >= L*) is within 5% of e(512) on >= 5/6 dev systems, on id and "
     "sequence_ood.",
     methods=[f"lora@window:window={L}" for L in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)],
     reference=["lora@window:window=512"], tier="sequence_ood", priority=5)

node("C3-faults", "C1-where", "When does a learned KKL observer beat classical filters? "
     "Sensor faults, input faults, poor init, plant drift", "C3",
     "Against EKF, UKF, PF, model-based KKL and NLOX (nominal model, oracle noise for clean), "
     "and against robust (Huber) and condition-oracle filters, under which conditions do the "
     "learned observers win?",
     "Filters win on clean, well-modelled data. Learned observers win under plant drift and "
     "gross sensor faults against NOMINAL filters; robust/oracle filters close part of that gap.",
     "Per condition, count dev systems where the best learned observer beats the best filter "
     "(nominal, and separately the best of robust/oracle) with a paired CI excluding 0. A "
     "'learned wins' regime needs >= 4/6 systems against the best filter of either kind.",
     set={"eval.conditions": CONDITIONS}, methods=["latent", "lora", "hyper"],
     baselines=FILTERS + ROBUST, seeds=[0, 1, 2, 3, 4], reference=["ekf", "hyper"],
     tier="sequence_ood", metrics=["nrmse", "nrmse_transient"], priority=8)

for m, tag in ((0.1, "10"), (0.2, "20"), (0.3, "30"), (0.5, "50")):
    node(f"C3-mismatch{tag}", "C1-where",
         f"Model mismatch: the real plant's physical parameters are {int(m * 100)}% off nominal",
         "C3", "Filters use the nominal model; learned observers are trained on data from the "
         "real plant. When does learning from data beat a wrong model?",
         "The filter-vs-learned crossover sits between 10% and 30% mismatch.",
         "Learned wins at a level if the best learned observer beats the best filter with CI "
         "excluding 0 on >= 4/6 systems, on id and sequence_ood.",
         set={"system.mismatch": m}, methods=["latent", "lora", "hyper"],
         baselines=FILTERS + ROBUST, reference=["ekf", "hyper"], tier="sequence_ood",
         priority=7 if tag in ("10", "30") else 5)

for n_tr in (50, 150):
    node(f"C3-data{n_tr}", "C1-where", f"Learning curve: {n_tr} training episodes (default 600)",
         "C3", "How much data does a learned observer need before it pays off?",
         "Error falls roughly as a power law in n_train; hyper needs more data than latent.",
         "Report error vs n_train (50/150/600) per method; crossover with the filters under "
         "30% mismatch read from C3-mismatch30.",
         set={"data.n_train": n_tr}, methods=["latent", "lora", "hyper"], baselines=[],
         reference=["hyper"], priority=4)


# --- incomplete knowledge and non-Gaussian noise (user request 2026-10-25) ---------------
# knowledge=partial: every model-based component (filters, kkl_mb, the phase-1 PDE term)
# gets f_known (one physically identified term missing, systems.Plant.UNMODELLED); the real
# plant keeps f. Learned observers additionally see data from the real plant.
# colored_t sensor: AR(1)-filtered Student-t noise (rho 0.9, nu 3), the sensor's real
# nature in training AND deployment; nominal filters assume white Gaussian R.
CT = {"kind": "colored_t", "rho": 0.9, "nu": 3.0}

node("C3-partial", "C1-where", "Incomplete knowledge: the model lacks one physical term", "C3",
     "When the modeller knows only part of the dynamics (a missing or simplified term, not "
     "a wrong parameter), do learned observers trained on real-plant data beat filters "
     "built on the partial model?",
     "Learned observers win on systems where the unmodelled term is large relative to f "
     "(bioreactor, vdp, lorenz); filters keep winning where it is small (duffing, quad_tank).",
     "Per system, best learned vs best filter (nominal and robust/oracle), paired CI on id and "
     "sequence_ood; report against the relative model error median |f - f_known| / |f|.",
     set={"system.knowledge": "partial"}, methods=["latent", "lora", "hyper"],
     baselines=FILTERS + ROBUST, reference=["ekf", "hyper"], tier="sequence_ood", priority=8)

node("C3-partial-modelfree", "C3-partial", "Incomplete knowledge, model-free phase 1", "C3",
     "Is a partial model in the phase-1 PDE term still worth using, or does it hurt?",
     "The partial model still helps (structure) unless the model error is large.",
     "Model helps if C3-partial beats this node with CI excluding 0 on >= 4/6 systems.",
     set={"system.knowledge": "partial", "phase1.pde": False},
     methods=["latent", "lora", "hyper"], reference=["hyper"], tier="sequence_ood", priority=6)

node("C3-noise-ct", "C1-where", "Non-Gaussian colored sensor noise (AR(1) Student-t)", "C3",
     "The sensor's noise is colored and heavy-tailed in training and deployment. Filters "
     "assume white Gaussian noise (nominal R; oracle filters use its true variance).",
     "Learned observers trained on the real sensor beat nominal and variance-oracle filters; "
     "Huber filters close part of the gap (they address tails, not color).",
     "Best learned vs best filter of each kind, paired CI on id and sequence_ood, >= 4/6.",
     set={"data.sensor": {"noise": 0.03, **CT}}, methods=["latent", "lora", "hyper"],
     baselines=FILTERS + ROBUST, reference=["ekf", "hyper"], tier="sequence_ood", priority=7)

node("C3-realistic", "C1-where", "Realistic deployment: partial model + 30% parameter error + "
     "colored non-Gaussian noise", "C3",
     "All three kinds of incomplete knowledge at once, as on a real plant.",
     "Learned observers win on >= 4/6 systems against every filter variant.",
     "Best learned vs best filter (any variant), paired CI, id and sequence_ood, >= 4/6.",
     set={"system.knowledge": "partial", "system.mismatch": 0.3,
          "data.sensor": {"noise": 0.03, **CT}},
     methods=["latent", "lora", "hyper"], baselines=FILTERS + ROBUST,
     reference=["ekf", "hyper"], tier="sequence_ood", seeds=[0, 1, 2, 3, 4], priority=8)


if __name__ == "__main__":
    for d in NODES:
        p = TREE / d["id"]
        p.mkdir(parents=True, exist_ok=True)
        (p / "node.yaml").write_text(yaml.safe_dump(d, sort_keys=False, width=100))
    print(f"wrote {len(NODES)} nodes to {TREE}")
