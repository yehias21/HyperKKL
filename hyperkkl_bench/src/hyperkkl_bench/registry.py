"""Plant and knowledge-level registry."""

from __future__ import annotations

# knowledge level -> (data variant on disk, modeller's model: knowledge, parameters, HF config suffix)
KNOWLEDGE = {
    "exact":     {"variant": "white",      "model_knowledge": "full",    "hf": "C1",
                  "about": "true dynamics at true parameters; white Gaussian sensor noise (3% y_std)"},
    "partial":   {"variant": "white",      "model_knowledge": "partial", "hf": "C3-partial",
                  "about": "model misses one physically identified term (f_known); same data as exact"},
    "mismatch":  {"variant": "mismatch30", "model_knowledge": "full",    "hf": "C3-mismatch30",
                  "about": "true structure, NOMINAL parameters; real plant parameters off by exp(+-0.3)"},
    "realistic": {"variant": "realistic",  "model_knowledge": "partial", "hf": "C3-realistic",
                  "about": "f_known at nominal parameters, 30% parameter mismatch, AR(1) colored Student-t(3) noise"},
}

DEV = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
TEST = ["cartpole_b", "fhn", "highway_traffic", "network", "rossler", "two_link_arm"]

PLANTS = {p: {"role": "dev", "frozen": True} for p in DEV}
PLANTS.update({p: {"role": "test", "frozen": True} for p in TEST})
# legacy: unbounded cart-pole, FAILED the boundedness inclusion rule; kept only to reproduce old numbers
PLANTS["legacy_cartpole"] = {"role": "legacy", "frozen": True, "excluded": True,
                             "why": "unbounded cart position (spread grows 9.4x from T to 5T); "
                                    "failed the boundedness rule (INCLUSION.md)"}

CORE = DEV + TEST      # the core track: 12 plants x 4 knowledge levels

# deployment track (experiments/casestudies), frozen in data_extra/ by tools/export_casestudies.py
DEPLOYMENT = ["quad_planar", "cstr_exo"]
PLANTS.update({p: {"role": "deployment", "frozen": True, "extra_root": True} for p in DEPLOYMENT})
# scaling track (experiments/scaling): FHN ring, n_x = 4 .. 64, frozen by tools/export_scaling.py
SCALING = [f"fhnring_n{n}" for n in (4, 8, 16, 32, 64)]
PLANTS.update({p: {"role": "scaling", "frozen": True, "extra_root": True} for p in SCALING})


def check_plant_name(plant: str) -> None:
    if plant not in PLANTS:
        raise KeyError(f"unknown plant {plant!r}; available: {sorted(PLANTS)}")


def variant_of(knowledge: str) -> str:
    return KNOWLEDGE[knowledge]["variant"]


def list_plants(include_legacy: bool = False, role: str | None = None) -> list:
    """Core plants by default; role='deployment' (or 'all') for the deployment-track plants."""
    out = [p for p, d in PLANTS.items() if include_legacy or not d.get("excluded")]
    if role is None:
        out = [p for p in out if PLANTS[p]["role"] in ("dev", "test", "legacy")]
    elif role == "all":
        role = None
    if role:
        out = [p for p in out if PLANTS[p]["role"] == role]
    return out
