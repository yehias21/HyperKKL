"""Plant cards: metadata + difficulty axes (with provenance) + track membership."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .data import metadata
from .paths import extra_root
from .registry import KNOWLEDGE, PLANTS, check_plant_name

RES = Path(__file__).resolve().parent / "resources"


@lru_cache(maxsize=1)
def _difficulty() -> dict:
    p = RES / "difficulty.json"
    return json.loads(p.read_text())["plants"] if p.exists() else {}


def plant_card(plant: str, root=None) -> dict:
    check_plant_name(plant)
    meta = metadata(plant, root)
    keep = ("plant", "role", "n_x", "n_y", "n_u", "state_names", "sensors", "sensor_names", "unmeasured",
            "group", "unmodelled_term", "nominal_params", "dt", "N", "T", "t_ref", "settle_index",
            "input_box", "actuator", "ic_box", "x_std", "y_std")
    card = {k: meta[k] for k in keep if k in meta}
    card["status"] = PLANTS[plant]
    card["knowledge_levels"] = {k: v["about"] for k, v in KNOWLEDGE.items()}
    card["episodes"] = {v: dict(s["n"]) for v, s in meta["settings"].items()}
    man = extra_root() / "MANIFEST.json"
    if man.exists():
        for r in json.loads(man.read_text())["splits"]:
            if r["plant"] == plant and r["variant"] in card["episodes"]:
                card["episodes"][r["variant"]][r["split"]] = r["episodes"]
    card["difficulty"] = _difficulty().get(plant)
    return card


def card_markdown(plant: str, root=None) -> str:
    c = plant_card(plant, root)
    d = c.get("difficulty") or {}
    ib = d.get("info_budget", {})
    f = lambda v, fmt="{:.4g}": "n/a" if v is None else fmt.format(v)
    L = [f"# Plant card: `{plant}`", "",
         f"- role: **{c['status']['role']}**" + ("  (EXCLUDED: " + c['status'].get('why', '') + ")" if c['status'].get('excluded') else ""),
         f"- n_x = {c['n_x']}, n_y = {c['n_y']}, n_u = {c['n_u']}; dt = {c['dt']:.6g} s, N = {c['N']}, T = {c['T']:g} s, t_ref = {c['t_ref']:g} s",
         f"- measured: {', '.join(c['sensor_names'])}",
         f"- unmeasured: {', '.join(c['state_names'][j] for j in c['unmeasured'])}",
         f"- partial model misses: {c.get('unmodelled_term')}",
         f"- input box: u in {c['input_box']['u_range']} (actuator: {c['actuator'].get('kind')})",
         f"- information group: {c.get('group')}", "",
         "## Difficulty axes", "",
         "| axis | value | provenance |", "|---|---|---|"]
    for lvl in ("white", "realistic"):
        b = ib.get(lvl)
        if b:
            L.append(f"| info bound, unmeasured NRMSE ({lvl}): window 3/alpha / full past / ratio | "
                     f"{f(b['window_3_over_alpha'])} / {f(b['full_past'])} / {f(b['window_ratio'], '{:.2f}')} | `{b['provenance']}` |")
    r = d.get("repairability")
    if r:
        L.append(f"| repairability rho (1 = grey-box residual removes all output model error) | {f(r['rho'], '{:.3f}')} | `{r['provenance']}` |")
    ly = d.get("lyapunov")
    if ly:
        L.append(f"| largest Lyapunov exponent (1/s, id inputs) | {f(ly['lambda_max_mean_id'], '{:.3f}')} | `{ly['provenance']}` |")
    o = d.get("observability")
    if o:
        L.append(f"| observability Gramian cond (median) / full-rank fraction | {f(o['gramian_cond_median'], '{:.3g}')} / {f(o['frac_full_rank'], '{:.2f}')} | `{o['provenance']}` |")
    inc = d.get("inclusion")
    if inc:
        L.append(f"| inclusion: observable / weak / bounded | {inc['observable']} / {inc['weakly_observable']} / {inc['bounded']} | `{inc['provenance']}` |")
    cols = list(next(iter(c['episodes'].values())).keys())
    L += ["", "## Episodes per split", "", "| data variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for v, n in c["episodes"].items():
        L.append(f"| {v} | " + " | ".join(str(n.get(k, "")) for k in cols) + " |")
    L += ["", "Knowledge levels: " + "; ".join(f"`{k}` = {v}" for k, v in c["knowledge_levels"].items())]
    return "\n".join(L) + "\n"
