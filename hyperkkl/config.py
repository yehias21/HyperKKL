"""Configuration: nested dicts from YAML, deep-merged, with dotted overrides.

    cfg = load(system="vdp", overrides={"phase2.lr": 3e-3})
    cfg["phase2"]["lr"]            # plain dict access
    get(cfg, "phase2.lr")          # dotted access
    digest(cfg, "system", "data")  # stable hash of the named sections, for caching

Resolution order (later wins): configs/base.yaml, configs/systems/<system>.yaml (under
the key ``system``), then any number of override dicts (an experiment node and its
ancestors contribute these).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(cfg: dict, key: str, default=None):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def apply_dotted(cfg: dict, overrides: dict) -> dict:
    """Apply {"a.b.c": v} overrides; plain nested dicts are deep-merged."""
    out = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        if "." in k:
            set_(out, k, copy.deepcopy(v))
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def system_names() -> list:
    return sorted(p.stem for p in (CONFIG_DIR / "systems").glob("*.yaml"))


def load(system: str, overrides: list | dict | None = None) -> dict:
    base = yaml.safe_load(open(CONFIG_DIR / "base.yaml"))
    sysc = yaml.safe_load(open(CONFIG_DIR / "systems" / f"{system}.yaml"))
    cfg = deep_merge(base, {"system": sysc})
    if isinstance(overrides, dict):
        overrides = [overrides]
    for ov in overrides or []:
        cfg = apply_dotted(cfg, ov)
    return cfg


_SCI = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def parse_value(s: str):
    """CLI value parsing: YAML scalars/lists, so 1e-3, true, [a, b] all work. YAML 1.1
    reads '1e-3' (no dot) as a STRING, which would reach the optimiser as lr='1e-3'."""
    v = yaml.safe_load(s)
    fix = lambda x: float(x) if isinstance(x, str) and _SCI.fullmatch(x) else x
    return [fix(x) for x in v] if isinstance(v, list) else fix(v)


def digest(cfg: dict, *sections: str, extra=None) -> str:
    payload = {s: get(cfg, s) for s in sections}
    if extra is not None:
        payload["_extra"] = extra
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]
