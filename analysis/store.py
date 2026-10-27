"""Read-only access to the experiment tree for the paper's results layer.

Everything here READS experiments/tree/*/runs/*/seed*/{results.json, traj.npz, log.txt}
and the node.yaml files; nothing under experiments/ is ever written. A slim copy of each
results.json (only the fields the paper uses) is memoised in analysis/.cache keyed by the
file's (size, mtime), so re-runs at 05:00 / 08:00 only parse new or changed runs.

Conventions used by every caller
    runs(node)            {system: {seed: slim payload}}   (only finished runs)
    expected(node)        [(system, seed)] the node's job grid (tree.jobs_of)
    episodes(r, tier, cond, method, key='nrmse') -> 1-D array over that seed's episodes
    climatology(...)      NRMSE of the constant mean-state predictor (see `climatology`)
    rel_model_error(...)  median ||f_real - f_model|| / ||f_real|| over a Sobol sample
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyperkkl import tree as Tr                     # noqa: E402  (read-only helpers)
from hyperkkl.config import get as cfg_get          # noqa: E402

TREE = ROOT / "experiments" / "tree"
CACHE = Path(__file__).resolve().parent / ".cache"

# ---------------------------------------------------------------------------
# Method families (one place; figures, tables and rules all use these)
# ---------------------------------------------------------------------------
# Three observer groups (used everywhere):
#   KKL           learned KKL observers (this paper)
#   MODEL-BASED   filters that hold the modeller's model (FILTERS = nominal + robust/oracle)
#   DATA-DRIVEN   same-data baselines: learned from the same real-plant data, not KKL-structured
LOCATIONS = ("latent", "concat", "film", "lora", "hyper")         # C1 conditioning locations
KKL = ("autonomous",) + LOCATIONS
LEARNED_C3 = LOCATIONS                                            # "best KKL" in C3 (C3 nodes run latent/lora/hyper)
FILTERS_NOMINAL = ("ekf", "ukf", "pf", "kkl_mb")
FILTERS_ROBUST = ("ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle", "ekf_oracle_huber")
FILTERS = FILTERS_NOMINAL + FILTERS_ROBUST                        # = MODEL-BASED
MODEL_BASED = FILTERS
DATA_DRIVEN_NEW = ("koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
DATA_DRIVEN = ("nlox",) + DATA_DRIVEN_NEW
LEARNED_BASELINES = DATA_DRIVEN


DISPLAY = {"lora": "HyperKKL (r=4)", "hyper": "dense-subspace generator (ablation)"}
SHORT_DISPLAY = {"lora": "HyperKKL", "hyper": "dense (abl.)"}
MACRO_NAME = {"lora": "HyperKKL", "hyper": "DenseAblation"}


def disp(m: str, short: bool = False) -> str:
    return (SHORT_DISPLAY if short else DISPLAY).get(m, m)


def group_of(method: str) -> str:
    m = method.split("@")[0]
    return "KKL" if m in KKL else "model-based" if m in FILTERS else "data-driven" if m in DATA_DRIVEN else "other"
TIERS = ("id", "shape_ood", "sequence_ood")
OOD_TIERS = ("shape_ood", "sequence_ood")
KEEP_EP = ("nrmse", "nrmse_unmeasured", "diverged")


class use_tree:
    """Context manager: point hyperkkl.tree (node store, run dirs, queue) at another tree root,
    e.g. the archived protocol-v1 tree experiments/tree_v1/tree. Read-only use only."""

    def __init__(self, tree_dir: Path, queue_dir: Path = None):
        self.t, self.q = Path(tree_dir), Path(queue_dir) if queue_dir else Path(tree_dir).parent / "queue"

    def __enter__(self):
        self.old = (Tr.TREE, Tr.QUEUE)
        Tr.TREE, Tr.QUEUE = self.t, self.q
        return self

    def __exit__(self, *a):
        Tr.TREE, Tr.QUEUE = self.old


def system_roles() -> dict:
    """{system: {'role', 'nx', 'ny', 't_ref', 'T', 'nz', 'alpha', 'cfg'}} from configs/systems."""
    out = {}
    for p in sorted((ROOT / "configs" / "systems").glob("*.yaml")):
        c = yaml.safe_load(p.read_text())
        out[c["name"]] = {"role": c.get("role"), "nx": len(c["ic_box"]), "ny": len(c.get("sensors") or []),
                          "t_ref": c.get("t_ref"), "T": c.get("T"), "nz": c.get("nz"),
                          "alpha": c.get("alpha"), "cfg": c}
    return out


ROLES = system_roles()
DEV_ALL = tuple(s for s, v in ROLES.items() if v["role"] == "dev")
TEST = tuple(s for s, v in ROLES.items() if v["role"] == "test")

# ---------------------------------------------------------------------------
# Post-hoc exclusion (author decision 2026-10-26, ~14:00 Dubai). ONE setting, applied to every headline
# count, verdict, macro, main figure and main table (k/5 instead of k/6). The excluded systems' results stay
# in the appendix (tab_excluded_duffing), and every verdict keeps its as-pre-registered k/6 reading next to the
# k/5 one (tab4_decisions, outcomes.json: original_label / excluded_reading). Pre-registered thresholds k/6 are
# scaled to ceil(k/6 * 5): >= 4/6 -> >= 4/5, >= 5/6 -> >= 5/5.
# ---------------------------------------------------------------------------
EXCLUDED_SYSTEMS = {"duffing"}
EXCLUSION_NOTE = ("post-hoc exclusion (author decision 2026-10-26): duffing excluded as non-discriminating - a single "
                  "missing cubic term that a small learned residual repairs trivially; decided after seeing results")
EXCLUDE = {"on": True}            # headline reading; evidence.analyse_all_both switches it off for the k/6 passes


def excluded(system: str) -> bool:
    return EXCLUDE["on"] and system in EXCLUDED_SYSTEMS


def dev() -> tuple:
    """Dev systems of the CURRENT reading (all six in the pre-registered pass, five in the headline)."""
    return tuple(s for s in DEV_ALL if not excluded(s))


DEV = tuple(s for s in DEV_ALL if s not in EXCLUDED_SYSTEMS)    # headline dev set (figures, main tables)


def _slim(r: dict) -> dict:
    """The subset of a results.json the paper layer uses."""
    cfg = r.get("config") or {}
    res = {}
    for tier, conds in (r.get("results") or {}).items():
        for cname, b in conds.items():
            res.setdefault(tier, {})[cname] = {
                "families": b.get("families"),
                "methods": {m: {k: v[k] for k in KEEP_EP if k in v} for m, v in (b.get("methods") or {}).items()},
                "scorecard": b.get("scorecard") or {}}
    sc = r.get("scales") or {}
    return {
        "node": r.get("node"), "system": r.get("system"), "seed": r.get("seed"), "code": r.get("code"),
        "config": cfg,
        "scales": {k: sc.get(k) for k in ("x_std", "x_mean", "dt", "N", "domain")}
                  | {"alpha": (sc.get("design_check") or {}).get("alpha")},
        "phase1": (r.get("phase1") or {}).get("diag") or {},
        "train": {m: {k: v.get(k) for k in ("n_params", "width", "ctx_params", "wall_s", "val", "val_nrmse", "lr",
                                            "rank", "arch") if k in v}
                  for m, v in (r.get("train") or {}).items()},
        "memory_probe": r.get("memory_probe"),
        "cost": r.get("cost") or {},
        "oracle_tuning": r.get("oracle_tuning"),
        "results": res,
    }


class Store:
    """Lazy, memoised reader. ``problems`` collects every data issue met on the way."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._runs = {}
        self._clim = {}
        self.problems = []          # (node, message)
        CACHE.mkdir(parents=True, exist_ok=True)

    # -- nodes -------------------------------------------------------------
    @staticmethod
    def nodes() -> dict:
        return Tr.all_nodes()

    @staticmethod
    def node(nid: str) -> dict:
        return Tr.load_node(nid)

    @staticmethod
    def expected(nid: str) -> list:
        return [(s, k) for s, k in Tr.jobs_of(nid) if not excluded(s)]

    @staticmethod
    def progress(nid: str) -> dict:
        return Tr.progress(nid)

    def problem(self, nid, msg):
        if (nid, msg) not in self.problems:
            self.problems.append((nid, msg))

    # -- runs --------------------------------------------------------------
    def _read(self, p: Path):
        st = p.stat()
        key = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", str(p.resolve().relative_to(ROOT / "experiments"))) + ".pkl")
        stamp = (st.st_size, st.st_mtime_ns)
        if self.use_cache and key.exists():
            try:
                with open(key, "rb") as fh:
                    s, obj = pickle.load(fh)
                if s == stamp:
                    return obj
            except Exception:
                pass
        obj = _slim(json.loads(p.read_text()))
        with open(key, "wb") as fh:
            pickle.dump((stamp, obj), fh)
        return obj

    def runs(self, nid: str) -> dict:
        """Finished runs of the node; systems excluded in the current reading are left out."""
        full = self._runs_all(nid)
        return {s: v for s, v in full.items() if not excluded(s)}

    def _runs_all(self, nid: str) -> dict:
        if nid in self._runs:
            return self._runs[nid]
        out = {}
        for s, k in Tr.jobs_of(nid):
            p = Tr.run_dir(nid, s, k) / "results.json"
            if not p.exists():
                continue
            try:
                out.setdefault(s, {})[k] = self._read(p)
            except Exception as e:                       # half-written file during the wave
                self.problem(nid, f"{s}/seed{k}: results.json unreadable ({type(e).__name__}); treated as pending")
        # integrity: one code version and one config per system (tree.report's warning)
        codes = sorted({r["code"] for ss in out.values() for r in ss.values()})
        if len(codes) > 1:
            by = {}
            for s_, ss in out.items():
                for k, r in ss.items():
                    by.setdefault(r["code"], []).append(f"{s_}/s{k}")
            self.problem(nid, f"runs come from {len(codes)} code versions: " +
                         "; ".join(f"{c}: {', '.join(sorted(v))}" for c, v in sorted(by.items())))
        for s, ss in out.items():
            d = {Tr._config_digest(r["config"]) for r in ss.values()}
            if len(d) > 1:
                self.problem(nid, f"{s}: seeds ran with {len(d)} different configs")
        self._runs[nid] = out
        return out

    def pooled(self, nids) -> dict:
        """Seeds of several nodes merged per system (control node + its '-s2' third seed).
        Within-seed pairing is unaffected; a seed present in two nodes keeps the first."""
        out = {}
        for nid in nids:
            for s, ss in self.runs(nid).items():
                for k, r in ss.items():
                    out.setdefault(s, {}).setdefault(k, r)
        return out

    def expected_pooled(self, nids) -> list:
        return sorted({j for n in nids for j in self.expected(n)})

    # -- episode access ----------------------------------------------------
    @staticmethod
    def block(r, tier, cond="clean"):
        return ((r.get("results") or {}).get(tier) or {}).get(cond)

    @staticmethod
    def episodes(r, tier, cond, method, key="nrmse"):
        b = Store.block(r, tier, cond)
        if not b or method not in b["methods"] or key not in b["methods"][method]:
            return None
        return np.asarray(b["methods"][method][key], float)

    def per_seed(self, seeds: dict, tier, cond, method, key="nrmse", seed_list=None):
        """[episode array per seed] in seed order, or None if any seed lacks the block."""
        ks = sorted(seeds) if seed_list is None else seed_list
        out = []
        for k in ks:
            v = self.episodes(seeds[k], tier, cond, method, key)
            if v is None:
                return None
            out.append(v)
        return out

    @staticmethod
    def methods(seeds: dict, tier, cond="clean") -> list:
        out = []
        for r in seeds.values():
            b = Store.block(r, tier, cond)
            for m in (b or {}).get("methods", {}):
                if m not in out:
                    out.append(m)
        return out

    # -- climatology ---------------------------------------------------------
    def climatology(self, nid: str, system: str, tier: str) -> tuple:
        """(value, how). NRMSE of the CONSTANT predictor xhat = mean real-plant state, with
        the same normalisation (sigma_j = scales.x_std) and the same settled window as every
        method, averaged over episodes like the methods' NRMSE.

        OOD tiers: the mean and the episode NRMSEs come from the saved TRUE evaluation
        trajectories of this node (traj.npz, 4 clean episodes per seed, pooled over seeds).
        id (no saved trajectories): analytic value 1 when the real plant is the nominal one
        (sigma_j IS the per-state std of id trajectories of the nominal plant, so the mean
        predictor scores 1); for a perturbed real plant, the mean of the two OOD-tier values
        (same amplitude box; flagged 'proxy')."""
        key = (nid, system, tier)
        if key in self._clim:
            return self._clim[key]
        seeds = self.runs(nid).get(system, {})
        val, how = float("nan"), "unavailable"
        if tier == "id":
            any_r = next(iter(seeds.values()), None)
            mism = float(cfg_get(any_r["config"], "system.mismatch") or 0.0) if any_r else 0.0
            if any_r is None:
                pass
            elif mism == 0.0:
                val, how = 1.0, "analytic (nominal plant)"
            else:
                v = [self.climatology(nid, system, t)[0] for t in OOD_TIERS]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    val, how = float(np.mean(v)), "proxy (mean of OOD tiers; perturbed plant)"
        else:
            X = []
            xs = None
            for k, r in sorted(seeds.items()):
                p = Tr.run_dir(nid, system, k) / "traj.npz"
                xs = np.asarray(r["scales"]["x_std"], float)
                if not p.exists():
                    continue
                try:
                    z = np.load(p)
                    name = f"{tier}/clean/true"
                    if name in z.files:
                        X.append(z[name])
                except Exception:
                    self.problem(nid, f"{system}/seed{k}: traj.npz unreadable")
            if X:
                X = np.concatenate(X, 0)                    # (E, N+1, nx)
                st = int(0.5 * (X.shape[1] - 1))
                S = X[:, st:]
                mu = S.reshape(-1, S.shape[-1]).mean(0)
                ne = (S - mu) / xs
                val = float(np.mean(np.sqrt((ne ** 2).mean((1, 2)))))
                how = f"empirical ({X.shape[0]} true episodes)"
            elif seeds:
                val, how = 1.0, "analytic fallback (no traj.npz)"
                self.problem(nid, f"{system}/{tier}: no saved trajectories; climatology = 1 (analytic)")
        self._clim[key] = (val, how)
        return val, how


# ---------------------------------------------------------------------------
# Relative model error of the modeller's model against the real plant
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _rme(system: str, knowledge: str, mismatch: float, box_key: tuple, n: int = 4096) -> float:
    import torch
    from scipy.stats import qmc
    from hyperkkl.systems import build_plant
    sc = ROLES[system]["cfg"]
    real = build_plant(sc, perturb=mismatch)
    model = build_plant(sc, knowledge=knowledge)
    box = np.asarray(box_key, float)
    m = int(math.ceil(math.log2(n)))
    Z = qmc.Sobol(d=len(box) + real.nu, scramble=True, seed=0).random_base2(m)[:n]
    x = torch.as_tensor(box[:, 0] + Z[:, :len(box)] * (box[:, 1] - box[:, 0]), dtype=torch.float64)
    u = torch.as_tensor(float(sc.get("u_scale", 1.0)) * (2 * Z[:, len(box):] - 1), dtype=torch.float64)
    with torch.no_grad():
        fr, fm = real.f(x, u), model.f(x, u)
    ok = torch.isfinite(fr).all(1) & torch.isfinite(fm).all(1) & (fr.norm(dim=1) > 0)
    r = (fr - fm)[ok].norm(dim=1) / fr[ok].norm(dim=1)
    return float(r.median())


def rel_model_error(system: str, knowledge: str = "full", mismatch: float = 0.0, domain=None) -> float:
    """Median over a Sobol sample of (x in the operating box, u in the input box) of
    ||f_real(x,u) - f_model(x,u)|| / ||f_real(x,u)||. f_real: the real plant (parameters
    off by `mismatch`, full dynamics); f_model: nominal parameters with `knowledge`.
    Box: the system's ic_box (this reproduces the interim digest's medians for 5/6 dev
    systems; scales.domain can be passed instead). 0 for the exact model."""
    if knowledge == "full" and not mismatch:
        return 0.0
    box = domain if domain is not None else ROLES[system]["cfg"]["ic_box"]
    try:
        return _rme(system, knowledge, float(mismatch), tuple(tuple(map(float, b)) for b in box))
    except Exception:
        return float("nan")


def _readme_systems() -> dict:
    """{system: {'input': ..., 'unmodelled': ...}} from the README systems table."""
    out = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0] in ROLES:
            out[cells[0]] = {"input": cells[3], "unmodelled": cells[5].replace("\\|", "|")}
    return out


def unmodelled(system: str) -> str:
    """The partial-knowledge model's missing term: Plant.UNMODELLED, else the README table."""
    from hyperkkl import systems_extra  # noqa: F401  (registers battery, network)
    from hyperkkl.systems import REGISTRY
    plant = ROLES[system]["cfg"]["plant"]
    txt = getattr(REGISTRY.get(plant), "UNMODELLED", "") or ""
    return txt or _readme_systems().get(system, {}).get("unmodelled", "")


def log_rejections(nid: str, system: str, seed: int) -> dict:
    """{tier: n_rejected} parsed from log.txt ('state coverage <tier>: ... n_rej k')."""
    p = Tr.run_dir(nid, system, seed) / "log.txt"
    out = {}
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            m = re.search(r"state coverage (\w+)\s*:.*n_rej (\d+)", line)
            if m:
                out[m.group(1)] = int(m.group(2))
    return out
