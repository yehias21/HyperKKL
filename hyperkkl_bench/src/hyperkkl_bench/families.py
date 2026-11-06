"""hyperkkl-bench 2.0: the two families of state estimation.

    import hyperkkl_bench as hkb
    tr  = hkb.load("driven/battery/realistic", "train", recipe="mixed")    # operating profiles + event-structured
    te  = hkb.load("driven/battery/realistic", "test_op_ood")              # different operator, same actuator / box
    ta  = hkb.load("autonomous/vdp/partial", "test_ic_ood")               # u = 0, held-out IC shell
    rep = hkb.evaluate_v2(est, "driven/battery/realistic")                 # every v2 tier x condition, bench scorer

Spec "family/plant/level": family in {autonomous, driven}; level in {exact, partial, mismatch, realistic} (data variant
white / white / mismatch30 / realistic, as v1). Splits: train, val, train_event, val_event (the event-structured
recipe), val_ood, phase1, the test tiers of the family (TIERS) and hidden_*. recipe="mixed" concatenates
train + train_event (val + val_event for "val"); "standard" (default) returns the split as stored.
Data root: HYPERKKL_BENCH_DATA_V2 (folder with data/ and metadata/ of the v2 export).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

FAMILIES = ("autonomous", "driven")
LEVELS = {"exact": ("white", "full"), "partial": ("white", "partial"), "mismatch": ("mismatch30", "full"),
          "realistic": ("realistic", "partial")}
TIERS = {"driven": ("test_op_id", "test_op_ood", "test_legacy_id", "test_legacy_shape_ood", "test_legacy_sequence_ood"),
         "autonomous": ("test_id", "test_ic_ood", "test_param_shift")}
HEADLINE_TIER = {"driven": "test_op_ood", "autonomous": "test_id"}
CONDITIONS = {
    "driven": {"test_op_id": ("clean", "outliers", "burst_outage", "poor_init", "input_realistic",
                              "input_cmd_vs_applied", "input_quant"),
               "test_op_ood": ("clean", "outliers", "burst_outage", "input_realistic"),
               "test_legacy_id": ("clean",), "test_legacy_shape_ood": ("clean",), "test_legacy_sequence_ood": ("clean",)},
    "autonomous": {"test_id": ("clean", "noise10", "outliers", "dropout", "burst_outage", "poor_init"),
                   "test_ic_ood": ("clean",), "test_param_shift": ("clean",)}}
# family A (DESIGN.md section 2): plants with non-trivial unforced dynamics; lorenz only at the nominal-data levels
AUTONOMOUS = ("vdp", "fhn", "rossler", "lorenz", "lorenz96", "network", "lorenz96_n10")
NOMINAL_ONLY = ("lorenz", "rossler")
DRIVEN = ("battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp", "cartpole_b", "fhn", "highway_v2",
          "network", "rossler", "two_link_arm_v2", "lorenz96")


def data_root_v2(root=None) -> Path:
    r = root or os.environ.get("HYPERKKL_BENCH_DATA_V2")
    if r is None:
        here = Path(__file__).resolve()
        for cand in (here.parents[2] / "data_v2", here.parents[3] / "experiments" / "bench_v2" / "export" / "data_v2",
                     here.parents[4] / "export" / "data_v2"):
            if (cand / "metadata").exists():
                return cand
        raise FileNotFoundError("v2 data not found: set HYPERKKL_BENCH_DATA_V2")
    return Path(r)


def parse(spec: str):
    parts = spec.strip("/").split("/")
    if len(parts) != 3:
        raise ValueError(f"spec must be 'family/plant/level', got {spec!r}")
    fam, plant, level = parts
    if fam not in FAMILIES:
        raise ValueError(f"unknown family {fam!r}; choose from {FAMILIES}")
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose from {list(LEVELS)}")
    allowed = AUTONOMOUS if fam == "autonomous" else DRIVEN
    if plant not in allowed:
        why = (" (it comes to rest at u = 0: fails the 'non-trivial unforced dynamics' inclusion check)"
               if fam == "autonomous" and plant in DRIVEN else "")
        raise ValueError(f"{plant!r} is not {'an' if fam == 'autonomous' else 'a'} {fam} plant{why}; {fam} plants: {allowed}")
    if fam == "autonomous" and plant in NOMINAL_ONLY and LEVELS[level][0] != "white":
        why = {"lorenz": "the +0.3 perturbed plant collapses to a fixed point",
               "rossler": "from the held-out IC shell the +0.3 plant leaves the envelope; the shell and param-shift tiers cannot be built"}[plant]
        raise ValueError(f"autonomous/{plant}: no {level} cell ({why})")
    return fam, plant, level


@lru_cache(maxsize=64)
def _meta(fam, plant, root=None):
    return json.loads((data_root_v2(root) / "metadata" / fam / f"{plant}.json").read_text())


def metadata_v2(spec: str, root=None) -> dict:
    fam, plant, level = parse(spec)
    m = dict(_meta(fam, plant, root))
    v = m["variants"][LEVELS[level][0]]
    m.update(level=level, variant=LEVELS[level][0], x_std=v["x_std"], y_std=v["y_std"], x_mean=v["x_mean"],
             model_knowledge=LEVELS[level][1])
    return m


def load(spec: str, split: str = "train", recipe: str = "standard", root=None):
    """-> hyperkkl_bench.data.EpisodeSet (u (E, N+1), y, x or None, dt, ...); profile test tiers carry ``cmd``."""
    import pyarrow.parquet as pq
    from hyperkkl_bench.data import EpisodeSet, _stack
    fam, plant, level = parse(spec)
    if recipe not in ("standard", "mixed"):
        raise ValueError(recipe)
    if recipe == "mixed" and split in ("train", "val"):
        a, b = load(spec, split, root=root), load(spec, f"{split}_event", root=root)
        return _cat(a, b)
    m = metadata_v2(spec, root)
    f = data_root_v2(root) / "data" / fam / plant / m["variant"] / f"{split}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} (splits: train, val, train_event, val_event, val_ood, phase1, {TIERS[fam]}, hidden_*)")
    tb = pq.read_table(f)
    nx, ny, names = int(m["n_x"]), int(m["n_y"]), tb.column_names
    es = EpisodeSet(u=_stack(tb.column("u")), y=_stack(tb.column("y"), ny),
                    x=_stack(tb.column("x"), nx) if "x" in names else None, dt=float(tb.column("dt")[0].as_py()),
                    plant=plant, knowledge=level, split=split, families=tb.column("family").to_pylist(),
                    signals=tb.column("signal").to_pylist(),
                    x0=_stack(tb.column("x0")) if "x0" in names else np.full((tb.num_rows, nx), np.nan, np.float32),
                    episode_ids=np.asarray(tb.column("episode").to_numpy()), source=str(f))
    es.cmd = _stack(tb.column("cmd")) if "cmd" in names else None
    es.family_v2 = fam
    return es


def _cat(a, b):
    from hyperkkl_bench.data import EpisodeSet
    out = EpisodeSet(np.concatenate([a.u, b.u]), np.concatenate([a.y, b.y]), np.concatenate([a.x, b.x]), a.dt,
                     a.plant, a.knowledge, a.split + "+event", list(a.families) + list(b.families),
                     list(a.signals) + list(b.signals), np.concatenate([a.x0, b.x0]),
                     np.concatenate([a.episode_ids, b.episode_ids + len(a)]), a.source + " + " + b.source)
    out.cmd, out.family_v2 = None, getattr(a, "family_v2", None)
    return out


def context_v2(spec: str, root=None) -> dict:
    """What a method may know (as hyperkkl_bench.context.build_context): metadata without the real-plant parameters,
    nominal noise, prior, and the modeller's model of the level (NOMINAL parameters; f_known at partial / realistic)."""
    from hyperkkl_bench.plants import build_plant
    fam, plant, level = parse(spec)
    m = metadata_v2(spec, root)
    box = np.asarray(m["ic_box"], float)
    model = build_plant(m["system_config"], knowledge=LEVELS[level][1])
    sensor = {"noise": 0.03} if level != "realistic" else {"noise": 0.03, "kind": "colored_t", "rho": 0.9, "nu": 3.0}
    meta = {k: m[k] for k in ("family", "plant", "n_x", "n_y", "n_u", "sensors", "unmeasured", "dt", "N", "T", "t_ref",
                              "nz", "alpha", "settle_index", "ic_box", "x_std", "y_std", "x_mean", "process_noise_rel")}
    meta["input_box"] = {"u_scale": float(m["system_config"].get("u_scale", 1.0)), **({"box": m["input"]["box"]} if fam == "driven" else {})}
    return {"plant": plant, "family": fam, "knowledge": level, "meta": meta, "dt": float(m["dt"]),
            "sensors": list(m["sensors"]), "unmeasured": list(m["unmeasured"]),
            "x_std": np.asarray(m["x_std"], float), "y_std": np.asarray(m["y_std"], float),
            "x_mean": np.asarray(m["x_mean"], float),
            "q_std": float(m["process_noise_rel"]) * np.asarray(m["x_std"], float),
            "r_std": 0.03 * np.asarray(m["y_std"], float), "sensor_model": sensor,
            "prior_mean": box.mean(1), "prior_std": (box[:, 1] - box[:, 0]) / np.sqrt(12.0),
            "model": model, "model_knowledge": LEVELS[level][1]}


def _cmd_for(es, plant, dt, T):
    """The operator command of each profile episode, regenerated from its profile seed (float64, as the study);
    falls back to the stored float32 ``cmd`` column."""
    from .plants import profiles_v2 as PV
    out = []
    for i, sg in enumerate(es.signals):
        m = json.loads(sg) if isinstance(sg, str) else dict(sg)
        if "profile_seed" not in m:
            return None if getattr(es, "cmd", None) is None else es.cmd.astype(np.float64)
        sp = m["profile_split"]
        out.append(PV.generate(plant, "train" if sp == "train" else sp, int(m["profile_seed"]), T=T, dt=dt).cmd)
    return np.stack(out)


def apply_condition(cond: str, es, spec: str, root=None):
    """-> (U_obs, Y, X0, P0std) for one condition. EXACTLY the conditions of the v2 reference study
    (experiments/bench_v2/code/conditions_v2.py): torch CPU generator seeded by sha256(family|plant|variant|split|cond),
    the same draw order, float32 arithmetic on y, poor prior = centre + 0.9 half-width x random signs. Draws depend on
    the episode count, so scoring a subset (n_episodes) gives other realisations than the 64-episode reference.

    sensor: outliers (2 % of samples +-12 y_std), burst_outage (frozen at 0.6 T for 2 t_ref), poor_init, noise10
    (white noise raised to 10 % y_std; family A), dropout (10 % held samples; family A).
    input record (family B): input_realistic (plant preset), input_cmd_vs_applied (observer gets the operator command),
    input_quant (12-bit over the actuator span). Only the observer's input changes; the plant got the applied u."""
    import hashlib
    import math
    import torch
    from .plants import profiles_v2 as PV
    fam, plant, level = parse(spec)
    m = metadata_v2(spec, root)
    split = es.split.replace("+event", "")
    s = int(hashlib.sha256("|".join(map(str, (fam, plant, m["variant"], split, cond))).encode()).hexdigest(), 16) % (2 ** 31)
    Y = torch.as_tensor(np.array(es.y, dtype=np.float32, copy=True))
    U = es.u.astype(np.float64).copy()
    E, N1, ny = Y.shape
    ys = torch.as_tensor(np.asarray(m["y_std"], float), dtype=Y.dtype)
    box = np.asarray(m["ic_box"], float)
    ctr, half = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    X0 = np.tile(ctr, (E, 1)).astype(np.float32).astype(np.float64)
    P0 = np.tile((box[:, 1] - box[:, 0]) / np.sqrt(12.0), (E, 1))
    g = torch.Generator().manual_seed(s)
    dt, t_ref = float(m["dt"]), float(m["t_ref"])
    if cond == "clean":
        pass
    elif cond == "outliers":
        hit = torch.rand(Y.shape, generator=g) < 0.02
        sg = torch.where(torch.rand(Y.shape, generator=g) < 0.5, -1.0, 1.0)
        Y = torch.where(hit, Y + 12.0 * ys * sg, Y)
    elif cond == "burst_outage":
        k0 = int(round(0.6 * (N1 - 1)))
        L = max(0, min(int(round(2.0 * t_ref / dt)), N1 - 1 - k0))
        if L > 0:
            Y[:, k0 + 1:k0 + L + 1] = Y[:, k0:k0 + 1]
    elif cond == "poor_init":
        rng = np.random.RandomState(s)
        X0 = (ctr + 0.9 * half * rng.choice([-1.0, 1.0], size=(E, len(ctr)))).astype(np.float32).astype(np.float64)
    elif cond == "noise10":
        Y = Y + math.sqrt(0.10 ** 2 - 0.03 ** 2) * ys * torch.randn(Y.shape, generator=g)
    elif cond == "dropout":
        hold = torch.rand(E, N1, generator=g) < 0.10
        for k in range(1, N1):
            Y[:, k] = torch.where(hold[:, k, None], Y[:, k - 1], Y[:, k])
    elif cond.startswith("input_"):
        if fam != "driven":
            raise ValueError("input conditions exist only in the driven family")
        P = PV.SPECS[plant]
        span = float(P["hi"] - P["lo"])
        imp = {"input_realistic": P["imperf"], "input_cmd_vs_applied": PV.Imperfection(use_command=True),
               "input_quant": PV.Imperfection(quant=span / 4096)}[cond]
        cmd = None
        if imp.use_command:
            cmd = _cmd_for(es, plant, dt, float(m["T"]))
            if cmd is None:
                raise ValueError(f"{cond} needs the operator command (profile tiers only)")
        U = np.stack([PV.observed_input(U[i], dt, imp, span, s + i, cmd=None if cmd is None else cmd[i])
                      for i in range(E)])
    else:
        raise ValueError(f"unknown condition {cond!r}")
    return U, Y.double().numpy(), X0, P0


def evaluate_v2(est, spec: str, tiers=None, fit: bool = True, recipe: str = "mixed", n_episodes=None, root=None,
                verbose=False) -> dict:
    """Fit on the cell's training data (recipe 'mixed' = the default of the v2 study) and score every tier x
    condition with the bench scorer (settled NRMSE, unmeasured headline). Returns a dict report."""
    import time
    from hyperkkl_bench import metrics as M
    from hyperkkl_bench.estimator import run_episodes
    from hyperkkl_bench.version import BENCH_VERSION
    fam, plant, level = parse(spec)
    ctx = context_v2(spec, root)
    t0 = time.time()
    if fit:
        est.fit(load(spec, "train", recipe, root), load(spec, "val", recipe, root), ctx)
    res = {}
    for t in tiers or TIERS[fam]:
        es = load(spec, t, root=root)
        if n_episodes:
            cmd = getattr(es, "cmd", None)
            es = es.subset(np.arange(min(n_episodes, len(es))))
            es.cmd = None if cmd is None else cmd[:len(es)]
        res[t] = {}
        for c in CONDITIONS[fam][t]:
            U, Y, X0, P0 = apply_condition(c, es, spec, root)
            Xh, _ = run_episodes(est, U, Y, X0, P0)
            mm = M.episode_metrics(es.x, Xh, ctx["x_std"], int(ctx["meta"]["settle_index"]), ctx["sensors"])
            res[t][c] = {"summary": M.summarize(mm), "per_episode": {k: np.asarray(v).tolist() for k, v in mm.items()}}
            if verbose:
                print(f"  {spec} {t}/{c}: nrmse_unmeasured {res[t][c]['summary'].get('nrmse_unmeasured', float('nan')):.4g}")
    return {"meta": {"estimator": getattr(est, "name", type(est).__name__), "spec": spec, "recipe": recipe,
                     "bench_version": BENCH_VERSION, "headline_tier": HEADLINE_TIER[fam], "fit_s": time.time() - t0},
            "results": res}


# ---------------------------------------------------------------------------------------------------- hidden test (v2)
HIDDEN = {"driven": ("hidden_op_id", "hidden_op_ood"), "autonomous": ("hidden_id", "hidden_ic_ood")}


def predict_v2(est, spec: str, out_dir, fit: bool = True, recipe: str = "mixed", root=None):
    """Participant side: fit (train/val of the recipe), run every hidden split of the cell from the standard prior,
    write <out_dir>/<family>__<plant>__<level>__<split>.npz (x_hat) and SUBMISSION.json. Labels are never read."""
    from hyperkkl_bench.estimator import run_episodes
    from hyperkkl_bench.version import BENCH_VERSION
    fam, plant, level = parse(spec)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if fit:
        est.fit(load(spec, "train", recipe, root), load(spec, "val", recipe, root), context_v2(spec, root))
    files = []
    for s in HIDDEN[fam]:
        es = load(spec, s, root=root)
        U, Y, X0, P0 = apply_condition("clean", es, spec, root)
        Xh, _ = run_episodes(est, U, Y, X0, P0)
        f = out / f"{fam}__{plant}__{level}__{s}.npz"
        np.savez_compressed(f, x_hat=Xh.astype(np.float32))
        files.append(f.name)
    man = out / "SUBMISSION.json"
    old = json.loads(man.read_text()) if man.exists() else {"files": []}
    old.update(estimator=getattr(est, "name", type(est).__name__), bench_version=BENCH_VERSION,
               files=sorted(set(old["files"]) | set(files)))
    man.write_text(json.dumps(old, indent=1))
    return files


def score_hidden_v2(submission, labels_root=None, root=None) -> dict:
    """Host side: score every v2 hidden file of a submission against the private labels (hidden_labels/v2)."""
    import pyarrow.parquet as pq
    from hyperkkl_bench.data import _stack
    from hyperkkl_bench import metrics as M
    from hyperkkl_bench.paths import labels_root as _lr
    lr = Path(labels_root) if labels_root else _lr() / "v2"
    out = {}
    for f in sorted(Path(submission).glob("*__*__*__hidden_*.npz")):
        fam, plant, level, split = f.stem.split("__")
        spec = f"{fam}/{plant}/{level}"
        m = metadata_v2(spec, root)
        tb = pq.read_table(lr / "data" / fam / plant / m["variant"] / f"{split}.parquet")
        X = _stack(tb.column("x"), int(m["n_x"]))
        Xh = np.load(f)["x_hat"]
        if Xh.shape != X.shape:
            raise ValueError(f"{f.name}: shape {Xh.shape} != labels {X.shape}")
        mm = M.episode_metrics(X, Xh, m["x_std"], int(m["settle_index"]), m["sensors"])
        out[f"{spec}/{split}"] = M.summarize(mm)
    return out
