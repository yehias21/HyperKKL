"""Read-only loader of finished runs + their cached models (no training, no cache writes).
Mirrors job.run_job's cache keys. Any cache miss raises FileNotFoundError (run skipped)."""
import copy, json, os, sys
from pathlib import Path
ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
# import the package from the runs' own code snapshot (experiments/code/a1e5d75cce66.tar.gz, extracted):
# the live hyperkkl/nets.py was edited at 01:14 CET by another process, which changes MODEL_MODULES'
# hash (cache keys) and possibly the unpickled classes. Data/cache paths are pointed at the real tree.
SNAP = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/pkg_frozen")   # live package copied 01:35 CET, nets.py restored from the a1e5d75cce66 snapshot
sys.path.insert(0, str(SNAP))
import torch
from hyperkkl import data as Dm, cache as C
from hyperkkl import systems_extra  # noqa: F401
from hyperkkl.cache import MODEL_MODULES, module_hash
from hyperkkl.config import digest
from hyperkkl.job import method_cfg, _to


def ro_cached(path, build=None, **kw):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache miss (read-only mode): {path.name}")
    return torch.load(path, weights_only=False, map_location="cpu")


assert Path(Dm.__file__).resolve().is_relative_to(SNAP), Dm.__file__
Dm.CACHE = ROOT / "experiments" / "cache"
Dm._cached = ro_cached
C.cached = ro_cached
_STORED = {}
Dm.system_scales = lambda cfg, device="cpu": _STORED["scales"]   # stored scales of the run (no writes)

TREE = ROOT / "experiments" / "tree"
MC = Dm.CACHE / "models"


def load_run(node, system, seed, device, methods=None):
    p = TREE / node / "runs" / system / f"seed{seed}" / "results.json"
    payload = json.loads(p.read_text())
    cfg = copy.deepcopy(payload["config"])
    _STORED["scales"] = payload["scales"]
    scales = payload["scales"]
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, seed, device)
    mh = module_hash(MODEL_MODULES)
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
    base = _to(ro_cached(MC / f"{system}_p1_{k1}.pt"), device)
    trained = {}
    for m in (methods or cfg["methods"]):
        if m not in cfg["methods"]:
            continue
        kind, mc = method_cfg(cfg, m)
        k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
        trained[m] = _to(ro_cached(MC / f"{system}_p2_{kind}_{k2}.pt"), device)
    return payload, cfg, scales, des, data, base, trained
