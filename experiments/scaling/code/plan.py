"""The pre-registered job grid (PREREG.md sections 4 and 7), in launch priority order."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SC = HERE.parent
PY = str(SC.parent.parent / ".venv" / "bin" / "python")
NX = (4, 8, 16, 32, 64)
LEVELS = ("partial", "realistic", "exact", "off30")
DATASETS = ("D0", "DR", "D30")
SEEDS = (0, 1, 2)


def _out(nx, seed, unit, data_nx=None):
    sub = f"fixed{data_nx}/" if data_nx else ""
    return SC / "runs" / f"{sub}nx{nx}" / f"seed{seed}" / f"{unit.replace(':', '_')}.json"


def included(nx):
    """Observer runs at n_x only after its inclusion check passed (observable and bounded)."""
    import json
    f = SC / "inclusion" / f"ring{nx}.json"
    if not f.exists():
        return False
    r = json.loads(f.read_text())
    return bool(r["observable"] and r["bounded"])


def _job(nx, seed, unit, data_nx=None, gpu=True, deps=()):
    o = _out(nx, seed, unit, data_nx)
    name = f"{'F' + str(data_nx) + '_' if data_nx else ''}nx{nx}_s{seed}_{unit.replace(':', '_')}"
    cmd = [PY, str(HERE / "job.py"), "--nx", str(nx), "--seed", str(seed), "--unit", unit]
    if data_nx:
        cmd += ["--data-nx", str(data_nx)]
    deps = list(deps) + ([lambda nx=nx: included(nx)] if gpu else [])
    return {"name": name, "cmd": cmd, "gpu": gpu, "done": (lambda o=o: o.exists()), "deps": deps}


def cell(nx, seed, data_nx=None, levels=LEVELS, datasets=DATASETS):
    out = []
    data = {}
    lev_ds = {"exact": "D0", "partial": "D0", "off30": "D30", "realistic": "DR"}
    need = sorted(set(datasets) | {lev_ds[lev] for lev in levels})
    for ds in need:
        d = _job(nx, seed, f"data:{ds}", data_nx, gpu=False)
        data[ds] = d["done"]
        out.append(d)
    for lev in levels:
        out.append(_job(nx, seed, f"kkl:{lev}", data_nx, deps=[data[lev_ds[lev]]]))
    for ds in datasets:
        out.append(_job(nx, seed, f"mf:{ds}", data_nx, deps=[data[ds]]))
    for lev in levels:
        out.append(_job(nx, seed, f"gb:{lev}", data_nx, deps=[data[lev_ds[lev]]]))
    return out


def jobs():
    J = []
    for seed in SEEDS:
        for nx in NX:
            J += cell(nx, seed)
    for seed in SEEDS:                                    # fixed-data variant (largest n_x only)
        J += cell(64, seed, data_nx=4, levels=("partial", "realistic"), datasets=("D0", "DR"))
    for seed in (3, 4):                                   # extra seeds at the two smallest n_x (if time)
        for nx in (4, 8):
            J += cell(nx, seed)
    seen, out = set(), []
    for j in J:
        if j["name"] not in seen:
            seen.add(j["name"]); out.append(j)
    return out
