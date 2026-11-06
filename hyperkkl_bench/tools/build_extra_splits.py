"""Generate the package's extra frozen splits on CPU (run ONCE per data release; outputs are frozen
and checksummed in data_extra/MANIFEST.json):

  val_ood   PUBLIC OOD validation (64 episodes / plant / data variant): the 4 val-OOD input
            generators of experiments/battery_deep/code/valood.py (regen_decay, gated_sine,
            spline, urban_cycle), none of them a training or test generator. Seeds public.
  hidden_*  label-withheld test draws (32 episodes per tier) of the SAME tier generators as the
            public test tiers, with SECRET seeds derived from HYPERKKL_HIDDEN_SECRET. The public
            file has no x and no x0; labels (x, x0) go to the private labels root.

Everything else is exactly the data generator of the paper (hyperkkl.data.generate): same plant
config, real plant (perturb 0 or 0.3), process noise 0.005 x_std, k_cross = 1, rejection at 1.5x
the envelope, sensor of the variant (white 3 % / colored Student-t). The mismatch30 variant reuses
the realistic trajectories (same seeds, same real plant) with white noise re-drawn, as in the
export (EXPORT.md).

Needs the research repo (``hyperkkl`` package + experiments/information/infolib.py) - this is a
maintainer tool, not part of the installed package.

    CUDA_VISIBLE_DEVICES="" HYPERKKL_HIDDEN_SECRET=... python tools/build_extra_splits.py [plants...]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parent
REPO = CHECKOUT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "information"))
sys.path.insert(0, str(HERE))

DEV = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
TEST = ["cartpole_b", "fhn", "highway_traffic", "network", "rossler", "two_link_arm"]
N_VALOOD, N_HIDDEN = 64, 32
VALOOD_BASE, HIDDEN_BASE = 4_400_000, 7_700_000


def nodes(p):
    if p in DEV:
        return {"white": "C1-where", "realistic": "C3-realistic"}
    if p == "cartpole_b":
        return {"white": "TEST-C1-B", "realistic": "TEST-C3-B"}
    return {"white": "TEST-C1", "realistic": "TEST-C3"}


def seed_of(base, plant, split, secret=""):
    key = f"{secret}|{plant}|{split}".encode()
    return base + int(hashlib.sha256(key).hexdigest(), 16) % 1_000_000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_plant_splits(p):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    torch.set_num_threads(2)
    import infolib as L
    import valood_gen as VG
    from hyperkkl import data as Dm
    from hyperkkl import signals as S
    from hyperkkl import systems_bounded  # noqa: F401
    from hyperkkl.systems import build_plant

    secret = os.environ["HYPERKKL_HIDDEN_SECRET"]
    out_pub = CHECKOUT / "data_extra" / "data" / p
    out_lab = CHECKOUT / "hidden_labels" / "data" / p
    log = []

    def nested(a):
        E, T, d = a.shape
        vals = pa.array(a.reshape(-1).astype(np.float32), type=pa.float32())
        inner = pa.ListArray.from_arrays(pa.array(np.arange(0, E * T * d + 1, d, dtype=np.int32)), vals)
        return pa.ListArray.from_arrays(pa.array(np.arange(0, E * T + 1, T, dtype=np.int32)), inner)

    def flat(a):
        E, T = a.shape
        vals = pa.array(a.reshape(-1).astype(np.float32), type=pa.float32())
        return pa.ListArray.from_arrays(pa.array(np.arange(0, E * T + 1, T, dtype=np.int32)), vals)

    def write(es, y, path, with_labels=True):
        E = len(es)
        cols = {"episode": pa.array(np.arange(E, dtype=np.int32)),
                "family": pa.array(list(es.families), type=pa.string()),
                "signal": pa.array([json.dumps({k: v for k, v in m.items() if k != "ic"}, default=str)
                                    for m in es.meta]),
                "dt": pa.array(np.full(E, es.dt, dtype=np.float64)),
                "u": flat(es.u[..., 0].numpy()), "y": nested(y.numpy())}
        x0 = nested(np.stack([np.asarray(m["ic"], np.float32) for m in es.meta])[:, None, :]).flatten()
        if with_labels:
            cols["x0"] = x0
            cols["x"] = nested(es.x.numpy())
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), path, compression="zstd", compression_level=6, row_group_size=32)
        return E, x0

    def write_labels(es, x0, path):
        E = len(es)
        tbl = pa.table({"episode": pa.array(np.arange(E, dtype=np.int32)), "x0": x0, "x": nested(es.x.numpy())})
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, path, compression="zstd", compression_level=6, row_group_size=32)

    for var_src in ("white", "realistic"):
        pay = L.run_payload(p, nodes(p)[var_src])
        cfg, scales = pay["config"], pay["scales"]
        scfg = {k: v for k, v in cfg["system"].items() if k != "knowledge"}
        des = Dm.make_design(scfg)
        plant = build_plant(scfg, perturb=float(cfg["system"].get("mismatch", 0.0)))
        dcfg = dict(cfg["data"], k_cross=1)
        t_ref = float(scfg.get("t_ref", 1.0))
        jobs = [("val_ood", None, N_VALOOD, seed_of(VALOOD_BASE, p, "val_ood"))]
        jobs += [(f"hidden_{t}", t, N_HIDDEN, seed_of(HIDDEN_BASE, p, f"hidden_{t}", secret)) for t in S.TIERS]
        for split, tier, n, seed in jobs:
            orig = Dm._signals
            if tier is None:
                Dm._signals = VG.valood_signals
                fams, regime = VG.FAMILIES, "train"
            else:
                fams, regime = S.families_for(tier), tier
            try:
                es = Dm.generate(plant, des, scales, scfg, dcfg, fams, n, regime, seed, "cpu")
            finally:
                Dm._signals = orig
            variants = {"white": es.y} if var_src == "white" else {
                "realistic": es.y,
                "mismatch30": Dm.measure(es.y_clean.double(), {"noise": 0.03}, scales["y_std"], seed,
                                         dt=des.dt, t_ref=t_ref).float()}
            for v, y in variants.items():
                hidden = split.startswith("hidden_")
                E, x0 = write(es, y, out_pub / v / f"{split}.parquet", with_labels=not hidden)
                if hidden:
                    write_labels(es, x0, out_lab / v / f"{split}.parquet")
                log.append({"plant": p, "variant": v, "split": split, "episodes": E,
                            "rejected_redrawn": int(es.n_rejected), "seed": None if hidden else seed,
                            "N1": int(es.x.shape[1])})
                print(p, v, split, E, es.x.shape, flush=True)
    return log


def main():
    if not os.environ.get("HYPERKKL_HIDDEN_SECRET"):
        sys.exit("set HYPERKKL_HIDDEN_SECRET (keep it with the hidden labels, never publish it)")
    plants = sys.argv[1:] or DEV + TEST
    with Pool(min(len(plants), 12)) as pool:
        logs = pool.map(build_plant_splits, plants)
    man = {"generated_by": "tools/build_extra_splits.py", "n_val_ood": N_VALOOD, "n_hidden_per_tier": N_HIDDEN,
           "splits": [r for lg in logs for r in lg], "files": {}, "label_files": {}}
    for root, key in ((CHECKOUT / "data_extra", "files"), (CHECKOUT / "hidden_labels", "label_files")):
        for f in sorted((root / "data").rglob("*.parquet")):
            man[key][str(f.relative_to(root))] = sha256(f)
    old = CHECKOUT / "data_extra" / "MANIFEST.json"
    if old.exists() and len(plants) < 12:
        prev = json.loads(old.read_text())
        prev["splits"] = [r for r in prev["splits"] if r["plant"] not in plants] + man["splits"]
        prev["files"].update(man["files"]); prev["label_files"].update(man["label_files"])
        man = prev
    lab = man.pop("label_files")
    old.write_text(json.dumps(man, indent=1))
    (CHECKOUT / "hidden_labels" / "MANIFEST.json").write_text(json.dumps({"label_files": lab}, indent=1))


if __name__ == "__main__":
    main()
