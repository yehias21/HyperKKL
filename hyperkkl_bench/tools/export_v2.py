"""Export the frozen bench_v2 episodes to the benchmark's Parquet layout (maintainer tool; CPU).

    python export_v2.py [--out DIR] [cells...]          cells like driven/battery or autonomous/vdp (default: all built)

Layout (data_v2/):
  data/<family>/<plant>/<variant>/<split>.parquet     one row per episode: episode, family (operator for profile
                                                      episodes), signal (JSON), dt, u (N+1), y (N+1, ny), x (N+1, nx),
                                                      x0 (nx); profile tiers also cmd (N+1) = the operator's command
  data/<family>/<plant>/<variant>/hidden_*.parquet    no x, no x0, no cmd (labels in <labels>/...)
  metadata/<family>/<plant>.json                      names, sensors, unmeasured, x_std / y_std / x_mean per variant,
                                                      settle_index, design, input box + actuator + operators, tiers,
                                                      conditions, recipe, counts, sha256 per file
Never writes into hyperkkl_bench/ or the research cache. Hidden labels go to --labels (private).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "experiments" / "bench_v2"   # the v2 study (generator + frozen .pt)
sys.path.insert(0, str(HERE / "code"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

import v2cfg  # noqa: E402

TIERS = {"driven": ["test_op_id", "test_op_ood", "test_legacy_id", "test_legacy_shape_ood", "test_legacy_sequence_ood"],
         "autonomous": ["test_id", "test_ic_ood", "test_param_shift"]}
HIDDEN = {"driven": ["hidden_op_id", "hidden_op_ood"], "autonomous": ["hidden_id", "hidden_ic_ood"]}
PUBLIC = ["train", "train_event", "val", "val_event", "val_ood", "phase1"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nested(a):
    import pyarrow as pa
    E, T, d = a.shape
    vals = pa.array(a.reshape(-1).astype(np.float32), type=pa.float32())
    inner = pa.ListArray.from_arrays(pa.array(np.arange(0, E * T * d + 1, d, dtype=np.int32)), vals)
    return pa.ListArray.from_arrays(pa.array(np.arange(0, E * T + 1, T, dtype=np.int32)), inner)


def flat(a):
    import pyarrow as pa
    E, T = a.shape
    vals = pa.array(a.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.ListArray.from_arrays(pa.array(np.arange(0, E * T + 1, T, dtype=np.int32)), vals)


def commands(plant, es, des):
    import profiles_v2 as PV
    out = []
    for m in es.meta:
        sp = m["profile_split"]
        pr = PV.generate(plant, "train" if sp == "train" else sp, int(m["profile_seed"]), T=des.T, dt=des.dt)
        out.append(pr.cmd)
    return np.stack(out)


def write(es, path, plant, des, labels=True, cmd=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    E = len(es)
    fam = [m.get("operator", es.families[i]) for i, m in enumerate(es.meta)]
    cols = {"episode": pa.array(np.arange(E, dtype=np.int32)), "family": pa.array(fam, type=pa.string()),
            "signal": pa.array([json.dumps({k: v for k, v in m.items() if k != "ic"}, default=str) for m in es.meta]),
            "dt": pa.array(np.full(E, es.dt, dtype=np.float64)),
            "u": flat(es.u[..., 0].numpy()), "y": nested(es.y.numpy())}
    if labels:
        cols["x"] = nested(es.x.numpy())
        cols["x0"] = flat(np.stack([np.asarray(m["ic"], np.float32) for m in es.meta]))
        if cmd:
            cols["cmd"] = flat(commands(plant, es, des).astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path, compression="zstd", compression_level=6, row_group_size=32)
    return sha256(path)


def export_cell(family, plant, out, labels_root):
    from hyperkkl import data as Dm
    from hyperkkl import systems_extra, systems_bounded, systems_chaos  # noqa: F401  (register every plant)
    import profiles_v2 as PV
    meta = None
    for variant in ("white", "mismatch30", "realistic"):
        d = HERE / "data" / family / plant / variant
        if not (d / "MANIFEST.json").exists():
            continue
        man = json.loads((d / "MANIFEST.json").read_text())
        sc = json.loads((d / "scales.json").read_text())
        cfg = v2cfg.config(family, plant, variant)
        des = Dm.make_design(cfg["system"])
        files = {}
        for s in PUBLIC + TIERS[family]:
            es = torch.load(d / f"{s}.pt", weights_only=False)
            files[s] = {"sha256": write(es, out / "data" / family / plant / variant / f"{s}.parquet", plant, des,
                                        cmd=s.startswith("test_op")), "n": len(es)}
        for s in HIDDEN[family]:
            es = torch.load(HERE / "hidden_labels" / family / plant / variant / f"{s}.pt", weights_only=False)
            files[s] = {"sha256": write(es, out / "data" / family / plant / variant / f"{s}.parquet", plant, des,
                                        labels=False), "n": len(es)}
            write(es, labels_root / "data" / family / plant / variant / f"{s}.parquet", plant, des, labels=True,
                  cmd=family == "driven")
        if meta is None:
            from hyperkkl.systems import build_plant
            pl = build_plant(cfg["system"])
            s_ = cfg["system"]
            meta = {"family": family, "plant": plant, "system": v2cfg.SYSTEM.get(plant, plant),
                    "n_x": len(sc["x_std"]), "n_y": len(pl.sensors), "n_u": 1, "sensors": list(pl.sensors),
                    "unmeasured": [j for j in range(len(sc["x_std"])) if j not in pl.sensors],
                    "unmodelled_term": getattr(pl, "UNMODELLED", None) or getattr(getattr(pl, "base", None), "UNMODELLED", None),
                    "dt": des.dt, "N": des.N, "T": des.T, "t_ref": float(s_.get("t_ref", 1.0)), "nz": int(des.A.shape[0]),
                    "alpha": des.alpha, "settle_index": int(0.5 * des.N), "ic_box": s_["ic_box"],
                    "system_config": {k: v for k, v in s_.items() if k not in ("mismatch",)},
                    "process_noise_rel": float(cfg["data"].get("noise_proc", 0.0)),
                    "tiers": TIERS[family], "hidden": HIDDEN[family], "recipes": {"standard": ["train", "val"],
                                                                                    "mixed": ["train", "train_event", "val", "val_event"]},
                    "variants": {}}
            if family == "driven":
                P = PV.SPECS[plant]
                meta["input"] = {"unit": P["unit"], "physical": list(P["physical"]), "box": [P["lo"], P["hi"]],
                                 "actuator": {k: getattr(P["act"], k) for k in ("lo", "hi", "rate", "tau", "quant", "deadband")},
                                 "train_operators": [o.__name__ for o in P["train"]],
                                 "ood_operators": [o.__name__ for o in P["ood"]],
                                 "record_preset": {k: getattr(P["imperf"], k) for k in
                                                   ("noise", "noise_tau", "bias", "delay", "quant", "aggregate", "dropout", "use_command")},
                                 "uncited": PV.UNCITED.get(plant)}
            else:
                meta["ic_regions"] = {"train/val/test_id": "core box, centre +- 0.6 half-width",
                                      "val_ood": "shell 0.6-0.8 (normalised inf-norm)", "test_ic_ood": "shell 0.8-1.0",
                                      "test_param_shift": "parameters exp(+-0.15) further from the data plant"}
        meta["variants"][variant] = {"x_std": sc["x_std"], "y_std": sc["y_std"], "x_mean": sc["x_mean"],
                                     "mismatch": cfg["system"]["mismatch"], "sensor": cfg["data"]["sensor"],
                                     "files": files, "built": man.get("built")}
    if meta is not None:
        p = out / "metadata" / family / f"{plant}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=1, default=str))
    return meta is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data_v2"))
    ap.add_argument("--labels", default=str(HERE / "hidden_labels" / "parquet"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    cells = a.cells or sorted({f"{p.parts[-3]}/{p.parts[-2]}" for p in (HERE / "data").glob("*/*/*")})
    for c in cells:
        fam, plant = c.split("/")
        ok = export_cell(fam, plant, Path(a.out), Path(a.labels))
        print(c, "exported" if ok else "not built", flush=True)


if __name__ == "__main__":
    main()
