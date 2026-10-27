"""Design checks for cartpole_b / two_link_arm_b (physical criteria only; no observer involved). CPU.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/cartpole_bounded/design_check.py cartpole_b
-> design_check_<plant>.json: linearised closed-loop eigenvalues at the operating point, t_ref rule
(median 1/|Re eig| along the zero-input trajectory), protocol rollouts (horizon T, 48 episodes per
input set: train / id / shape_ood / sequence_ood) with |x| quantiles and per-state stds, and the
same for the original plant for comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
sys.path.insert(0, str(ROOT))
from hyperkkl import data as Dm, signals as S  # noqa: E402
from hyperkkl.config import load  # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent
OP = {"cartpole_b": [0, 0, 0, 0], "cartpole": [0, 0, 0, 0], "two_link_arm_b": [-np.pi / 2, 0, 0, 0],
      "two_link_arm": [-np.pi / 2, 0, 0, 0]}


def lin(plant, x0):
    x = torch.tensor(x0, dtype=torch.float64)
    J = torch.func.jacrev(lambda xi: plant.f(xi[None], torch.zeros(1, 1, dtype=torch.float64))[0])(x)
    return np.linalg.eigvals(J.numpy())


def tref_rule(plant, cfg):
    des = Dm.make_design(cfg["system"])
    x0 = np.asarray(cfg["system"]["ic_box"], float).mean(1)[None]
    x, _, _ = Dm.integrate(plant, des, x0, [S.sample_signal("zero", "train", np.random.RandomState(0), des.dt, des.T)])
    taus = []
    for k in range(0, x.shape[1], max(1, x.shape[1] // 50)):
        ev = lin(plant, x[0, k].numpy())
        taus.append(np.median(1 / np.maximum(np.abs(ev.real), 1e-9)))
    return float(np.median(taus))


def rollouts(plant, cfg, n=48):
    scfg = cfg["system"]
    des = Dm.make_design(scfg)
    sp = 0.005 * np.ones(plant.nx)                       # small process noise (scale only for the check)
    x0 = Dm._sobol(np.asarray(scfg["ic_box"], float), n, 1)
    out = {}
    for i, t in enumerate(["train", "id", "shape_ood", "sequence_ood"]):
        fams = S.TRAIN_FAMILIES if t == "train" else S.families_for(t)
        sigs = Dm._signals(fams, n, t, np.random.RandomState(10 + i), des, scfg)
        x, _, _ = Dm.integrate(plant, des, x0, sigs, None)
        xs = x[:, x.shape[1] // 2:].reshape(-1, plant.nx)
        out[t] = {"absmax": x.abs().amax((0, 1)).tolist(), "abs_p99": np.percentile(x.abs().reshape(-1, plant.nx).numpy(), 99, 0).tolist(),
                  "std_settled": xs.std(0).tolist(), "finite": bool(torch.isfinite(x).all())}
    return out


def main(name):
    cfg = load(name)
    plant = build_plant(cfg["system"])
    known = build_plant(cfg["system"], knowledge="partial")
    res = {"plant": name, "op": OP[name], "eig_closed_loop": [complex(e).__repr__() for e in lin(plant, OP[name])],
           "eig_known_model": [complex(e).__repr__() for e in lin(known, OP[name])],
           "tau_slow_median_s": tref_rule(plant, cfg), "t_ref_yaml": cfg["system"]["t_ref"],
           "actuator": {"k": plant.k, "j": plant.j, "x_ref": plant.x_ref, "u_bar": plant.u_bar},
           "rollouts_T": rollouts(plant, cfg)}
    res["t_ref_rule"] = max(1, round(res["tau_slow_median_s"] / 3))
    (HERE / f"design_check_{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rollouts_T"}, indent=1))
    for t, v in res["rollouts_T"].items():
        print(t, "absmax", np.round(v["absmax"], 2), "p99", np.round(v["abs_p99"], 2), "std", np.round(v["std_settled"], 3))


if __name__ == "__main__":
    main(sys.argv[1])
