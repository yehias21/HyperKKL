"""Real-battery final pass (PREREG section 6 / addendum 2): an arm of the campaign trained with
experiments/real's data, design, masked loss and split, evaluated on the 25 degC test windows.

    python run_real.py train --arm A4+D3 --seed 0      (GPU)
    python run_real.py report --arm A4+D3
Reads experiments/real/cache (read-only); writes experiments/beat_greybox/real/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import beat as BG  # noqa: E402
from hyperkkl import realdata as R  # noqa: E402
from hyperkkl import realdata_run as RR  # noqa: E402
from hyperkkl.config import deep_merge  # noqa: E402
from hyperkkl.train import phase1  # noqa: E402

LIVE_REAL = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/real")
OUT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/beat_greybox/real")
MC = HERE / "models_real"
SEEDS = (0, 1, 2)
TEST = {"US06": ["test25/US06"], "HWFET": ["test25/HWFTa", "test25/HWFTb"], "NN": ["test25/NN"]}


def prep():
    return torch.load(LIVE_REAL / "cache" / "prep.pt", weights_only=False)


def spec_for(arm, cfg, des):
    import run_bg as RB
    tags = RB.arm_tags(arm)
    sp = RB.components(tags, cfg)
    assert not sp["design"], "design arms are not applied on the real design (none in the final arm)"
    if "window" in sp["p2"]:
        sp["p2"]["stride"] = max(1, int(round(6.0 / (des.alpha * des.dt) / 20.0)))
    return sp


def train(arm, seed, device, log=print):
    P = prep()
    cfg = RR.full_cfg(P)
    des = RR.design()
    sp = spec_for(arm, cfg, des)
    torch.manual_seed(seed); np.random.seed(seed)
    cfg1 = deep_merge(cfg, {"phase1": sp["p1"]})
    t0 = time.time()
    base = phase1(cfg1, P["sim"], P["scales"], des, device, seed, log)
    ms = R.masked_scales(P["scales"])
    data2 = {"train": P["sets"]["train"], "val": P["sets"]["val"]}
    r = BG.phase2_bg("lora", cfg1, data2, ms, des, base, device, seed, sp["p2"], log)
    MC.mkdir(exist_ok=True)
    torch.save({"base": base, "r": r, "spec": sp, "wall_s": time.time() - t0}, MC / f"{arm}_seed{seed}.pt")
    log(f"saved {arm} seed {seed} ({time.time() - t0:.0f}s)")


def _gm(P, seed):
    from hyperkkl import baselines_data as BD
    b = torch.load(LIVE_REAL / "cache" / f"baselines_seed{seed}.pt", weights_only=False, map_location="cpu")
    plant = R.Pan18650PF(sensors=(3, 4), **P["scfg"]["params"])
    return BD.GreyBoxModel(plant, b["greybox_mlp"])


def _prior1(es):
    """One prior per episode for the B2 training features: SOC0 + 0.2 (or - 0.2 if above 1)."""
    s0 = es.x[:, 0, 0]
    soc = torch.where(s0 + 0.2 <= 1.0, s0 + 0.2, s0 - 0.2)
    z = torch.zeros_like(s0)
    return torch.stack([soc, z, z, es.y[:, 0, 0], es.y[:, 0, 1]], 1).float()


def train_b2(base_arm, seed, device, log=print):
    from hyperkkl import beat_b as BB
    P = prep()
    cfg = RR.full_cfg(P)
    des = RR.design()
    m = torch.load(MC / f"{base_arm}_seed{seed}.pt", weights_only=False, map_location="cpu")
    to = lambda o: o.to(device) if isinstance(o, torch.nn.Module) else o
    base = {k: to(v) for k, v in m["base"].items()}
    r1 = {k: to(v) for k, v in m["r"].items()}
    ob1 = BG.make_observer(base_arm, des, base, r1, device)
    gm = _gm(P, seed)
    sig = torch.as_tensor(P["scales"]["x_std"], dtype=torch.float32, device=device)

    def ext_of(es):
        x0 = _prior1(es).to(device)
        from hyperkkl.evaluate import _run_chunks
        with torch.no_grad():
            xh = _run_chunks(ob1, es.y.to(device), es.u.to(device), x0, 256)
        return BB.ext_sequence(gm, xh, es.u.to(device), des.dt, sig).cpu()

    extra = {"train": ext_of(P["sets"]["train"]), "val": ext_of(P["sets"]["val"])}
    cfg1 = deep_merge(cfg, {"phase1": m["spec"]["p1"]})
    ms = R.masked_scales(P["scales"])
    data2 = {"train": P["sets"]["train"], "val": P["sets"]["val"]}
    r = BB.phase2_ext("lora", cfg1, data2, ms, des, base, device, seed, m["spec"]["p2"], log, extra=extra)
    torch.save({"base": m["base"], "r1": m["r"], "r": r, "spec": m["spec"]}, MC / f"B2-{base_arm}_seed{seed}.pt")
    log(f"saved B2-{base_arm} seed {seed}")


@torch.no_grad()
def predict(arm, seed, device):
    if arm.startswith("B2-"):
        return predict_b2(arm, seed, device)
    P = prep()
    des = RR.design()
    m = torch.load(MC / f"{arm}_seed{seed}.pt", weights_only=False, map_location="cpu")
    to = lambda o: o.to(device) if isinstance(o, torch.nn.Module) else o
    base = {k: to(v) for k, v in m["base"].items()}
    r = {k: to(v) for k, v in m["r"].items()}
    ob = BG.make_observer(arm, des, base, r, device)
    out = {}
    for n in ["val"] + [s for ss in TEST.values() for s in ss]:
        Y, U, S, X0, idx, sgn = RR.expand(P["sets"][n])
        out[n] = RR.run_soc(ob, Y, U, X0, device, chunk=10 ** 9)
    return out


@torch.no_grad()
def predict_b2(arm, seed, device):
    from hyperkkl import beat_b as BB
    from hyperkkl.observers import LatentStep
    P = prep()
    des = RR.design()
    m = torch.load(MC / f"{arm}_seed{seed}.pt", weights_only=False, map_location="cpu")
    to = lambda o: o.to(device) if isinstance(o, torch.nn.Module) else o
    base = {k: to(v) for k, v in m["base"].items()}
    r1 = {k: to(v) for k, v in m["r1"].items()}
    r = {k: to(v) for k, v in m["r"].items()}
    ob1 = BG.make_observer("s1", des, base, r1, device)
    sig = torch.as_tensor(P["scales"]["x_std"], dtype=torch.float32, device=device)
    ob = BB.ExtCtxObserver(arm, LatentStep(des, device), r["cond"], r["ctx"], r["hist"], r["gx"], ob1,
                           _gm(P, seed), des.dt, sig, z0_map=base["T0"])
    out = {}
    for n in ["val"] + [s for ss in TEST.values() for s in ss]:
        Y, U, S, X0, idx, sgn = RR.expand(P["sets"][n])
        out[n] = RR.run_soc(ob, Y, U, X0, device, chunk=10 ** 9)
    return out


def report(arm, device):
    from hyperkkl.realdata_report import block_bootstrap, per_window
    P = prep()
    stored = torch.load(LIVE_REAL / "cache" / "predictions.pt", weights_only=False)
    pred, truth, meta = dict(stored["pred"]), stored["truth"], stored["meta"]
    for p_ in sorted((LIVE_REAL / "cache").glob("predictions_baselines_seed*.pt")):
        pred.update(torch.load(p_, weights_only=False)["pred"])
    for s in SEEDS:
        for n, v in predict(arm, s, device).items():
            pred[(n, arm, s)] = v
    rm = lambda n, m, s: RR.soc_metrics(pred[(n, m, s)], truth[n])["rmse_settled"]

    def paired(a, b, sets):
        series, per_seed = [], []
        for n in sets:
            d = np.mean([rm(n, a, s) - rm(n, b, s) for s in SEEDS], 0)
            series.append(per_window([d], [meta[n]])[0][1])
        for s in SEEDS:
            per_seed.append(float(np.mean(np.concatenate([rm(n, a, s) - rm(n, b, s) for n in sets]))))
        mean, lo, hi = block_bootstrap(series)
        return {"mean_diff": mean, "ci95": [lo, hi], "per_seed_diff": per_seed,
                "a_better": bool(hi < 0 and all(d < 0 for d in per_seed)),
                "b_better": bool(lo > 0 and all(d > 0 for d in per_seed))}

    res = {"arm": arm, "settled_rmse": {}, "tests": {}}
    for f, ss in TEST.items():
        res["settled_rmse"][f] = {m: [float(np.mean(np.concatenate([rm(n, m, s) for n in ss]))) for s in SEEDS]
                                  for m in (arm, "lora", "greybox_ukf", "greybox_ekf")}
    for b in ("greybox_ukf", "greybox_ekf", "lora"):
        res["tests"][f"{arm}-{b}"] = {f: paired(arm, b, ss) for f, ss in TEST.items()}
        w = sum(v["a_better"] for v in res["tests"][f"{arm}-{b}"].values())
        l = sum(v["b_better"] for v in res["tests"][f"{arm}-{b}"].values())
        res["tests"][f"{arm}-{b}"]["verdict"] = {"arm_better_cycles": w, "comparator_better_cycles": l,
                                                 "arm_beats (>= 2 of 3)": w >= 2}
    res["settled_rmse"]["val"] = {arm: [float(np.mean(rm("val", arm, s))) for s in SEEDS]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{arm}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "tests"}, indent=1))
    for k, t in res["tests"].items():
        print(k, t["verdict"], {f: (round(v["mean_diff"], 3), [round(x, 3) for x in v["ci95"]]) for f, v in t.items() if f != "verdict"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "train_b2", "report"]); ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.stage == "train":
        train(a.arm, a.seed, a.device)
    elif a.stage == "train_b2":
        train_b2(a.arm, a.seed, a.device)
    else:
        report(a.arm, a.device)
