"""Cartpole diagnosis (eval only, no retraining): stored models under sensor-noise conditions.

    python diag_conditions.py --which old --seed 0 [--device cuda:0]   # old lora + greybox_ukf (refit), snapshot code
    python diag_conditions.py --which new --seed 0                     # A2, A4 (p2_rescore cached models)

Output: experiments/p2_rescore/diag/<which>_s<seed>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P = HERE.parent
LIVE = P.parent.parent
LIVE_CACHE = LIVE / "experiments" / "cache"
NODE, SYSTEM = "TEST-C3", "cartpole"
CONDS = {
    "stored": {},                                                  # original coloured Student-t draw
    "ct_redraw": {"sensor": {"kind": "colored_t"}},                # same law, new draw (seed + 1)
    "noiseless": {"sensor": {"noise": 0.0}},
    "gauss_white": {"sensor": {"kind": "gaussian"}},               # same variance (unit-variance w)
    "gauss_col": {"sensor": {"kind": "colored", "rho": 0.9}},      # colour, no heavy tails
}


def setup_conditions(cfg):
    cfg["conditions"] = dict(cfg["conditions"]); cfg["conditions"].update(CONDS)
    cfg["eval"] = dict(cfg["eval"]); cfg["eval"].update(conditions=list(CONDS), save_traj=False, memory_probe=False)
    return cfg


def summarize(res, names):
    out = {}
    for tier, cs in res.items():
        for c, blk in cs.items():
            for m in names:
                sc = blk["scorecard"][m]
                out.setdefault(m, {}).setdefault(tier, {})[c] = {
                    "nrmse": float(np.mean(blk["methods"][m]["nrmse"])), "unm": float(sc["nrmse_unmeasured"]),
                    "state": [float(v) for v in sc["nrmse_state"]], "div": float(sc["div_rate"])}
    return out


def run_old(seed, device):
    sys.path.insert(0, str(P / "diag" / "oldcode"))
    from hyperkkl import data as Dm
    from hyperkkl import baselines_data as BD
    from hyperkkl.cache import MODEL_MODULES, module_hash
    from hyperkkl.config import digest
    from hyperkkl.evaluate import build_observers, evaluate
    from hyperkkl.job import method_cfg
    from hyperkkl import systems_extra  # noqa: F401
    assert Dm.CACHE.resolve() != LIVE_CACHE.resolve()

    def _ro(path, build, **kw):
        live = LIVE_CACHE / Path(path).relative_to(Dm.CACHE)
        if live.exists():
            return torch.load(live, weights_only=False, map_location="cpu")
        raise FileNotFoundError(f"not in live cache: {live}")
    Dm._cached = _ro
    stored = json.load(open(LIVE / "experiments" / "tree" / NODE / "runs" / SYSTEM / f"seed{seed}" / "results.json"))
    cfg = stored["config"]; cfg["seed"] = seed
    scales = Dm.system_scales(cfg, device)
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, seed, device)
    mh = module_hash(MODEL_MODULES)
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
    kind, mc = method_cfg(cfg, "lora")
    k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
    f1 = LIVE_CACHE / "models" / f"{SYSTEM}_p1_{k1}.pt"; f2 = LIVE_CACHE / "models" / f"{SYSTEM}_p2_{kind}_{k2}.pt"
    print("checkpoints", f1.name, f1.exists(), f2.name, f2.exists(), flush=True)
    to = lambda d: {k: (v.to(device) if isinstance(v, torch.nn.Module) else v) for k, v in d.items()}
    base = to(torch.load(f1, weights_only=False, map_location="cpu"))
    r = to(torch.load(f2, weights_only=False, map_location="cpu"))
    cfg = setup_conditions(cfg)
    cfg["baselines"] = ["greybox_ukf"]
    t0 = time.time()
    fits = BD.fit_all(cfg, data, scales, des, device, seed, print)
    print(f"greybox fit {time.time() - t0:.0f}s", flush=True)
    obs = build_observers(cfg, des, scales, base, {"lora": r}, None, device, fits)
    res = evaluate(cfg, des, scales, obs, device, print)
    out = summarize(res, ["lora", "greybox_ukf"])
    chk = {m: {t: float(np.mean(stored["results"][t]["clean"]["methods"][m]["nrmse"])) for t in res} for m in ["lora", "greybox_ukf"]}
    return {"models": {"lora": {"l_dec_q95": base["diag"]["l_dec_q95"], "eps_dec": base["diag"]["eps_dec"]}},
            "results": out, "stored_clean_check": chk, "ckpt": [f1.name, f2.name]}


def run_new(seed, device):
    sys.path.insert(0, str(HERE))
    import run_p2 as R
    from hyperkkl.evaluate import evaluate
    j = R.Job(NODE, SYSTEM, seed, device)
    obs, models = {}, {}
    for a in ("A2", "A4"):
        ob, info = j.train_arm(a)            # cached models: loads, no training (checked below)
        assert info["load_or_train_s"]["p2"] < 120, "model cache miss: refusing to retrain"
        obs[a] = ob
        models[a] = {"l_dec_q95": info["diag"]["l_dec_q95"], "eps_dec": info["diag"]["eps_dec"],
                     "noise_gain_base": info["noise_gain_base"]}
    cfg = setup_conditions(j.cfg)
    res = evaluate(cfg, j.des0, j.scales, obs, device, print)
    chk = {a: {t: float(np.mean(json.load(open(P / "runs" / NODE / SYSTEM / f"seed{seed}" / f"{a}.json"))["results"][t]["nrmse"]))
               for t in res} for a in obs}
    return {"models": models, "results": summarize(res, list(obs)), "stored_clean_check": chk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["old", "new"], required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = (run_old if a.which == "old" else run_new)(a.seed, a.device)
    (P / "diag").mkdir(exist_ok=True)
    (P / "diag" / f"{a.which}_s{a.seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print("written", flush=True)


if __name__ == "__main__":
    main()
