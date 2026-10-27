"""Which grey-box setting is weak? Refit the one-step residual with other lr / steps (CPU) and compare
open-loop validation rollout errors (AUDIT_UDE.md section 6b)."""
import json, sys, torch
import b2  # noqa
from b2.job import load_cfg
from audit_ude import rollout_err, _F, KEY
from hyperkkl import baselines_data as BD, data as Dm
from hyperkkl import systems_extra, systems_bounded  # noqa
torch.set_num_threads(6)
cell = sys.argv[1]; node, system = cell.split("/")
cfg, _ = load_cfg(node, system, 0); sc = cfg["system"]
scales = Dm.system_scales(cfg, KEY); des = Dm.make_design(sc); data = Dm.training_data(cfg, 0, KEY)
model = Dm.model_plant(sc); tr, p1, va = data["train"], data["p1"], data["val"]
trn = BD.transitions([tr, p1], 1_000_000, 1, "cpu"); van = BD.transitions([va], 100_000, 2, "cpu")
sig = torch.as_tensor(scales["x_std"], dtype=torch.float64)
g = torch.Generator().manual_seed(1); N1 = tr.x.shape[1]
e = torch.randint(0, len(va), (2048,), generator=g); s0 = torch.randint(0, N1 - 33, (2048,), generator=g)
out = {}
for tag, over in (("stored: lr1e-3, 3000 steps", {}), ("lr 1e-2", {"gb_lr": 1e-2}),
                  ("lr 3e-3", {"gb_lr": 3e-3}), ("lr 1e-3, 12000 steps, patience 40", {"gb_steps": 12000, "gb_patience": 40})):
    opt = {**BD.options(cfg), **over}
    gb = BD.fit_greybox(model, trn, van, des.dt, scales, float(sc.get("u_scale", 1.0)), 0, "cpu", opt, log=lambda m: None)
    r = rollout_err(_F(gb["model"].f, model), va.x, va.u, e, s0, 32, des.dt, sig)
    out[tag] = {"steps": gb["info"]["steps"], "val_loss": gb["info"]["val_loss"], "k1": r[0], "k8": r[7], "k32": r[31]}
    print(cell, tag, out[tag], flush=True)
json.dump(out, open(f"../audit/gb_lr_{node}_{system}.json", "w"), indent=1)
