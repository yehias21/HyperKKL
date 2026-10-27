"""Train/deploy consistency of the one-step grey-box residual: it is fitted so that
Phi_model(x) + dt r(x) ~ x+ (the discrete map), but deployed as RK4 of f_model + r (AUDIT_UDE.md 6c)."""
import sys, torch
import b2  # noqa
from b2.job import load_cfg
from audit_ude import KEY
from hyperkkl import baselines_data as BD, data as Dm
from hyperkkl.observers import _rk4
from hyperkkl import systems_extra, systems_bounded  # noqa
torch.set_num_threads(6)
cell = sys.argv[1]; node, system = cell.split("/")
cfg, _ = load_cfg(node, system, 0); sc = cfg["system"]
scales = Dm.system_scales(cfg, KEY); des = Dm.make_design(sc); data = Dm.training_data(cfg, 0, KEY)
model = Dm.model_plant(sc); dt = des.dt
trn = BD.transitions([data["train"], data["p1"]], 1_000_000, 1, "cpu"); van = BD.transitions([data["val"]], 100_000, 2, "cpu")
gb = BD.fit_greybox(model, trn, van, dt, scales, float(sc.get("u_scale", 1.0)), 0, "cpu", BD.options(cfg), log=lambda m: None)["model"]
X, U, X1 = (a[:20000] for a in van)
sig = torch.as_tensor(scales["x_std"], dtype=torch.float64)
e = lambda xp: float((((xp - X1) / sig) ** 2).mean())
with torch.no_grad():
    disc = _rk4(model, X, U, U, U, dt) + dt * gb.residual(X, U)          # what the fit targets
    dep = _rk4(gb, X, U, U, U, dt)                                        # what the filter integrates
    known = _rk4(model, X, U, U, U, dt)
print(cell, f"one-step nMSE: f_known {e(known):.3e}  discrete-map (training form) {e(disc):.3e}  "
      f"RK4(f_known + r) (deployed form) {e(dep):.3e}", flush=True)
