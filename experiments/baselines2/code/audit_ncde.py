"""NCDE failure diagnostic: hidden-state norm and error over time on validation episodes (CPU)."""
import sys, json, torch, numpy as np
import b2
from b2 import HERE, models as M
from b2.job import load_cfg, NCDE_DZ
from hyperkkl import data as Dm
from hyperkkl import systems_extra, systems_bounded  # noqa
torch.set_num_threads(8)
for cell in sys.argv[1:]:
    node, system = cell.split("/")
    cfg, _ = load_cfg(node, system, 0); sc = cfg["system"]
    scales = Dm.system_scales(cfg, "cuda"); des = Dm.make_design(sc); data = Dm.training_data(cfg, 0, "cuda")
    model = Dm.model_plant(sc); sens = list(model.sensors)
    d = torch.load(HERE / "models" / node / system / "seed0" / "ncde140.pt", weights_only=False, map_location="cpu")
    xm = np.asarray(scales["x_mean"])
    norm = M._Norm(float(sc.get("u_scale", 1.0)), xm[sens], scales["y_std"], xm, scales["x_std"])
    net = M.NCDENet(model.nu, model.ny, model.nx, NCDE_DZ, d["arch"]["width"], norm, des.dt / float(sc.get("t_ref", 1.0)))
    net.load_state_dict(d["state"]); net.eval()
    va = data["val"]
    with torch.no_grad():
        X = net.X(va.u[:16], va.y[:16]); z = net.zeta(X[:, 0]); zn, err = [], []
        sig = torch.as_tensor(scales["x_std"])
        for k in range(X.shape[1] - 1):
            z = net.cde_step(z, X[:, k + 1] - X[:, k])
            if k % 200 == 0:
                zn.append(float(z.norm(dim=1).mean()))
                err.append(float((((net.norm.out(net.head(z)) - va.x[:16, k + 1]) / sig) ** 2).mean().sqrt()))
    print(cell, "lr", d["lr"], "|z| every 200 steps", [round(v, 1) for v in zn][:12], "nrmse", [round(v, 3) for v in err][:12], flush=True)
