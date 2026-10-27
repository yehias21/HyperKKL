"""A-priori feasibility probe (no training): washout time of the M1/M2 designs vs episode length."""
from common_md import *  # noqa
from hyperkkl import beat as BG, design as D
for node in ("C3-partial", "C3-mismatch30"):
    for s in ("quad_tank", "battery", "vdp", "bioreactor"):
        cfg = run_cfg(node, s, 0)
        sc = Dm.system_scales(cfg, "cuda")
        des0 = Dm.make_design(cfg["system"])
        d = Dm.training_data(cfg, 0, "cuda")
        T = des0.dt * (d["train"].x.shape[1] - 1)
        out = [f"{node} {s}: T {T:.1f} dt {des0.dt:.4g} N1 {d['train'].x.shape[1]} base t_burn {sc['t_burn']:.2f} burn {d['train'].burn} p1 {tuple(d['p1'].x.shape)}"]
        for nm, kw in (("M1", dict(alpha_scale=1/3)), ("M2", dict(alpha_scale=0.1, spread=10.0))):
            des = BG.custom_design(cfg["system"], des0.dt, des0.N, **kw)
            tr = BG.relabel(d["train"], des, 0)
            zf = tr.z[:, tr.z.shape[1] // 2:]
            zs = float(zf.double().pow(2).sum(-1).mean().sqrt())
            tb = D.washout_time(1.0, des.alpha, float(np.linalg.norm(des.B, 2)), float(sc["y_bar"]), zs, float(cfg["data"].get("eps_label", 1e-3)))
            lam = np.linalg.eigvals(des.A)
            out.append(f"   {nm}: alpha {des.alpha:.4g} |Re lam| {np.round(np.sort(-lam.real),4)} t_burn {tb:.1f} ({tb/T:.2f} T) max|lam|dt {np.abs(lam).max()*des.dt:.3f} ok {BG.design_checks(des)['ok_relaxed']}")
        print("\n".join(out), flush=True)
