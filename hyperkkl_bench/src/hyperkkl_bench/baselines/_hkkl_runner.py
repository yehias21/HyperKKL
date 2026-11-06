"""Subprocess runner for the stored HyperKKL (jitter_fix) checkpoints. It imports the experiment's own
package copy (a different ``hyperkkl``), so it runs in a separate interpreter:

    python -m hyperkkl_bench.baselines._hkkl_runner PLANT NAME SEED IN.npz OUT.npz

IN.npz: Y (E, N+1, n_y), U (E, N+1), X0 (E, n_x). OUT.npz: X_hat (E, N+1, n_x), n_params.
No data cache is touched, nothing is trained: the config comes from the stored run, the phase-1 maps
and the phase-2 conditioner are loaded from experiments/jitter_fix/models.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
JF = Path(__file__).resolve().parents[4] / "experiments" / "jitter_fix"


def main(plant, name, seed, fin, fout):
    sys.path.insert(0, str(JF / "code"))
    import jf_env
    P = jf_env.setup(plant)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import jf
    from jf_run import load_model
    from hyperkkl import data as Dm
    from hyperkkl.observers import run
    seed = int(seed)
    key_dev = "cuda:0" if plant == "quad" else "cpu"       # cache key of the stored scales (no GPU is used)
    cfg = P.cfg(seed, key_dev)
    des = Dm.make_design(cfg["system"])
    base = P.phase1(cfg, seed, "cpu", recipe="D0") if name.startswith("repro_") else P.phase1(cfg, seed, "cpu")
    fs = sorted((JF / "models" / plant / name / f"seed{seed}").glob("p2_*.pt"))
    assert len(fs) == 1, fs
    r = load_model(fs[0], "cpu")
    ob = jf.make_observer(f"hk_{name}", des, base, r, "cpu")
    d = np.load(fin)
    Y = torch.as_tensor(d["Y"], dtype=torch.float32)
    U = torch.as_tensor(d["U"], dtype=torch.float32)[..., None]
    X0 = torch.as_tensor(d["X0"], dtype=torch.float32)
    with torch.no_grad():
        xh = torch.cat([run(ob, Y[i:i + 64], U[i:i + 64], X0[i:i + 64]) for i in range(0, len(Y), 64)])
    npar = int(r.get("trainable_total") or 0) + sum(p.numel() for m in base.values() if isinstance(m, torch.nn.Module)
                                                    for p in m.parameters())
    np.savez(fout, X_hat=xh.double().numpy(), n_params=npar)


if __name__ == "__main__":
    main(*sys.argv[1:6])
