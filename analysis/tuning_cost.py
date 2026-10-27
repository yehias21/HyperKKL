"""C1 tuning study: training compute per variant (experiments/tuning/PROTOCOL.md, section 1b).

    .venv/bin/python -m analysis.tuning_cost [--systems vdp battery] [--out experiments/tuning/cost.json]

FLOPs per optimiser step: one phase-2 training loss + backward exactly as train.phase2 runs it
(train._decoder_loss for the decoder variants, train._rollout_loss with K = rollout_K for latent),
at the real shapes (system design, batch episodes_per_batch x times_per_episode, the SFM context
over the whole episode), counted by torch.utils.flop_counter.FlopCounterMode on the CPU. It
counts matmul-class ops only (mm, addmm, bmm, conv; no elementwise ops, no FFT), so it is a
lower bound dominated by the dense layers; the context-encoder part is the same for every
variant. Values do not depend on trained weights (dummy base maps of the phase-1 shapes,
random data). Measured wall-clock comes from the selection runs' train.<method>.wall_s
(shared GPUs, 4 jobs per GPU: indicative only), median over the dev systems.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyperkkl import data as Dm                                   # noqa: E402
from hyperkkl import train as T                                   # noqa: E402
from hyperkkl import tree as Tr                                   # noqa: E402
from hyperkkl.conditioning import arch_options, build_conditioner, match_width   # noqa: E402
from hyperkkl.config import load                                  # noqa: E402
from hyperkkl.job import method_cfg                               # noqa: E402
from hyperkkl.nets import BaseMap, build_context, n_params        # noqa: E402
from hyperkkl.observers import LatentStep                         # noqa: E402

from .select_tuned import DESIGN, configs                          # noqa: E402

OUT = ROOT / "experiments" / "tuning" / "cost.json"


def _maps(cfg, nx, nz):
    p = cfg["phase1"]
    st = lambda n: (torch.zeros(n), torch.ones(n))
    T0 = BaseMap(nx, nz, p["enc_width"], p["enc_depth"], "silu", st(nx), st(nz))
    Ti = BaseMap(nz, nx, p["dec_width"], p["dec_depth"], "tanh", st(nz), st(nx))
    for q in list(T0.parameters()) + list(Ti.parameters()):
        q.requires_grad_(False)
    return T0, Ti


def step_flops(system: str, method: str) -> dict:
    cfg0 = load(system)
    kind, cfg = method_cfg(cfg0, method)
    p = cfg["phase2"]
    des = Dm.make_design(cfg["system"])
    nz, nx = des.A.shape[0], len(cfg["system"]["ic_box"])
    ny = len(cfg["system"].get("sensors") or [0])
    T0, Ti = _maps(cfg, nx, nz)
    torch.manual_seed(0)
    ctx = build_context(p["context"]["kind"], p["context"], 1, des.dt, np.linalg.eigvals(des.A),
                        float(cfg["system"].get("u_scale", 1.0)))
    arch = arch_options(p)
    z_std = torch.ones(nz)
    w, npar = match_width(kind, Ti, ctx.dim, int(p["budget"]), int(p["rank"]), z_std, **arch)
    cond = build_conditioner(kind, Ti, ctx.dim, w, int(p["rank"]), z_std, **arch)
    Be, M, K = int(p["episodes_per_batch"]), int(p["times_per_episode"]), int(p.get("rollout_K", 256))
    N1 = int(des.N) + 1
    g = torch.Generator().manual_seed(0)
    es = SimpleNamespace(x=torch.randn(Be, N1, nx, generator=g), y=torch.randn(Be, N1, ny, generator=g),
                         u=torch.randn(Be, N1, 1, generator=g), z_obs=torch.randn(Be, N1, nz, generator=g))
    sig = torch.ones(nx)
    e = torch.arange(Be)
    params = [q for q in list(cond.parameters()) + list(ctx.parameters()) if q.requires_grad]
    for q in params:
        q.grad = None
    fc = FlopCounterMode(display=False)
    with fc:
        if kind == "latent":
            lat = LatentStep(des, "cpu")
            k0 = torch.randint(0, N1 - K - 1, (Be,), generator=g)
            loss = T._rollout_loss(cond, ctx, lat, T0, Ti, es, e, k0, K, sig)
        else:
            k = torch.randint(0, N1, (Be, M), generator=g)
            loss = T._decoder_loss(cond, ctx, es, e, k, sig)
        loss.backward()
    # the context encoder alone (forward + backward of its output), shared by every variant
    fc2 = FlopCounterMode(display=False)
    with fc2:
        ctx.seq(es.u).sum().backward()
    return {"system": system, "method": method, "kind": kind, "width": w, "n_params": npar,
            "ctx_params": n_params(ctx), "flops_step": int(fc.get_total_flops()),
            "flops_ctx": int(fc2.get_total_flops()), "samples_step": Be * (K if kind == "latent" else M),
            "N1": N1, "nz": nz, "nx": nx}


def measured_wall(method_node: dict) -> dict:
    """{method: {'median_s', 'per_system'}} from the selection runs (seed 0) that exist."""
    out = {}
    for m, node in method_node.items():
        per = {}
        for s, k in Tr.jobs_of(node) if Tr.node_path(node).exists() else []:
            p = Tr.run_dir(node, s, k) / "results.json"
            if p.exists():
                t = json.loads(p.read_text()).get("train", {}).get(m) or {}
                if t.get("wall_s") is not None:
                    per[s] = float(t["wall_s"])
        if per:
            out[m] = {"median_s": statistics.median(per.values()), "per_system": per}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["vdp", "battery"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    rows, mnode = [], {}
    for kind in DESIGN:
        for c in configs(kind):
            mnode[c["method"]] = c["node"]
            if c["lr"] != 0.002:                    # FLOPs do not depend on the learning rate
                continue
            for s in a.systems:
                r = step_flops(s, c["method"])
                rows.append(r)
                print(f"{s:10s} {c['method']:34s} width {r['width']:4d}  params {r['n_params']:6d}  "
                      f"GFLOP/step {r['flops_step'] / 1e9:8.3f}  (context {r['flops_ctx'] / 1e9:.3f})", flush=True)
    wall = measured_wall(mnode)
    for m, w in sorted(wall.items()):
        print(f"wall {m:34s} median {w['median_s']:7.0f} s over {len(w['per_system'])} systems")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"flops": rows, "wall": wall,
                               "note": "matmul-class FLOPs (torch.utils.flop_counter), forward+backward of one "
                                       "phase-2 training loss at the real shapes; context = SFM encoder alone"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
