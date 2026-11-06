"""Compute the bundled reference baselines on every frozen cell (CPU, no training) and store the full
reports in reference/runs/<baseline>/<plant>__<knowledge>.json. These are the yardstick of
tracks.track_score. Usage: python tools/run_reference_filters.py [baseline] [n_procs] [roles]"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
OUT = Path(__file__).resolve().parents[1] / "reference" / "runs"


def job(args):
    b, p, k = args
    import torch
    torch.set_num_threads(1)
    import hyperkkl_bench as hkb
    out = OUT / b / f"{p}__{k}.json"
    if out.exists():
        return str(out), "exists"
    t0 = time.time()
    try:
        rep = hkb.evaluate(hkb.baselines.make(b), p, k, verbose=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        rep.save(out)
        return str(out), f"{time.time() - t0:.0f}s"
    except Exception as e:  # noqa: BLE001
        return str(out), f"FAILED {e!r}"


if __name__ == "__main__":
    import hyperkkl_bench as hkb
    b = sys.argv[1] if len(sys.argv) > 1 else "ekf_gated"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    roles = sys.argv[3].split(",") if len(sys.argv) > 3 else None          # e.g. dev,test,deployment
    plants = [p for p in hkb.list_plants(role="all") if roles is None or hkb.registry.PLANTS[p]["role"] in roles]
    cells = [(b, p, k) for p in plants for k in hkb.KNOWLEDGE]
    cells.sort(key=lambda c: -hkb.metadata(c[1])["N"] * hkb.metadata(c[1])["n_x"])   # longest first
    with Pool(n) as pool:
        for o, s in pool.imap_unordered(job, cells):
            print(o, s, flush=True)
