"""Smoke test of the full job pipeline on a new plant (tiny steps; outputs and models to scratch).

    .venv/bin/python experiments/cartpole_bounded/smoke.py NODE SYSTEM DEVICE
Runs hyperkkl.job.run_job with the node's resolved config plus tiny training/eval overrides. The run
directory and the model cache are redirected to the scratchpad; nothing is written into the tree.
"""
import sys
from pathlib import Path

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/smoke")

from hyperkkl import job as J, tree as Tr  # noqa: E402

TINY = {"phase1.steps_enc": 40, "phase1.steps_dec": 40, "phase2.steps": 20, "phase2.val_every": 10,
        "nlox.steps": 20, "nlox.val_every": 10, "eval.n_episodes": 8, "eval.memory_probe": False,
        "data_baselines": {"gb_steps": 50, "n_fit": 20000, "gb_n_fit": 20000, "kmeans_points": 2000}}

node, system, device = sys.argv[1], sys.argv[2], sys.argv[3]
_ro = Tr.resolved_overrides
Tr.resolved_overrides = lambda nid: _ro(nid) + [TINY]
Tr.run_dir = lambda nid, s, seed: SCR / nid / s / f"seed{seed}"
J.MODEL_CACHE = SCR / "models"
p = J.run_job(node, system, 0, device)
print("methods evaluated:", sorted(p["results"]["id"]["clean"]["methods"]))   # no numbers: tiny-step models
print("SMOKE OK", node, system)
