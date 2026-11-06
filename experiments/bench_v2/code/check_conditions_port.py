"""The package's families.apply_condition must reproduce the study's conditions_v2.apply exactly (same U, Y, X0) on the
64-episode test tiers, for every condition of both families. Uses the LIVE hyperkkl_bench and the study's code/pkg."""
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pkg")); sys.path.insert(0, str(CODE))
import numpy as np
import torch
import hyperkkl_bench as hkb
from hyperkkl_bench import families as F
import conditions_v2 as C
import v2cfg
from hyperkkl import data as Dm

LEVEL = {"realistic": "realistic", "white": "partial"}
bad, n = 0, 0
for fam, plant, var in (("driven", "vdp", "realistic"), ("driven", "battery", "realistic"), ("driven", "cstr", None),
                        ("autonomous", "fhn", "white")):
    if var is None:
        continue
    cfg = v2cfg.config(fam, plant, var)

    class Cell:
        pass
    cell = Cell(); cell.family, cell.plant, cell.variant, cell.cfg = fam, plant, var, cfg
    cell.des = Dm.make_design(cfg["system"]); cell.t_ref = float(cfg["system"].get("t_ref", 1.0))
    cell.sc = json.loads((CODE.parent / "data" / fam / plant / var / "scales.json").read_text())
    spec = f"{fam}/{plant}/{LEVEL[var]}"
    for split, conds in F.CONDITIONS[fam].items():
        es_t = torch.load(CODE.parent / "data" / fam / plant / var / f"{split}.pt", weights_only=False)
        es_p = hkb.load(spec, split)
        for c in conds:
            _, Y1, U1, X1 = C.apply(cell, es_t, split, c)
            U2, Y2, X2, _ = F.apply_condition(c, es_p, spec)
            ok = (np.array_equal(Y1.double().numpy(), Y2) and np.allclose(U1.double().numpy()[..., 0], U2, rtol=0, atol=1e-6)
                  and np.array_equal(X1.double().numpy(), X2))
            n += 1; bad += not ok
            if not ok:
                print("MISMATCH", spec, split, c, float(np.abs(Y1.double().numpy() - Y2).max()),
                      float(np.abs(U1.double().numpy()[..., 0] - U2).max()), float(np.abs(X1.double().numpy() - X2).max()))
print("ALL EQUAL" if bad == 0 else f"{bad} MISMATCHES", f"({n} block x condition pairs)")
