"""Collect the stored per-episode NRMSE of the reference observers into references.json (seeds 0-2):
A0 = tree `lora`, stored 40k `nlox`, stored grey-box filters (tree), A3 / A4+D3 (beat_greybox campaign
runs, per-episode files in the campaign's scratch run directory; seed means are cross-checked against
experiments/beat_greybox/results.json)."""
import json
from pathlib import Path

import numpy as np

LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
BG = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/"
          "scratchpad/beat_greybox/runs")
OUT = LIVE / "experiments" / "baselines2" / "references.json"
from dispatch import CELLS  # noqa: E402

TIERS = ("id", "shape_ood", "sequence_ood")
KEYS = ("nrmse", "nrmse_unmeasured", "diverged")


def main():
    out = {}
    for node, sys_ in CELLS:
        for seed in (0, 1, 2):
            f = LIVE / "experiments/tree" / node / "runs" / sys_ / f"seed{seed}" / "results.json"
            if f.exists():
                r = json.load(open(f))["results"]
                for t in TIERS:
                    for m, v in r[t]["clean"]["methods"].items():
                        if m in ("lora", "nlox", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"):
                            name = {"lora": "A0", "nlox": "nlox40_stored"}.get(m, m + "_stored")
                            out.setdefault(f"{node}/{sys_}", {}).setdefault(name, {}).setdefault(t, {})[seed] = \
                                {k: v[k] for k in KEYS if k in v}
            for arm in ("A3", "A4+D3"):
                g = BG / node / sys_ / f"seed{seed}" / f"{arm}.json"
                if g.exists():
                    r = json.load(open(g))["results"]
                    for t in TIERS:
                        out.setdefault(f"{node}/{sys_}", {}).setdefault(arm, {}).setdefault(t, {})[seed] = \
                            {k: r[t][k] for k in KEYS if k in r[t]}
    # cross-check with the campaign's summary (seed-mean sequence_ood)
    s = json.load(open(LIVE / "experiments/beat_greybox/results.json"))
    chk = []
    for arm in ("A3", "A4+D3"):
        for key, v in s["main_c3"]["arms"][arm]["mean"].items():
            cell, tier = key.split("|")
            sysn, sc = cell.split("/")
            node = "C3-partial" if sc == "partial" else "C3-mismatch30"
            d = out.get(f"{node}/{sysn}", {}).get(arm, {}).get(tier, {})
            if len(d) == 3:
                m = float(np.mean([np.mean(d[k]["nrmse"]) for k in d]))
                chk.append(abs(m - v) / v)
    OUT.write_text(json.dumps({"refs": out, "crosscheck_max_rel_diff": max(chk) if chk else None,
                               "n_crosscheck": len(chk)}))
    print("cells", len(out), "crosscheck", max(chk) if chk else None, len(chk))
    for c, d in out.items():
        print(c, {a: {t: len(v) for t, v in x.items()} for a, x in d.items()})


if __name__ == "__main__":
    main()
