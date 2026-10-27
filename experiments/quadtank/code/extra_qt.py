import json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
NODES = ["C3-partial", "C3-mismatch30"]; ARMS = ["X0", "X1", "X2", "X3"]; SEEDS = [0, 1, 2]
for n in NODES:
    for s in SEEDS:
        d = json.load(open(HERE / "runs" / n / f"seed{s}" / "X1.json"))
        g = HERE / "runs" / n / f"seed{s}" / "X1_gb.json"
        if g.exists():
            gb = json.load(open(g))
            print(n, s, "X1 gb in-job vs gb-only (seq):", {m: (round(float(np.mean(d["results"]["sequence_ood"][m]["nrmse"])), 5),
                  round(float(np.mean(gb["results"]["sequence_ood"][m]["nrmse"])), 5)) for m in ("greybox_ekf", "greybox_ukf")})
out = {}
for n in NODES:
    for a in ARMS:
        J = [json.load(open(HERE / "runs" / n / f"seed{s}" / f"{a}.json")) for s in SEEDS]
        row = {"p1_s": float(np.mean([j["wall_s"]["p1"] for j in J])), "p2_s": float(np.mean([j["wall_s"]["p2"] for j in J])),
               "val_nrmse": [j["val"]["val_nrmse"] for j in J], "eps_dec": [j["diag_p1"]["eps_dec"] for j in J],
               "l_dec_q95": [j["diag_p1"]["l_dec_q95"] for j in J], "train_shape": J[0]["train_shape"],
               "n_rejected": [j["n_rejected"] for j in J]}
        out[f"{n}/{a}"] = row
        print(n, a, "p1 %.0fs p2 %.0fs" % (row["p1_s"], row["p2_s"]), "val", [round(x, 4) for x in row["val_nrmse"]],
              "eps_dec", [round(x, 3) for x in row["eps_dec"]], "l_dec_q95", [round(x, 1) for x in row["l_dec_q95"]], row["train_shape"], row["n_rejected"])
json.dump(out, open(HERE / "extra.json", "w"), indent=1)
