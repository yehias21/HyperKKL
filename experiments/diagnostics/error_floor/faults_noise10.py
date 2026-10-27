import json, glob, numpy as np, sys
T = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/tree/C3-faults/runs"
out = {}
for s in ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]:
    for k in range(3):
        f = f"{T}/{s}/seed{k}/results.json"
        try: d = json.load(open(f))
        except FileNotFoundError: continue
        for t in ["id", "sequence_ood"]:
            cl, n10 = d["results"][t]["clean"]["methods"], d["results"][t]["noise10"]["methods"]
            for m in cl:
                if m in n10:
                    out.setdefault(s, {}).setdefault(t, {}).setdefault(m, []).append(
                        (float(np.mean(cl[m]["nrmse"])), float(np.mean(n10[m]["nrmse"]))))
summ = {}
for s, ts in out.items():
    for t, ms in ts.items():
        for m, v in ms.items():
            a = np.array(v)
            summ.setdefault(s, {}).setdefault(t, {})[m] = {"clean": a[:, 0].mean(), "noise10": a[:, 1].mean(),
                "ratio": float(np.exp(np.mean(np.log(a[:, 1] / a[:, 0])))),
                "slope": float(np.mean(np.log(a[:, 1] / a[:, 0])) / np.log(10 / 3)), "n": len(a)}
json.dump(summ, open("faults_noise10.json", "w"), indent=1)
for s in summ:
    for t in summ[s]:
        print(s, t, "  ".join(f"{m}:{v['clean']:.3f}->{v['noise10']:.3f} (x{v['ratio']:.2f}, b={v['slope']:.2f})" for m, v in summ[s][t].items() if m in ("lora","hyper","latent","ekf","ukf","ekf_oracle","nlox","pf")))
