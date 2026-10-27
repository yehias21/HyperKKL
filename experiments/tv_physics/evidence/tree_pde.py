"""Current pipeline phase-1 PDE on/off from tree runs: <off node> (phase1.pde=false) vs <on node>.
Per (system, method, tier): per-seed relative change of scorecard nrmse_mean (clean), mean + Student-t 95% CI over seeds."""
import json, os, sys, math, statistics as st
from scipy import stats
T = '/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/tree'
def load(node, s, sd):
    p = f'{T}/{node}/runs/{s}/seed{sd}/results.json'
    return json.load(open(p)) if os.path.exists(p) else None
def ci(v):
    m = st.mean(v); 
    if len(v) < 2: return m, float('nan')
    return m, stats.t.ppf(.975, len(v) - 1) * st.stdev(v) / math.sqrt(len(v))
out = {}
for off, on, methods in [('C1-modelfree', 'C1-where', ['autonomous', 'latent', 'concat', 'film', 'lora', 'hyper']),
                         ('C3-partial-modelfree', 'C3-partial', ['latent', 'lora', 'hyper'])]:
    print(f'\n## {off} (PDE off) vs {on} (PDE on)')
    systems = sorted(os.listdir(f'{T}/{off}/runs'))
    for tier in ('sequence_ood', 'shape_ood', 'id'):
        print(f' tier {tier}')
        allrel = {m: [] for m in methods}
        for s in systems:
            row = []
            for m in methods:
                rel, pairs = [], []
                for sd in range(5):
                    a, b = load(off, s, sd), load(on, s, sd)
                    if not a or not b: continue
                    try:
                        va = a['results'][tier]['clean']['scorecard'][m]['nrmse_mean']; vb = b['results'][tier]['clean']['scorecard'][m]['nrmse_mean']
                    except KeyError: continue
                    rel.append(100 * (va / vb - 1)); pairs.append((sd, vb, va)); allrel[m].append(math.log(va / vb))
                if rel:
                    mm, h = ci(rel)
                    mon = st.mean(p[1] for p in pairs); moff = st.mean(p[2] for p in pairs)
                    sig = '*' if not math.isnan(h) and abs(mm) > h else ' '
                    row.append(f'{m} {mon:.4f}->{moff:.4f} {mm:+.0f}% [{mm-h:+.0f},{mm+h:+.0f}]{sig}')
                    out.setdefault(f'{off}|{tier}|{s}|{m}', {'n': len(rel), 'on': mon, 'off': moff, 'rel': mm, 'h': h, 'pairs': pairs})
            print(f'   {s:11s} ' + ' | '.join(row))
        for m in methods:
            v = allrel[m]
            if len(v) > 1:
                mm, h = ci(v)
                print(f'   pooled {m}: n={len(v)} (system,seed) geo-mean off/on {100*(math.exp(mm)-1):+.1f}% [{100*(math.exp(mm-h)-1):+.1f}, {100*(math.exp(mm+h)-1):+.1f}] (pooled t over pairs, systems not independent-weighted)')
json.dump(out, open('tree_pde.json', 'w'), indent=1)
