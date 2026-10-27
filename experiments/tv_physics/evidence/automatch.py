"""Find every pair of runs (same system, seed) whose configs differ in a physics knob and in at most 2 other keys."""
import json, collections, itertools
R = json.load(open('runs_HyperKKL.json'))
K = {'phase1.use_pde','phase1.lambda_pde','phase1.pde_u_scale','phase1.weight_residual','phase2.freeze_encoder_deltas','phase2.residual_mode','phase2.w_label'}
def norm(v): return {'False': False, 'True': True}.get(v, v) if isinstance(v, str) else v
g = collections.defaultdict(list)
for r in R: g[(r['system'], r['seed'])].append(r)
found = collections.Counter()
for key, rs in g.items():
    for a, b in itertools.combinations(rs, 2):
        ks = set(a['cfg']) | set(b['cfg'])
        d = [k for k in ks if a['cfg'].get(k) != b['cfg'].get(k) and not k.startswith('system.design') and k != 'phase2.window_size'
             and not (a['cfg'].get(k) is None or b['cfg'].get(k) is None)]
        phys = [k for k in d if k in K and norm(a['cfg'].get(k)) != norm(b['cfg'].get(k))]
        if phys and len(d) - len(phys) <= 2:
            t = tuple(sorted([a['tag'], b['tag']])); found[(t, tuple(sorted(phys)), tuple(sorted(set(d) - set(phys))))] += 1
for (t, p, o), c in sorted(found.items()): print(t, p, 'other:', o, 'pairs=', c)
