"""Read-only: harvest every results.json + config.yaml (flattened phase1/phase2/evaluation/system scalars).
usage: python -I harvest_full.py <root> <out.json>"""
import json, os, sys, statistics as st, yaml
root, out = sys.argv[1], sys.argv[2]
def flat(d, p=''):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + '.'))
        elif isinstance(v, list): o[p + k] = json.dumps(v) if len(json.dumps(v)) < 200 else 'LIST'
        else: o[p + k] = v
    return o
rows = []
for dp, dn, fn in os.walk(root):
    if 'results.json' not in fn: continue
    try: d = json.load(open(os.path.join(dp, 'results.json')))
    except Exception as e: print('bad', dp, e); continue
    cfg = {}
    cp = os.path.join(dp, 'config.yaml')
    if os.path.exists(cp):
        c = yaml.safe_load(open(cp)) or {}
        for sec in ('phase1', 'phase2', 'evaluation', 'data'):
            if isinstance(c.get(sec), dict): cfg.update(flat(c[sec], sec + '.'))
        s = c.get('system') or {}
        for k, v in s.items():
            if not isinstance(v, (dict, list)): cfg['system.' + k] = v
        cfg['system.design.alpha'] = (s.get('design') or {}).get('alpha')
        cfg['system.z_size'] = s.get('z_size')
    res = {}
    for tier, per in (d.get('results') or {}).items():
        for m, ag in per.items():
            bf = ag.get('by_family') or {}
            fam = {f: v.get('nrmse_steady') for f, v in bf.items()}
            res.setdefault(tier, {})[m] = {'pooled': ag.get('nrmse_steady'), 'fam': fam}
    ts = d.get('train_stats') or {}
    extra = {m: {k: v for k, v in (x.get('extra') or {}).items() if not isinstance(v, (dict, list))}
             for m, x in ts.items() if isinstance(x, dict)}
    rel = os.path.relpath(dp, root).split(os.sep)
    rows.append({'path': '/'.join(rel), 'tag': rel[0], 'system': d.get('system'), 'seed': d.get('seed'),
                 'mtime': os.path.getmtime(os.path.join(dp, 'results.json')), 'cfg': cfg, 'res': res,
                 'extra': extra, 'methods': d.get('methods_requested')})
json.dump(rows, open(out, 'w'), default=str)
print(len(rows), 'runs')
