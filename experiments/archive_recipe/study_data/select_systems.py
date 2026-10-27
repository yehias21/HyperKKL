"""Pre-registered system selection for the archived-recipe test (STUDY.md section 7).
usage: python -I select_systems.py <tree_dir> [out.json]
Rule: scenarios C3-partial and C3-mismatch30 (dev systems, nominal sensor, clean condition),
tier sequence_ood. L = min(mean lora, mean hyper) over the node's seeds (current defaults);
B = min over every non-HyperKKL method in the run (filters, oracle/Huber filters, pf, kkl_mb,
nlox, koopman_kf*, greybox_*). r = L / B. Per scenario: the 2 cells with the largest r (where
the current recipe LOSES most) and the 2 with the smallest r (where it WINS most); 8 cells.
Methods missing from some seeds are ignored for that cell (flagged)."""
import json, sys, os, glob, statistics as st, collections, time
root = sys.argv[1]
OURS = {'lora', 'hyper', 'latent', 'concat', 'film', 'autonomous'}
out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'tier': 'sequence_ood', 'cells': [], 'selected': []}
for node in ['C3-partial', 'C3-mismatch30']:
    cells = []
    for sd in sorted(glob.glob(os.path.join(root, node, 'runs', '*'))):
        sysn = os.path.basename(sd)
        per = collections.defaultdict(list); nseed = 0
        for f in sorted(glob.glob(os.path.join(sd, 'seed*', 'results.json'))):
            try: d = json.load(open(f))
            except Exception: continue
            nseed += 1
            ms = d['results']['sequence_ood']['clean']['methods']
            for m, v in ms.items():
                x = [a for a in v.get('nrmse', []) if a is not None and a == a]
                if x: per[m].append(st.mean(x))
        full = {m: st.mean(v) for m, v in per.items() if len(v) == nseed}
        partial = sorted(m for m, v in per.items() if len(v) != nseed)
        L = min((full[m], m) for m in ('lora', 'hyper') if m in full)
        B = min((v, m) for m, v in full.items() if m not in OURS)
        has_data = any(m.startswith(('greybox', 'koopman')) for m in full)
        cells.append({'node': node, 'system': sysn, 'seeds': nseed, 'best_ours': L[1], 'L': L[0],
                      'best_baseline': B[1], 'B': B[0], 'r': L[0] / B[0], 'data_baselines': has_data,
                      'incomplete_methods': partial})
    cells.sort(key=lambda c: -c['r'])
    out['cells'] += cells
    for c in cells[:2]: out['selected'].append(dict(c, role='LOSE'))
    for c in cells[-2:]: out['selected'].append(dict(c, role='WIN'))
for c in out['cells']:
    print('%-14s %-11s n=%d r=%6.2f  ours %-5s %.4g  best baseline %-18s %.4g  data_bl=%s %s' % (
        c['node'], c['system'], c['seeds'], c['r'], c['best_ours'], c['L'], c['best_baseline'], c['B'], c['data_baselines'],
        ('incomplete:' + ','.join(c['incomplete_methods'])) if c['incomplete_methods'] else ''))
print('SELECTED:', [(c['node'], c['system'], c['role']) for c in out['selected']])
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], 'w'), indent=1)
