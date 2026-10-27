"""Mean episode NRMSE per method x tier (clean) for v2 nodes; seeds averaged. usage: v2table.py <tree_dir> node [node...]"""
import json, sys, os, glob, statistics as st, collections
root = sys.argv[1]
for node in sys.argv[2:]:
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    seeds = collections.defaultdict(set)
    for f in glob.glob(os.path.join(root, node, 'runs', '*', 'seed*', 'results.json')):
        sysn = f.split(os.sep)[-3]
        try: d = json.load(open(f))
        except Exception: continue
        seeds[sysn].add(f.split(os.sep)[-2])
        for tier, tb in d['results'].items():
            cl = tb.get('clean') or {}
            for m, v in (cl.get('methods') or {}).items():
                x = [a for a in v.get('nrmse', []) if a is not None and a == a]
                if x: agg[(sysn, tier)][m].append(st.mean(x))
    print('#', node)
    for (s, t) in sorted(agg):
        row = agg[(s, t)]
        items = sorted(((st.mean(v), m, len(v)) for m, v in row.items()))
        print('  %-11s %-13s n=%d  ' % (s, t, len(seeds[s])) + '  '.join('%s=%.4g' % (m, a) for a, m, n in items))
