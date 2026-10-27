import json, sys, glob, os, statistics as st
root = sys.argv[1]
for tag in sys.argv[2:]:
    for s in ['bioreactor','vdp','duffing_mult','fhn','lorenz','rossler','quad_tank','cartpole']:
        v = []
        for f in sorted(glob.glob(os.path.join(root, tag, s, 'dev_seed*', 'results.json'))):
            d = json.load(open(f)).get('diagnostics') or {}
            v.append((d.get('eps_enc'), d.get('eps_dec'), d.get('l_dec_q95')))
        if v:
            print('%-11s %-12s eps_enc=%s eps_dec=%s l_dec_q95=%s' % (tag, s, *['%.3g' % st.mean(x[i] for x in v if x[i] is not None) if any(x[i] is not None for x in v) else None for i in range(3)]))
