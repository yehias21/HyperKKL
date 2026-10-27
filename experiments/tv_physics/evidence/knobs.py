import json, sys, collections
K = ['phase1.use_pde','phase1.lambda_pde','phase1.pde_u_scale','phase1.weight_residual','phase1.grad_method',
     'phase2.freeze_encoder_deltas','phase2.residual_mode','phase2.w_label','phase2.grad_method','phase2.w_recon']
for fn in sys.argv[1:]:
    R = json.load(open(fn)); print('==', fn, len(R))
    nocfg = [r['path'] for r in R if not r['cfg']]; print('no config:', len(nocfg), nocfg[:5])
    for k in K:
        c = collections.Counter(repr(r['cfg'].get(k)) for r in R); print(k, dict(c))
    # tags with non-default per knob
    D = {'phase1.use_pde': True, 'phase1.lambda_pde': 1.0, 'phase1.pde_u_scale': 0.0, 'phase1.weight_residual': True,
         'phase1.grad_method': 'sum', 'phase2.freeze_encoder_deltas': True, 'phase2.residual_mode': 'characteristic', 'phase2.w_label': 1.0}
    for k, dv in D.items():
        tags = collections.defaultdict(list)
        for r in R:
            v = r['cfg'].get(k)
            if v is not None and v != dv: tags[(r['tag'], repr(v))].append(r['system'] + ':' + str(r['seed']))
        print('\n#', k, '!=', dv)
        for (t, v), l in sorted(tags.items()): print('  ', t, v, len(l), sorted(set(l))[:12])
