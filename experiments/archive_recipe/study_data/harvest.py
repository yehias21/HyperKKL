"""Harvest archived HyperKKL run artefacts (read-only) into one JSON table.
usage: python -I harvest.py <autoresearch_dir> <out.json>"""
import json, os, sys, statistics as st
import yaml
root, out = sys.argv[1], sys.argv[2]
KN = {
 'p1.opt': ('phase1','optimizer'), 'p1.lr': ('phase1','lr'), 'p1.ep': ('phase1','epochs'),
 'p1.nep': ('phase1','n_episodes'), 'p1.ncoll': ('phase1','n_collocation'), 'p1.gm': ('phase1','grad_method'),
 'p1.pde': ('phase1','use_pde'), 'p1.lam': ('phase1','lambda_pde'), 'p1.syl': ('phase1','sylvester_init'),
 'p2.opt': ('phase2','optimizer'), 'p2.lr': ('phase2','lr'), 'p2.ep': ('phase2','epochs'),
 'p2.nep': ('phase2','n_train_episodes'), 'p2.frz': ('phase2','freeze_encoder_deltas'),
 'p2.res': ('phase2','residual_mode'), 'p2.rank': ('phase2','lora_rank'), 'p2.curv': ('phase2','delta_curv'),
 'p2.dreg': ('phase2','delta_reg'), 'p2.smooth': ('phase2','delta_smooth'), 'p2.gm': ('phase2','grad_method'),
 'p2.fam': ('phase2','train_families'), 'p2.cur': ('phase2','curriculum'), 'p2.W': ('phase2','window_size'),
 'p2.sfmb': ('phase2','sfm_blocks'), 'p2.wlab': ('phase2','w_label'),
 'nm': ('system','noise_meas'), 'np': ('system','noise_process'), 'nk': ('system','noise_kind'),
 'pk': ('system','process_kind'), 'outl': ('system','outlier_rate'), 'imode': ('system','input_mode'),
 'alpha_ov': ('system','alpha_override'), 'nz_ov': ('system','nz_override'), 'spec_ov': ('system','spectrum_override'),
 'pbias': ('system','param_bias'), 'epbias': ('system','eval_param_bias'), 'pjit': ('system','param_jitter'),
 'latinit': ('evaluation','latent_init'), 'evrep': ('evaluation','eval_rep'), 'neval': ('evaluation','n_episodes'),
 'obsinit': ('evaluation','obs_init'), 'rad': ('evaluation','obs_init_radius'),
}
rows = []
for dp, dn, fn in os.walk(root):
    if 'results.json' not in fn: continue
    rel = os.path.relpath(dp, root).split(os.sep)
    try:
        d = json.load(open(os.path.join(dp, 'results.json')))
    except Exception as e:
        print('bad', dp, e); continue
    cfg = {}
    cp = os.path.join(dp, 'config.yaml')
    if os.path.exists(cp):
        try:
            c = yaml.safe_load(open(cp))
            for k, (a, b) in KN.items():
                cfg[k] = (c.get(a) or {}).get(b)
        except Exception as e:
            print('badcfg', dp, e)
    mtime = os.path.getmtime(os.path.join(dp, 'results.json'))
    res = {}
    for tier, per in (d.get('results') or {}).items():
        for m, ag in per.items():
            bf = ag.get('by_family') or {}
            fam = {f: v.get('nrmse_steady') for f, v in bf.items()}
            ci = {f: v.get('nrmse_steady_ci') for f, v in bf.items()}
            res.setdefault(tier, {})[m] = {'pooled': ag.get('nrmse_steady'), 'fam': fam, 'ci': ci,
                                         'div': ag.get('divergence_rate')}
    ts = d.get('train_stats') or {}
    ntr = {m: v.get('n_trainable') for m, v in ts.items() if isinstance(v, dict) and 'n_trainable' in v}
    rows.append({'path': '/'.join(rel), 'tag': rel[0], 'system': d.get('system'), 'seed': d.get('seed'),
                 'split': d.get('split'), 'mtime': mtime, 'cfg': cfg, 'res': res, 'n_trainable': ntr,
                 'learning_flags': d.get('learning_flags'), 'methods': d.get('methods_requested')})
json.dump(rows, open(out, 'w'))
print(len(rows), 'runs')
