import json, os, statistics as st
T = '/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/tree'
for on, off in [('C1-where', 'C1-modelfree'), ('C3-partial', 'C3-partial-modelfree')]:
    print('\n##', on, 'vs', off, ' phase1 diag mean over seeds (on -> off); per-seed lora sequence_ood nrmse_mean (on -> off)')
    for s in sorted(os.listdir(f'{T}/{off}/runs')):
        line = []
        for k in ('eps_dec', 'eps_pde_rms', 'eps_pde_ss_rms', 'l_dec_q95'):
            v = {}
            for n in (on, off):
                xs = []
                for sd in range(3):
                    p = f'{T}/{n}/runs/{s}/seed{sd}/results.json'
                    if os.path.exists(p): xs.append(json.load(open(p))['phase1']['diag'].get(k))
                v[n] = st.mean(xs)
            line.append(f'{k} {v[on]:.3g}->{v[off]:.3g}')
        seeds = []
        for sd in range(3):
            a = json.load(open(f'{T}/{on}/runs/{s}/seed{sd}/results.json'))['results']['sequence_ood']['clean']['scorecard']['lora']['nrmse_mean']
            b = json.load(open(f'{T}/{off}/runs/{s}/seed{sd}/results.json'))['results']['sequence_ood']['clean']['scorecard']['lora']['nrmse_mean']
            seeds.append(f's{sd} {a:.4f}->{b:.4f}')
        print(f'  {s:11s}', ' | '.join(line), ' || lora seq:', ', '.join(seeds))
