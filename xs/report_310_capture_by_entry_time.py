"""Nov 18 2026 report 310. LEAD 1: why did capture fall in 2026? Legs DO store entry minute m (ET). Hypothesis (declared before running): trends front-load, so legs entering after ~10:30 find less move left; pts/trade by entry bucket (<=10:00, 10:00-11:30, 11:30-13:30, >13:30) should decay in 2025-26 for confirmation legs (M1, V15a, RULES) but not for opening legs (RA). Why it would be wrong: 2025 front-loaded too and captured best. Descriptive only; train 2019-21 | test 2018+2022+."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; L = pickle.load(open(SP + 'legs287_ext.pkl', 'rb'))
B = [(0, 600, '<=10:00'), (601, 690, '10-11:30'), (691, 810, '11:30-13:30'), (811, 999, '>13:30')]
for k in ('M1', 'V15a', 'RULES', 'RA', 'R1'):
    v = L[k].copy(); v['y'] = pd.to_datetime(v.day).dt.year; v['b'] = pd.cut(v.m, [-1, 600, 690, 810, 999], labels=[b[2] for b in B])
    print(f'\n{k}: pts/trade (n) by entry bucket x year')
    t = v.pivot_table(index='y', columns='b', values='pts', aggfunc='mean').round(1); n = v.pivot_table(index='y', columns='b', values='pts', aggfunc='size')
    print(pd.concat([t, n.add_prefix('n ')], axis=1).to_string())
    tr = (v.y >= 2019) & (v.y <= 2021); print('  train|test pts/trade by bucket:', {b: f'{v[tr & (v.b == b)].pts.mean():+.1f}|{v[~tr & (v.b == b)].pts.mean():+.1f}' for b in v.b.cat.categories})
