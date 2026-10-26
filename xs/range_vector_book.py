import numpy as np, pandas as pd, glob
src = open('/tmp/claude-0/ml/stopbook.py').read(); exec(src[:src.index("S = pd.concat")])
exec(src[src.index('def sim('):src.index("B = S[S['var']")])
S = pd.concat(pd.read_parquet(f) for f in glob.glob('/tmp/claude-0/ml/rv/*.parquet')); S['d'] = S.entry.dt.normalize()
S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].sort_values('d')
S = pd.merge_asof(S, R, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)]; S['w'] = 1.0
B = S[S['var'] == 'live']; print('live trades', len(B))
print(f'\n{"rule":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', B)
for v, lab in (('stop_lod', 'stop: 15m close under the day low at entry'), ('tp13', 'take profit: day range >= 1.3 budgets, high looks in'),
               ('tp17', 'take profit: day range >= 1.7 budgets, high looks in'), ('wtp13', 'take profit: week range >= 1.3 wk budgets, high in')):
    show(lab, S[S['var'] == v])
    z = S[S['var'] == v][['tk', 'h1', 'entry', 'ret']].rename(columns={'ret': 'r2'}); h = B.merge(z, on=['tk', 'h1', 'entry'], how='left')
    h['ret'] = .5 * h.ret + .5 * h.r2.fillna(h.ret); show('  half of it (other half rides live)', h)
print('\nentry context (live trades): avg return %, n')
for c, cut in (('both', None), ('wk', None), ('age', [-1, 1, 3, 8, 99])):
    k = B[c] if cut is None else pd.cut(B[c], cut)
    print(B.groupby([k, 'h1']).ret.agg(['mean', 'size']).assign(mean=lambda x: x['mean'] * 100).round(2).to_string(), '\n')
show('only trades where week low looks in (wk)', B[B.wk])
show('skip 15m entries when the 1h is not bull', B[B.h1 | B.both])
