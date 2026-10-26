import numpy as np, pandas as pd, glob
src = open('/tmp/claude-0/ml/stopbook.py').read(); exec(src[:src.index("S = pd.concat")]); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
def prep(fs):
    S = pd.concat(pd.read_parquet(f) for f in fs); S['d'] = S.entry.dt.normalize()
    S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].sort_values('d')
    S = pd.merge_asof(S, R, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)]; S['w'] = 1.0; return S
S = prep(glob.glob('xs/ext/pb/*.parquet')); L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec']
print(f'{"variant":<44}{"trades":>7}{"avg":>8}{"win":>6}{"hold d":>7}   {"2019-21":>14}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
def row(nm, Z): print(f'{nm:<44}{len(Z):>7}{Z.ret.mean()*100:>+7.2f}%{(Z.ret>0).mean():>6.0%}{(Z.exit-Z.entry).dt.total_seconds().median()/86400:>7.1f}  ' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
row('LIVE vector book', L); print('-- alone (6 slots):')
for v in sorted(S['var'].unique()): row(v, S[S['var'] == v])
print('-- added to the live book (same 6 slots; live signals first on ties):')
for v in sorted(S['var'].unique()):
    Z = pd.concat([L.assign(ratio=L.ratio + 100), S[S['var'] == v]]); row('live + ' + v, Z)
