import numpy as np, pandas as pd, sys
src = open('/home/claude/fartpie/xs/hm_fill.py').read().split("P = [(np.datetime64")[0]
sys.argv = ['x', '/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet', '/mnt/user-data/uploads/fartpie/lse_clean']
exec(src)                                              # T (live), sim(), opx
LIVE = T.copy(); W = pd.read_parquet(sys.argv[1]); W['d'] = W.entry.dt.normalize()
R = W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
S = pd.read_parquet('/mnt/user-data/outputs/ZZSS.parquet'); S['entry'] = pd.to_datetime(S.entry); S['exit'] = pd.to_datetime(S.exit); S['d'] = S.entry.dt.normalize()
S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].sort_values('d')
S = pd.merge_asof(S, R, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5))
S = S[(S.sp == True) & (S.ratio >= 1.5)]
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
print(f'{"bars that build the vector":<50}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
for v, lab in (('sess_pre', 'premarket + RTH (live)'), ('sess_rth', 'RTH only'), ('sess_ext', 'premarket + RTH + after-hours')):
    z = S[S['var'] == v].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable').reset_index(drop=True)
    z['px0'] = [opx(t, e) for t, e in zip(z.tk, z.entry.values)]; T = z
    res = [sim(a, b) for a, b in P]
    print(f'  {lab + f" ({len(z)}, {z.ret.mean():+.2%})":<48}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d, _ in res), flush=True)
