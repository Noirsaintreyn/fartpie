import numpy as np, pandas as pd, glob
src = open('/tmp/claude-0/ml/stopbook.py').read(); exec(src[:src.index("S = pd.concat")]); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
S = pd.concat(pd.read_parquet(f) for f in glob.glob('/tmp/claude-0/ml/tm/*.parquet')); S['d'] = S.entry.dt.normalize()
S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].sort_values('d')
S = pd.merge_asof(S, R, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)]; S['w'] = 1.0
print(f'\n{"model":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}   trades  avg   yrs>vec')
Y = [(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01')) for y in range(2019, 2027)]
base = [sim(S[S['var'] == 'vec'], a, b)[0] for a, b in Y]
for v in ['vec', 'donch20', 'donch55', 'turtle20_10', 'supertrend', 'ema21_55', 'kama', 'hma55', 'tsmom', 'kalman']:
    Z = S[S['var'] == v]; yr = [sim(Z, a, b)[0] for a, b in Y]
    print(f'  {v:<56}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)) + f'  {len(Z):>7} {Z.ret.mean()*100:+.2f}%  {sum(x > y for x, y in zip(yr, base))}/8')
