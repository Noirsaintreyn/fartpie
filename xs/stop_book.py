import numpy as np, pandas as pd, glob
from collections import deque
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W['d'] = W.entry.dt.normalize()
R = W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
S = pd.concat(pd.read_parquet(f) for f in glob.glob('sl/*.parquet')); S['entry'] = pd.to_datetime(S.entry); S['exit'] = pd.to_datetime(S.exit); S['d'] = S.entry.dt.normalize()
S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].sort_values('d')
S = pd.merge_asof(S, R, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5)); S = S[(S.sp == True) & (S.ratio >= 1.5)]
def sim(Z, a, b, n=6, brk=.08):
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_, ww in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.w.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, w in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_, ww / n))
    for xt, rr, w in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
B = S[S['var'] == 'cap_none'].copy(); B['w'] = 1.0
print(f'live trades {len(B)} | stop distance at entry: median {B.sd.median():.1%}, 75th {B.sd.quantile(.75):.1%}, 90th {B.sd.quantile(.9):.1%}  (15m {B[~B.h1].sd.median():.1%}, 1h {B[B.h1].sd.median():.1%})')
q = pd.qcut(B.sd.rank(method='first'), 4, labels=['tightest', 'q2', 'q3', 'widest']); g = B.groupby(q).ret.agg(['mean', lambda r: (r >= .03).mean(), lambda r: r.quantile(.05)]); g.columns = ['avg', 'P(>=3%)', 'worst 5%']
print('by stop distance quartile:'); print((g * 100).round(2).to_string())
print(f'\n{"rule":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live (vector stop only)', B)
for v, lab in (('cap_3', 'hard stop: close 3% under entry'), ('cap_4', 'hard stop: close 4% under entry'), ('cap_6', 'hard stop: close 6% under entry')):
    z = S[S['var'] == v].copy(); z['w'] = 1.0; show(lab, z)
for th in (.03, .04, .05):
    show(f'skip trades whose stop is more than {th:.0%} away ({(B.sd > th).sum()})', B[B.sd <= th])
for tr in ('median', .02):
    z = B.copy(); t_ = z.sd.median() if tr == 'median' else tr; w = np.clip(t_ / z.sd.clip(lower=.002), .5, 1.5); z['w'] = w / w.mean()
    show(f'equal risk: size = {"median" if tr == "median" else "2%"} / stop distance (avg 1x)', z)
