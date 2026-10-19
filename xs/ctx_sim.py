import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
O = pd.read_csv('xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RAT = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1); last = O.groupby('tk').date.max()
T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ctx_trades/*.parquet')); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[T.entry >= '2019-01-01']; T = T[T.entry <= T.tk.map(last) + pd.Timedelta(days=30)]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]
def sim(Z, n=8):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
    for e, x, r, w in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.w.values):
        op = [o for o in op if o > e]
        if len(op) < n: op.append(x); ev.append((x, r * w / n))
    return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
def st(s):
    e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'))
NM = {'live': 'live (premarket vector only)', 'A': 'A: enter only if RTH agrees', 'B': 'B: also exit on RTH flip', 'C': 'C: half size if RTH disagrees'}
for thr in (1.5, 2.5):
    print(f'\n== ratio >= {thr}, 8 slots, ratio tie-break ==        {"2019-21":>14}{"2022-26":>16}{"2019-26":>16}   trades  avg/trade')
    for md in ('live', 'A', 'B', 'C'):
        Z = T[(T['mode'] == md) & (T.ratio >= thr)]
        res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
        print(f'  {NM[md]:<34}' + ''.join(f'{c:>+8.1%} /{d:>5.0%}' for c, d in res) + f'{len(Z):>9,}{(Z.ret * Z.w).mean():>+10.2%}', flush=True)
L = T[(T['mode'] == 'live') & (T.ratio >= 1.5)]; A = T[(T['mode'] == 'A') & (T.ratio >= 1.5)]
k = set(zip(A.tk, A.h1, A.entry)); sk = L[[ (t, h, e) not in k for t, h, e in zip(L.tk, L.h1, L.entry)]]
print(f'\nentries where the RTH vector disagreed (skipped by A): {len(sk):,} of {len(L):,} ({len(sk)/len(L):.0%}), avg {sk.ret.mean():+.2%}, win {(sk.ret>0).mean():.0%}'
      f'  vs agreed: avg {L.drop(sk.index, errors="ignore").ret.mean() if False else L[[ (t, h, e) in k for t, h, e in zip(L.tk, L.h1, L.entry)]].ret.mean():+.2%}')
