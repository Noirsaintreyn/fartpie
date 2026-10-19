"""Bigger universe without diluting: (a) fixed cutoffs, (b) top-K names by options ratio each day (adaptive cutoff),
(c) tiered: >=2.5 names take any slot, 1.0/1.5-2.5 names only fill up to M slots. 6 slots, ratio tie-break, live model trades, booked at exit."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
O = pd.read_csv('xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RAT = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
last = O.groupby('tk').date.max()
ETF = {'QQQ', 'SPY', 'IWM', 'DIA', 'SMH', 'XLE', 'XLF', 'XLK', 'GLD', 'SLV', 'TLT', 'IEF'}
RAT = RAT[[c for c in RAT.columns if c not in ETF]]
for t in RAT.columns: RAT.loc[RAT.index > last[t] + pd.Timedelta(days=30), t] = np.nan
RK = RAT.rank(axis=1, ascending=False)
out = []
for tk in RAT.columns:
    f = f'xs/ext/rot_trades/{tk}.parquet'
    if not os.path.exists(f) or not (RAT[tk] >= 0.8).any(): continue
    T = pd.read_parquet(f); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit); T = T[T.entry >= '2019-01-01'].copy()
    d = T.entry.dt.normalize(); T['ratio'] = RAT[tk].reindex(d, method='ffill').values; T['rank'] = RK[tk].reindex(d, method='ffill').values; T['tk'] = tk
    out.append(T[T.ratio.notna()])
T = pd.concat(out); T['h1'] = T['var'].str.startswith('H1')
def sim(Z, n=6, cap_low=None, hi=2.5):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
    for e, x, r, q in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.ratio.values):
        op = [o for o in op if o[0] > e]
        if len(op) >= n: continue
        low = q < hi
        if cap_low is not None and low and sum(o[1] for o in op) >= cap_low: continue
        op.append((x, low)); ev.append((x, r / n))
    return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
def st(s):
    e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'))
def run(nm, Z, **k):
    res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)], **k)) for a, b in P]
    nn = Z.groupby(Z.entry.dt.year).tk.nunique().mean()
    print(f'  {nm:<40}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res) + f'{nn:>8.0f}', flush=True)
print(f'{len(T):,} trades, {T.tk.nunique()} names\n{"6 slots, ratio tie-break":<42}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}  names/yr')
for c in (2.5, 2.0, 1.5, 1.0): run(f'cutoff >= {c}', T[T.ratio >= c])
for k in (10, 15, 20, 25, 30, 40): run(f'top {k} by ratio each day', T[T['rank'] <= k])
for lo in (1.5, 1.0):
    for m in (1, 2, 3): run(f'>=2.5 any slot + {lo}-2.5 in max {m} slot(s)', T[T.ratio >= lo], cap_low=m)
