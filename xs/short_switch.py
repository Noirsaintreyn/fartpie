"""Short switch: take the bear 15m flips as shorts (mirror of the 15m track: short at the RTH open after a bear flip, cover after the
bull flip) only when volatility is high.  Gates: VIX (prior close) and the stock's own 20-day realized vol vs its trailing-year range
(percentile, lagged a day). Longs = live (ahx base+sticky). Ratio >= 2.5, 6 slots shared by longs and shorts."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.argv = ['x', '2']
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
L = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); L = L[L['var'] == 'base+sticky'].drop(columns='var')
S = pd.read_parquet('xs/ext/short_trades.parquet'); S['h1'] = False; S['side'] = 'short'; L['side'] = 'long'
T = pd.concat([L, S]); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5].reset_index(drop=True)
vix = pd.read_parquet('etf_daily/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).normalize()
T['vix'] = vix.shift(1).reindex(T.entry.dt.normalize(), method='ffill').values
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
T['vp'] = np.nan
for tk, idx in T.groupby('tk').groups.items():
    H = pd.read_parquet(fs[tk], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)]
    dc = H.groupby(H.dt.dt.normalize()).close.last(); rv = np.log(dc).diff().rolling(20).std()
    pct = rv.rolling(252, min_periods=120).rank(pct=True).shift(1)
    T.loc[idx, 'vp'] = pct.reindex(T.loc[idx, 'entry'].dt.normalize(), method='ffill').values
Sh = T[T.side == 'short']; per = np.where(Sh.entry.dt.year >= 2022, '22-26', '19-21')
for f, bins in (('vix', [0, 15, 20, 25, 30, 99]), ('vp', [0, .5, .8, .9, 1.01])):
    g = Sh.groupby([pd.cut(Sh[f], bins), per]).ret.agg(['mean', 'count']); g['mean'] *= 100
    print(f'\nshorts: avg trade % by {"VIX" if f == "vix" else "own vol percentile"} (19-21 | 22-26):'); print(g.unstack().round(2).to_string())
def sim(Z, n=6):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
    for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
        op = [o for o in op if o > e]
        if len(op) < n: op.append(x); ev.append((x, r / n))
    return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
def st(s):
    e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))
print(f'\n{"6 slots, ratio >= 2.5, sticky":<44}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
def show(nm, Z):
    res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
    print(f'  {nm:<42}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
Lg = T.side == 'long'; Sg = ~Lg
show('live (long only)', T[Lg])
for nm, c in (('+ shorts when VIX > 25', T.vix > 25), ('+ shorts when VIX > 30', T.vix > 30), ('+ shorts when own vol top 20%', T.vp > .8),
              ('+ shorts when own vol top 10%', T.vp > .9), ('+ shorts when VIX > 25 and own vol top 20%', (T.vix > 25) & (T.vp > .8))):
    show(nm, T[Lg | (Sg & c)])
    show('   switch (no longs while on)', T[(Lg & ~c) | (Sg & c)])
print('\n--- placebo: same shorts, but their return set to 0 (they only occupy slots) ---')
for nm, c in (('+ shorts when own vol top 20%', T.vp > .8), ('+ shorts when own vol top 10%', T.vp > .9)):
    Z = T[Lg | (Sg & c)].copy(); Z.loc[Z.side == 'short', 'ret'] = 0.0; show(nm + ' (placebo)', Z)
    x = T[Sg & c]; print(f'     those shorts: {len(x):,} trades, avg {x.ret.mean():+.3%}, win {(x.ret > 0).mean():.0%}')
print('\n--- benchmarks ---')
for b in ('SPY', 'QQQ'):
    D = pd.read_parquet(f'etf_daily/{b}.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); c = D.drop_duplicates('d').set_index('d').adjclose
    r = c.pct_change().dropna(); print(f'  {b:<42}' + ''.join(f'{x:>+9.1%} /{y:>5.0%}' for x, y in [st(r[(r.index >= a) & (r.index < b_)]) for a, b_ in P]) + f'   (data thru {c.index[-1].date()})')
print('\n--- VIX weekend rule ---')
exec(open('xs/weekend_test.py').read().split("V = pd.read_parquet")[0].split("vix = pd.read_parquet")[0].replace("sys.argv = ['x', '2']", ""))
