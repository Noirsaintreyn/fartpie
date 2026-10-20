"""Flat into weekends: any live trade still open at Friday's close is sold at Friday's 16:00 close (no re-entry; the flip is used up).
Variants: always, only when VIX (prior close) > 20 / 25.  Live trades (xs/ext/ahx 'base+sticky'), ratio >= 2.5, 6 slots."""
import sys, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.argv = ['x', '2']
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5].reset_index(drop=True)
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
T['fri'] = pd.NaT; T['fret'] = np.nan
for tk, idx in T.groupby('tk').groups.items():
    H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960) & (H.dt.dt.dayofweek < 5)].drop_duplicates('dt').sort_values('dt')
    dc = H.groupby(H.dt.dt.normalize()).close.last(); op = pd.Series(H.open.values, index=H.dt.values)
    days = dc.index
    for i in idx:
        e, x = T.at[i, 'entry'], T.at[i, 'exit']
        span = days[(days >= e.normalize()) & (days < x.normalize())]
        gap = [d for k, d in enumerate(span) if k + 1 < len(span) and (span[k + 1] - d).days > 1] + ([span[-1]] if len(span) and (x.normalize() - span[-1]).days > 1 else [])
        if gap:
            f = gap[0]; px = op.asof(e); T.at[i, 'fri'] = f + pd.Timedelta(hours=16); T.at[i, 'fret'] = dc[f] / px - 1 - 2e-4
vix = pd.read_parquet('etf_daily/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).normalize()
T['vix'] = vix.shift(1).reindex(pd.to_datetime(T.fri).dt.normalize(), method='ffill').values
wk = T.fri.notna(); print(f'{wk.mean():.0%} of trades are held over a weekend (or holiday); their avg: live {T[wk].ret.mean():+.2%} vs sold Friday {T[wk].fret.mean():+.2%}')
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
print(f'\n{"6 slots, ratio >= 2.5, sticky":<40}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
def show(nm, Z):
    res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
    print(f'  {nm:<38}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
show('live', T)
for nm, cond in (('flat every weekend', wk), ('flat weekends when VIX > 20', wk & (T.vix > 20)), ('flat weekends when VIX > 25', wk & (T.vix > 25)), ('flat weekends when VIX > 30', wk & (T.vix > 30))):
    Z = T.copy(); Z.loc[cond, 'ret'] = Z.loc[cond, 'fret']; Z.loc[cond, 'exit'] = Z.loc[cond, 'fri']; Z['exit'] = pd.to_datetime(Z.exit); show(nm, Z)
