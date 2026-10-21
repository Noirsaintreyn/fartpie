"""Site backtest (Oct 21 2026), honest version: point-in-time S&P 500 members, options ratio >= CUT (252d median, lagged), live rules
(sticky 1h stop, LOCK, 6 slots, highest-ratio tie-break, 8% breaker on realized P&L, no buys after 14:00). Trades: xs/ext/wide_T.parquet
(clean bars). Daily mark-to-market from lse_clean bars, 2 bp per trade. Writes xs/ext/site_wide_stats.json and site_series 'scr'."""
import os, glob, json, numpy as np, pandas as pd, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
from collections import deque
CUT = float(os.getenv('CUT', '1.5')); N = 6; BRK = .08; COST = 2e-4
T = pd.read_parquet('xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= CUT)].copy()
T = T.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
def taken():
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for e, x, r, tk in zip(T.entry.values, T.exit.values, T.ret.values, T.tk.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= N: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-BRK): continue
        op.append(x); pend.append((x, r)); out.append((tk, pd.Timestamp(e), pd.Timestamp(x), r))
    return out
TK = taken(); print(len(TK), 'trades taken')
OP, CL = {}, {}
for tk in sorted({t[0] for t in TK}):
    f = glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'open', 'close', 'volume']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt')
    OP[tk] = H.set_index('dt').open; CL[tk] = H.set_index('dt').close.groupby(lambda x: x.normalize()).last()
px = {}
for b in ('SPY', 'QQQ'):
    Dd = pd.read_parquet(f'etf_daily/{b}.parquet'); Dd['d'] = pd.to_datetime(Dd.datetime).dt.tz_localize(None).dt.normalize(); px[b] = Dd.drop_duplicates('d').set_index('d').adjclose
Bm = pd.DataFrame({k: v.pct_change() for k, v in px.items()})
D = Bm.index[(Bm.index >= '2019-01-01')]; D = D[D <= max(t[2] for t in TK).normalize()]
Rr = pd.Series(0.0, index=D); miss = 0
for tk, e, x, r in TK:
    if tk not in OP: miss += 1; Rr[Rr.index >= x.normalize()][:1] += 0; continue
    p0, p1 = OP[tk].get(e), OP[tk].get(x)
    if p0 is None or p1 is None or not p0 > 0: p0, p1 = 1.0, 1.0 + r                     # fall back to the trade's own return
    d0, d1 = e.normalize(), x.normalize(); c = CL[tk]
    mid = c[(c.index >= d0) & (c.index < d1)] if p1 != 1.0 + r or tk in OP else c.iloc[:0]
    path = pd.concat([pd.Series([p0], index=[d0 - pd.Timedelta(seconds=1)]), mid, pd.Series([p1 * (1 - COST)], index=[d1])])
    rr = path.pct_change().dropna(); rr.index = rr.index.normalize(); rr = rr.groupby(level=0).apply(lambda z: (1 + z).prod() - 1)
    Rr = Rr.add(rr.reindex(D).fillna(0) / N, fill_value=0)
print('trades without bars:', miss)
J = pd.concat([Rr.rename('m'), Bm], axis=1).reindex(D).fillna(0)
def stats(s):
    e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25
    return dict(cagr=e.iloc[-1] ** (1 / y) - 1, dd=(e / e.cummax() - 1).min(), sharpe=s.mean() / s.std() * 252 ** .5)
OUT = {}
for p, a, b in (('19-21', '2019', '2021'), ('22-26', '2022', '2026'), ('19-26', '2019', '2026'), ('2026', '2026', '2026'), ('19-23', '2019', '2023'), ('23-26', '2023', '2026')):
    x = J.loc[a:b]; OUT[p] = {k: stats(x[c]) for k, c in (('model', 'm'), ('SPY', 'SPY'), ('QQQ', 'QQQ'))}
    print(f'  {p}: ' + '  '.join(f"{k} {v['cagr']:+.1%}/{v['dd']:.0%}/S{v['sharpe']:.2f}" for k, v in OUT[p].items()))
f = sm.OLS(J.m, sm.add_constant(J[['SPY', 'QQQ']])).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
OUT['alpha'] = dict(a=f.params['const'] * 252, t=f.tvalues['const'], bspy=f.params['SPY'], bqqq=f.params['QQQ'])
print(f"  alpha vs SPY+QQQ: {OUT['alpha']['a']:+.1%}/yr t {OUT['alpha']['t']:.1f}  betas {f.params['SPY']:.2f}/{f.params['QQQ']:.2f}")
eq = (1 + J.m).cumprod(); print('  $100 since 2019 ->', round(100 * eq.iloc[-1]), ' through', J.index[-1].date())
OUT['thru'] = str(J.index[-1].date()); OUT['n'] = len(TK); json.dump(OUT, open('xs/ext/site_wide_stats.json', 'w'), indent=1, default=float)
J.to_parquet('xs/ext/site_wide_daily.parquet')
if os.getenv('WRITE', '0') == '1':
    S = json.load(open('live_data/site_series.json')); d = pd.PeriodIndex(S['scr']['d'], freq='M')
    lvl = [float(eq[eq.index < p.start_time].iloc[-1]) if (eq.index < p.start_time).any() else 1.0 for p in d]
    S['scr']['b'] = [round(v / lvl[0] * 100, 2) for v in lvl]
    S['scr']['note'] = f'point-in-time S&P 500, options ratio >= {CUT}, 6 slots, 8% breaker, no buys after 14:00, daily mark-to-market'
    json.dump(S, open('live_data/site_series.json', 'w')); print('site_series scr rebuilt, $100 ->', S['scr']['b'][-1])
