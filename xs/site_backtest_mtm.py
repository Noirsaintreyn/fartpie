"""Daily mark-to-market book for the site: live model (ahx base+sticky, ratio >= 2.5, 6 slots) + 8% drawdown breaker (decided on realized P&L,
as live). Stats vs SPY / QQQ, alpha (HAC) vs SPY+QQQ and vs an equal-weight hold of the same eligible names; rebuilds site_series 'scr'."""
import sys, glob, os, json, numpy as np, pandas as pd, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
exec(open('xs/dyn_slots.py').read().split("if __name__ == '__main__':")[0])
TK = Z0.tk.values
def taken(brk):
    op = []; out = []; pend = []; real = 0.0; hist = []
    for e, x, r, tk in zip(E, X, R, TK):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / 6); hist.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= 6: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D'); inw = [h for t_, h in hist if t_ >= t0]; pre = [h for t_, h in hist if t_ < t0]
            if real - max([pre[-1] if pre else 0.0] + inw) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r)); out.append((tk, pd.Timestamp(e), pd.Timestamp(x)))
    return out
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
OP, CL = {}, {}
for tk in sorted(set(TK)):
    H = pd.read_parquet(fs[tk], columns=['datetime', 'open', 'close', 'volume']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960) & (H.volume > 0)].drop_duplicates('dt').sort_values('dt')
    OP[tk] = H.set_index('dt').open; CL[tk] = H.set_index('dt').close.groupby(lambda x: x.normalize()).last()
CL = pd.DataFrame(CL).sort_index(); D = CL.index[CL.index >= '2019-01-01']
def book(tr):
    Rr = pd.Series(0.0, index=D)
    for tk, e, x in tr:
        p0 = OP[tk].get(e); p1 = OP[tk].get(x)
        if p0 is None or p1 is None: continue
        d0, d1 = e.normalize(), x.normalize(); c = CL[tk]
        path = pd.concat([pd.Series([p0], index=[d0 - pd.Timedelta(seconds=1)]), c[(c.index >= d0) & (c.index < d1)], pd.Series([p1], index=[d1])])
        r = path.pct_change().dropna(); r.index = r.index.normalize(); r = r.groupby(level=0).apply(lambda z: (1 + z).prod() - 1)
        Rr = Rr.add(r.reindex(D).fillna(0) / 6, fill_value=0)
    return Rr
px = {}
for b in ('SPY', 'QQQ'):
    Dd = pd.read_parquet(f'etf_daily/{b}.parquet'); Dd['d'] = pd.to_datetime(Dd.datetime).dt.tz_localize(None).dt.normalize(); px[b] = Dd.drop_duplicates('d').set_index('d').adjclose
Bm = pd.DataFrame({k: v.pct_change() for k, v in px.items()})
elig = T.groupby(T.entry.dt.normalize()).tk.agg(set)                                    # equal-weight hold of the names eligible that day
ew = CL.pct_change().reindex(D); names = pd.Series([set() for _ in D], index=D)
cur = set(); lastd = {}
for d0, s in elig.items(): lastd.update({t: d0 for t in s})
ewr = pd.Series([ew.loc[d, [t for t, l in lastd.items() if l <= d and (d - l).days < 60]].mean() if True else 0 for d in D], index=D).fillna(0)
def stats(s):
    e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25
    return dict(cagr=e.iloc[-1] ** (1 / y) - 1, dd=(e / e.cummax() - 1).min(), sharpe=s.mean() / s.std() * 252 ** .5)
res = {}
for brk, nm in ((0.0, 'live'), (.08, 'breaker')):
    r = book(taken(brk)); res[nm] = r
    J = pd.concat([r, Bm, ewr.rename('EW')], axis=1).dropna(); J.columns = ['m', 'SPY', 'QQQ', 'EW']
    print(f'\n== {nm}')
    for p, a, b in (('19-21', '2019', '2021'), ('22-26', '2022', '2026'), ('19-26', '2019', '2026'), ('2026', '2026', '2026')):
        x = J.loc[a:b]; print(f'  {p}: ' + '  '.join(f"{k} {v['cagr']:+.1%}/{v['dd']:.0%}/S{v['sharpe']:.2f}" for k, v in ((k, stats(x[c])) for k, c in (('model', 'm'), ('SPY', 'SPY'), ('QQQ', 'QQQ')))))
    for X_ in (['SPY', 'QQQ'], ['EW']):
        f = sm.OLS(J.m, sm.add_constant(J[X_])).fit(cov_type='HAC', cov_kwds={'maxlags': 5}); print(f"  alpha vs {'+'.join(X_)}: {f.params['const'] * 252:+.1%}/yr t {f.tvalues['const']:.1f}")
r = res['breaker']; eq = (1 + r).cumprod()
S = json.load(open('live_data/site_series.json')); d = pd.PeriodIndex(S['scr']['d'], freq='M')
lvl = [float(eq[eq.index < p.start_time].iloc[-1]) if (eq.index < p.start_time).any() else 1.0 for p in d]
S['scr']['b'] = [round(v / lvl[0] * 100, 2) for v in lvl]; S['scr']['note'] = 'live model + 8% drawdown breaker (Oct 20 2026), daily mark-to-market'
json.dump(S, open('live_data/site_series.json', 'w')); print('\nsite_series scr rebuilt, $100 ->', S['scr']['b'][-1])
