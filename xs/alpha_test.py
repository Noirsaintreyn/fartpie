"""Does the live model have alpha?  Daily mark-to-market book (8 slots, ratio tie-break, live model trades) regressed on
SPY, QQQ, and an equal-weight buy-and-hold of the same eligible names (does the vector timing add anything over just owning them?).
HAC (Newey-West, 5 lags) t-stats. Alpha annualised (x252)."""
import glob, os, numpy as np, pandas as pd, warnings, statsmodels.api as sm; warnings.filterwarnings('ignore')
exec(open('xs/ctx_sim.py').read().split('def sim(')[0]); T = T[T['mode'] == 'live'].copy()
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
OP, CL = {}, {}
for tk in sorted(set(T.tk)) + ['SPY', 'QQQ']:
    H = pd.read_parquet(fs[tk]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960) & (H.volume > 0)].drop_duplicates('dt').sort_values('dt')
    OP[tk] = H.set_index('dt').open; CL[tk] = H.set_index('dt').close.groupby(lambda x: x.normalize()).last()
CL = pd.DataFrame(CL).sort_index(); D = CL.index[(CL.index >= '2019-01-01')]
def taken(Z, n=8):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, out = [], []
    for row in Z.itertuples():
        op = [o for o in op if o > row.entry]
        if len(op) < n: op.append(row.exit); out.append(row)
    return out
def book(tr, n=8):
    R = pd.Series(0.0, index=D)
    for t in tr:
        p0 = OP[t.tk].get(t.entry); p1 = OP[t.tk].get(t.exit)
        if p0 is None or p1 is None: continue
        d0, d1 = t.entry.normalize(), t.exit.normalize(); c = CL[t.tk]
        path = pd.concat([pd.Series([p0], index=[d0 - pd.Timedelta(seconds=1)]), c[(c.index >= d0) & (c.index < d1)], pd.Series([p1], index=[d1])])
        r = path.pct_change().dropna(); r.index = r.index.normalize(); r = r.groupby(level=0).apply(lambda x: (1 + x).prod() - 1)
        R = R.add(r.reindex(D).fillna(0) / n, fill_value=0)
    return R
ret = CL.pct_change()
for thr in (1.5, 2.5):
    tr = taken(T[T.ratio >= thr]); B = book(tr)
    el = pd.DataFrame({tk: RAT[tk].reindex(D, method='ffill') >= thr for tk in set(T.tk)})
    EW = (ret[el.columns].reindex(D) * el.shift(0)).sum(axis=1) / el.sum(axis=1).replace(0, np.nan)
    X = pd.DataFrame({'book': B, 'SPY': ret.SPY.reindex(D), 'QQQ': ret.QQQ.reindex(D), 'EW': EW}).dropna()
    print(f'\n== ratio >= {thr}: {len(tr):,} trades taken, avg exposure {X.book.abs().gt(0).mean():.0%} of days ==')
    for lab, a, b in (('2019-21', '2019', '2022'), ('2022-26', '2022', '2027'), ('2019-26', '2019', '2027')):
        x = X[(X.index >= a) & (X.index < b)]; line = f'  {lab}: '
        for fac in (['SPY'], ['QQQ'], ['EW'], ['SPY', 'EW']):
            f = sm.OLS(x.book, sm.add_constant(x[fac])).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            line += f"| vs {'+'.join(fac):<6} alpha {f.params['const']*252:+.1%}/yr (t {f.tvalues['const']:.1f}) beta {f.params[fac[0]]:.2f}  "
        print(line, flush=True)
    s = X.book; print(f'  Sharpe book {s.mean()/s.std()*np.sqrt(252):.2f} | SPY {X.SPY.mean()/X.SPY.std()*np.sqrt(252):.2f} | QQQ {X.QQQ.mean()/X.QQQ.std()*np.sqrt(252):.2f} | EW names {X.EW.mean()/X.EW.std()*np.sqrt(252):.2f}')
