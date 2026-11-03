"""Nov 3 2026: does corporate news that lands WHILE a trade is open (8-K item types, SEC acceptance timestamps) predict what the trade does next?
For every backtest trade (17,447) and every 8-K filing by that ticker accepted after entry and before exit: value of the trade at the first bar open after acceptance,
remaining return to exit = (1+ret)/value - 1. Compared with the same quantity at ALL other times (mean remaining return of trades at a random time inside the hold).
Two periods. DISCOVERY of which item types carry information; any rule is a separate counted test."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
E = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/8k_items.csv'); E['acc'] = pd.to_datetime(E.accept, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
print('8-K rows', len(E), 'tickers', E.ticker.nunique(), 'overlap with trade universe', len(set(E.ticker) & set(X.tk)), '/', X.tk.nunique(), ' range', E.acc.min().date(), E.acc.max().date())
BB = {}
for tk in X.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'open']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); BB[tk] = (H.dt.values, H.open.values)
rows = []; Eg = {k: g.sort_values('acc') for k, g in E.groupby('ticker')}
for tk, g in X.groupby('tk'):
    if tk not in Eg or tk not in BB: continue
    tt, oo = BB[tk]; ea = Eg[tk]; ac = ea.acc.values
    for i, x in g.iterrows():
        j0 = np.searchsorted(tt, np.datetime64(x.entry)); p0 = oo[j0] if j0 < len(oo) else np.nan
        lo, hi = np.searchsorted(ac, np.datetime64(x.entry), side='right'), np.searchsorted(ac, np.datetime64(x.exit), side='left')
        for _, f in ea.iloc[lo:hi].iterrows():
            k = np.searchsorted(tt, np.datetime64(f.acc)); 
            if k >= len(tt) or tt[k] >= np.datetime64(x.exit): continue
            v = oo[k] / p0; rows.append((tk, x.entry, f.acc, str(f['items']), (1 + x.ret) / v - 1, v - 1, x.ret, x.h1))
F = pd.DataFrame(rows, columns=['tk', 'entry', 'acc', 'items', 'rem', 'sofar', 'ret', 'h1']); F['p'] = np.where(F.entry.dt.year <= 2021, 0, 1)
F.to_parquet('/tmp/claude-0/ml/ev8k_F.parquet'); print('filings inside a hold:', len(F), 'on', F[['tk', 'entry']].drop_duplicates().shape[0], 'trades')
print('baseline mean trade', f'{X.ret.mean():+.3%}', ' mean remaining return across all filings', f'{F.rem.mean():+.3%}', '| by period', F.groupby('p').rem.mean().round(4).to_dict())
F['it'] = F['items'].str.split(',')
Ex = F.explode('it'); Ex = Ex[~Ex.it.isin(['9.01'])]
print('\nitem   n(2019-21/2022-26)  mean remaining return after the filing (2019-21 | 2022-26)   share of remaining < -3% (2019-21 | 2022-26)')
for it, g in sorted(Ex.groupby('it'), key=lambda z: -len(z[1])):
    if len(g) < 120: continue
    r = [g[g.p == p] for p in (0, 1)]; print(f'  {it:<5} {len(r[0]):5d}/{len(r[1]):5d}   {r[0].rem.mean():+.2%} | {r[1].rem.mean():+.2%}      {np.mean(r[0].rem < -.03):.1%} | {np.mean(r[1].rem < -.03):.1%}')
