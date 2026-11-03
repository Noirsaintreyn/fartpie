"""Nov 3 2026: can the trade ITSELF tell us it is a runner? Path of every backtest trade at day-checkpoints.
Declared before results: R = final return >= +15% (1.3% of trades, ~100% of P&L). For checkpoint k in (1,2,3,5,10) trading days after entry,
bucket by return-so-far; report P(R | bucket) and lift vs base, both periods; plus the 'give-back' stat (drawdown from peak so far).
Pass = lift >= 3 in BOTH periods with n >= 100 for the top bucket."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
CK = (1, 2, 3, 5, 10); out = np.full((len(X), len(CK)), np.nan); PK = np.full((len(X), len(CK)), np.nan)
for tk, g in X.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'open', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt')
    dc = H.groupby(H.dt.dt.normalize()).close.last(); di = dc.index.values; dv = dc.values; t = H.dt.values; o = H.open.values
    for i, x in g.iterrows():
        j = np.searchsorted(t, np.datetime64(x.entry))
        if j >= len(t): continue
        p0 = o[j]; d0 = np.searchsorted(di, np.datetime64(x.entry.normalize()))
        for c, k in enumerate(CK):
            if d0 + k < len(dv) and di[d0 + k] <= np.datetime64(x.exit):
                out[i, c] = dv[d0 + k] / p0 - 1; PK[i, c] = dv[d0:d0 + k + 1].max() / p0 - 1
X['R'] = X.ret >= .15; X['p'] = np.where(X.entry.dt.year <= 2021, 0, 1); base = X.groupby('p').R.mean()
print('base P(R):', {int(k): f'{v:.2%}' for k, v in base.items()}, ' n trades', len(X))
for c, k in enumerate(CK):
    v = out[:, c]; ok = ~np.isnan(v); print(f'\nday {k}: still open {ok.mean():.0%} of trades')
    for lo, hi in ((-9, -.02), (-.02, 0), (0, .03), (.03, .06), (.06, .10), (.10, 9)):
        row = []
        for p in (0, 1):
            m = ok & (v > lo) & (v <= hi) & (X.p.values == p); n = m.sum()
            row.append(f'P={X.R[m].mean():5.1%} lift {X.R[m].mean() / base[p]:4.1f} n{n:5d} (med final {X.ret[m].median():+.1%})' if n > 20 else f'n{n}')
        print(f'  ret so far ({lo:+.0%},{hi:+.0%}]  ' + '  |  '.join(row))
np.save('/tmp/claude-0/ml/runner_out.npy', out)
for t in ('CVNA', 'ORCL', 'AMD'):
    g = X[X.tk == t]; print(t, 'trades', len(g), 'R-share', f'{g.R.mean():.1%}', 'mean', f'{g.ret.mean():+.2%}', ' (base R %.1f%%)' % (100 * X.R.mean()))
