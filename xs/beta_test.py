"""Vector flips (live model, 15m track, rot_trades) by the stock's beta to SPY (252-day, daily, lagged a day), all S&P names with bars.
Is the edge in high beta? Avg trade and win rate by beta tier, and by beta x options ratio."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D = pd.read_parquet('etf_daily/SPY.parquet'); D['d'] = pd.to_datetime(D.datetime).dt.tz_localize(None).dt.normalize(); spy = np.log(D.drop_duplicates('d').set_index('d').adjclose).diff()
O = pd.read_csv('xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RAT = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
old = pd.read_csv('xs/ext/optvol_ratio.csv', parse_dates=['date']); old = old[old.sh > 0]; old['r'] = old.opt_contracts * 100 / old.sh
RAT2 = old.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
ETF = {'QQQ', 'SPY', 'IWM', 'DIA', 'SMH', 'XLE', 'XLF', 'XLK', 'GLD', 'SLV', 'TLT', 'IEF', 'GDX', 'HYG'}
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
out = []
for f in glob.glob('xs/ext/rot_trades/*.parquet'):
    tk = os.path.basename(f)[:-8]
    if tk in ETF or tk not in fs: continue
    T = pd.read_parquet(f); T = T[T['var'] == 'LOCK5/0.5'].copy(); T['entry'] = pd.to_datetime(T.entry); T = T[T.entry >= '2019-01-01']
    if not len(T): continue
    H = pd.read_parquet(fs[tk], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)]
    r = np.log(H.groupby(H.dt.dt.normalize()).close.last()).diff()
    J = pd.concat([r, spy], axis=1, keys=['s', 'm']).dropna()
    beta = (J.s.rolling(252, min_periods=120).cov(J.m) / J.m.rolling(252, min_periods=120).var()).shift(1)
    d = T.entry.dt.normalize(); T['beta'] = beta.reindex(d, method='ffill').values
    R = RAT if tk in RAT else (RAT2 if tk in RAT2 else None)
    T['ratio'] = R[tk].reindex(d, method='ffill').values if R is not None else np.nan
    T['tk'] = tk; out.append(T[['tk', 'entry', 'ret', 'beta', 'ratio']])
T = pd.concat(out).dropna(subset=['beta']); T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21')
print(f'{len(T):,} flips, {T.tk.nunique()} names')
b = pd.cut(T.beta, [-9, .6, .8, 1.0, 1.2, 1.5, 2.0, 9])
g = T.groupby([b, 'per']).ret.agg(['mean', 'count', lambda s: (s > 0).mean()]); g.columns = ['avg', 'n', 'win']; g['avg'] *= 100
print('\navg flip by beta (19-21 | 22-26):'); print(g.unstack().round(2).to_string())
T['rt'] = pd.cut(T.ratio, [-1, 1, 2.5, 99], labels=['ratio<1', '1-2.5', '>=2.5']); T['bt'] = pd.cut(T.beta, [-9, 1.0, 1.5, 9], labels=['beta<1', '1-1.5', '>1.5'])
g = T.groupby(['rt', 'bt', 'per']).ret.agg(['mean', 'count']); g['mean'] *= 100
print('\navg flip by options ratio x beta (19-21 | 22-26):'); print(g.unstack().round(2).to_string())
