"""Put-heavy flow on the FULL universe: every S&P name with options data (xs/ext/rot_trades, live model 15m + 1h + lock, no ratio filter)."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
old = pd.read_csv('xs/ext/optvol_ratio.csv', usecols=['tk', 'date', 'call_c', 'opt_contracts', 'sh'], parse_dates=['date'])
new = pd.concat(pd.read_csv(f, usecols=['tk', 'date', 'call_c', 'opt_contracts']) for f in glob.glob('data/theta_vol/*.csv')).dropna(subset=['date'])
new['date'] = pd.to_datetime(new.date); F = pd.concat([old.drop(columns='sh'), new]).drop_duplicates(['tk', 'date'], keep='last')
F = F[F.opt_contracts > 0]; F['cs'] = F.call_c / F.opt_contracts
CS = F.pivot_table(index='date', columns='tk', values='cs').sort_index()
CSZ = ((CS - CS.rolling(60, min_periods=30).mean()) / CS.rolling(60, min_periods=30).std()).shift(1)
O = pd.read_csv('xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RAT = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1)
ETF = {'QQQ', 'SPY', 'IWM', 'DIA', 'SMH', 'XLE', 'XLF', 'XLK', 'GLD', 'SLV', 'TLT', 'IEF', 'GDX', 'HYG'}
out = []
for f in glob.glob('xs/ext/rot_trades/*.parquet'):
    tk = os.path.basename(f)[:-8]
    if tk in ETF or tk not in CSZ: continue
    T = pd.read_parquet(f); T['entry'] = pd.to_datetime(T.entry); T = T[T.entry >= '2019-01-01']
    d = T.entry.dt.normalize(); z = CSZ[tk]; T['csz'] = z.reindex(d, method='ffill').values
    T['ratio'] = RAT[tk].reindex(d, method='ffill').values if tk in RAT else np.nan; T['tk'] = tk; out.append(T)
T = pd.concat(out); T = T[T.csz.notna()]; T['per'] = np.where(T.entry < '2022-01-01', '19-21', '22-26')
T['tier'] = pd.cut(T.ratio, [-1, 0.5, 1, 2.5, 99], labels=['ratio <0.5', '0.5-1', '1-2.5', '>=2.5'])
print(f'{len(T):,} trades, {T.tk.nunique()} names, 2019-2026')
b = pd.cut(T.csz, [-9, -1.5, -0.5, 0.5, 1.5, 9])
g = T.groupby([b, 'per']).ret.agg(['mean', 'count']).unstack('per')
print('\nALL NAMES: avg trade by call-share z (neg = puts piling in)'); print((g['mean'] * 100).round(2).astype(str).add('%  n=').add(g['count'].astype(int).astype(str)).to_string())
T['putheavy'] = np.where(T.csz <= -1.5, 'puts piling in (z<=-1.5)', 'rest')
g = T.groupby(['tier', 'putheavy', 'per']).ret.agg(['mean', 'count']).unstack('per')
print('\nby options-ratio tier'); print((g['mean'] * 100).round(2).astype(str).add('%  n=').add(g['count'].astype(int).astype(str)).to_string())
# per-name consistency: share of names where put-heavy flips beat the name's other flips
s = T.groupby(['tk', 'putheavy']).ret.mean().unstack(); s = s.dropna()
print(f"\nnames where put-heavy flips beat that name's other flips: {(s.iloc[:, 0] > s.iloc[:, 1]).mean():.0%} of {len(s)}")
yr = T.groupby([T.entry.dt.year, 'putheavy']).ret.mean().unstack() * 100
print('\nby year (avg %):'); print(yr.round(2).to_string())
