"""Report 507 (Nov 27 2026): does the VIX level at the prior close change what the flips earn? Mechanism (Citi bulletin claim + our persistence work): after a fear spike the flip edge is larger (mean reversion) but the tail is fatter; the test is the mean and the left tail by VIX bucket.
Large-cap flips: wide_T base+sticky, sp, ratio>=1.5, 15m+1h, and the PCV subset. Small-cap: flip_history (sub-$5 daily flips; reach = hit). Prior-close VIX, train 2019-21 / test 2022+, month-clustered t."""
import pandas as pd, numpy as np
UP = '/mnt/user-data/uploads/fartpie/'
v = pd.read_parquet('data/yahoo/daily_v/VIX.parquet').close; v.index = pd.to_datetime(v.index).tz_localize(None).normalize(); v = v.sort_index()
vz = (v - v.rolling(252).mean()) / v.rolling(252).std()
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W = W[W.sp & (W.ratio >= 1.5)].copy(); W['entry'] = pd.to_datetime(W.entry); W['d'] = W.entry.dt.normalize()
W['vix'] = v.shift(1).reindex(W.d, method='ffill').values; W['vz'] = vz.shift(1).reindex(W.d, method='ffill').values; W['mon'] = W.entry.dt.strftime('%Y-%m'); W['era'] = np.where(W.entry < '2022-01-01', 'train 19-21', 'test 22-26')
M = pd.read_csv('live_data/model_flips.csv', parse_dates=['entry']); M['h1'] = M.tf == '1h'
pcv = M[M.tag == 'PCV'][['tk', 'entry', 'h1']].drop_duplicates(); W = W.merge(pcv.assign(PCV=True), on=['tk', 'entry', 'h1'], how='left'); W['PCV'] = W.PCV.fillna(False)
W['b'] = pd.cut(W.vix, [0, 16, 20, 25, 30, 100], labels=['<16', '16-20', '20-25', '25-30', '30+'])
def stat(x):
    m = x.groupby('mon').ret.mean(); t = m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 2 and m.std() > 0 else np.nan
    return pd.Series(dict(n=len(x), bp=x.ret.mean() * 1e4, win=(x.ret > 0).mean() * 100, p5=x.ret.quantile(.05) * 1e4, t=t))
for nm, D in (('ALL large-cap flips', W), ('PCV flips', W[W.PCV])):
    print('\n', nm); print(D.groupby(['era', 'b'], observed=True).apply(stat).round(1).to_string())
    print(' fear z>1.5 vs not:'); print(D.assign(f=D.vz > 1.5).groupby(['era', 'f']).apply(stat).round(1).to_string())
# small caps
H = pd.read_csv('flip_history.csv', parse_dates=['d']); H['vix'] = v.shift(1).reindex(H.d, method='ffill').values; H['b'] = pd.cut(H.vix, [0, 16, 20, 25, 100], labels=['<16', '16-20', '20-25', '25+']); H['era'] = np.where(H.d < '2022-01-01', 'train 19-21', 'test 22-26')
print('\n small-cap reach by VIX'); print(H.groupby(['era', 'b'], observed=True).agg(n=('hit', 'size'), reach=('hit', lambda x: round(x.mean() * 100)), med_r=('r', lambda x: round(x.median() * 100, 1))).to_string())
