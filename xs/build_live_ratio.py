"""Build the screener's options-ratio file from BOTH pulls: the old per-contract pulls (xs/ext/optvol_ratio.csv) + the overnight volume pull
(data/theta_vol, finished names only), shares from the 15m LSE bars. Writes xs/ext/optvol_ratio_live.csv (tk, date, opt_contracts, sh) -
the live file xs/ext/optvol_ratio.csv is NOT touched. Rerun after each monthly fetch_theta_volume.py refresh."""
import glob, os, pandas as pd
old = pd.read_csv('xs/ext/optvol_ratio.csv', parse_dates=['date'])[['tk', 'date', 'opt_contracts', 'sh']]
fs = {os.path.basename(f).split('_15m_')[0]: f for f in sorted(glob.glob('lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'))}
new = []
for f in glob.glob('data/theta_vol/*.csv'):
    v = pd.read_csv(f)
    tk = v.tk.iloc[0]
    if tk not in fs: continue
    v = v.dropna(subset=['date']); v['date'] = pd.to_datetime(v.date)
    if tk in set(old.tk): v = v[v.date > old[old.tk == tk].date.max()]          # refresh: only months after the old pull
    elif v.month.nunique() < 100: continue
    if not len(v): continue
    H = pd.read_parquet(fs[tk], columns=['datetime', 'volume']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)]; sh = H.groupby(H.dt.dt.normalize()).volume.sum().rename('sh')
    new.append(v.merge(sh, left_on='date', right_index=True)[['tk', 'date', 'opt_contracts', 'sh']])
out = pd.concat([old] + new).sort_values(['tk', 'date']); out.to_csv('xs/ext/optvol_ratio_live.csv', index=False)
print(f'{out.tk.nunique()} names ({len(new)} new), data through {out.date.max():%Y-%m-%d}')
# compact copy for the cloud (git-tracked): last ~15 months, only names whose trailing median ratio ever reached 1.0
x = out[out.sh > 0].copy(); x['r'] = x.opt_contracts * 100 / x.sh
med = x.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median()
keep = [t for t in med.columns if (med[t] >= 1.0).any()]
if os.getenv('INCLUDE_NEW') != '1':                                        # universe switch needs Rey's OK: until then only the names already live
    keep = [t for t in keep if t in set(pd.read_csv('xs/ext/optvol_ratio.csv', usecols=['tk']).tk)]
c = out[out.tk.isin(keep) & (out.date >= out.date.max() - pd.Timedelta(days=460))]
os.makedirs('live_data', exist_ok=True); c.to_csv('live_data/optvol_ratio_live.csv', index=False)
print(f'live_data/optvol_ratio_live.csv: {c.tk.nunique()} names, {len(c):,} rows, through {c.date.max():%Y-%m-%d}')
