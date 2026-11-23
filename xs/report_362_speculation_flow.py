"""Nov 22 2026 report 362. Rey: 'big moves probably have a lot of speculation or some insider flow - test volume/OI-type ideas and VWAP, try all of them; the O/C ratio did not work'.
Declared before running (9 hypotheses; same split as always: train 2019-21, test 2022-26; terciles cut on TRAIN flips of the 38 option names; flips = b302 rows with an option chain).
DATA NOTE: no open-interest history is on disk (theta_compact has per-contract VOLUME, bid/ask, strike, expiry only), so OI and next-day OI change cannot be tested here; they need a ThetaData pull (separate step).  Option features are the PRIOR trading day (lagged one day), own-history z-scores over the trailing 126 sessions so a name's level cannot stand in for the signal.
 H1 OTM CALL SHARE (call volume with strike >= 1.05 x spot / all call volume; lottery demand): higher -> BIGGER (mean and share >= +5%).
 H2 SHORT-DATED CALL SHARE (call volume with <= 7 days to expiry / all call volume; leverage-seeking, Bollen-Whaley, Pan-Poteshman): higher -> BIGGER.
 H3 ABNORMAL CALL VOLUME (call contracts / own 60d median): higher -> BETTER.
 H4 ABNORMAL TOTAL OPTION VOLUME (all contracts / own 60d median, the 'relative option volume'): higher -> BETTER.
 H5 STRIKE CONCENTRATION (share of call volume in the single busiest strike; block / targeted flow): higher -> BETTER.
 H6 OTM PUT SHARE (put volume with strike <= 0.95 x spot / all put volume; informed bearish or hedging, Pan-Poteshman): higher -> WORSE.
 H7 CALL / PUT VOLUME RATIO: higher -> BETTER.
 H8 VWAP: flip-bar close vs the RTH session VWAP so far (15m bars, 09:30 on; flips in the first bar have none): close ABOVE VWAP (acceptance above the day's average price) -> BETTER.
 H9 FLIP-BAR RELATIVE VOLUME (flip-bar volume / mean volume of the same time-of-day bar over the prior 20 sessions): higher -> BETTER (attention / participation).
Pass rule (strict): same sign in both halves in the declared direction and |t| >= 2 (week-clustered) in at least one half; with 9 hypotheses a single |t| >= 2.7 is the solid bar.  Outputs per hypothesis: top-minus-bottom tercile mean trade return (train | test), t, share >= +5% top vs bottom."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
B = pd.read_parquet(SP + 'b302.parquet'); B['entry'] = pd.to_datetime(B.entry); B['day'] = pd.to_datetime(B.day)
names = sorted(os.path.basename(f)[:-8] for f in glob.glob(U + 'data/theta_compact/*.parquet')); names = [n for n in names if n in set(B.tk)]
rows = []
for tk in names:
    t = pd.read_parquet(U + f'data/theta_compact/{tk}.parquet', columns=['d', 'e', 'K', 'r', 'vol', 'dte', 'S']); t = t[t.vol > 0]; t['mn'] = t.K / t.S
    c = t[t.r == 1]; p = t[t.r == -1]
    g = pd.DataFrame({'call': c.groupby('d').vol.sum(), 'put': p.groupby('d').vol.sum()}).fillna(0)
    g['otmc'] = c[c.mn >= 1.05].groupby('d').vol.sum(); g['shortc'] = c[c.dte <= 7].groupby('d').vol.sum(); g['otmp'] = p[p.mn <= .95].groupby('d').vol.sum()
    g['top1'] = c.groupby(['d', 'K']).vol.sum().groupby('d').max(); g = g.fillna(0); g = g[(g.call > 0) & (g.put > 0)]
    f = pd.DataFrame({'otmc': g.otmc / g.call, 'shortc': g.shortc / g.call, 'abn_c': g.call, 'abn_t': g.call + g.put, 'conc': g.top1 / g.call, 'otmp': g.otmp / g.put, 'cp': np.log(g.call / g.put)})
    f['abn_c'] = f.abn_c / f.abn_c.rolling(60, min_periods=30).median(); f['abn_t'] = f.abn_t / f.abn_t.rolling(60, min_periods=30).median()
    for k in ('abn_c', 'abn_t'): f[k] = np.log(f[k])
    for k in f.columns:
        s = f[k]; f[k + '_z'] = (s - s.rolling(126, min_periods=60).mean()) / s.rolling(126, min_periods=60).std()
    f = f[[c_ for c_ in f.columns if c_.endswith('_z')]].shift(1); f['tk'] = tk; rows.append(f.reset_index().rename(columns={'d': 'date', 'index': 'date'}))
F = pd.concat(rows); F['date'] = pd.to_datetime(F.date)
M = B[B.tk.isin(names)].copy().sort_values('day'); F = F.sort_values('date')
M = pd.merge_asof(M.sort_values('day'), F.sort_values('date'), left_on='day', right_on='date', by='tk', direction='backward', tolerance=pd.Timedelta(days=5))
# VWAP and flip-bar relative volume from the 15m cache
vw = np.full(len(M), np.nan); rv = np.full(len(M), np.nan); M = M.reset_index(drop=True)
for tk in names:
    fl = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(fl); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    H['d'] = H.index.normalize(); H['tod'] = H.index.strftime('%H:%M'); R = H[(H.index.hour * 60 + H.index.minute >= 570) & (H.index.hour * 60 + H.index.minute < 960)].copy()
    R['pv'] = R.close * R.volume; R['cpv'] = R.groupby('d').pv.cumsum(); R['cv'] = R.groupby('d').volume.cumsum(); R['vwap'] = R.cpv / R.cv
    R['rvol'] = R.volume / R.groupby('tod').volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    idx = R.index.values; m = np.where(M.tk.values == tk)[0]
    for j in m:
        a = np.searchsorted(idx, np.datetime64(M.entry[j])) - 1
        if a < 0 or idx[a] >= np.datetime64(M.entry[j]) or R.d.values[a] != np.datetime64(M.entry[j].normalize()): continue
        vw[j] = R.close.values[a] / R.vwap.values[a] - 1; rv[j] = R.rvol.values[a]
M['vwap_dist'] = vw; M['frvol'] = np.log(rv)
print('flips with option features %d of %d; VWAP available %d; flip-bar rvol %d' % (M.otmc_z.notna().sum(), len(M), M.vwap_dist.notna().sum(), M.frvol.notna().sum()), flush=True)
H_ = [('H1 OTM call share', 'otmc_z', 1), ('H2 short-dated call share', 'shortc_z', 1), ('H3 abnormal call volume', 'abn_c_z', 1), ('H4 abnormal total option volume', 'abn_t_z', 1), ('H5 strike concentration', 'conc_z', 1), ('H6 OTM put share', 'otmp_z', -1), ('H7 call/put ratio', 'cp_z', 1), ('H8 close vs VWAP', 'vwap_dist', 1), ('H9 flip-bar relative volume', 'frvol', 1)]
M['wk'] = M.entry.dt.to_period('W').astype(str); M['half'] = np.where(M.entry < '2022-01-01', 'train', 'test'); M['big'] = (M.ret > .05).astype(float); tr = M[M.half == 'train']
print('\n%-30s %-28s %-28s  pass' % ('', 'TRAIN  top-bot mean (t) | >=5% top/bot', 'TEST  top-bot mean (t) | >=5% top/bot'))
out = []
for lab, col, sgn in H_:
    lo, hi = tr[col].quantile([1 / 3, 2 / 3]); res = {}
    for h in ('train', 'test'):
        D = M[(M.half == h) & M[col].notna()]; T = D[D[col] >= hi]; Bo = D[D[col] <= lo]; Z = pd.concat([T.assign(x=1.0), Bo.assign(x=0.0)])
        mod = sm.OLS(Z.ret.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); res[h] = (mod.params[1] * 100, mod.tvalues[1], T.big.mean() * 100, Bo.big.mean() * 100, len(T), len(Bo))
    s = lambda h: '%+.2f%% (t %+.1f) | %.1f/%.1f%% n%d/%d' % res[h]
    same = np.sign(res['train'][0]) == np.sign(res['test'][0]) == sgn; tmax = max(abs(res['train'][1]), abs(res['test'][1])); ok = same and tmax >= 2
    print('%-30s %-48s %-48s %s' % (lab, s('train'), s('test'), 'PASS' if ok else ('same sign, |t|<2' if same else 'fail')), flush=True); out.append((lab, res, ok))
M.to_parquet(SP + 'M362.parquet')
