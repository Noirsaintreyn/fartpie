"""Nov 22 2026 report 375. An outside list of 'structurally different' conditions for when a vector flip has room to run. Testing the ones not already covered (reports 96/98 survival, 138/179 variance ratio, 374 change-point/semivariance), 15m, 12,360 live flips, features at the flip bar close (no look-ahead).
VT  = log(rv10 / rv100)   (volatility transition: short vs long realised vol of 15m log returns); VTD = VT - VT 10 bars ago (change in it)
VREB= first volatility expansion: VT > 0.2 now and VT < 0 for each of the prior 5..30 bars (rebound after a quiet spell)   [flag]
VR4, VR16 = variance ratio of 4- and 16-bar returns over the last 200 bars (persistence map; >1 persistent, <1 reverting)
ES  = directional-efficiency shock: z-score of the 20-bar efficiency ratio vs its own trailing 100-bar mean/sd
TI  = tail imbalance: (# returns > +2 sd) - (# < -2 sd) over the last 100 bars
VMEM= volatility memory: autocorrelation of |r| at lags 1..10 averaged, over the last 200 bars
Two-sided (the list gave no signs). Terciles on train 2019-21; top-minus-bottom mean trade, week-clustered t; flag = yes vs no. SOLID = same sign and |t|>=2 in both halves; WEAK = same sign, |t|>=2 in one. 7 features: expect ~0.1 SOLID and ~0.5 WEAK by luck."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry); cols = {}
for tk in sorted(b.tk.unique()):
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    c = H.close; r = np.log(c).diff(); a = r.abs(); Fd = {}
    rv10 = r.rolling(10).std(); rv100 = r.rolling(100).std(); vt = np.log(rv10 / rv100); Fd['VT'] = vt; Fd['VTD'] = vt - vt.shift(10)
    quiet = (vt < 0).astype(float).rolling(26).sum().shift(5) >= 26; Fd['VREB'] = ((vt > .2) & quiet).astype(float)
    for q in (4, 16): rq = r.rolling(q).sum(); Fd['VR%d' % q] = rq.rolling(200).var() / (q * r.rolling(200).var())
    er = (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum(); Fd['ES'] = (er - er.rolling(100).mean()) / er.rolling(100).std()
    sd = r.rolling(100).std().shift(1); Fd['TI'] = ((r > 2 * sd).astype(float).rolling(100).sum() - (r < -2 * sd).astype(float).rolling(100).sum())
    Fd['VMEM'] = sum(a.rolling(200).corr(a.shift(k)) for k in range(1, 11)) / 10
    m = np.where(b.tk.values == tk)[0]; ix = H.index.get_indexer(b.entry.values[m]) - 1; ok = ix >= 0
    for k, s in Fd.items(): cols.setdefault(k, np.full(len(b), np.nan))[m[ok]] = s.values[ix[ok]]
    print(tk, end=' ', flush=True)
Fe = pd.DataFrame(cols); Fe.to_parquet(SP + 'F375.parquet'); print('\nVREB prevalence %.1f%%' % (Fe.VREB.mean() * 100))
b['wk'] = b.entry.dt.to_period('W').astype(str); b['half'] = np.where(b.entry < '2022-01-01', 'train', 'test'); tr_ = b.half == 'train'
def clus(T, R):
    Z = pd.concat([T.assign(x=1.0), R.assign(x=0.0)]); m = sm.OLS(Z.ret.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); return m.params[1] * 100, m.tvalues[1]
for k in Fe.columns:
    v = Fe[k]; flag = k == 'VREB'
    if flag: lo = hi = .5
    else: lo, hi = v[tr_].quantile([1 / 3, 2 / 3])
    out = {}
    for h in ('train', 'test'):
        D = b[(b.half == h) & v.notna()]; vv = v[D.index]; T = D[(vv >= .5) if flag else (vv >= hi)]; R = D[(vv < .5) if flag else (vv <= lo)]; out[h] = clus(T, R) + (len(T), T.ret.mean() * 100, R.ret.mean() * 100)
    same = np.sign(out['train'][0]) == np.sign(out['test'][0]); tier = 'SOLID' if same and min(abs(out['train'][1]), abs(out['test'][1])) >= 2 else ('WEAK' if same and max(abs(out['train'][1]), abs(out['test'][1])) >= 2 else '')
    print('%-5s train top %+.2f%% vs bottom %+.2f%% diff %+.2f t %+.1f (n %d) | test top %+.2f%% vs bottom %+.2f%% diff %+.2f t %+.1f (n %d) %s' % (k, out['train'][3], out['train'][4], out['train'][0], out['train'][1], out['train'][2], out['test'][3], out['test'][4], out['test'][0], out['test'][1], out['test'][2], tier))
