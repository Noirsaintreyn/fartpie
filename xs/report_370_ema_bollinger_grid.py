"""Nov 22 2026 report 370. Rey: 'try different EMA timeframes and counts, then an EMA z-score or Bollinger bands'.
Declared before running.  Flip-bar features for each of the 12,360 historical bull flips (b302), measured at the flip bar's close from the last COMPLETED bar of each timeframe (15m, 1h, daily; the 1h/daily bar still forming is never used):
 DIST  (close - EMA_n) / ATR14 of that timeframe, n in {20, 50, 100, 200}               = 12 features
 ZEMA  (close - EMA_n) / rolling std of (close - EMA_n) over the last 100 bars          = 12 features (the 'EMA z-score')
 PCTB  Bollinger %B (SMA20, 2 sd)  and BWID Bollinger bandwidth (4 sd / SMA20)           = 6 features
 STK   EMA20>EMA50 and EMA50>EMA200 flags                                                = 6 features   (36 total)
Split train 2019-21 | test 2022-26; terciles (flags: yes/no) cut on TRAIN; mean-trade difference top minus bottom, week-clustered t.  TWO-SIDED (sign not declared; a feature counts only if the sign is the same in both halves).  Tiers: SOLID = same sign and |t| >= 2 in BOTH halves; WEAK = same sign and |t| >= 2 in one.  With 36 correlated features expect ~1-2 WEAK by luck and ~0.2 SOLID.  A SOLID feature then gets the book test (skip the unfavourable third) with a placebo of skipping the same number of random flips."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry)
def agg1h(H): g = H.resample('60min', origin='start_day', offset='30min'); return pd.DataFrame({'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last()}).dropna()
def aggd(H): g = H.groupby(H.index.normalize()); return pd.DataFrame({'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last()})
def tf_feats(X, span_ema=(20, 50, 100, 200)):
    c = X.close; tr = np.maximum(X.high - X.low, np.maximum((X.high - c.shift()).abs(), (X.low - c.shift()).abs())); atr = tr.rolling(14).mean(); F = {}
    for n in span_ema:
        e = c.ewm(span=n, adjust=False).mean(); d = c - e; F['dist%d' % n] = d / atr; F['zema%d' % n] = d / d.rolling(100).std()
    s = c.rolling(20).mean(); sd = c.rolling(20).std(); F['pctb'] = (c - (s - 2 * sd)) / (4 * sd); F['bwid'] = 4 * sd / s
    e20, e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200)); F['stk2050'] = (e20 > e50).astype(float); F['stk50200'] = (e50 > e200).astype(float)
    return pd.DataFrame(F)
names = sorted(b.tk.unique()); cols = {}
for tk in names:
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    m = np.where(b.tk.values == tk)[0]; ix = H.index.get_indexer(b.entry.values[m]) - 1; ok = ix >= 0; t_flip = H.index[ix[ok]]; rows = m[ok]
    t15 = tf_feats(H); out = {('15m', k): t15[k].values[ix[ok]] for k in t15.columns}
    H1 = agg1h(H); F1 = tf_feats(H1); end1 = H1.index + pd.Timedelta(minutes=60); pos = np.searchsorted(end1.values, (t_flip + pd.Timedelta(minutes=15)).values, side='right') - 1
    for k in F1.columns: out[('1h', k)] = np.where(pos >= 0, F1[k].values[np.clip(pos, 0, None)], np.nan)
    Hd = aggd(H); Fd = tf_feats(Hd); pd_ = np.searchsorted(Hd.index.values, t_flip.normalize().values, side='left') - 1
    for k in Fd.columns: out[('1d', k)] = np.where(pd_ >= 0, Fd[k].values[np.clip(pd_, 0, None)], np.nan)
    for k, v in out.items():
        cols.setdefault(k, np.full(len(b), np.nan))[rows] = v
    print(tk, end=' ', flush=True)
Fe = pd.DataFrame({'%s_%s' % k: v for k, v in cols.items()}); print('\nfeatures', Fe.shape[1]); Fe.to_parquet(SP + 'F370.parquet')
b['wk'] = b.entry.dt.to_period('W').astype(str); b['half'] = np.where(b.entry < '2022-01-01', 'train', 'test'); b['win'] = (b.ret > 0).astype(float); tr_ = b.half == 'train'
def clus(T, R_):
    Z = pd.concat([T.assign(x=1.0), R_.assign(x=0.0)]); m = sm.OLS(Z.ret.values, sm.add_constant(Z.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Z.wk)[0]}); return m.params[1] * 100, m.tvalues[1], T.win.mean() * 100, R_.win.mean() * 100
res = []
for k in Fe.columns:
    v = Fe[k]; flag = k.endswith('stk2050') or k.endswith('stk50200')
    if flag: lo = hi = .5
    else: lo, hi = v[tr_].quantile([1 / 3, 2 / 3])
    out = {}
    for h in ('train', 'test'):
        D = b[(b.half == h) & v.notna()]; vv = v[D.index]
        T = D[(vv >= .5) if flag else (vv >= hi)]; R_ = D[(vv < .5) if flag else (vv <= lo)]
        out[h] = clus(T, R_) if len(T) > 30 and len(R_) > 30 else (np.nan,) * 4
    same = np.sign(out['train'][0]) == np.sign(out['test'][0]); tier = 'SOLID' if same and abs(out['train'][1]) >= 2 and abs(out['test'][1]) >= 2 else ('WEAK' if same and max(abs(out['train'][1]), abs(out['test'][1])) >= 2 else '')
    res.append((k, out['train'][0], out['train'][1], out['train'][2] - out['train'][3], out['test'][0], out['test'][1], out['test'][2] - out['test'][3], tier))
R = pd.DataFrame(res, columns=['feature', 'tr_diff%', 'tr_t', 'tr_winpts', 'te_diff%', 'te_t', 'te_winpts', 'tier']); R['absmax'] = R[['tr_t', 'te_t']].abs().max(axis=1); R = R.sort_values('absmax', ascending=False)
pd.set_option('display.width', 250); print(R.drop(columns='absmax').round(2).head(20).to_string(index=False)); print('\nSOLID: %d | WEAK: %d of %d features (luck expects ~0.2 SOLID, ~1-2 WEAK)' % ((R.tier == 'SOLID').sum(), (R.tier == 'WEAK').sum(), len(R)))
R.to_csv(SP + 'tab370.csv', index=False)
