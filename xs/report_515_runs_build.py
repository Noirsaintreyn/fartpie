"""Report 515 build (Nov 27 2026): what precedes a 100%+ run in sub-$5 stocks. Rey: "go through all 100% runs, find a reason: catalysts, HMM / hidden state, fundamental or macro".
DECLARED BEFORE RESULTS.
Unit: stock-day d with close<5 (split-adjusted Yahoo daily, 1,656-name wide file, 2018-2026-11; survivorship: current-ticker list, disclosed).
Label y(d) = max(close[d+1..d+40]) >= 2*close[d]   (a 100% run from d; the 40-day window defines the EVENT, it is not a trade exit).
Features known at the close of d (all lagged to be point-in-time):
  VOL   ATRP20 (carries the reach edge per reports 133/464)   -- the benchmark every other feature must beat
  TECH  r20 (20d return), DD252 (close/252d high-1), VX5 (5d dollar vol / 60d dollar vol), DVOL (20d median dollar volume)
  CAT   8-K in prior 7d: DEAL (1.01/2.01), PR (7.01/8.01 w/o 2.02), EARN (2.02), NEG (3.02/1.02/1.03/2.05)  [known by close d: filed d or earlier, accepted <20Z]
  SHORT FINRA short interest / avg daily volume (days to cover, published +9d), SVR5 (5d short-volume ratio)
  MACRO VIX, dVIX20, HY_OAS, dHY20, USD20 chg, T10Y2Y, NFCI, market breadth, index regime
  WAVE  speculative breadth: share of sub-$5 names up >30% over prior 20d (risk-appetite state of the penny market)
  HMM   (a) stock-level 3-state Gaussian HMM on [ret, log vol/60d med, log range] pooled, FIT 2019-21 only, forward-filtered (no smoothing);
        (b) market-level 3-state HMM on [EW sub-$5 ret, cross-section dispersion, dVIX], FIT 2019-21 only, forward-filtered.
PASS (declared): feature adds to ATRP on P(y) in EVERY era (2019-21, 2022, 2023-26) with month-clustered |t|>=2 on the within-ATRP-bucket lift.
"""
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); H = 40
K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K['filed'] = pd.to_datetime(K.filed); K['acc'] = pd.to_datetime(K.accept, utc=True, errors='coerce')
K['known'] = pd.to_datetime(np.where(K.acc.dt.hour < 20, K.filed, K.filed + pd.Timedelta(days=1))).astype('datetime64[ns]'); it = K['items'].astype(str).str.split(',')
K['DEAL'] = it.apply(lambda l: '1.01' in l or '2.01' in l); K['EARN'] = it.apply(lambda l: '2.02' in l); K['PR'] = it.apply(lambda l: ('7.01' in l or '8.01' in l) and '2.02' not in l)
K['NEG'] = it.apply(lambda l: any(x in l for x in ('3.02', '1.02', '1.03', '2.05'))); K['ANY'] = True
KG = {t: g for t, g in K.groupby('ticker')}
SI = pd.read_pickle(X + 'shortint.pkl')[['symbolCode', 'settlementDate', 'currentShortPositionQuantity', 'averageDailyVolumeQuantity']]; SI['use'] = pd.to_datetime(SI.settlementDate).astype('datetime64[ns]') + pd.Timedelta(days=9)
SIG = {t: g.sort_values('use') for t, g in SI.groupby('symbolCode')}
SV = pd.read_pickle(X + 'finra_sv_wide.pkl'); SV['Date'] = pd.to_datetime(SV.Date); SV['r'] = SV.ShortVolume / SV.TotalVolume.replace(0, np.nan); SVG = {t: g.set_index('Date').r for t, g in SV.groupby('Symbol')}
rows = []; seq = {}
for tk, q in D.items():
    q = q.dropna(subset=['close']); q = q[q.close > 0]
    if len(q) < 300 or not (q.close < 5).any(): continue
    q = q.copy(); q.index = pd.DatetimeIndex(q.index).astype('datetime64[ns]'); c = q.close; v = q.volume.astype(float); dv = c * v
    ret = c.pct_change(); rng = np.log(q.high / q.low.clip(lower=1e-6)).clip(0, 3); tr = pd.concat([q.high - q.low, (q.high - c.shift()).abs(), (q.low - c.shift()).abs()], axis=1).max(axis=1)
    f = pd.DataFrame(dict(tk=tk, c=c, ATRP=tr.rolling(20).mean() / c, r20=c / c.shift(20) - 1, DD252=c / c.rolling(252, min_periods=120).max() - 1, VX5=dv.rolling(5).mean() / dv.rolling(60).mean().shift(5), DVOL=dv.rolling(20).median(), ret=ret))
    f.index.name = 'd'
    fwd = pd.concat([c.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1); f['y'] = (fwd >= 2 * c).astype(float); f.loc[fwd.isna(), 'y'] = np.nan
    f['fmax'] = fwd / c - 1
    g = KG.get(tk)
    for col in ('DEAL', 'PR', 'EARN', 'NEG', 'ANY'):
        f[col] = 0
        if g is not None:
            kk = np.sort(g.loc[g[col], 'known'].values); idx = f.index.values
            if len(kk): j = np.searchsorted(kk, idx, side='right'); prev = np.where(j > 0, kk[np.clip(j - 1, 0, len(kk) - 1)], np.datetime64('NaT')); f[col] = ((idx - prev) <= np.timedelta64(7, 'D')).astype(int) & (j > 0)
    s = SIG.get(tk)
    if s is not None:
        m = pd.merge_asof(f[[]].reset_index().sort_values('d').astype({'d': 'datetime64[ns]'}), s[['use', 'currentShortPositionQuantity', 'averageDailyVolumeQuantity']].rename(columns={'use': 'd'}).astype({'d': 'datetime64[ns]'}), on='d', direction='backward', tolerance=pd.Timedelta(days=30))
        f['DTC'] = (m.currentShortPositionQuantity / m.averageDailyVolumeQuantity.replace(0, np.nan)).values
    else: f['DTC'] = np.nan
    sv = SVG.get(tk); f['SVR5'] = sv.reindex(f.index).rolling(5, min_periods=3).mean().shift(0).values if sv is not None else np.nan
    f['lv'] = np.log((v / v.rolling(60).median().shift(1)).clip(lower=.05)); f['rg'] = rng; f['ret1'] = ret.clip(-.5, 1.5)
    f = f[f.c < 5]
    rows.append(f)
P = pd.concat(rows); P.index.name = 'd'; P = P.reset_index(); P['d'] = P.d.astype('datetime64[ns]'); print('panel', P.shape, 'names', P.tk.nunique(), 'y rate', P.y.mean().round(4))
# macro (date level)
F = pd.read_pickle(X + 'fred.pkl'); F['date'] = pd.to_datetime(F.date).astype('datetime64[ns]'); F = F.set_index('date').sort_index().ffill()
M = pd.DataFrame(dict(VIX=F.VIX, dVIX20=F.VIX.pct_change(20), HY=F.HY_OAS, dHY20=F.HY_OAS.diff(20), USD20=F.USD.pct_change(20), T10Y2Y=F.T10Y2Y, NFCI=F.NFCI, DGS2d=F.DGS2.diff(60)))
MS = pd.read_pickle(X + 'market_state.pkl'); M = M.join(MS[['breadth', 'idx_reg', 'idx_dd']], how='outer').ffill()
# wave + market-level inputs
W = P.assign(up=(P.r20 > .3).astype(float)).groupby('d').agg(WAVE=('up', 'mean'), EWR=('ret', 'mean'), DISP=('ret', 'std'), N=('tk', 'size'))
W = W[W.N >= 30]; M = M.join(W[['WAVE', 'EWR', 'DISP']], how='outer')
M.index = pd.DatetimeIndex(M.index).astype('datetime64[ns]'); M.index.name = 'd'; P = P.merge(M.reset_index(), on='d', how='left'); P.to_pickle(SP + 'ext/runs515_panel.pkl'); M.to_pickle(SP + 'ext/runs515_macro.pkl'); print('saved')
