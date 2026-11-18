"""Nov 18 2026 report 294 (Rey: is the options-ratio edge speculation? + do we hold winners too long). DECLARED BEFORE RESULTS.
A) Candidate trades (wide_T: S&P PIT, ratio >= 1.5, sticky 1h stop). Regress trade return on log(ratio) alone, then with controls known the prior close: beta252 vs SPY, YZ20 vol, log 50d $ volume,
   20d return, close/252d high. Day-clustered SE. 2019-22 (train) vs 2023-26 (test). If the ratio coefficient survives -> it is not just vol/beta/liquidity/momentum.
B) GIVEBACK: trades entered after 2023-12-12 (hourly bars exist). Entry = open of the entry bar; baseline = the recorded return. MFE = max high between entry and exit / entry - 1. Giveback = MFE - ret.
   Declared ATR-scaled trail on top of the vector exit: exit at max(highest high since entry - k x ATR14(daily, prior close), no look-ahead within the bar: checked on bar lows) for k = 2, 3, 4; also a 'lock' of
   half the MFE once MFE >= 2 ATR. Compare mean return/trade and win rate vs baseline (same trades)."""
import numpy as np, pandas as pd, warnings, statsmodels.api as sm; warnings.filterwarnings('ignore')
T = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5) & (T.exit < pd.Timestamp('2026-09-26'))].copy()
T['day'] = T.entry.dt.normalize(); J = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); spy = J.SPY.copy(); spy.index = pd.DatetimeIndex(spy.index)
FE = []
for tk in sorted(T.tk.unique()):
    try: Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet')
    except Exception: continue
    Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); r = np.log(Y.close).diff(); sr = np.log1p(spy.reindex(Y.index).fillna(0))
    cov = r.rolling(252, min_periods=120).cov(sr); beta = cov / sr.rolling(252, min_periods=120).var()
    o, h, l, c = (np.log(Y[k]) for k in ('open', 'high', 'low', 'close')); n = 20; co = o - c.shift(1); oc = c - o; rs = (h - c) * (h - o) + (l - c) * (l - o); k_ = .34 / (1.34 + (n + 1) / (n - 1))
    yz = np.sqrt(co.rolling(n).var() + k_ * oc.rolling(n).var() + (1 - k_) * rs.rolling(n).mean())
    tr = pd.concat([Y.high - Y.low, (Y.high - Y.close.shift(1)).abs(), (Y.low - Y.close.shift(1)).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    f = pd.DataFrame({'beta': beta, 'yz': yz, 'ldv': np.log((Y.close * Y.volume).rolling(50).mean()), 'mom20': c.diff(20), 'hi52': Y.close / Y.close.rolling(252, min_periods=120).max(), 'atr': atr, 'atrp': atr / Y.close}).shift(1)
    f['tk'] = tk; f['day'] = f.index; FE.append(f)
FE = pd.concat(FE); T = T.merge(FE, on=['tk', 'day'], how='left'); T['lr'] = np.log(T.ratio)
cols = ['beta', 'yz', 'ldv', 'mom20', 'hi52']
def reg(S, xs):
    S = S.dropna(subset=xs + ['ret']); X = sm.add_constant((S[xs] - S[xs].mean()) / S[xs].std()); m = sm.OLS(S.ret.values, X.values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(S.day)[0]})
    return dict(zip(['const'] + xs, zip(m.params, m.tvalues))), len(S)
print('A) trade return (%, per 1 sd of each feature), cluster-t in brackets')
for lab, a, b in (('train 2019-22', '2019-01-01', '2023-01-01'), ('test 2023-26', '2023-01-01', '2027-01-01')):
    S = T[(T.day >= a) & (T.day < b)]
    for nm, xs in (('ratio alone', ['lr']), ('ratio + controls', ['lr'] + cols), ('controls only', cols)):
        r, n = reg(S, xs); print(f'  {lab:14s} {nm:17s} n={n:5d} ' + ' '.join(f'{k}:{v[0] * 100:+.2f}[{v[1]:+.1f}]' for k, v in r.items() if k != 'const'))
S = T.dropna(subset=cols); print('  corr(log ratio, controls) all:', {c: round(S.lr.corr(S[c]), 2) for c in cols})
# B) giveback
H = {}
def hourly(tk):
    if tk not in H:
        try: x = pd.read_parquet(f'data/yahoo/h1/{tk}.parquet'); x.index = pd.DatetimeIndex(x.index).tz_localize(None); H[tk] = x
        except Exception: H[tk] = None
    return H[tk]
rows = []
for r in T[T.entry >= '2023-12-13'].itertuples():
    x = hourly(r.tk)
    if x is None or not np.isfinite(r.atr): continue
    seg = x[(x.index >= r.entry.floor('h') - pd.Timedelta(minutes=30)) & (x.index <= r.exit)]
    seg = seg[seg.index >= r.entry - pd.Timedelta(minutes=30)]
    if len(seg) < 1: continue
    e = seg.open.iloc[0]; hh = seg.high.cummax(); mfe = hh.iloc[-1] / e - 1; res = dict(tk=r.tk, day=r.day, ret=r.ret, mfe=mfe, n=len(seg), atrp=r.atr / e)
    for k in (2, 3, 4):
        px = None
        for i in range(len(seg)):
            lvl = (hh.iloc[i - 1] if i > 0 else seg.open.iloc[0]) - k * r.atr          # trail from the highest high BEFORE this bar (no look-ahead)
            if seg.low.iloc[i] <= lvl and i > 0: px = max(min(lvl, seg.open.iloc[i]), 0); break
        res[f'trail{k}'] = (px / e - 1 - 2e-4) if px is not None else r.ret
    px = None
    for i in range(1, len(seg)):
        mf = hh.iloc[i - 1] / e - 1
        if mf >= 2 * r.atr / e and seg.low.iloc[i] <= e * (1 + mf / 2): px = min(seg.open.iloc[i], e * (1 + mf / 2)); break
    res['lock50'] = (px / e - 1 - 2e-4) if px is not None else r.ret; rows.append(res)
G = pd.DataFrame(rows); print(f'\nB) giveback, {len(G)} trades entered Dec 2023+ with hourly bars')
print(f'  baseline: mean {G.ret.mean():+.2%} win {np.mean(G.ret > 0):.0%} | mean MFE {G.mfe.mean():+.2%} | mean giveback (MFE - ret) {(G.mfe - G.ret).mean():.2%} | winners giveback {(G.mfe - G.ret)[G.ret > 0].mean():.2%}')
for v in ('trail2', 'trail3', 'trail4', 'lock50'): print(f'  {v:8s}: mean {G[v].mean():+.2%} win {np.mean(G[v] > 0):.0%} | changed {np.mean(np.abs(G[v] - G.ret) > 1e-9):.0%} of trades | winners-only mean {G[v][G.ret > 0].mean():+.2%} vs {G.ret[G.ret > 0].mean():+.2%}')
G['yr'] = G.day.dt.year; print(G.groupby('yr')[['ret', 'trail3', 'lock50']].mean().round(4).to_string())
