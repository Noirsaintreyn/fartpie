"""Report 539 (Nov 28 2026): Rey: "or candle speed at flip". DECLARED BEFORE RESULTS.
Speed definitions at the FLIP bar (known at its close): S_BODY = |close-open| / ATR14; S_RANGE = (high-low) / ATR14; S_BREAK = distance the flip closed beyond the vector level / ATR14 (NQ: R / ATR1H; R = |entry open - vector level|); S_IMP (NQ only) = largest 5m bar range inside the flip hour / mean 5m range of the same clock hour over the prior 20 sessions (impulse speed); SIGNED variants use the bar direction (flip direction only, long flips so close>open).
Hypothesis (mechanism: a fast, decisive displacement through the level = commitment; a slow grind through the level = a weak flip that fails): top third beats bottom third. Contrasts: TOP-BOT net pts (NQ 1H flips, 1:1, 1pt cost) and, for sub-$5 daily flips (flip_history, 1,504 sub-$5 flips with bars), reach (PEAK>=+10%) / mean r within ATRP terciles. TRAIN 2019-21 cuts | TEST 2018+2022+ (small caps 2022+). Pass: same sign both eras and t>=2 (NQ day-clustered; small caps Welch).
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vx_trades import vx
T = vx(); T['yr'] = T.day.dt.year; T['ts'] = T.day + pd.to_timedelta(T.m, unit='m')
b = pd.read_parquet('/home/claude/nqdrive/nq5_drive_et_ext.parquet'); b = b[b.dt >= '2017-06-01'].set_index('dt')
g = b.resample('60min', label='right', closed='left').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(); pc = g.close.shift(); tr_ = pd.concat([g.high - g.low, (g.high - pc).abs(), (g.low - pc).abs()], axis=1).max(axis=1); atr = tr_.rolling(14).mean()
g['S_BODY'] = (g.close - g.open).abs() / atr; g['S_RANGE'] = (g.high - g.low) / atr
r5 = (b.high - b.low); hr5 = r5.index.hour; mx = r5.resample('60min', label='right', closed='left').max(); mn = r5.groupby(hr5).transform(lambda s: s.rolling(20 * 12, min_periods=60).mean().shift(12)).resample('60min', label='right', closed='left').mean(); g['S_IMP'] = (mx / mn).reindex(g.index); g['ATR1H'] = atr
T = pd.merge_asof(T.sort_values('ts'), g[['S_BODY', 'S_RANGE', 'S_IMP', 'ATR1H']].reset_index().rename(columns={'dt': 'ts'}).sort_values('ts'), on='ts', direction='backward', tolerance=pd.Timedelta(minutes=5)); T['S_BREAK'] = T.R / T.ATR1H; T = T.dropna(subset=['S_BODY', 'S_BREAK']); T['neg'] = T.neg.astype(bool)
trm = T.yr.between(2019, 2021); tem = ~trm
def cell(S):
    if len(S) < 5: return f'n{len(S):4d} (few)'
    dm = S.groupby('day').pts.sum(); t = dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 2 and dm.std() > 0 else np.nan; return f'n{len(S):4d} {S.pts.mean():+6.1f} (t{t:+.1f})'
print(f'NQ 1H flips with speed: {len(T)}; net pts/trade TOP third | BOT third  (train | test), train cuts')
for f in ('S_BODY', 'S_RANGE', 'S_BREAK', 'S_IMP'):
    q = T.loc[trm, f].quantile([.33, .67])
    for name, U in (('ALL', T), ('NEG', T[T.neg])):
        top, bot = U[f] > q.iloc[1], U[f] <= q.iloc[0]; ut, ue = trm.loc[U.index], tem.loc[U.index]
        print(f'{f:8} {name:3} TOP {cell(U[top & ut])} | {cell(U[top & ue])}   ||  BOT {cell(U[bot & ut])} | {cell(U[bot & ue])}')
# small caps daily flips
H = pd.read_csv('flip_history.csv', parse_dates=['d']); H = H[H.px < 5].copy(); X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); rows = []
for tk, gH in H.groupby('tk'):
    q_ = D.get(tk)
    if q_ is None: continue
    q_ = q_.dropna(subset=['close']).copy(); q_.index = pd.DatetimeIndex(q_.index).astype('datetime64[ns]'); pcl = q_.close.shift(); a14 = pd.concat([q_.high - q_.low, (q_.high - pcl).abs(), (q_.low - pcl).abs()], axis=1).max(axis=1).rolling(14).mean()
    f = pd.DataFrame(dict(S_BODY=(q_.close - q_.open).abs() / a14, S_RANGE=(q_.high - q_.low) / a14, S_GAP=(q_.open / pcl - 1).abs() / (a14 / q_.close)))
    m = gH.merge(f, left_on='d', right_index=True, how='inner')
    rows.append(m)
S = pd.concat(rows); S['yr'] = S.d.dt.year; S['hit'] = S.hit.astype(float); trs = S.yr.between(2019, 2021); S['at'] = pd.qcut(S.ATRP.rank(method='first'), 3, labels=False)
print(f'\nSUB-$5 daily flips with bar features: {len(S)} (train {trs.sum()}, test {(~trs).sum()})')
for f in ('S_BODY', 'S_RANGE', 'S_GAP'):
    q = S.loc[trs, f].quantile([.33, .67]); S['T_' + f] = np.where(S[f] > q.iloc[1], 1, np.where(S[f] <= q.iloc[0], 0, np.nan)); o = f'{f:8}'
    for nm, mk in (('train', trs), ('test', ~trs)):
        d = S[mk & S['T_' + f].notna()]; a, c = d[d['T_' + f] == 1], d[d['T_' + f] == 0]
        wr = []
        for k in (0, 1, 2):
            x, y = a[a['at'] == k], c[c['at'] == k]
            if len(x) > 5 and len(y) > 5: wr.append((x.hit.mean() - y.hit.mean(), x.r.mean() - y.r.mean()))
        t = (a.r.mean() - c.r.mean()) / np.sqrt(a.r.var() / len(a) + c.r.var() / len(c)); o += f' | {nm}: reach TOP-BOT within ATRP terciles {np.mean([w[0] for w in wr])*100:+.1f}pp, mean r {np.mean([w[1] for w in wr])*100:+.1f}pp (raw t {t:+.1f}, n {len(a)}/{len(c)})'
    print(o)
