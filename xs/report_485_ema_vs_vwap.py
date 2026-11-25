"""Report 485 (Nov 25 2026): Rey - 'we probably don't need VWAP on the chart, just the EMAs with the best backtest, unless a VWAP (different timeframe) backtested better.' Daily flips (structure_small), base5 top-30% group (train ECDF/cut per price bucket), px<20/<10/<5. Test 2022+, selection on TRAIN 2019-21 only.
DECLARED BEFORE RESULTS. Hypothesis: a flip whose close is already ABOVE a slower average has support under it, so it holds up better (the live-alert reading Rey wants from the chart). Candidates, all known at the flip close: close > EMA(n) for n in 8, 13, 21, 50, 100, 200 (daily) and close > rolling VWAP(n) = sum(typical*vol)/sum(vol) over the last n sessions for n in 5, 10, 20, 50, 100 (the only VWAPs computable from daily bars; session VWAP resets every day and carries no information at a daily close). Also the NEAR variant (close within 1 ATR of it, above or below) for the best of each family.
Selection on train: for each candidate, mean(r | above) - mean(r | below) and REACH difference; the best EMA and best VWAP = largest train mean gap with >= 40 flips on each side and the same sign of REACH gap. Test: report both sides, era split 2022 / 2023-26, month-clustered t of the above-minus-below gap. A candidate PASSES if train-chosen and test gap > 0 in BOTH eras with |t| >= 2."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O['tr'] = O.d.dt.year <= 2021; W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl')
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
EM = (8, 13, 21, 50, 100, 200); VW = (5, 10, 20, 50, 100); cols = [f'E{n}' for n in EM] + [f'V{n}' for n in VW]
X = {c: np.full(len(O), np.nan) for c in cols}; XA = np.full(len(O), np.nan)
for tk, g in O.groupby('tk'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index; c = df.close.values; tp = (df.high.values + df.low.values + c) / 3; v = df.volume.values.astype(float); pv = pd.Series(tp * v)
    E = {n: pd.Series(c).ewm(span=n, adjust=False).mean().values for n in EM}; V = {n: (pv.rolling(n).sum() / pd.Series(v).rolling(n).sum()).values for n in VW}
    for i, d in zip(g.index, g.d):
        j = ix.searchsorted(pd.Timestamp(d))
        if j < 200 or j >= len(ix) or ix[j] != pd.Timestamp(d): continue
        for n in EM: X[f'E{n}'][i] = c[j] / E[n][j] - 1
        for n in VW: X[f'V{n}'][i] = c[j] / V[n][j] - 1
        XA[i] = g.ATRP[i] if 'ATRP' in g else np.nan
for k, x in X.items(): O[k] = x
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+6.1%} med {x.r.median():+.1%}'
def tmo(a, b):
    ma = a.groupby('mo').r.mean(); mb = b.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b; return d.mean(), (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 else np.nan), len(d)
for cap in (20, 10, 5):
    S = O[(O.px < cap)].dropna(subset=['r', 'PEAK'] + cols).copy(); S['s_'] = score(S); S['mo'] = S.d.dt.to_period('M').astype(str); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut]; trn = G[G.tr]; tst = G[~G.tr]
    print(f'\n######## sub-${cap} base5 group: train n {len(trn)} | test {st(tst)}'); rows = []
    for c_ in cols:
        a = trn[trn[c_] > 0]; b = trn[trn[c_] <= 0]; rows.append((c_, len(a), len(b), a.r.mean() - b.r.mean() if len(a) and len(b) else np.nan, (a.PEAK >= .1).mean() - (b.PEAK >= .1).mean() if len(a) and len(b) else np.nan))
    T = pd.DataFrame(rows, columns=['c', 'na', 'nb', 'gap', 'rgap']); print('  TRAIN gaps (above - below): ' + ' | '.join(f"{r.c} {r.gap:+.1%}/{r.rgap:+.0%} (n {r.na}/{r.nb})" for r in T.itertuples()))
    for fam in ('E', 'V'):
        t = T[T.c.str.startswith(fam) & (T.na >= 40) & (T.nb >= 40) & (T.gap > 0) & (T.rgap > 0)].sort_values('gap', ascending=False)
        if not len(t): print(f'  best {fam}: none with a positive train gap on mean and REACH'); continue
        c_ = t.c.iloc[0]; a = tst[tst[c_] > 0]; b = tst[tst[c_] <= 0]; g, tt, nm = tmo(a, b); e22 = (a[a.d.dt.year == 2022].r.mean(), b[b.d.dt.year == 2022].r.mean()); e23 = (a[a.d.dt.year >= 2023].r.mean(), b[b.d.dt.year >= 2023].r.mean())
        print(f'  best {fam} (train) = {c_}: TEST above {st(a)} | below {st(b)} | gap {g:+.1%} t {tt:+.2f} ({nm} months) | 2022 above {e22[0]:+.1%} below {e22[1]:+.1%} | 2023-26 above {e23[0]:+.1%} below {e23[1]:+.1%}')
print('\n######## POST-HOC (direction NOT declared; train showed negative gaps for the slow averages in all 3 buckets): TEST above vs below the slow EMAs / VWAPs')
for cap in (20, 10, 5):
    S = O[(O.px < cap)].dropna(subset=['r', 'PEAK'] + cols).copy(); S['s_'] = score(S); S['mo'] = S.d.dt.to_period('M').astype(str); cut = np.quantile(S[S.tr].s_.dropna(), .7); tst = S[(S.s_ >= cut) & ~S.tr]
    for c_ in ('E50', 'E100', 'E200', 'V100'):
        a = tst[tst[c_] > 0]; b = tst[tst[c_] <= 0]; g, tt, nm = tmo(a, b)
        print(f'  sub-${cap} {c_}: above {st(a)} | below {st(b)} | gap {g:+.1%} t {tt:+.2f} | 2022 {a[a.d.dt.year == 2022].r.mean():+.1%}/{b[b.d.dt.year == 2022].r.mean():+.1%} | 2023-26 {a[a.d.dt.year >= 2023].r.mean():+.1%}/{b[b.d.dt.year >= 2023].r.mean():+.1%}')
