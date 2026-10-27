"""Oct 27 2026: scale the whole book up / down by context (all known at the prior close), applied to the daily marked-to-market book.
Contexts: SPY trend efficiency (20d net move / sum of |daily moves|, low = chop), SPY daily-vector flips in 20 days (chop), SPY 20d realized
vol, VIX level, VIX/VIX3M, SPY downtrend (vector bear AND under 200d), SPY vs 200d, COT asset managers, breadth, the book's own last-20-day
return, the book's drawdown from its high, the book's 20-day realized vol.
For each: thirds cut on 2019-21; the third with the best / worst NEXT-20-day book return in 2019-21 gets 1.5x / 0.5x, the middle 1x
(direction fixed on 2019-21). Then a combined vote of every context that was consistent. Judged on 2022-26 (unseen) by return / max DD."""
import numpy as np, pandas as pd, warnings, sys, os; warnings.filterwarnings('ignore')
ms = open('xs/market_signal.py').read(); exec(ms[:ms.index("PER = [('2016-06-01'")])
S = OUT['SPY'].copy()
B = pd.read_parquet('xs/ext/vt/series.parquet')['tm:vec']; B = B[B.index >= '2018-06-01']
spy = etf('SPY').close; r = spy.pct_change()
C = pd.DataFrame(index=B.index)
C['eff'] = ((spy - spy.shift(20)).abs() / r.abs().mul(spy.shift(1)).rolling(20).sum()).shift(1).reindex(B.index, method='ffill')
fl = (S.vec_d != S.vec_d.shift(1)).astype(int).rolling(20).sum(); C['flips'] = fl.reindex(B.index, method='ffill')    # S is already prior-close
C['rv20'] = (r.rolling(20).std() * np.sqrt(252)).shift(1).reindex(B.index, method='ffill')
P = pd.read_parquet('/tmp/claude-0/ml/macro/px.parquet'); P.index = pd.to_datetime(P.index).tz_localize(None)
C['vix'] = P['^VIX'].shift(1).reindex(B.index, method='ffill'); C['vixts'] = (P['^VIX'] / P['^VIX3M']).shift(1).reindex(B.index, method='ffill')
C['down'] = ((S.vec_d < 0) & (S.sma200 < 0)).astype(float).reindex(B.index, method='ffill')
C['spy200'] = (spy / spy.rolling(200).mean() - 1).shift(1).reindex(B.index, method='ffill')
C['cot'] = S.cot.reindex(B.index, method='ffill'); C['breadth'] = S.breadth.reindex(B.index, method='ffill')
eq = (1 + B).cumprod(); C['bk20'] = (eq / eq.shift(20) - 1).shift(1); C['bkdd'] = (eq / eq.cummax() - 1).shift(1); C['bkvol'] = B.rolling(20).std().shift(1)
fwd = (eq.shift(-20) / eq - 1)                                                         # next-20-day book return (for choosing direction on 2019-21 only)
tr = (C.index >= '2019-01-01') & (C.index < '2022-01-01')
def stats(x):
    out = []
    for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01')):
        y = x[a:b]; e = (1 + y).cumprod(); yr = len(y) / 252; c = e.iloc[-1] ** (1 / yr) - 1; d = (e / e.cummax() - 1).min(); out.append((c, d))
    return out
def show(nm, w):
    R = stats(B * w)
    print(f'  {nm:<44}' + ''.join(f'{c:>+8.1%} /{d:>5.0%}' for c, d in R) + f'   ret/DD {R[0][0] / -R[0][1]:.1f} {R[1][0] / -R[1][1]:.1f}   avg exposure {w["2022":].mean():.2f}', flush=True)
print(f'{"":<46}{"2019-21":>15}{"2022-26":>15}{"2026 YTD":>15}'); show('live (1x always)', pd.Series(1.0, index=B.index))
W = {}
for c in C.columns:
    x = C[c]
    if x.nunique() <= 2: lo = hi = .5; q = x
    else: lo, hi = x[tr].quantile([1 / 3, 2 / 3])
    grp = np.where(x <= lo, 0, np.where(x > hi, 2, 1)) if x.nunique() > 2 else x.fillna(0).astype(int).values
    g = pd.Series(grp, index=x.index); m = fwd[tr].groupby(g[tr]).mean()
    if len(m) < 2: continue
    best, worst = m.idxmax(), m.idxmin(); w = pd.Series(1.0, index=x.index); w[g == best] = 1.5; w[g == worst] = .5; w[x.isna()] = 1.0
    W[c] = w; show(f'{c}: 1.5x in its best third, 0.5x in its worst', w)
V = pd.concat(W, axis=1); show('all contexts averaged (equal weight)', V.mean(axis=1))
show('chop only: eff + flips averaged', (W['eff'] + W['flips']) / 2)
show('downtrend 1.5x + chop 0.5x (eff low third)', pd.Series(np.where(C.down > 0, 1.5, np.where(C.eff <= C.eff[tr].quantile(1 / 3), .5, 1.0)), index=B.index))
