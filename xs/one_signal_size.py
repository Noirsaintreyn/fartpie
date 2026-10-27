"""Oct 27 2026: one sizing signal per trade from three contexts known at entry (prior close): SPY downtrend (daily vector bear AND under its
200-day), the macro score (asset managers' COT, VIX/VIX3M, SPY vs 200d; xs/edge_combo.py), and the book's own 20-day volatility.
Directions and thirds fixed on 2019-21 (next-20-day book return for book vol; admitted-trade average for macro), judged on 2022-26.
size = 1 + 0.5 x downtrend + 0.25 x macro (+1 top third / -1 bottom) + 0.25 x book-vol (+1 best third / -1 worst), clipped 0.5..2."""
import numpy as np, pandas as pd
ms = open('xs/market_signal.py').read(); exec(ms[:ms.index("PER = [('2016-06-01'")]); MK = OUT
exec(open('xs/edge_combo.py').read().split("R0 = L[L.inbook")[0])
R0 = L[L.inbook & (L.half == '19-21')]
S = MK['SPY']; down = ((S.vec_d < 0) & (S.sma200 < 0)).astype(float).reindex(L.d.values, method='ffill').values
m1, m2 = R0.macro.quantile([1 / 3, 2 / 3]); msgn = np.where(L.macro > m2, 1, np.where(L.macro < m1, -1, 0))
Bk = pd.read_parquet('xs/ext/vt/series.parquet')['tm:vec']; eqb = (1 + Bk).cumprod(); bv = Bk.rolling(20).std().shift(1); fwd = eqb.shift(-20) / eqb - 1
tr = (bv.index >= '2019-01-01') & (bv.index < '2022-01-01'); v1, v2 = bv[tr].quantile([1 / 3, 2 / 3])
g = pd.Series(np.where(bv <= v1, 0, np.where(bv > v2, 2, 1)), index=bv.index); mm = fwd[tr].groupby(g[tr]).mean(); best, worst = mm.idxmax(), mm.idxmin()
print('book-vol thirds (low/mid/high) next-20d book return 2019-21:', (mm * 100).round(2).to_dict(), '-> best', best, 'worst', worst)
gl = g.reindex(L.d.values, method='ffill').values; vsgn = np.where(gl == best, 1, np.where(gl == worst, -1, 0))
def show2(nm, w):
    z = L.copy(); z['cap'] = 6; z['wn'] = 6; z['w'] = w; R_ = [sim_cw(z, a, b) for a, b in PER]
    print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + '  ret/DD ' + ' '.join(f'{c / -d:.1f}' for c, d in R_[:2]) + f'  avg size {np.mean(w):.2f}', flush=True)
print(f'{"":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
one = np.ones(len(L)); show2('live (1x)', one)
show2('downtrend 1.5x only', 1 + .5 * down)
show2('macro +-0.25 only', 1 + .25 * msgn)
show2('book vol +-0.25 only', 1 + .25 * vsgn)
show2('downtrend + macro', np.clip(1 + .5 * down + .25 * msgn, .5, 2))
show2('downtrend + book vol', np.clip(1 + .5 * down + .25 * vsgn, .5, 2))
show2('macro + book vol', np.clip(1 + .25 * msgn + .25 * vsgn, .5, 2))
show2('ALL THREE (one signal)', np.clip(1 + .5 * down + .25 * msgn + .25 * vsgn, .5, 2))
show2('ALL THREE, half strength', np.clip(1 + .25 * down + .125 * msgn + .125 * vsgn, .5, 2))
show2('control: always 1.25x', one * 1.25)
