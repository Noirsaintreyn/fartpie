import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
X = pd.read_parquet('xs/ext/vt/series.parquet'); live = X['tm:vec']
S2 = pd.read_parquet('xs/ext/s2/daily.parquet')
PER = [('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01')]
def st(r, a='2019-01-01', b='2027-01-01'):
    r = r[(r.index >= a) & (r.index < b)].dropna(); eq = (1 + r).cumprod(); y = len(r) / 252
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252), r.min()
def line(nm, r): print(f'{nm:<46}' + ''.join(f'{c:>+8.1%} /{d:>5.0%}' for c, d, *_ in (st(r, a, b) for a, b in PER)) + '   Sharpe %.2f  worst day %+.1f%%' % (st(r)[2], st(r)[3] * 100), flush=True)
print(f'{"":<46}{"2019-21":>15}{"2022-26":>15}{"2026 YTD":>15}')
line('live book (MTM)', live)
print('\n== 1. volatility targeting the whole book')
v = live[live.index >= '2019-01-01'].std() * np.sqrt(252)
for win in (10, 20):
    for cap in (1.0, 1.5):
        for tgt in (v, .8 * v):
            sig = live.rolling(win, min_periods=5).std().shift(1) * np.sqrt(252); w = (tgt / sig).clip(upper=cap).fillna(1)
            line(f'  {win}d vol, target {tgt:.0%}, max {cap}x (avg {w.mean():.2f}x)', live * w)
ew = (live.ewm(halflife=10).std().shift(1) * np.sqrt(252)); w = (v / ew).clip(upper=1.0).fillna(1); line(f'  EWMA hl10, target {v:.0%}, max 1x (avg {w.mean():.2f}x)', live * w)
w = pd.Series(np.where(live.shift(1) < -.02, .5, 1.0), index=live.index); line('  half size the day after a -2% day', live * w)
print('\n== 2. deflated Sharpe ratio')
r = live[live.index >= '2019-01-01']; T = len(r); sr = r.mean() / r.std(); g3, g4 = skew(r), kurtosis(r, fisher=False)
trials = X[X.index >= '2019-01-01']; srs = trials.mean() / trials.std(); N_obs = trials.shape[1]
for N in (N_obs, 100, 150):
    vs = srs.var(); emax = np.sqrt(vs) * ((1 - 0.5772) * norm.ppf(1 - 1 / N) + 0.5772 * norm.ppf(1 - 1 / (N * np.e)))
    dsr = norm.cdf((sr - emax) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    print(f'  trials {N:>4}: SR {sr*np.sqrt(252):.2f}/yr, expected best of {N} no-skill trials {emax*np.sqrt(252):.2f}/yr -> DSR {dsr:.3f}')
print(f'  ({N_obs} variant books; skew {g3:.2f}, kurtosis {g4:.1f}; variant SR median {srs.median()*np.sqrt(252):.2f}, max {srs.max()*np.sqrt(252):.2f})')
print(f'  SR 2019-21 {st(live,"2019-01-01","2022-01-01")[2]:.2f}, 2022-26 {st(live,"2022-01-01","2027-01-01")[2]:.2f}')
print('\n== 3. second strategies')
J = S2.reindex(live.index)
for c in S2.columns:
    ok = J[c].notna(); cor = live[ok].corr(J[c][ok]); line(f'  {c} (corr {cor:+.2f})', J[c].dropna())
for c in ('REV5', 'REV5v', 'REV5r', 'SKEW_ls', 'PCR_ls', 'IVCH_ls'):
    line(f'  live + 30% {c}', live + .3 * J[c].fillna(0))
