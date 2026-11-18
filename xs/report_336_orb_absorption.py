"""Nov 18 2026 report 336. ABSORPTION / EFFORT-vs-RESULT at the first 5m candle (Rey: 'surprised absorption does not add anything').  Report 334 only tested first-candle aggressor-flow ALIGNMENT on 214 MNQ trades (2024-12 ->); that is not absorption, and the 255/264 absorption flags were tested as stand-alone REVERSALS.  Here: full-history (2014-2026) price-volume proxy on the ORB's own first candle.  Declared before running.
 rv0 = first-candle volume / median of the prior 20 first-candle volumes (effort); body = |close-open|/range (result); ABS = rv0 >= 1.5 and body <= .30 (lots of volume, little progress = aggression absorbed by passive size).
THEORY: absorbed aggression leaves the absorber positioned at the extreme (stop-run fuel); the break that follows is the stored energy releasing -> HIGHER pts/trade after an ABS candle.  Alternatively high effort WITH result (rv0 high, body large) = initiative flow that continues (also higher).  Both fixed; opposite = low volume candle breaks are weaker.  Same ORB engine, pts hold-to-15:55 (orb335 parquet), train 2019-21 | test 2018+2022-26.
Why it would fail: the first candle after the open is always high effort (auction opens), 5m volume cannot separate aggressor from passive, and absorption is a footprint concept (needs bid/ask volume at price) that OHLCV cannot see."""
import numpy as np, pandas as pd
T = pd.read_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/orb335.parquet')
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m == 570)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); F = N.set_index('day').sort_index()
F['rv0'] = F.volume / F.volume.rolling(20).median().shift(1); F['body'] = (F.close - F.open).abs() / (F.high - F.low).replace(0, np.nan); T = T.join(F[['rv0', 'body']]).dropna(subset=['rv0', 'body'])
T['ABS'] = (T.rv0 >= 1.5) & (T.body <= .3); T['INIT'] = (T.rv0 >= 1.5) & (T.body >= .6)
tr = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); te = (T.index >= '2018-01-01') & ~tr
def f(m, c='hold'):
    s = T.loc[m, c].dropna(); return '   n/a' if len(s) < 20 else f'{s.mean():+6.2f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f},n{len(s):4d})'
print(f'trades {len(T)}; share ABS {T.ABS.mean():.0%}, INIT {T.INIT.mean():.0%}; base pts: train {f(tr)} | test {f(te)}')
for lab, c in (('ABS (high effort, small body)', T.ABS), ('INIT (high effort, big body)', T.INIT), ('neither', ~T.ABS & ~T.INIT)):
    print(f'  {lab:30s} pts train {f(tr & c)} | test {f(te & c)}      R train {f(tr & c, "R")} | test {f(te & c, "R")}')
q = T.loc[tr, 'rv0'].quantile([1 / 3, 2 / 3]).values
for lab, c in (('rv0 low', T.rv0 <= q[0]), ('rv0 mid', (T.rv0 > q[0]) & (T.rv0 < q[1])), ('rv0 high', T.rv0 >= q[1])): print(f'  {lab:30s} pts train {f(tr & c)} | test {f(te & c)}')
for lab, g in (('GAP-aligned', T.gal),):
    for l2, c in (('ABS', T.ABS), ('INIT', T.INIT), ('rv0 high', T.rv0 >= q[1])): print(f'  {lab} & {l2:9s} pts train {f(tr & g & c)} | test {f(te & g & c)}')
