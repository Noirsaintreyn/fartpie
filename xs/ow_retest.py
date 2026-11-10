"""Nov 10 2026: RE-TEST OF THE OVERNIGHT-RANGE (WIDE) FLAG ON THE CORRECTED ET CLOCK. DECLARED BEFORE RESULTS.
overnight range of day d = high - low over the non-RTH bars after the prior day's 15:55 until 09:25 ET (nq5_real.parquet, rth == False, same contract). WIDE = range > 2/3 quantile of the previous 20 days' overnight ranges (the old rule).
Per engine (RA, M1, R1, RULES, V15a; trades with sz > 0 from the S6 V2 table) and for the noise engine: points per trade (per day for NZ) WIDE vs not; PASS = same sign, |t| >= 2, both periods (2014-21 | 2022-26; NZ 2018-21 | 2022-26 not needed - uses 2014+)."""
import sys, numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X['dt'] = pd.to_datetime(X.dt)
ON = X[~X.rth].copy(); ON['day'] = ON.dt.dt.normalize(); ON.loc[ON.dt.dt.hour >= 16, 'day'] += pd.Timedelta(days=1)       # evening session belongs to the next trading day
ON = ON[ON.dt.dt.hour * 60 + ON.dt.dt.minute < 565].copy() if True else ON
g = ON.groupby('day'); ONR = (g.high.max() - g.low.min()); WIDE = ONR > ONR.shift(1).rolling(20).quantile(2 / 3)
print(f'overnight days {len(ONR)}, wide share {WIDE.mean():.0%}')
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb'))
print('\n=== per-trade points (V2 live table), WIDE vs not; 2014-21 | 2022-26 ===')
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n in ('ON+', 'ON-'): continue
    t = t[t.sz > 0].copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); t = t[t.day <= '2026-01-30']; w = WIDE.reindex(t.day).values; t = t[~pd.isna(w)]; w = WIDE.reindex(t.day).values.astype(bool); res = []
    for f in (t.day < '2022-01-01', t.day >= '2022-01-01'):
        a, b = t.pts[f & w], t.pts[f & ~w]; res.append((a.mean(), b.mean(), tt(a, b), int((f & w).sum()), int((f & ~w).sum())))
    ok = all(abs(r[2]) >= 2 for r in res) and np.sign(res[0][0] - res[0][1]) == np.sign(res[1][0] - res[1][1])
    print(f'  {n:<6}' + ' | '.join(f'WIDE {r[0]:+6.1f} (n {r[3]}) vs not {r[1]:+6.1f} (n {r[4]}) t {r[2]:+5.2f}' for r in res) + ('   <== PASS' if ok else ''))
import nz_core as Z
rows = [r for d in Z.DAYS for r in Z.noise(d)]; T = pd.DataFrame(rows, columns=['day', 'j', 'dir', 'pts', 'mae']); s = T.groupby('day').pts.sum().reindex(Z.DAYS).fillna(0.0); s.index = pd.to_datetime(s.index)
w = WIDE.reindex(s.index); ok = w.notna(); print('\n=== noise engine pts / DAY, WIDE vs not ===')
res = []
for f in (s.index < '2022-01-01', s.index >= '2022-01-01'):
    a, b = s[ok & f & (w == True)], s[ok & f & (w == False)]; res.append((a.mean(), b.mean(), tt(a, b), len(a), len(b)))
print('  NZ    ' + ' | '.join(f'WIDE {r[0]:+6.2f} (n {r[3]}) vs not {r[1]:+6.2f} (n {r[4]}) t {r[2]:+5.2f}' for r in res) + ('   <== PASS' if all(abs(r[2]) >= 2 for r in res) and np.sign(res[0][0] - res[0][1]) == np.sign(res[1][0] - res[1][1]) else ''))
