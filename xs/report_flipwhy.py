"""WHY DOES THE FLIP-ONLY V15a WIN MORE AND LOSE MORE? NQ 15m RTH vector flips, pure vector exit (exit at the opposite flip next open, else the close), 2014-2026. Nov 11 2026.
(1) recoverability: of trades whose adverse excursion reaches k*ADR, how many still win?  (2) who are the big losers (< -0.35 ADR) vs the rest, by entry-time features: flip number of day, previous trade result, time of day, entry-bar size, distance from VWAP / day open, gap alignment, hold time.
Terciles cut on 2014-21; PASS = same sign & |t|>=2 in 2014-21 and 2022-26 on points per trade."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']; NSL = {}
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0], NSL); sys.argv = _a; vec = NSL['vec']
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X = X[X.rth].copy(); X['day'] = pd.to_datetime(X.day).dt.normalize()
X['tp'] = (X.high + X.low + X.close) / 3; X['pv'] = X.tp * X.volume; X['vw'] = X.groupby('day').pv.cumsum() / X.groupby('day').volume.cumsum().replace(0, np.nan)
X['b'] = X.m // 15 * 15
B = X.groupby(['day', 'b']).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), vw=('vw', 'last')).reset_index().sort_values(['day', 'b']).reset_index(drop=True)
B['reg'] = np.asarray(vec(B[['open', 'high', 'low', 'close']])); rng = B.groupby('day').apply(lambda g: g.high.max() - g.low.min()); adr = rng.shift(1).rolling(20).mean()
dopen = B.groupby('day').open.first(); dclose = B.groupby('day').close.last(); gap = (dopen - dclose.shift(1))
o, h, l, c, vw = B.open.values, B.high.values, B.low.values, B.close.values, B.vw.values; dy = B.day.values; reg = B.reg.values; bm = B.b.values
rows = []; i = 1; seq = {}; prev = {}
while i < len(B) - 1:
    if dy[i] != dy[i - 1] or reg[i] == reg[i - 1] or reg[i] == 0 or not (585 <= bm[i] + 15 <= 930) or dy[i + 1] != dy[i]: i += 1; continue
    d = pd.Timestamp(dy[i]); a = adr.get(d, np.nan)
    if not np.isfinite(a): i += 1; continue
    dr = int(reg[i]); e = o[i + 1]; j = i + 1; brk = False
    while j < len(B) and dy[j] == dy[i]:
        if j + 1 < len(B) and dy[j + 1] == dy[i] and reg[j] == -dr and reg[j - 1] != reg[j]: brk = True; break
        j += 1
    ex = o[j + 1] if brk else c[j - 1]; kk = j if brk else j - 1
    mae = min(((l[q] - e) if dr > 0 else (e - h[q])) for q in range(i + 1, kk + 1)); mfe = max(((h[q] - e) if dr > 0 else (e - l[q])) for q in range(i + 1, kk + 1))
    n = seq.get(d, 0) + 1; seq[d] = n
    rows.append(dict(day=d, m=int(bm[i] + 15), dr=dr, pts=dr * (ex - e) - .5, mae=min(mae, 0), mfe=max(mfe, 0), bars=kk - i, adr=a, seq=n, prev=prev.get(d, 0),
                     barsz=(h[i] - l[i]) / a, vwd=dr * (e - vw[i]) / a, dopen=dr * (e - dopen[d]) / a, gapal=dr * gap.get(d, np.nan) / a))
    prev[d] = np.sign(rows[-1]['pts']); i = max(j, i + 1)
R = pd.DataFrame(rows); R['maeA'] = R.mae / R.adr; R['ptsA'] = R.pts / R.adr; R['era'] = np.where(R.day < '2022-01-01', 'A', 'B')
def tstat(x): x = np.asarray(x, float); x = x[~np.isnan(x)]; return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
print(f'trades {len(R)} | win {np.mean(R.pts>0):.0%} | mean {R.pts.mean():+.1f} pts | median ADR {R.adr.median():.0f} pts | median hold {R.bars.median():.0f} bars (15m)')
print('\n== (1) RECOVERABILITY: trades whose adverse excursion reached k*ADR')
for k in (.05, .1, .15, .25, .35, .5):
    s = R[R.maeA <= -k]; r = R[R.maeA > -k]
    print(f' MAE reached -{k:.2f} ADR: n {len(s):4d} ({len(s)/len(R):.0%}) win {np.mean(s.pts>0):3.0%} mean final {s.pts.mean():+7.1f} pts | never reached: n {len(r):4d} win {np.mean(r.pts>0):3.0%} mean {r.pts.mean():+6.1f}')
print('\n== winners vs losers: path shape (median)')
for nm, s in (('winners', R[R.pts > 0]), ('losers', R[R.pts <= 0])): print(f' {nm:8} n {len(s):4d} mean {s.pts.mean():+7.1f} | median MAE {s.mae.median():+6.1f} ({s.maeA.median():+.2f} ADR) | median MFE {s.mfe.median():+6.1f} | median hold {s.bars.median():.0f} bars')
big = R.ptsA <= -.35; print(f'\nBIG LOSERS (<= -0.35 ADR): {big.sum()} trades ({big.mean():.1%}) account for {R[big].pts.sum():+.0f} of the total {R.pts.sum():+.0f} pts; non-big mean {R[~big].pts.mean():+.1f}')
print('\n== (2) FEATURE TERCILES (cuts 2014-21): n, win, mean pts, big-loser rate | era A mean (t) | era B mean (t)')
good = []
for f in ('seq', 'prev', 'm', 'barsz', 'vwd', 'dopen', 'gapal'):
    col = R[f]
    if f in ('seq', 'prev'): groups = {str(v): col == v for v in sorted(col.unique()) if (col == v).sum() > 60 and v < 5}
    else: q = R[R.era == 'A'][f].quantile([1 / 3, 2 / 3]).values; groups = {'low': col <= q[0], 'mid': (col > q[0]) & (col <= q[1]), 'high': col > q[1]}
    for nm, sel in groups.items():
        s = R[sel]; a = s[s.era == 'A'].pts; b = s[s.era == 'B'].pts
        ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
        print(f' {f:6} {nm:5} n {len(s):4d} win {np.mean(s.pts>0):3.0%} mean {s.pts.mean():+7.1f} bigloss {np.mean(s.ptsA<=-.35):4.1%} | A {a.mean():+6.1f} ({tstat(a):+.1f}) | B {b.mean():+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
        if ok: good.append((f, nm))
print('\nPASS list:', good); R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/flipwhy.pkl')
