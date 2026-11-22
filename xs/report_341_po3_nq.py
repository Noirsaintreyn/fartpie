"""Nov 22 2026 report 341. AMD Po3 (WillyAlgoTrader v2.4 compression -> sweep -> return -> reference trade) ported to NQ 5m and tested, with Rey's critique tested: the compression test (Donchian-20 width percentile) cannot tell a CONSOLIDATION from a SLOW TREND (20 bars drifting 15 pts has a small width).  Declared before running.
WARM-UP FIX (run 2): percentile history crosses session breaks, window needs 21 bars inside the segment.  Port (absolute-boundary mode, indicator defaults): range = Donchian(20) when width <= 25th percentile of the last 200 bars and >= 0.15% of price; range matures >= 12 bars, expires > 96; breach = high/low beyond boundary + 10% width tolerance; both sides = reset; sweep must close back inside within 6 bars; trade opens at the D bar: entry = NEXT bar open (indicator uses D close; we use executable price), stop = sweep extreme -/+ 0.4 x ATR14 (at range start), target = fib 1.5 extension of the sweep leg (rangeHigh + 0.5 x leg for longs), timeout 64 bars, stop-first when both touch. Cost 0.75 pt.  FSM resets at data gaps > 30 min / contract change.  Direction: sweep of the low -> long, sweep of the high -> short.
Hypothesis (Rey): the edge, if any, is in TRUE consolidations.  Slow-trend measures on the accumulation window, fixed before running: ER = |net close change| / sum |close changes| (Kaufman efficiency, window bars), DRIFT = |net| / width.  Low ER / low DRIFT = consolidation.  Terciles cut on the TRAIN set (2019-21); test = 2018 + 2022-26.  Also: breakdown by 90-min session quarter (Quarterly Theory idea: Q1-Q4 of the 90m cycle, 18:00 ET + k x 90 min; data clock is ET).
Why it should fail: a mean-reversion-after-sweep trade into a trend-day engine market; compression + sweep is a very common pattern with no stated edge; the stop is wide relative to the target so a win rate near 40% is needed."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); import sys; RTHONLY = len(sys.argv) > 1 and sys.argv[1] == 'rth'
N = N[N.dt >= '2018-01-01']; N = (N[N.rth] if RTHONLY else N).reset_index(drop=True); print('RTH-only' if RTHONLY else '24h')
dt = pd.to_datetime(N.dt).values; o, h, l, c = (N[k].values.astype(float) for k in ('open', 'high', 'low', 'close')); sym = N.sym.values; n = len(N)
brk = np.zeros(n, bool); brk[1:] = ((dt[1:] - dt[:-1]) > np.timedelta64(30, 'm')) | (sym[1:] != sym[:-1])
hi20 = pd.Series(h).rolling(20).max().values; lo20 = pd.Series(l).rolling(20).min().values; chW = hi20 - lo20
tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]]))); atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
def prank(a, w=200):
    s = pd.Series(a); return s.rolling(w + 1).apply(lambda x: (x[:-1] <= x[-1]).mean() * 100, raw=True).values
pr = prank(chW)
seg = np.cumsum(brk); segpos = np.zeros(n, int)
for i in range(1, n): segpos[i] = 0 if brk[i] else segpos[i - 1] + 1
ER = np.full(n, np.nan); DR = np.full(n, np.nan)
dc = np.abs(np.diff(c, prepend=c[0])); cs = pd.Series(dc).rolling(19).sum().values  # path of last 20 closes (19 steps)
def feat(i0, i1):  # window bars i0..i1
    path = np.abs(np.diff(c[i0:i1 + 1])).sum(); net = abs(c[i1] - c[i0]); w = h[i0:i1 + 1].max() - l[i0:i1 + 1].min(); return (net / path if path > 0 else np.nan), (net / w if w > 0 else np.nan)
rows = []; state = 0; lastEnd = -100; cd = 10
i = 0
while i < n - 70:
    if brk[i]: state = 0
    if state == 0:
        if i >= 230 and segpos[i] >= 21 and pr[i] <= 25 and chW[i] >= .0015 * c[i] and i - lastEnd >= cd:
            s0 = i - 19; rH, rL = hi20[i], lo20[i]
            if c[i] <= rH and c[i] >= rL:
                w = rH - rL; tol = .10 * w; start = s0; er, dr = feat(s0, i); atrA = atr[s0 - 1]; state = 1; rows_meta = dict(er=er, dr=dr, w=w)
    elif state == 1:
        age = i - start
        if age > 96: state = 0; lastEnd = i
        else:
            bh = h[i] > rH + tol; bl = l[i] < rL - tol
            if bh and bl: state = 0; lastEnd = i
            elif bh or bl:
                if age < 12: state = 0; lastEnd = i
                else:
                    side = 1 if bh else -1; ext = h[i] if bh else l[i]; sb = i
                    # wait up to 6 bars for return inside; same-bar return allowed
                    ok = False; j = i; broke2 = False
                    while True:
                        if j > i:
                            ext = max(ext, h[j]) if side == 1 else min(ext, l[j])
                            if (side == 1 and l[j] < rL - tol) or (side == -1 and h[j] > rH + tol): broke2 = True; break
                            if j - sb > 6: break
                        if rL <= c[j] <= rH: ok = True; break
                        j += 1
                        if j >= n - 70 or brk[j]: break
                    if ok and not broke2 and j + 1 < n and not brk[j + 1]:
                        d = -side; leg = (rH - ext) if d == 1 else (ext - rL)
                        ent = o[j + 1]; tgt = (rH + .5 * leg) if d == 1 else (rL - .5 * leg); stp = (ext - .4 * atrA) if d == 1 else (ext + .4 * atrA); risk = abs(ent - stp)
                        if leg > 0 and (tgt - ent) * d > 0 and (ent - stp) * d > 0:
                            res = None
                            for k in range(j + 1, min(j + 1 + 64, n)):
                                if brk[k]: res = (c[k - 1] - ent) * d; break
                                hs = (l[k] <= stp) if d == 1 else (h[k] >= stp); ht = (h[k] >= tgt) if d == 1 else (l[k] <= tgt)
                                if hs: res = (stp - ent) * d; break
                                if ht: res = (tgt - ent) * d; break
                            if res is None: res = (c[min(j + 64, n - 1)] - ent) * d
                            tm = pd.Timestamp(dt[j]); rows.append(dict(t=tm, d=d, pts=res - .75, R=(res - .75) / risk, risk=risk, w=rows_meta['w'], er=rows_meta['er'], dr=rows_meta['dr'], leg=leg / rows_meta['w'], q=int(((tm.hour * 60 + tm.minute - 18 * 60) % 1440) // 90)))
                    state = 0; lastEnd = j; i = max(i, j)
    i += 1
T = pd.DataFrame(rows).set_index('t'); T['yr'] = T.index.year; trn = (T.index >= '2019-01-01') & (T.index < '2022-01-01'); tst = ~trn
def f(m, col='pts'):
    s = T.loc[m, col]; return '   n/a' if len(s) < 20 else f'{s.mean():+6.2f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s):4d}, win {(T.loc[m, "pts"] > 0).mean():.0%})'
print(f'Po3 trades {len(T)} ({len(T) / ((T.index[-1] - T.index[0]).days / 365.25):.0f}/yr); long {int((T.d == 1).sum())}, short {int((T.d == -1).sum())}')
print(f'all   pts/trade: train {f(trn)} | test {f(tst)}\n      R/trade  : train {f(trn, "R")} | test {f(tst, "R")}')
for lab, m in (('long', T.d == 1), ('short', T.d == -1)): print(f'{lab:5s} pts: train {f(trn & m)} | test {f(tst & m)}')
print('by year pts/trade:', {int(y): round(T[T.yr == y].pts.mean(), 2) for y in sorted(T.yr.unique())}, ' n:', T.yr.value_counts().sort_index().to_dict())
def terc(col, lab):
    q = T.loc[trn, col].quantile([1 / 3, 2 / 3]).values; L, M, H = T[col] <= q[0], (T[col] > q[0]) & (T[col] < q[1]), T[col] >= q[1]
    print(f'{lab:22s} cuts {q.round(2)}\n   train low {f(trn & L)} mid {f(trn & M)} high {f(trn & H)}\n   test  low {f(tst & L)} mid {f(tst & M)} high {f(tst & H)}')
print('\nSLOW-TREND vs CONSOLIDATION (accumulation window), pts/trade'); terc('er', 'ER (efficiency)'); terc('dr', 'DRIFT (net/width)'); terc('leg', 'sweep leg / width')
print('\n90m session quarter of the D bar (Q0 = 18:00-19:30 ET ... ), pts/trade:'); 
for qq in range(16): print(f'   slot {qq:2d} ({(18 * 60 + qq * 90) // 60 % 24:02d}:{(18 * 60 + qq * 90) % 60:02d}): train {f(trn & (T.q == qq))} | test {f(tst & (T.q == qq))}')
T.to_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/po3_341' + ('rth' if RTHONLY else '') + '.parquet')
