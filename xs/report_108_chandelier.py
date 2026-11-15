"""Nov 15 2026 CHANDELIER EXITS (Rey). Declared before results. Chandelier = highest high since entry minus M x ATR(22, Wilder); exit when the CLOSE falls below it (daily: exit at next open; 1H: exit at that bar close), with the vector bear flip as backstop (earlier of the two). PRIMARY M = 3 (classic, declared in advance); M = 2, 4, 5 reported as sensitivity, not selected. Daily: report_90 flips, entry next open, 30bp, TRAIN 2019-21 / TEST 2022+, survivorship-biased. 1H: the report_105 flip set (entry at flip-bar close, 100bp, 60 days Aug 31-Nov 12, ONE regime, lead only), judged on first/second half split at Oct 6. Compare to the vector-only exit on mean/median/win10/huge/ex-top3/avg hold. Lead bar: improves mean net on BOTH eras (or halves) versus vector-only and keeps the huge-tail share within 25% of vector-only."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SC = X[:-4]; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); Ms = (2, 3, 4, 5)
def rma(x, n):
    o = np.full(len(x), np.nan); o[n - 1] = np.nanmean(x[:n]); a = 1 / n
    for k in range(n, len(x)): o[k] = o[k - 1] * (1 - a) + x[k] * a
    return o
def atr22(h, l, c): pc = np.r_[c[0], c[:-1]]; return rma(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), 22)
def chand(h, c, a, i, j, m, start):
    hh = -np.inf
    for k in range(start, j + 1):
        hh = max(hh, h[k])
        if not np.isnan(a[k]) and c[k] < hh - m * a[k]: return k
    return j
rows = []
for tk, g in F.groupby('tk'):
    q = D.get(tk)
    if q is None or len(q) < 300: continue
    q = q[['open', 'high', 'low', 'close', 'volume']].copy(); q = q[~q.index.duplicated()].dropna(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize(); o = q.open.values; h = q.high.values; l = q.low.values; c = q.close.values; n = len(q); reg, _, _, _ = run_state_machine(q.assign(datetime=q.index)); a = atr22(h, l, c); pos = {d: k for k, d in enumerate(q.index)}
    for _, r in g.iterrows():
        i = pos.get(r.d)
        if i is None or i + 2 >= n: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        if j + 1 >= n: continue
        e = o[i + 1]; row = dict(tk=tk, d=r.d, test=r.test, v=o[j + 1] / e - 1, hv=j - i)
        for m in Ms: k = chand(h, c, a, i, j, m, i + 1); row[f'c{m}'] = o[k + 1] / e - 1; row[f'h{m}'] = k - i
        rows.append(row)
A = pd.DataFrame(rows); print('daily flips', len(A))
def stt(r, hold): r = r - .003; s = r.sort_values(ascending=False); return f'mean {r.mean()*100:+6.2f}% med {r.median()*100:+6.2f}% win10 {(r>=.1).mean()*100:5.1f}% huge {(r>=1).mean()*100:4.2f}% ex-top3 {s.iloc[3:].mean()*100:+6.2f}% worst1% {r.quantile(.01)*100:+6.1f}% avg hold {hold.mean():5.1f}d'
for era, z in (('TRAIN 2019-21', A[~A.test]), ('TEST 2022+', A[A.test])):
    print(f'\n DAILY {era} n {len(z)}'); print('  vector-only  ', stt(z.v, z.hv))
    for m in Ms: print(f'  chandelier {m}x ', stt(z[f'c{m}'], z[f'h{m}']))
cut = F[~F.test].lr.quantile(.8); A = A.merge(F[['tk', 'd', 'lr']], on=['tk', 'd'], how='left'); s = A[(A.lr >= cut) & A.test]; print(f'\n DAILY TEST, frozen score top 20% n {len(s)}'); print('  vector-only  ', stt(s.v, s.hv))
for m in Ms: print(f'  chandelier {m}x ', stt(s[f'c{m}'], s[f'h{m}']))
A.to_pickle(SC + 'ext/chand_daily.pkl')
# 1H
H = pd.read_pickle(SC + 'ext/flips_1h_filters.pkl'); M5 = pickle.load(open(SC + 'lowpx_5m.pkl', 'rb'))
def cl(x):
    x = x[['Open', 'High', 'Low', 'Close', 'Volume']].copy(); x.columns = ['open', 'high', 'low', 'close', 'volume']; x.index = x.index.tz_convert('America/New_York') if x.index.tz is not None else x.index; x = x[~x.index.duplicated()]; m = x.index.hour * 60 + x.index.minute; return x[(m >= 570) & (m < 960)]
out = []
for tk, g in H.groupby('tk'):
    x = cl(M5[tk]); x = x[x.volume > 0]; B = x.resample('60min', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open']); B = B[B.index.hour * 60 + B.index.minute >= 570]; h = B.high.values; l = B.low.values; c = B.close.values; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); a = atr22(h, l, c); ix = {t: k for k, t in enumerate(B.index)}
    for _, r in g.iterrows():
        i = ix.get(r.t)
        if i is None: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        j = min(j, n - 1); e = c[i]; row = dict(d=r.d, v=c[j] / e - 1 - .01, hv=j - i)
        for m in Ms: k = chand(h, c, a, i, j, m, i + 1); row[f'c{m}'] = c[k] / e - 1 - .01; row[f'h{m}'] = k - i
        out.append(row)
Hh = pd.DataFrame(out); Hh['half'] = np.where(Hh.d < '2026-10-06', 'train', 'test'); print('\n1H flips', len(Hh), '(close-to-close, 100bp, 60 days, ONE regime)')
def s1(r, hold): s = r.sort_values(ascending=False); return f'mean {r.mean()*100:+6.2f}% med {r.median()*100:+6.2f}% win {(r>0).mean()*100:5.1f}% >=10% {(r>=.1).mean()*100:4.1f}% worst1% {r.quantile(.01)*100:+6.1f}% avg hold {hold.mean():5.1f} bars'
for half in ('train', 'test'):
    z = Hh[Hh.half == half]; print(f'\n 1H {half} n {len(z)}'); print('  vector-only  ', s1(z.v, z.hv))
    for m in Ms: print(f'  chandelier {m}x ', s1(z[f'c{m}'], z[f'h{m}']))
