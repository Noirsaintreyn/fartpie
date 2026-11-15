"""Nov 15 2026 LARGE-CAP 1H AND 15m VECTOR WITH CHANDELIER EXITS (Rey: his current large-cap screener runs 15m and 1H; 'try the chandelier on that'). Declared before results. Universe: top-150 dollar-volume names from the daily file (price >= $20, ETFs excluded = names without a SIC code), Yahoo RTH bars: 1H 2023-12..2026-11 (730d cap), 15m only 60 days (Aug 21-Nov 13 2026, ONE regime, lead only). Long-only vector flips, entry at the CLOSE of the flip bar, cost 10bp round trip (large caps; 25bp sensitivity), survivorship-biased (names listed today). Exits: vector bear flip (current), chandelier = highest high since entry - M x ATR22 on the same timeframe, close-based, vector bear flip as backstop, M = 3 primary (declared), 4 and 5 sensitivity. 1H split: TRAIN flip date <= 2024-12-31, TEST 2025-01+. Control: the same names' average per-bar drift times the trade's bars held (excess = trade - drift x bars). Report n, mean/median net, win, >=+10% share, excess mean, worst 1%, average bars held, and equal-weight slot-limited portfolio (max 10 positions, 10% each, first come, additive) return/maxDD/Sharpe on TEST for vector-only vs chandelier 3x/4x. Lead bar: chandelier improves mean net and maxDD in both eras with TEST day-clustered t of (chandelier minus vector) >= 2."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SC + 'ext/'; sic = pd.read_csv(X + 'sic.csv').drop_duplicates('tk').set_index('tk')
def rma(x, n):
    o = np.full(len(x), np.nan)
    if len(x) < n + 1: return o
    o[n - 1] = np.nanmean(x[:n]); a = 1 / n
    for k in range(n, len(x)): o[k] = o[k - 1] * (1 - a) + x[k] * a
    return o
def atr22(h, l, c): pc = np.r_[c[0], c[:-1]]; return rma(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), 22)
def chand(h, c, a, j, m, start):
    hh = -np.inf
    for k in range(start, j + 1):
        hh = max(hh, h[k])
        if not np.isnan(a[k]) and c[k] < hh - m * a[k]: return k
    return j
def build(path, nm, cost=.001):
    P = pickle.load(open(path, 'rb')); rows = []; nn = 0
    for tk, B in P.items():
        if tk not in sic.index: continue
        B = B[~B.index.duplicated()]; B = B[B.volume > 0]
        if len(B) < 200: continue
        nn += 1; reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); h = B.high.values; l = B.low.values; c = B.close.values; n = len(B); a = atr22(h, l, c); dr = np.mean(np.diff(np.log(c))); idx = B.index; dts = idx.normalize().tz_localize(None)
        for i in range(2, n - 1):
            if not (reg[i] == 1 and reg[i - 1] != 1): continue
            j = i + 1
            while j < n and reg[j] != -1: j += 1
            j = min(j, n - 1); e = c[i]; row = dict(tk=tk, t=idx[i], d=dts[i], te=idx[min(j, n - 1)], v=c[j] / e - 1 - cost, hv=j - i, drift=dr)
            for m in (3, 4, 5): k = chand(h, c, a, j, m, i + 1); row[f'c{m}'] = c[k] / e - 1 - cost; row[f'h{m}'] = k - i; row[f'te{m}'] = idx[k]
            rows.append(row)
    A = pd.DataFrame(rows); print(f'\n{nm}: names {nn}, flips {len(A)}'); return A
def stt(r, hold, drift):
    ex = r - drift * hold; s = r.sort_values(ascending=False); return f'mean {r.mean()*100:+6.2f}% med {r.median()*100:+6.2f}% win {(r>0).mean()*100:4.1f}% >=10% {(r>=.1).mean()*100:4.1f}% excess {ex.mean()*100:+6.2f}% worst1% {r.quantile(.01)*100:+6.1f}% hold {hold.mean():5.1f} bars'
def dday(z, a, b): g = z.assign(x=z[a] - z[b]).groupby(z.d.astype(str)).x.mean(); return g.mean() * 100, g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 5 else np.nan
def port(A, ex, te, N=10, f=.10):
    ev = []; 
    cur = []; tot = []; 
    P = A.sort_values('t'); pos = []; pnl = []; dates = sorted(set(P.d)); eq = 0; curve = {}
    open_ = []
    for d in dates:
        open_ = [p for p in open_ if p[1].normalize().tz_localize(None) > d]
        for _, r in P[P.d == d].iterrows():
            if len(open_) < N: open_.append((r.name, r[te])); tot.append(f * r[ex])
        curve[d] = sum(tot)
    c = pd.Series(curve); dd = (c.cummax() - c).max(); dly = c.diff().dropna(); return sum(tot) * 100, dd * 100, (dly.mean() / dly.std() * np.sqrt(252) if dly.std() > 0 else np.nan)
H = build(X + 'largecap_1h.pkl', '1H (2023-12..2026-11)'); H['era'] = np.where(H.d <= '2024-12-31', 'TRAIN', 'TEST')
for era in ('TRAIN', 'TEST'):
    z = H[H.era == era]; print(f'\n 1H {era} n {len(z)}'); print('  vector-only  ', stt(z.v, z.hv, z.drift))
    for m in (3, 4, 5): print(f'  chandelier {m}x ', stt(z[f'c{m}'], z[f'h{m}'], z.drift), f'| vs vector day-clustered {dday(z, f"c{m}", "v")[0]:+.2f}pp t {dday(z, f"c{m}", "v")[1]:+.2f}')
z = H[H.era == 'TEST']; print('\n 1H TEST slot-limited portfolio (max 10 positions, 10% each, additive, first-come):')
for lab, ex, te in (('vector-only', 'v', 'te'), ('chandelier 3x', 'c3', 'te3'), ('chandelier 4x', 'c4', 'te4'), ('chandelier 5x', 'c5', 'te5')): r, dd, sh = port(z, ex, te); print(f'  {lab:14s} total {r:+6.1f}% of capital  maxDD {dd:5.1f} pts  Sharpe {sh:5.2f}')
print('\n 1H cost sensitivity 25bp (TEST, vector vs chandelier 4x):'); print(f'  vector {((z.v-.0015).mean())*100:+.2f}%  chand4x {((z.c4-.0015).mean())*100:+.2f}%')
M = build(X + 'largecap_15m.pkl', '15m (Aug 21-Nov 13 2026, ONE regime)'); print(' 15m ALL (lead only)'); print('  vector-only  ', stt(M.v, M.hv, M.drift))
for m in (3, 4, 5): print(f'  chandelier {m}x ', stt(M[f'c{m}'], M[f'h{m}'], M.drift), f'| vs vector day-clustered {dday(M, f"c{m}", "v")[0]:+.2f}pp t {dday(M, f"c{m}", "v")[1]:+.2f}')
H.to_pickle(X + 'lc_1h.pkl'); M.to_pickle(X + 'lc_15m.pkl')
