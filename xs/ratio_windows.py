"""Nov 12 2026: options-ratio WINDOW test. DECLARED BEFORE RESULTS. Rey: why a 252d median - try week / month / quarter rolling?
Book as live (xs/stop_1h_ev.py sim): S&P names, wide_T base+sticky live exits, entries <14:00, 6 slots, higher-ratio-first, 8% breaker, 40 matched runs (5% random drops), CAGR/DD/ret-DD 2019-21 | 2022-26.
Only the ratio changes: median over w sessions (min_periods=min(60,w)), shift(1), asof entry date; gate >= 1.5 as live; w in 5,21,63,126,252. Parity: w=252 must match the stored ratio.
Variants: G = gate AND rank both by window-w value | R = gate fixed at stored 252 membership, RANK by window-w value | Q = window-w gate set to keep the same trade share as the 252 gate."""
import numpy as np, pandas as pd, zlib, warnings; warnings.filterwarnings('ignore')
from collections import deque
src = open('xs/stop_1h_ev.py').read()
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[(W['var'] == 'base+sticky') & ((W.entry.dt.hour * 60 + W.entry.dt.minute) < 840) & (W.entry >= '2019-01-01') & (W.sp == True)].copy().sort_values('entry').reset_index(drop=True)
IDX = pd.bdate_range('2019-01-01', W.exit.max()); P2 = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
exec(src[src.index('def stat2'):src.index('def line(')])
P = '/mnt/user-data/uploads/fartpie/xs/ext/'
L = pd.read_csv(P + 'optvol_ratio_live.csv', parse_dates=['date']); L = L[L.sh > 0]; L['r'] = L.opt_contracts * 100 / L.sh
S = pd.read_csv(P + 'optvol_ratio_sp.csv', parse_dates=['date'])
O = pd.concat([L[['tk', 'date', 'r']], S]).drop_duplicates(['tk', 'date'], keep='first')
R = O.pivot_table(index='date', columns='tk', values='r').sort_index(); print('ratio history', R.index.min().date(), R.index.max().date(), R.shape[1], 'tickers; wide_T S&P trades', len(W), 'tickers', W.tk.nunique())
def win(w): return R.rolling(w, min_periods=min(60, w)).median().shift(1)
d = W.entry.dt.normalize().values
def attach(M):
    return np.array([M[t].asof(pd.Timestamp(x)) if t in M else np.nan for t, x in zip(W.tk, d)])
V = {w: attach(win(w)) for w in (5, 21, 63, 126, 252)}
ok = ~np.isnan(V[252]); print('parity w=252 vs stored: corr', np.corrcoef(V[252][ok & W.ratio.notna().values], W.ratio.values[ok & W.ratio.notna().values])[0, 1], ' trades gated both ways', int(((V[252] >= 1.5) & (W.ratio >= 1.5)).sum()), 'mine', int((V[252] >= 1.5).sum()), 'stored', int((W.ratio >= 1.5).sum()))
def ev2(Y, K):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); res = []
    for i in range(1, 41):
        m = ~(((HSH * 2654435761 + i * 40503) % 1000) < 50); res.append(stat2(sim(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m])))
    return np.array(res)
def show(nm, Y, K, base=None):
    r = ev2(Y, K); m = np.median(r, axis=0); b = '' if base is None else f'  beats252 {(r[:, 2] > base[:, 2]).mean():.0%} / {(r[:, 5] > base[:, 5]).mean():.0%}'
    print(f'  {nm:<26} {len(Y):>6} tr | CAGR {m[0]:+.1%} / {m[3]:+.1%}  DD {m[1]:.0%} / {m[4]:.0%}  ret/DD {m[2]:.2f} / {m[5]:.2f}{b}', flush=True); return r
g252 = V[252] >= 1.5; Y0 = W[g252].reset_index(drop=True); base = show('252 gate+rank (live)', Y0, V[252][g252])
print('== G: gate>=1.5 and rank by window-w value')
for w in (5, 21, 63, 126):
    g = V[w] >= 1.5; show(f'G w={w}', W[g].reset_index(drop=True), V[w][g], base)
print('== R: gate = live 252 membership, rank by window-w')
for w in (5, 21, 63, 126): show(f'R w={w}', Y0, V[w][g252], base)
print('== Q: window-w gate keeping the same trade share as the 252 gate')
share = g252.mean()
for w in (5, 21, 63, 126):
    v = pd.Series(V[w]); thr = v.quantile(1 - share); g = (v >= thr).values; show(f'Q w={w} (thr {thr:.2f})', W[g].reset_index(drop=True), V[w][g], base)
print('== per-trade (gate set only): mean ret, win rate, by period')
for nm, g in [('252', g252)] + [(f'w={w}', V[w] >= 1.5) for w in (5, 21, 63)]:
    Y = W[g]; a, b = Y[Y.entry < '2022'], Y[Y.entry >= '2022']; print(f'  {nm:<6} {len(Y):>6} | {a.ret.mean()*100:+.2f}% {a.ret.gt(0).mean():.0%} / {b.ret.mean()*100:+.2f}% {b.ret.gt(0).mean():.0%}')
