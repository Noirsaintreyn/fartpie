"""Report 531 (Nov 28 2026): Rey: "lower time frame vector on some so u can catch more of the move, 1h or 4h".
DECLARED BEFORE RESULTS. Mechanism: the daily vector confirms late (pivot_len 8 daily bars + EMA filter), so it gives back part of the move and enters after the first leg; the same state machine on shorter bars should enter/exit earlier -> higher capture (r / PEAK), at the price of more trades, more noise, and more cost.
Same vector_core.run_state_machine, SAME parameters (bar-count based), on RTH bars rebuilt from LSE 15m: 1D (control), 4H-ish (2 bars/day: 09:30-13:00, 13:00-16:00), 1H (09:30-10:30 ... 15:30-16:00 six+ bars).
Sample: 131 tickers with a sub-$5 daily flip since 2025-01 (flip_history); window 2024-11-01..2026-11 (15m history). Long entries at the flip bar close when close<5; exit at the vector's next flip-to-resistance close; open trades at the end dropped. Metrics per timeframe: trades/ticker-yr, reach (PEAK>=+10% before exit, the flip_history 'hit'), mean/median r, capture = sum r / sum PEAK, avg hold in days, and net of a declared cost of 1.0% round trip per trade (sub-$5 spreads/slippage) AND gross.
Also by ATRP tercile (train cut from flip_history_cuts.json) since the daily edge is high-vol names. No fitting: single parameter set, no tuning of the intraday timeframe parameters (if it needs tuning that is a separate, declared test).
PASS: net-of-cost mean r per trade AND total net return per ticker-year both above daily on test-style split (first half / second half of window agree in sign of the difference).
"""
import os, sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'xs'); import lse_data; from vector_core import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; C = SP + 'lse15m/'; os.makedirs(C, exist_ok=True)
H = pd.read_csv('flip_history.csv', parse_dates=['d']); tks = sorted(H[H.d >= '2025-01-01'].tk.unique()); cuts = json.load(open('flip_history_cuts.json'))['atrp']
START, END = pd.Timestamp('2024-11-01'), pd.Timestamp('2026-11-28')
def load(tk):
    f = C + tk + '.pkl'
    if os.path.exists(f): return pd.read_pickle(f)
    D = lse_data.frame(tk, '15m', START, END)
    if D is not None: D.to_pickle(f)
    return D
def rth(D):
    t = D.index.hour * 60 + D.index.minute; return D[(t >= 570) & (t < 960)]
def agg(R, tf):
    if tf == '15m': return R
    idx = R.index; mins = idx.hour * 60 + idx.minute
    if tf == '1D': key = idx.normalize()
    elif tf == '4H': key = idx.normalize() + pd.to_timedelta(np.where(mins < 780, 570, 780), unit='m')
    elif tf == '1H': key = idx.normalize() + pd.to_timedelta(np.where(mins < 630, 570, (mins // 60) * 60), unit='m')
    g = R.groupby(key); return g.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'))
def trades(B, tk, tf):
    if len(B) < 120: return []
    reg, lev, atr, flips = run_state_machine(B); c, h = B.close.values, B.high.values; out = []; fl = [f for f in flips]
    for k, (i, dr) in enumerate(fl):
        if dr != 1 or not (c[i] < 5): continue
        j = next((jj for jj, d2 in fl[k + 1:] if d2 == -1), None)
        if j is None: continue
        pk = h[i + 1:j + 1].max() / c[i] - 1 if j > i else 0.0
        out.append(dict(tk=tk, tf=tf, t0=B.index[i], t1=B.index[j], px=c[i], ATRP=atr[i] / c[i], r=c[j] / c[i] - 1, PEAK=pk, days=(B.index[j] - B.index[i]).total_seconds() / 86400))
    return out
rows = []; n = 0
for tk in tks:
    D = load(tk)
    if D is None or not len(D): continue
    R = rth(D)
    if not len(R): continue
    for tf in ('1D', '4H', '1H'): rows += trades(agg(R, tf), tk, tf)
    n += 1
T = pd.DataFrame(rows); T['hit'] = (T.PEAK >= .10).astype(int); T['bucket'] = np.select([T.ATRP <= cuts[0], T.ATRP <= cuts[1]], ['lo', 'mid'], 'hi'); T.to_pickle(SP + 'ext/intraday531_trades.pkl'); print('tickers loaded', n, 'of', len(tks), 'trades', len(T))
COST = .01; yrs = (END - START).days / 365.25 * 0 + 2.0; mid = pd.Timestamp('2025-11-01'); pd.set_option('display.width', 220)
def summ(S, nt):
    if not len(S): return {}
    net = S.r - COST; return dict(n=len(S), per_tk_yr=round(len(S) / nt / yrs, 1), reach=round(S.hit.mean() * 100), mean_gross=round(S.r.mean() * 100, 2), mean_net=round(net.mean() * 100, 2), med=round(S.r.median() * 100, 2), capture=round(S.r.sum() / max(S.PEAK.sum(), 1e-9), 3), hold_d=round(S.days.mean(), 1), net_tot_per_tk_yr=round(net.sum() / nt / yrs * 100, 1))
for lab, m in (('ALL sub-$5', T.tf.notna()), ('high-ATRP only', T.bucket == 'hi'), ('first half (<2025-11)', T.t0 < mid), ('second half', T.t0 >= mid)):
    print('\n==', lab); print(pd.DataFrame({tf: summ(T[m & (T.tf == tf)], n) for tf in ('1D', '4H', '1H')}).T.to_string())
