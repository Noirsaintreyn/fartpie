"""Oct 25 2026: a safer sister model for trading OPTIONS on the screener's flips: higher options ratio, capped winners (take profit),
tighter stop, short holds (theta). Goal: high win rate, low drawdown, not max return.
Trades = the live screener's flips (full S&P, PIT). For each: take profit when price first touches +TP, stop at -SL (or the vector exit,
whichever first), and a time stop after D trading days (sell at that day's close). Stock-level results, plus a rough option P&L:
~30-day at-the-money call, delta 0.5, premium ~ 0.4 x vol x sqrt(30/365) of the stock price, theta ~ premium / 60 per day."""
from collections import deque
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
T = T[~T.h1].reset_index(drop=True)                                              # 15m flips only (one trade per flip)
VOLA = (CL.pct_change().rolling(20).std() * np.sqrt(252)).shift(1)
T['vola'] = [VOLA[t].asof(d) if t in VOLA.columns else np.nan for t, d in zip(T.tk, T.entry.dt.normalize())]
def run(tp, sl, D):
    r_out, x_out = T.ret.values.copy(), T.exit.values.copy(); days_held = np.zeros(len(T))
    for tk, g in T.groupby('tk'):
        B = bars(tk)
        if B is None: continue
        dt = B.dt.values; o, h, l, c = B.open.values, B.high.values, B.low.values, B.close.values; day = B.dt.dt.normalize().values
        for i, e, x in zip(g.index, g.entry.values, g.exit.values):
            a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
            if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: continue
            p0 = o[a]; ud = np.unique(day[a:b + 1]); lastday = ud[min(D, len(ud) - 1)] if D else None
            res, xt = o[b] / p0 - 1, dt[b]
            for j in range(a, b):
                if lastday is not None and day[j] > lastday: res, xt = c[j - 1] / p0 - 1, dt[j - 1]; break
                if sl and l[j] <= p0 * (1 - sl): res, xt = min(-sl, o[j] / p0 - 1) if j > a else -sl, dt[j]; break
                if tp and h[j] >= p0 * (1 + tp): res, xt = max(tp, o[j] / p0 - 1) if j > a else tp, dt[j]; break
            r_out[i] = res - 2e-4; x_out[i] = xt; days_held[i] = len(np.unique(day[a:np.searchsorted(dt, xt) + 1]))
    return r_out, x_out, days_held
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'))]
def opt_ret(r, vola, days):
    prem = 0.4 * vola * np.sqrt(30 / 365); return (0.5 * r - prem / 60 * days) / prem      # per-trade return on the premium (rough)
print(f'{"variant (15m flips)":<40}{"trades/yr":>10}{"win %":>8}{"avg %":>8}{"avg win":>9}{"avg loss":>9}{"opt win %":>10}{"opt avg %":>10}   book 19-21 / 22-26 / 19-26 (stock, 6 slots)')
for cut in (1.5, 3.0):
    base = T[T.ratio >= cut]
    for tp, sl, D in ((None, None, None), (.015, None, None), (.02, None, None), (.03, None, None), (.02, .015, None), (.02, .02, 5), (.03, .02, 5), (.015, .015, 3), (.02, .01, 3)):
        r, x, dh = run(tp, sl, D)
        Z = T.copy(); Z['ret'], Z['exit'] = r, x; Z['dh'] = dh; Z = Z[Z.ratio >= cut]
        yrs = (Z.entry.max() - Z.entry.min()).days / 365.25
        o = opt_ret(Z.ret.values, Z.vola.fillna(.35).values, np.maximum(Z.dh.values, 1))
        bk = '  '.join(f'{c_:+.0%}/{d_:.0%}' for c_, d_ in (sim3(Z, a, b) for a, b in P))
        nm = f'ratio>={cut} ' + ('rule exits (live)' if tp is None else f'TP {tp:.1%}' + (f' SL {sl:.1%}' if sl else ' vector stop') + (f' max {D}d' if D else ''))
        print(f'{nm:<40}{len(Z) / yrs:>10.0f}{(Z.ret > 0).mean():>8.0%}{Z.ret.mean() * 100:>8.2f}{Z.ret[Z.ret > 0].mean() * 100:>9.2f}{Z.ret[Z.ret <= 0].mean() * 100:>9.2f}'
              f'{(o > 0).mean():>10.0%}{np.mean(o) * 100:>10.1f}   {bk}', flush=True)
