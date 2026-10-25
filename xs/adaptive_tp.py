"""Oct 25 2026: stock-specific profit targets. Each stock has its own 'normal good move' (AAPL ~5%, ORCL maybe 8-10%); once a trade
reaches it, tighten: trail at peak - r x gain (exit at the next bar's open after a 15m close below it), instead of the live
lock (+5% peak, keep half). Target per stock, known at entry:
 vol  : m x daily sigma x sqrt(5)   (the stock's typical 1-week move)
 hist : median peak gain of that stock's EARLIER winning trades (expanding, >= 8 prior winners, else the vol target)
 band : intraday - today's range has reached the 80% band (low-to-high 1.9 daily sigma) while the trade is >= +2% -> exit next open
Full-S&P screener, ratio >= 1.5, live rules, 6 slots, 8% breaker."""
from collections import deque
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
exec('def sim3' + open('xs/equity_models.py').read().split('def sim3')[1].split('STRAT_MED =')[0]); STRAT_MED = 1
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
SIG = CL.pct_change().rolling(20).std().shift(1)
T['sig'] = [SIG[t].asof(d) if t in SIG.columns else np.nan for t, d in zip(T.tk, T.entry.dt.normalize())]; T['sig'] = T.sig.fillna(T.sig.median())
# peak gain of every trade (for the 'hist' target) from the same bars
PK = np.full(len(T), np.nan)
for tk, g in T.groupby('tk'):
    B = bars(tk)
    if B is None: continue
    dt = B.dt.values; o, h = B.open.values, B.high.values
    for i, e, x in zip(g.index, g.entry.values, g.exit.values):
        a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
        if a < len(dt) and dt[a] == e and b < len(dt) and b > a: PK[i] = h[a:b].max() / o[a] - 1
T['peak'] = PK
hist_t = np.full(len(T), np.nan); seen = {}
for i, (tk, e, x, r, pk) in enumerate(zip(T.tk, T.entry.values, T.exit.values, T.ret, T.peak)):
    prev = [p for (xx, rr, p) in seen.get(tk, []) if xx <= e and rr > 0 and p == p]
    hist_t[i] = np.median(prev) if len(prev) >= 8 else np.nan
    seen.setdefault(tk, []).append((x, r, pk))
T['hist_t'] = hist_t
print('typical winner peak by stock (median of winners, all years):'); W = T[(T.ret > 0) & T.peak.notna()]
print('  ' + '  '.join(f'{t} {v * 100:.1f}%' for t, v in W.groupby('tk').peak.median().sort_values().items() if (W.tk == t).sum() >= 20))
def run(kind, m=1.5, r=.3):
    r_out, x_out = T.ret.values.copy(), T.exit.values.copy(); n = 0
    for tk, g in T.groupby('tk'):
        B = bars(tk)
        if B is None: continue
        dt = B.dt.values; o, h, l, c = B.open.values, B.high.values, B.low.values, B.close.values; day = B.dt.dt.normalize().values
        for i, e, x in zip(g.index, g.entry.values, g.exit.values):
            a = np.searchsorted(dt, e); b = np.searchsorted(dt, x)
            if a >= len(dt) or dt[a] != e or b >= len(dt) or b <= a: continue
            p0 = o[a]; s = T.sig[i]
            tgt = m * s * np.sqrt(5) if kind == 'vol' else (T.hist_t[i] if kind == 'hist' and T.hist_t[i] == T.hist_t[i] else m * s * np.sqrt(5))
            peak = p0; dlo, dhi, dcur = None, None, None
            for j in range(a, b - 1):
                peak = max(peak, h[j]); pg = peak / p0 - 1
                if kind == 'band':
                    if day[j] != dcur: dcur, dlo, dhi = day[j], l[j], h[j]
                    dlo, dhi = min(dlo, l[j]), max(dhi, h[j])
                    if c[j] / p0 - 1 >= .02 and dhi / dlo - 1 >= 1.9 * s: r_out[i] += o[j + 1] / p0 - o[b] / p0; x_out[i] = dt[j + 1]; n += 1; break
                elif pg >= tgt and c[j] <= p0 * (1 + (1 - r) * pg):
                    r_out[i] += o[j + 1] / p0 - o[b] / p0; x_out[i] = dt[j + 1]; n += 1; break
    return r_out, x_out, n
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<56}' + ''.join(f'{cc:>+9.1%} /{dd:>5.0%}' for cc, dd in (sim3(Z, a, b) for a, b in P)), flush=True)
print(f'\n{"stock-specific targets":<58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', T)
for kind, m, r in [('vol', 1.0, .3), ('vol', 1.5, .3), ('vol', 2.0, .3), ('vol', 1.5, .2), ('vol', 2.0, .4), ('hist', 0, .3), ('hist', 0, .2), ('hist', 0, .4), ('band', 0, 0)]:
    rr, xx, n = run(kind, m, r); Z = T.copy(); Z['ret'], Z['exit'] = rr, xx
    nm = {'vol': f'target {m:g}x weekly vol, then trail giving back {r:.0%}', 'hist': f'target = stock\'s median winner peak, give back {r:.0%}', 'band': 'day range hits the 80% band while up 2% -> sell next open'}[kind]
    show(f'{nm} [{n}]', Z)
