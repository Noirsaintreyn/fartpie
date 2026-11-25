"""Report 459 (Nov 24 2026): portfolio books built from the 458 signal models on the options-ratio flip universe (23 names with theta so far).
Book: N slots (default 6) each 1/N of current equity, one position per name, first flip in time order takes a free slot, exit at the flip's own exit (vector flip / sticky stop already in wide_T `ret`, net of its cost convention), REALIZED-P&L equity (open trades not marked), compounding, 2019-01..2026-10.
Books: MAIN = all gated flips (benchmark); A = negative gamma; A+B/C/D = negative gamma plus the put/call volume (15m), skew and charm models as extra signals; B/C/D alone and B|C|D.
Per book: trades, trades/yr, total return, CAGR, max realized drawdown, win rate, by-year returns.  Run: python3 xs/report_459_signal_books.py"""
import pandas as pd, numpy as np, os
exec(open('xs/report_457_optfeat_flips.py').read().split("NAMES = dict")[0].split('"""')[2])
W['exit'] = pd.to_datetime(W.exit)
def book(T, N=6):
    T = T.sort_values('entry'); open_ = []; eq = 1.0; hist = []; held = set(); taken = []
    ev = []
    for _, r in T.iterrows():
        open_ = [o for o in open_ if (o[0] > r.entry) or (hist.append((o[0], o[1])) or eq.__class__ is None) and False]
    return None
def run_book(T, N=6):
    T = T.sort_values('entry').reset_index(drop=True); pos = []; closed = []; eq = 1.0; curve = []
    def settle(upto):
        nonlocal eq, pos
        done = sorted([p for p in pos if p['exit'] <= upto], key=lambda p: p['exit'])
        for p in done: eq *= 1 + p['ret'] / N; curve.append((p['exit'], eq)); closed.append(p)
        pos = [p for p in pos if p['exit'] > upto]
    for _, r in T.iterrows():
        settle(r.entry)
        if len(pos) < N and all(p['tk'] != r.tk for p in pos): pos.append(dict(tk=r.tk, exit=r.exit, ret=r.ret, entry=r.entry))
    settle(pd.Timestamp('2030-01-01'))
    c = pd.Series({t: e for t, e in curve}).sort_index(); c = c[~c.index.duplicated(keep='last')]
    return closed, c
def stats(closed, c, label):
    if not closed: print(f'{label:28s} no trades'); return
    yrs = (c.index[-1] - pd.Timestamp('2019-01-01')).days / 365.25; tot = c.iloc[-1] - 1; cagr = c.iloc[-1] ** (1 / yrs) - 1
    dd = (c / c.cummax() - 1).min(); win = np.mean([p['ret'] > 0 for p in closed]); by = c.groupby(c.index.year).last(); by = by / by.shift(1).fillna(1) - 1
    print(f'{label:28s} trades {len(closed):4d} ({len(closed) / yrs:4.0f}/yr)  total {tot:+6.1%}  CAGR {cagr:+5.1%}  maxDD {dd:+5.1%}  win {win:.0%} | ' + ' '.join(f'{y % 100}:{v:+.0%}' for y, v in by.items()))
for h1 in [False, True]:
    D = W[(W.h1 == h1) & W.gex.notna()].copy(); tf = '1h' if h1 else '15m'
    D['A'] = D.gex < 0; D['B'] = D.r_pc_vol >= 2 / 3; D['C'] = D.r_skew >= 2 / 3; D['D'] = D.r_charm >= 2 / 3
    print(f'\n=== {tf} track, 6 slots, 23 names (flips with features: {len(D)}) ===')
    sets = [('MAIN all gated flips', D), ('A negative gamma', D[D.A]), ('B put/call vol (15m)' if not h1 else None, D[D.B] if not h1 else None), ('C skew', D[D.C]), ('D charm', D[D.D]), ('B|C|D', D[(D.B & (not h1)) | D.C | D.D]), ('A + B|C|D', D[D.A | (D.B & (not h1)) | D.C | D.D])]
    for lab, T in sets:
        if lab is None: continue
        cl, c = run_book(T); stats(cl, c, lab)
