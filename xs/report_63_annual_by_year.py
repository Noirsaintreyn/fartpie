"""Nov 14 2026 ANNUAL RETURN BY YEAR of the kept rule (>=1 officer/director Form 4 buy in 30d + flip-bar volume >=1.5x) as a portfolio, vs volume-only and vs equal-weight buy-and-hold of the same universe. Book mechanics (fixed before results): long-only, K=10 slots, each position = (equity / 10) at entry, no leverage, idle cash earns 0, at most K open at once (same-day ties broken by highest flip-bar volume), vector exit, 30bp round trip per trade, P&L realized at the exit date (exit = entry + hold business days), equity compounded, year return = equity(Dec 31)/equity(Jan 1) - 1 with only realized P&L (open positions not marked). 2026 = Jan-Mar only (insider data ends 2026-03-31). Universe has survivorship bias (names still listed today)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_ins.pkl'); P = P[P.t >= '2019-01-01'].copy(); P['rn'] = P.ret - 0.003
P['ent'] = P.t.dt.normalize(); P['ext'] = [e + pd.offsets.BDay(int(h)) for e, h in zip(P.ent, P.hold)]
def book(S, K=10):
    S = S.sort_values(['ent', 'BV'], ascending=[True, False]); ev = []; eq = 1.0; opn = []; pnl = []; taken = 0
    for r in S.itertuples():
        opn = [(x, w, rr) for (x, w, rr) in opn if x > r.ent or (pnl.append((x, w * rr)) or False)]  # realize exits
        if len(opn) < K: opn.append((r.ext, 1 / K, r.rn)); taken += 1
    for (x, w, rr) in opn: pnl.append((x, w * rr))
    return pnl, taken
def annual(pnl):
    s = pd.DataFrame(pnl, columns=['d', 'c']).sort_values('d'); eq = 1.0; out = {}; yr = None; start = 1.0
    for d, c in zip(s.d, s.c):
        if d.year != yr:
            if yr is not None: out[yr] = eq / start - 1
            yr = d.year; start = eq
        eq *= (1 + c * 1.0)
    out[yr] = eq / start - 1; return out
# NOTE: weight w=1/K applied to the equity at EXIT time (compounding approximation: each slot is 1/K of equity)
R0 = P[(P.NI >= 1) & (P.BV >= 1.5)]; top2 = R0.rn.nlargest(2).index
sig = {'insider+volume (rule)': R0, 'rule EXCL. top-2 winners (CELH, OPRX)': R0.drop(top2), 'volume only (BV>=1.5)': P[P.BV >= 1.5]}
res = {}
for k, S in sig.items():
    pnl, n = book(S); res[k] = (annual(pnl), n, len(S))
D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); bh = {}
for y in range(2019, 2027):
    r = []
    for tk, x in D.items():
        a = x.close[(x.index >= f'{y}-01-01') & (x.index <= (f'{y}-12-31' if y < 2026 else '2026-03-31'))]
        if len(a) > 20: r.append(a.iloc[-1] / a.iloc[0] - 1)
    bh[y] = np.mean(r)
yrs = list(range(2019, 2027)); print('year | ' + ' | '.join(res) + ' | equal-weight buy&hold (same names)')
for y in yrs: print(f' {y}{"*" if y == 2026 else " "} | ' + ' | '.join(f'{res[k][0].get(y, float("nan"))*100:+7.1f}%' for k in res) + f' | {bh[y]*100:+7.1f}%')
for k in res: a = res[k][0]; g = np.prod([1 + a.get(y, 0) for y in yrs]); print(f'{k}: signals {res[k][2]}, trades taken (K=10) {res[k][1]}, compounded 2019-26Q1 {(g-1)*100:+.0f}%, worst year {min(a.values())*100:+.1f}%, years positive {sum(v>0 for v in a.values())}/{len(a)}')
print('* 2026 = Jan-Mar only, realized P&L only')
