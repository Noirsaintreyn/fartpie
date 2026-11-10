"""Nov 9 2026: THE SCREENER IN CALL OPTIONS WITH $115.33 A SLOT (Rey: 'what would happen if you used options for the screener but 115.33 per basket' = $692 / 6).
DECLARED BEFORE RESULTS.
Book: the live candidates (S&P point-in-time, options ratio >= 1.5) in the 23 names with ThetaData quotes (data/theta_compact), highest ratio first, 6 slots,
start $692 on Jan 2 2019 and again on Jan 3 2023, slot budget = equity / 6 (equity = cash + cost of open positions), WHOLE contracts only:
a flip whose contract costs more than the slot budget is skipped and the slot stays free for the next flip. No 8% breaker (option equity swings too much for it).
Pricing: the stock trade's own entry -> exit return (intraday 15m entries / exits) applied to the option by Black-Scholes, with the contract's implied vol from that
day's closing mid (entry: the entry day; exit: the exit day), r = 3%, plus half the closing bid-ask spread on each side, $0.10 a contract a side in fees.
An option expiring before the screener's exit is closed at expiry for intrinsic value.
CONTRACTS:  A  ~30 days (21-45), the first strike at or above spot; skip if one contract costs more than the slot
            B  ~30 days, the first strike at or above spot that one contract fits in the slot (goes further out of the money until it fits)
            C  weekly (5-10 days), same fitting rule as B
            D  ~60 days (45-75), first strike at or above spot; skip if it does not fit
Reference: the same flips in SHARES (fractional), same slots and budget, and the live stock book on all 54 names.
Reported: ending value of $692, CAGR, worst drawdown on closed trades, trades taken, flips skipped as unaffordable, win rate, average return per trade, by year."""
import numpy as np, pandas as pd, math, os, io, contextlib
from collections import deque
_o = open('/home/claude/fartpie/xs/opt_flips.py').read()
with contextlib.redirect_stdout(io.StringIO()): exec(_o[:_o.index('def entry_all(')])
from math import erf, exp, log, sqrt
N_ = lambda x: .5 * (1 + erf(x / sqrt(2)))
def bs(S, K, t, v, r=.03):
    if t <= 0 or v <= 0: return max(S - K, 0.0)
    d1 = (log(S / K) + (r + v * v / 2) * t) / (v * sqrt(t)); return S * N_(d1) - K * exp(-r * t) * N_(d1 - v * sqrt(t))
def iv(p, S, K, t, r=.03):
    if t <= 0 or p <= max(S - K * exp(-r * t), 0) + 1e-6: return .05
    lo, hi = .01, 5.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs(S, K, t, m, r) > p: hi = m
        else: lo = m
    return (lo + hi) / 2
SPEC = {'A': ((21, 45, 30), 'atm'), 'B': ((21, 45, 30), 'fit'), 'C': ((5, 10, 7), 'fit'), 'D': ((45, 75, 60), 'atm')}
DI = lambda x: np.datetime64(pd.Timestamp(x).normalize(), 'D').astype(np.int64)
def quote(bk, d, e, K):
    if d not in bk.rng: return None
    a, b = bk.rng[d]; w = np.where((bk.e[a:b] == e) & (bk.r[a:b] == 1) & (bk.K[a:b] == K))[0]
    return (bk.bid[a + w[0]], bk.ask[a + w[0]]) if len(w) else None
def option_trade(bk, d0, d1, ret, spec, budget):
    """-> (contracts, cost, proceeds) or None when no contract qualifies / fits"""
    if d0 not in bk.rng: return None
    S0 = bk.S[d0]; (lo, hi, tg), how = SPEC[spec]; a, b = bk.rng[d0]; dte = bk.dte[a:b]; ok_ = (dte >= lo) & (dte <= hi)
    if not np.isfinite(S0) or not ok_.any(): return None
    dsel = dte[ok_][np.argmin(np.abs(dte[ok_] - tg))]
    ix = a + np.where((dte == dsel) & (bk.r[a:b] == 1) & (bk.ask[a:b] > 0) & (bk.bid[a:b] > 0) & (bk.K[a:b] >= S0))[0]
    if not len(ix): return None
    ix = ix[np.argsort(bk.K[ix])]
    if how == 'atm': j = ix[0]
    else:
        fit = ix[bk.ask[ix] * 100 + .10 <= budget]
        if not len(fit): return None
        j = fit[0]
    K, e, bi, ak = bk.K[j], bk.e[j], bk.bid[j], bk.ask[j]
    if ak - bi > .5 * (ak + bi) / 2: return None
    mid = (ak + bi) / 2; t0 = dsel / 365; v0 = iv(mid, S0, K, t0); p0 = bs(S0, K, t0, v0) + (ak - bi) / 2      # entry at the stock's entry, plus half spread
    n = int((budget) // (p0 * 100 + .10))
    if n < 1: return None
    S1 = S0 * (1 + ret)
    if d1 >= e: px1 = max(bk.spot_at(e) * (1 + 0) - K, 0.0) if d1 > e else max(S1 - K, 0.0)               # held to expiry
    else:
        q = quote(bk, d1, e, K); t1 = (e - d1) / 365
        v1 = iv((q[0] + q[1]) / 2, bk.S.get(d1, S1), K, t1) if q and q[1] > 0 else v0; hs = (q[1] - q[0]) / 2 if q else (ak - bi) / 2
        px1 = max(bs(S1, K, t1, v1) - hs, 0.0)
    return n, n * (p0 * 100 + .10), n * (px1 * 100 - .10)
def book(C, mode, start, end='2027-01-01', cash0=692.0, N=6):
    """mode: 'stock' or a contract spec. Event-driven, ratio order, whole contracts."""
    X = C[(C.entry >= start) & (C.entry < end)]; cash = cash0; op = []; out = []; skipped = 0
    for r in X.itertuples():
        e = r.entry
        for p in sorted([p for p in op if p[0] <= e], key=lambda z: z[0]): cash += p[1]; out.append((p[0], p[1] - p[2], p[2])); op.remove(p)
        if len(op) >= N: continue
        eq = cash + sum(p[2] for p in op); budget = min(eq / N, cash)
        if mode == 'stock':
            if budget < 1: continue
            op.append((r.exit, budget * (1 + r.ret) * (1 - 2e-4), budget)); cash -= budget; continue
        bk = B.get(r.tk); res = option_trade(bk, DI(r.entry), DI(r.exit), r.ret, mode, budget) if bk else None
        if res is None: skipped += 1; continue
        n, cost, proc = res
        if cost > cash: skipped += 1; continue
        op.append((r.exit, proc, cost)); cash -= cost
    for p in op: cash += p[1]; out.append((p[0], p[1] - p[2], p[2]))
    Z = pd.DataFrame(out, columns=['t', 'pnl', 'cost']).sort_values('t'); Z['equity'] = cash0 + Z.pnl.cumsum()
    yrs = (pd.Timestamp(min(pd.Timestamp(end), X.exit.max())) - pd.Timestamp(start)).days / 365.25
    dd = (Z.equity / Z.equity.cummax().clip(lower=cash0) - 1).min() if len(Z) else 0
    return dict(end=cash, cagr=(cash / cash0) ** (1 / yrs) - 1 if cash > 0 else -1, dd=dd, n=len(Z), skip=skipped, win=(Z.pnl > 0).mean(), avg=(Z.pnl / Z.cost).mean(),
                yr=Z.groupby(Z.t.dt.year).pnl.sum(), Z=Z)
if __name__ == '__main__':
    ALL = T[T.exit < '2026-10-21'].sort_values(['entry', 'ratio', 'tk'], ascending=[True, False, True], kind='stable'); C = ALL[ALL.tk.isin(B)]
    print(f'candidates in the 23 option names: {len(C):,} ({C.tk.nunique()} names), {C.entry.min():%Y-%m-%d} - {C.exit.max():%Y-%m-%d}', flush=True)
    lab = {'stock': 'shares (fractional)', 'A': 'A  30-day ATM call, skip if > slot', 'B': 'B  30-day call, strike moved out until it fits',
           'C': 'C  weekly call, strike moved out until it fits', 'D': 'D  60-day ATM call, skip if > slot'}
    for start in ('2019-01-02', '2023-01-03'):
        print(f'\n=== $692 from {start[:4]} (6 slots, $115.33 each to start) | end value | CAGR | worst DD (closed trades) | trades | flips skipped (unaffordable) | win | avg per trade ===', flush=True)
        for m in ('stock', 'A', 'B', 'C', 'D'):
            R = book(C, m, start)
            print(f'  {lab[m]:<48} ${R["end"]:>9,.0f} | {R["cagr"]:+6.1%} | {R["dd"]:6.1%} | {R["n"]:4,} | {R["skip"]:5,} | {R["win"]:3.0%} | {R["avg"]:+6.1%}', flush=True)
            print(f'      by year P&L: ' + ' | '.join(f'{y} ${v:+,.0f}' for y, v in R['yr'].items()), flush=True)
        R = book(ALL, 'stock', start); print(f'  {"reference: shares, all 54 names (live universe)":<48} ${R["end"]:>9,.0f} | {R["cagr"]:+6.1%} | {R["dd"]:6.1%} | {R["n"]:4,}', flush=True)
