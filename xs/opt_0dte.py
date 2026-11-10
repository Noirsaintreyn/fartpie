"""Nov 9 2026: THE SCREENER'S FLIPS IN 0DTE / THIS-WEEK CALLS, CLOSED AT THE CONTRACT'S CLOSE (Rey: '0DTE or low time frame call, just close at contract close').
DECLARED BEFORE RESULTS. Flips: the live candidates in the 23 names with ThetaData quotes (data/theta_compact), one per name per day, entered at the flip's own 15m entry
(price from the Mac's 15m bars: data/flip_intraday.csv from xs/flip_intraday.py).
CONTRACTS (calls only; single stocks mostly expire on Fridays - some on Mon / Wed lately - so 0DTE exists only on those days):
  0DTE   expiry = the flip's day; closed at that day's close (expiry)
  WEEK   the nearest expiry (0-7 days); EOD = closed at that day's close | EXP = held to the contract's expiry close
  Strike: ATM = first strike at or above the stock's entry price | FIT = first strike at or above it that one contract fits in the slot ($115.33 to start)
PRICES: entry = Black-Scholes at the flip's entry price and time (time left in trading sessions), with the contract's implied vol from the PRIOR close's mid,
  plus half the prior close's bid-ask spread; exit = the real closing BID of that contract (expiry day: the closing bid, else intrinsic). $0.10 a contract a side.
REPORT: per trade (1 contract, return on premium): n, win rate, average, median, 2019-22 vs 2023-26, next to the stock's own move over the same window; then the $692 book
(6 slots, budget = equity / 6, whole contracts, unaffordable flips skipped) from 2019 and from 2023."""
import numpy as np, pandas as pd, io, contextlib, os
_o = open('/home/claude/fartpie/xs/opt_flips.py').read()
with contextlib.redirect_stdout(io.StringIO()): exec(_o[:_o.index('def entry_all(')])
from math import erf, exp, log, sqrt
N_ = lambda x: .5 * (1 + erf(x / sqrt(2)))
def bs(S, K, t, v, r=.03):
    if t <= 0 or v <= 0: return max(S - K, 0.0)
    d1 = (log(S / K) + (r + v * v / 2) * t) / (v * sqrt(t)); return S * N_(d1) - K * exp(-r * t) * N_(d1 - v * sqrt(t))
def iv(p, S, K, t, r=.03):
    if t <= 0 or p <= max(S - K * exp(-r * t), 0) + 1e-6: return np.nan
    lo, hi = .01, 8.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs(S, K, t, m, r) > p: hi = m
        else: lo = m
    return (lo + hi) / 2
I = pd.read_csv('/home/claude/fartpie/data/flip_intraday.csv', parse_dates=['entry']).set_index(['tk', 'entry'])
F = T[T.tk.isin(B)].copy(); F = F.join(I, on=['tk', 'entry']); F = F[F.px_e.notna() & F.px_c.notna() & (F.exit < '2026-10-21')]
F = F.sort_values(['entry', 'ratio', 'tk'], ascending=[True, False, True], kind='stable')
DI = lambda x: np.datetime64(pd.Timestamp(x).normalize(), 'D').astype(np.int64)
def sess(bk, a, b): return int(np.searchsorted(bk.days, b, 'right') - np.searchsorted(bk.days, a, 'right'))     # sessions in (a, b]
def quote(bk, d, e, K):
    if d not in bk.rng: return None
    a, b = bk.rng[d]; w = np.where((bk.e[a:b] == e) & (bk.r[a:b] == 1) & (bk.K[a:b] == K))[0]
    return (float(bk.bid[a + w[0]]), float(bk.ask[a + w[0]])) if len(w) else None
def setup(r, kind, strike, budget):
    """-> dict(prem per share incl. half spread, value at exit per share, stock move over the window) or a reason string"""
    bk = B[r.tk]; d0 = DI(r.entry)
    if d0 not in bk.rng: return 'no chain'
    k0 = np.searchsorted(bk.days, d0); dp = bk.days[k0 - 1] if k0 > 0 else None
    if dp is None: return 'no prior day'
    a, b = bk.rng[d0]; dte = bk.dte[a:b]; ok_ = (dte >= 0) & (dte <= 7)
    if not ok_.any(): return 'no short expiry'
    dsel = 0 if kind == '0DTE' else dte[ok_].min()
    if kind == '0DTE' and not (dte == 0).any(): return 'no 0DTE today'
    S_close = bk.S[d0]; S_e = S_close * r.px_e / r.px_c
    ix = a + np.where((dte == dsel) & (bk.r[a:b] == 1) & (bk.K[a:b] >= S_e))[0]; ix = ix[np.argsort(bk.K[ix])]
    if not len(ix): return 'no strike'
    e = bk.e[ix[0]]; Sp = bk.S[dp]; tod = (960 - (r.entry.hour * 60 + r.entry.minute)) / 390
    T_prev = max(sess(bk, dp, e), 1) / 252; T_in = (sess(bk, d0, e) + tod) / 252
    for j in ix:
        K = float(bk.K[j]); qp = quote(bk, dp, e, K)
        if not qp or qp[1] <= 0 or qp[0] <= 0: continue
        v = iv((qp[0] + qp[1]) / 2, Sp, K, T_prev)
        if not np.isfinite(v): continue
        prem = bs(S_e, K, T_in, v) + (qp[1] - qp[0]) / 2
        if prem < .01: continue
        if strike == 'FIT' and prem * 100 + .10 > budget: continue
        return dict(K=K, e=e, prem=prem, d0=d0, S_e=S_e, S_close=S_close)
    return 'nothing fits' if strike == 'FIT' else 'no quote'
def exit_value(r, s, how):
    bk = B[r.tk]
    if how == 'EOD' or s['e'] == s['d0']:
        q = quote(bk, s['d0'], s['e'], s['K']); intr = max(s['S_close'] - s['K'], 0.0)
        return (q[0] if q and q[0] > 0 else intr) if s['e'] > s['d0'] else (q[0] if q and q[0] > 0 else intr), r.px_c / r.px_e - 1
    q = quote(bk, s['e'], s['e'], s['K']); Sx = bk.S.get(s['e'], np.nan)
    if not np.isfinite(Sx): Sx = bk.spot_at(s['e'])
    return (q[0] if q and q[0] > 0 else max(Sx - s['K'], 0.0)), Sx / s['S_e'] - 1
VAR = {'0DTE ATM': ('0DTE', 'ATM', 'EOD'), '0DTE FIT': ('0DTE', 'FIT', 'EOD'), 'WEEK ATM, out at the close': ('WEEK', 'ATM', 'EOD'),
       'WEEK FIT, out at the close': ('WEEK', 'FIT', 'EOD'), 'WEEK ATM, held to expiry': ('WEEK', 'ATM', 'EXP'), 'WEEK FIT, held to expiry': ('WEEK', 'FIT', 'EXP')}
def per_trade(kind, strike, how):
    rows = []
    for r in F.itertuples():
        s = setup(r, kind, strike, 115.33)
        if isinstance(s, str): continue
        v, smove = exit_value(r, s, how); rows.append((r.entry, (v * 100 - .10) / (s['prem'] * 100 + .10) - 1, smove, s['prem'] * 100))
    return pd.DataFrame(rows, columns=['t', 'ret', 'stock', 'cost'])
def book(kind, strike, how, start, cash0=692.0, N=6):
    X = F[F.entry >= start]; cash = cash0; op = []; out = []; skip = 0
    for r in X.itertuples():
        e = r.entry
        for p in sorted([p for p in op if p[0] <= e], key=lambda z: z[0]): cash += p[1]; out.append((p[0], p[1] - p[2], p[2])); op.remove(p)
        if len(op) >= N: continue
        budget = min((cash + sum(p[2] for p in op)) / N, cash); s = setup(r, kind, strike, budget)
        if isinstance(s, str): skip += s in ('nothing fits',); continue
        n = int(budget // (s['prem'] * 100 + .10))
        if n < 1: skip += 1; continue
        v, _ = exit_value(r, s, how); t_out = pd.Timestamp(np.datetime64(int(s['d0'] if how == 'EOD' else s['e']), 'D')) + pd.Timedelta(hours=16)
        cost = n * (s['prem'] * 100 + .10); op.append((t_out, n * (v * 100 - .10), cost)); cash -= cost
    for p in op: cash += p[1]; out.append((p[0], p[1] - p[2], p[2]))
    Z = pd.DataFrame(out, columns=['t', 'pnl', 'cost']).sort_values('t'); eqc = cash0 + Z.pnl.cumsum()
    yrs = (pd.Timestamp('2026-10-20') - pd.Timestamp(start)).days / 365.25
    return dict(end=cash, cagr=(cash / cash0) ** (1 / yrs) - 1 if cash > 0 else -1, dd=(eqc / eqc.cummax().clip(lower=cash0) - 1).min() if len(Z) else 0.0, n=len(Z), skip=skip,
                yr=Z.groupby(Z.t.dt.year).pnl.sum())
if __name__ == '__main__':
    print(f'flips in the 23 option names with 15m entry prices: {len(F):,} | flip days with a 0DTE expiry available: '
          f'{np.mean([isinstance(setup(r, "0DTE", "ATM", 1e9), dict) for r in F.itertuples()]):.0%}', flush=True)
    sm = F.px_c / F.px_e - 1
    print(f'  the stock itself, flip entry -> that day\'s close: 2019-22 avg {sm[F.entry < "2023"].mean():+.2%} (up {np.mean(sm[F.entry < "2023"] > 0):.0%}) | '
          f'2023-26 avg {sm[F.entry >= "2023"].mean():+.2%} (up {np.mean(sm[F.entry >= "2023"] > 0):.0%})', flush=True)
    print('\n=== per trade, 1 contract, return on premium | 2019-22 | 2023-26 (n, win, avg, median, avg cost of 1 contract; stock over the same window) ===', flush=True)
    for nm, (k_, s_, h_) in VAR.items():
        Z = per_trade(k_, s_, h_); L = []
        for lab, z in (('2019-22', Z[Z.t < '2023-01-01']), ('2023-26', Z[Z.t >= '2023-01-01'])):
            L.append(f'n {len(z):5,} win {np.mean(z.ret > 0):3.0%} avg {z.ret.mean():+6.1%} med {z.ret.median():+6.1%} cost ${z.cost.median():5,.0f} (stock {z.stock.mean():+.2%})')
        print(f'  {nm:<28} ' + ' | '.join(L), flush=True)
    for start in ('2019-01-02', '2023-01-03'):
        print(f'\n=== $692 from {start[:4]}, 6 slots, whole contracts | end value | CAGR | worst DD (closed trades) | trades | skipped (nothing fit the slot) ===', flush=True)
        for nm, (k_, s_, h_) in VAR.items():
            R = book(k_, s_, h_, start)
            print(f'  {nm:<28} ${R["end"]:>8,.0f} | {R["cagr"]:+6.1%} | {R["dd"]:6.1%} | {R["n"]:5,} | {R["skip"]:5,} | ' + ' '.join(f'{y}: ${v:+,.0f}' for y, v in R['yr'].items()), flush=True)
