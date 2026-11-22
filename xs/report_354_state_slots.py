"""Nov 22 2026 report 354. Rey: 'go on both' - use the QQQ weekly x daily state (report 351) as SLOT context.  Declared before running (3 rules, nothing else).
State at entry (last completed QQQ weekly + daily vector): NON-BULL = CORRECTION / BEAR / REBOUND (per-trade +0.25..+1.36% vs BULL +0.16..+0.39%, report 351).
 P1 PRIORITY: within the same entry bar, non-BULL flips are taken before BULL flips (then the live order: brank, ratio).
 P2 SWAP: slots full and a NON-BULL flip arrives -> replace the open BULL-state position with the lowest unrealized return, only if that is < +0.5% (marked at the entry bar open, 2bp exit cost).
 P3 RESERVE: BULL-state flips may fill at most 4 of the 6 slots (2 are kept for non-BULL flips); non-BULL flips use any slot.
THEORY: report 351 per-trade return rises as the tape gets worse (in both halves), and the book is capacity-limited (35% of flips taken), so spending scarce slots on the better state raises the mean of the taken trades.
Why it may fail: 71% of flips are BULL (non-BULL is rare, so the rules rarely bind); a BULL slot that is weak today is often the pullback before the move (report 349).
Measured: 6-slot book CAGR / DD / Sharpe, trades, 2019-21 | 2022-26 halves, swaps made."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
from vector_pine import run_state_machine; import yfinance as yf
Q = yf.download('QQQ', start='1999-03-10', auto_adjust=True, progress=False); Q.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Q.columns]; Q = Q.dropna(); W = Q.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def regat(df, dates):
    regs = run_state_machine(df.reset_index(drop=True))[0]; k = df.index.searchsorted(pd.DatetimeIndex(dates).normalize().values, side='left') - 1; return np.where(k >= 0, regs[np.clip(k, 0, len(regs) - 1)], 0)
B0['wk'] = regat(W, B0.entry); B0['dy'] = regat(Q, B0.entry); B0['nb'] = ~((B0.wk == 1) & (B0.dy == 1))
print(f'non-BULL share of candidate flips {B0.nb.mean():.1%} | candidates {len(B0)}')
def sim_state(Bs, prio=False, swap=False, reserve=None, N=6, brk=.08):
    key = Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values; Bs = Bs.assign(_k=key, _s=(~Bs.nb.values).astype(int))
    T = Bs.sort_values(['entry', '_s', '_k', 'tk'] if prio else ['entry', '_k', 'tk'], kind='stable')
    op = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []; swaps = []
    for e, x, r, tk, ra, nb in zip(T.entry.values, T.exit.values, T.ret.values, T.tk.values, T.ratio.values, T.nb.values):
        done = sorted([p for p in op if p[0] <= e], key=lambda z: z[0]); op = [p for p in op if p[0] > e]
        for p in done: real += np.log1p(p[1] / N); hist.append((p[0], real)); win.append((p[0], real))
        if reserve is not None and not nb and sum(1 for p in op if not p[7]) >= reserve: continue
        if len(op) >= N:
            if not (swap and nb): continue
            cand = [(px_at(p[2], e) / p[5] - 1, j) for j, p in enumerate(op) if not p[7]]
            if not cand: continue
            un, v = min(cand)
            if un >= .005: continue
            p = op.pop(v); rr = un - .0002; p[0] = pd.Timestamp(e); p[1] = rr; p[6][2] = pd.Timestamp(e); p[6][3] = rr; real += np.log1p(rr / N); hist.append((e, real)); win.append((e, real)); swaps.append((rr, r))
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [hh for _, hh in win]) < np.log1p(-brk): continue
        rec = [tk, pd.Timestamp(e), pd.Timestamp(x), r, ra]; op.append([x, r, tk, pd.Timestamp(e), ra, px_at(tk, e), rec, bool(nb)]); out.append(rec)
    return [tuple(z) for z in out], swaps
def nbshare(TK, Bs):
    k = set(zip(Bs.tk, pd.to_datetime(Bs.entry))); nbm = {(a, pd.Timestamp(b)): c for a, b, c in zip(Bs.tk, Bs.entry, Bs.nb)}; return np.mean([nbm.get((t[0], t[1]), False) for t in TK])
TK, _ = sim_state(B0); show('LIVE order', TK); print(f'   non-BULL share of taken {nbshare(TK, B0):.1%}')
for lab, kw in (('P1 priority', dict(prio=True)), ('P2 swap', dict(swap=True)), ('P1+P2', dict(prio=True, swap=True)), ('P3 reserve 4 BULL', dict(reserve=4)), ('P1+P3', dict(prio=True, reserve=4))):
    TK, sw = sim_state(B0, **kw); show(lab, TK, sw); print(f'   non-BULL share of taken {nbshare(TK, B0):.1%}')
print('\nNEIGHBOURHOOD reserve (max BULL slots of 6):')
for rv in (2, 3, 5): TK, _ = sim_state(B0, reserve=rv); show(f'reserve {rv} BULL', TK)
print('\nPLACEBO: same reserve-4 rule but the non-BULL label is a random 29.4% of flips (50 draws): CAGR / DD / Sharpe / 2022-26 CAGR')
rng = np.random.default_rng(7); res = []
for i in range(50):
    Bp = B0.copy(); Bp['nb'] = rng.random(len(Bp)) < B0.nb.mean(); TK, _ = sim_state(Bp, reserve=4); z = spread(TK); z = z[z.index >= '2019-01-01']; b = z[z.index >= '2022-01-01']
    res.append((cagr(z), mdd(z), sharpe(z.values), cagr(b)))
res = np.array(res); real = None
TK, _ = sim_state(B0, reserve=4); z = spread(TK); z = z[z.index >= '2019-01-01']; b = z[z.index >= '2022-01-01']
print(f'   placebo mean CAGR {res[:,0].mean():.1%} (5-95% {np.percentile(res[:,0],5):.1%}..{np.percentile(res[:,0],95):.1%}), DD {res[:,1].mean():.1%}, Sharpe {res[:,2].mean():.2f} (95th {np.percentile(res[:,2],95):.2f}), 2022-26 CAGR {res[:,3].mean():.1%}')
print(f'   real state label: CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}, 2022-26 {cagr(b):.1%} | share of placebo draws with Sharpe >= real: {(res[:,2] >= sharpe(z.values)).mean():.0%}')
