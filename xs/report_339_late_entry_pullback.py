"""Nov 18 2026 report 339. Rey: 'I usually don't enter late unless we pull back and I'm still confident'.  Test his rule on the large-cap vector flips.  Declared before running.
Late entry at the open of the k-th 15m bar after the flip entry bar (k = 1, 2, 4), taken ONLY IF price has pulled back: open_k <= entry open (pullback of any size) or <= entry x (1 - 0.25%) (a real pullback); otherwise the flip is skipped (chasing).  Exit unchanged (the vector's own bear flip); if the flip already exited before bar k, no trade.
THEORY: the flip is a breakout (report 338: a 15m delay halves CAGR because the first bars carry the follow-through).  A pullback entry buys the same trade at a price at or below the signal price, so if the flip survives (no bear flip yet), the entry is at least as good as the original; the cost is the trades that never pull back (the strongest ones) are skipped.  Why it would fail: the flips that pull back are the weaker ones (no follow-through) and the strongest breakouts never come back.
Measured: (a) per candidate trade, mean return of the pullback-late entries vs the ORIGINAL entry on the SAME trades and vs the skipped (chased) trades; share taken; (b) the 6-slot book (as 338) with only the pullback-late trades.
Same cost as 338 (2+2 bp).  CAGR / max DD / Sharpe, 2019-26."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; B0 = pd.read_parquet(SP + 'b302.parquet'); B0 = B0[B0.exit < END].copy(); B0['entry'] = pd.to_datetime(B0.entry); B0['exit'] = pd.to_datetime(B0.exit); B0['h1'] = False; B0 = B0.reset_index(drop=True)
def sim(Bs, N=6, brk=.08):
    global T0
    key = Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values; T0 = Bs.assign(_k=key).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(N, brk); R = np.zeros(len(D))
    for tk, e, x, r, ra in tk_:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize())
        if a >= len(D): continue
        b = min(b, len(D) - 1); n = b - a + 1; w = 1 / N; R[a:b + 1] += w * ((1 + r) * (1 - 2e-4)) ** (1 / n) - w
    return pd.Series(R, index=D)
def cagr(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def mdd(s): e = (1 + s).cumprod(); return float((e / e.cummax() - 1).min())
O = {}
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))
    H = pd.read_parquet(f[-1], columns=['datetime', 'open']); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); O[tk] = H.open.sort_index()
s0 = sim(B0); s0 = s0[s0.index >= '2019-01-01']; print(f'baseline (enter at the flip bar open): CAGR {cagr(s0):.1%}, DD {mdd(s0):.1%}, Sharpe {sharpe(s0.values):.2f}; trades {len(B0)}')
for k in (1, 2, 4):
    e0 = np.full(len(B0), np.nan); ek = np.full(len(B0), np.nan); xk = np.full(len(B0), np.nan); alive = np.zeros(len(B0), bool)
    for tk, o in O.items():
        m = np.where((B0.tk == tk).values)[0]; ie = o.index.searchsorted(B0.entry.values[m]); ix = o.index.searchsorted(B0.exit.values[m]); ov = o.values; ik = np.minimum(ie + k, len(o) - 1)
        e0[m] = ov[np.minimum(ie, len(o) - 1)]; ek[m] = ov[ik]; xk[m] = ov[np.minimum(ix, len(o) - 1)]; alive[m] = ik < ix
    r0 = xk / e0 - 1 - .0002; rk = xk / ek - 1 - .0002; pb_any = alive & (ek <= e0); pb_real = alive & (ek <= e0 * (1 - .0025)); chase = alive & ~pb_any
    print(f'\n--- delay k={k} bar(s) ({15 * k} min): flip still alive at the entry bar {alive.mean():.0%}')
    for lab, c in (('pulled back (<= entry)', pb_any), ('pulled back >= 0.25%', pb_real), ('chased (above entry)', chase)):
        print(f'   {lab:24s} share {c.mean():5.1%} | same trades, original entry {r0[c].mean():+.3%} | late entry {rk[c].mean():+.3%} (win {np.mean(rk[c] > 0):.0%}, orig win {np.mean(r0[c] > 0):.0%})')
    for lab, c in (('pullback (any)', pb_any), ('pullback >= 0.25%', pb_real)):
        Bs = B0[c].copy(); Bs['ret'] = rk[c]; z = sim(Bs); z = z[z.index >= '2019-01-01']; print(f'   BOOK, only {lab:18s}: {c.sum()} trades, CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
