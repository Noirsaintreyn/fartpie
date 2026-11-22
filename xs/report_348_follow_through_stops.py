"""Nov 22 2026 report 348. Rey: don't try to separate good from bad flips (the model is a classifier); PROTECT them: move the stop to safety faster on flips that don't follow through, which also frees slots.  Declared before running (8 rules, no tuning grid beyond these).
Path = every 15m bar from the entry bar to the vector exit (original exit = open of the bear-flip next bar).  Stop checks use bar lows (gap through = the bar's open), stop-first ordering: a bar that arms a rule cannot also trigger it.
 BE g: once a PRIOR bar's high >= entry x (1+g), stop = entry (breakeven), g in {0.5%, 1%, 2%}.
 TS k,m: TIME STOP: if the close of the bar before bar k is < entry x (1+m), exit at bar k's open.  (4,0%) (8,0.25%) (16,0.5%).
 TT k,m,s: TIGHTENED STOP: if not followed through by bar k (close < entry x (1+m)), the stop becomes entry x (1-s) instead of the vector level, else the vector exit stands.  (8,0.25%,0.3%) (16,0.5%,0.5%).
Measured: per-trade return (train 2019-21 | test 2018+2022-26), win rate, hold in 15m bars, share exited early; and the 6-slot book (2+2 bp, 8% breaker) CAGR / DD / Sharpe, trades taken (freed slots change what the book picks up).
THEORY: a flip with no early follow-through is a failed breakout; its expected path is flat/down while the vector level is far below, so exiting early saves the give-back and frees the slot for a fresh flip; the vector exit is cut only for these.  Why it fails: report 339 found the best flips pull back first (a time stop/BE shakes them out), and the vector's wide stop is what lets the rare big winners run (heavy-tail winners)."""
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
    return pd.Series(R, index=D), len(tk_)
def cagr(s): e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def mdd(s): e = (1 + s).cumprod(); return float((e / e.cummax() - 1).min())
RULES = [('BE .5%', 'BE', .005), ('BE 1%', 'BE', .01), ('BE 2%', 'BE', .02), ('TS 4,0', 'TS', 4, 0.0), ('TS 8,.25%', 'TS', 8, .0025), ('TS 16,.5%', 'TS', 16, .005), ('TT 8,.25%,.3%', 'TT', 8, .0025, .003), ('TT 16,.5%,.5%', 'TT', 16, .005, .005)]
n = len(B0); RET = {r[0]: np.full(n, np.nan) for r in RULES}; XT = {r[0]: np.empty(n, dtype='datetime64[ns]') for r in RULES}; HOLD = {r[0]: np.zeros(n) for r in RULES}; base_hold = np.zeros(n)
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); H = pd.read_parquet(f[-1]); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); o, h, l, c = H.open.values, H.high.values, H.low.values, H.close.values; idx = H.index.values
    m = np.where((B0.tk == tk).values)[0]; ie = np.searchsorted(idx, B0.entry.values[m]); ix = np.searchsorted(idx, B0.exit.values[m])
    for j, a, b in zip(m, ie, ix):
        e = o[a]; k = b - a; base_hold[j] = k
        if k < 1: 
            for r in RULES: RET[r[0]][j] = B0.ret.values[j]; XT[r[0]][j] = idx[b]; HOLD[r[0]][j] = k
            continue
        hh, ll, cc, oo = h[a:b], l[a:b], c[a:b], o[a:b]; basep = o[b]
        for r in RULES:
            nm, kind = r[0], r[1]; px, tx, hold = basep, idx[b], k
            if kind == 'BE':
                cm = np.maximum.accumulate(hh); armed = np.r_[False, cm[:-1] >= e * (1 + r[2])]; hit = np.where(armed & (ll <= e))[0]
                if len(hit): jj = hit[0]; px = min(oo[jj], e); tx = idx[a + jj]; hold = jj
            elif kind == 'TS':
                kk, mm = r[2], r[3]
                if kk < k and cc[kk - 1] < e * (1 + mm): px = oo[kk]; tx = idx[a + kk]; hold = kk
            else:
                kk, mm, ss = r[2], r[3], r[4]
                if kk < k and cc[kk - 1] < e * (1 + mm):
                    hit = np.where(ll[kk:] <= e * (1 - ss))[0]
                    if len(hit): jj = kk + hit[0]; px = min(oo[jj], e * (1 - ss)); tx = idx[a + jj]; hold = jj
            RET[nm][j] = px / e - 1 - .0002; XT[nm][j] = tx; HOLD[nm][j] = hold
tr = ((B0.entry >= '2019-01-01') & (B0.entry < '2022-01-01')).values; te = ~tr
def line(lab, X, ret, hold):
    z, nt = sim(X); z = z[z.index >= '2019-01-01']; early = np.mean(hold < base_hold - .5)
    print(f'{lab:15s} per trade train {ret[tr].mean():+.3%} | test {ret[te].mean():+.3%}; win {np.mean(ret > 0):.0%}; hold {hold.mean():5.1f} bars (live {base_hold.mean():.1f}); early exits {early:.0%} || BOOK taken {nt}, CAGR {cagr(z):.1%}, DD {mdd(z):.1%}, Sharpe {sharpe(z.values):.2f}')
line('LIVE (vector)', B0, B0.ret.values, base_hold)
for r in RULES: X = B0.copy(); X['ret'] = RET[r[0]]; X['exit'] = pd.to_datetime(XT[r[0]]); line(r[0], X, X.ret.values, HOLD[r[0]])
