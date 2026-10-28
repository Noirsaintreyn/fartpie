"""Oct 28 2026 (ORCL missed: its flips came while the book was full; a slot freed 30 min later): when a slot frees, take a recent flip
that was skipped and is still in its trade (same rules otherwise), entering at that bar's open. W = how old the skipped flip may be."""
import os, glob, numpy as np, pandas as pd
from collections import deque
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
OP = {}
def opx(tk, t):
    if tk not in OP: b = bars(tk); OP[tk] = None if b is None else b[0]
    s = OP[tk]
    if s is None: return np.nan
    k = s.index.searchsorted(t)
    return s.iloc[k] if k < len(s) else np.nan
def run(Z, a, b, W=None, n=6, brk=.08):
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable').reset_index(drop=True)
    ev = list(zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.tk.values, Z.ratio.values, Z.h1.values))
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; skipped = []; late = 0
    def settle(t):
        nonlocal real, pend
        done = sorted([p for p in pend if p[0] <= t], key=lambda z: z[0]); pend = [p for p in pend if p[0] > t]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
    def broke(t):
        nonlocal lvl0
        if not (brk and hist): return False
        t0 = t - np.timedelta64(28, 'D')
        while win and win[0][0] < t0: lvl0 = win.popleft()[1]
        return real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk)
    def fill_late(t):                                   # slots freed by exits up to t: give them to queued flips (best ratio first)
        nonlocal late, skipped
        if W is None: return
        for x in sorted([o for o in op if o <= t]):
            op.remove(x); settle(x)
            if broke(x): continue
            c = [s for s in skipped if s[0] >= x - W and s[0] < x and s[1] > x and all(q[3] != s[3] or q[5] != s[5] for q in held)]
            if not c: continue
            s = max(c, key=lambda z: z[4]); skipped.remove(s)
            p0 = opx(s[3], x); pe = opx(s[3], s[0])
            if not (p0 == p0 and pe == pe and p0 > 0): continue
            r = pe * (1 + s[2]) / p0 - 1; op.append(s[1]); pend.append((s[1], r)); held.append(s); late += 1
    held = []
    for e_ in ev:
        e, x, r_, tk, ra, h1 = e_
        fill_late(e); settle(e); op[:] = [o for o in op if o > e]
        held[:] = [h for h in held if h[1] > e]
        if len(op) >= n: skipped.append(e_); skipped = [s for s in skipped if s[1] > e]; continue
        if broke(e): continue
        op.append(x); pend.append((x, r_)); held.append(e_)
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min(), late
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
for nm, W in (('live (no queue)', None), ('queue 30 min', np.timedelta64(30, 'm')), ('queue 1 hour', np.timedelta64(60, 'm')), ('queue 2 hours', np.timedelta64(120, 'm')),
              ('queue same day ~6h', np.timedelta64(390, 'm')), ('queue 1 day', np.timedelta64(1, 'D')), ('queue 3 days', np.timedelta64(3, 'D'))):
    R = [run(L, a, b, W) for a, b in P]
    print(f'  {nm:<20}' + ''.join(f'{c:>+8.1%} /{d:>4.0%} ({k} late)   ' for c, d, k in R), flush=True)
