"""When a slot frees, buy a still-open honorable mention (a flip skipped earlier for full slots) instead of waiting for a new flip?
Live book clone (6 slots, ratio tie-break, 8% breaker on realized P&L, no buys after 14:00, 2 bp). A late fill enters at the open of the
bar where the slot frees and exits where the skipped trade's own rule exit is (approximation: its stops were set from its original entry).
Real flips arriving at the same moment keep priority. python3 xs/hm_fill.py WIDE_T BARS_DIR"""
import sys, glob, os, numpy as np, pandas as pd
from collections import deque
W = pd.read_parquet(sys.argv[1]); T = W[W.sp & (W.ratio >= 1.5)].copy(); T = T.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable').reset_index(drop=True)
OB = {}
for f in glob.glob(f'{sys.argv[2]}/*_15m_*.parquet'):
    tk = os.path.basename(f).split('_15m_')[0]; B = pd.read_parquet(f, columns=['datetime', 'open']); B['dt'] = pd.to_datetime(B.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = B.dt.dt.hour * 60 + B.dt.dt.minute; B = B[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); OB[tk] = (B.dt.values, B.open.values)
def opx(tk, t):
    d, o = OB[tk]; k = np.searchsorted(d, t); return o[k] if k < len(o) else np.nan
T['px0'] = [opx(t, e) for t, e in zip(T.tk, T.entry.values)]
C = 2e-4
def sim(a, b, mode=None, maxage=None, cap=None, floor=None, n=6, brk=.08, cut=840, fillcut=840):
    Z = T[(T.entry >= a) & (T.entry < b) & ((T.entry.dt.hour * 60 + T.entry.dt.minute) < cut)]; E, X, R, TK, RA, P0 = Z.entry.values, Z.exit.values, Z.ret.values, Z.tk.values, Z.ratio.values, Z.px0.values
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = [0.0]; w = 1 / n; skipped = []; nfill = 0
    def realize(upto):
        nonlocal real, pend
        done = sorted([p for p in pend if p[0] <= upto], key=lambda z: z[0]); pend = [p for p in pend if p[0] > upto]
        for xt, rr in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
    def brk_on(t):
        if not hist: return False
        t0 = t - np.timedelta64(28, 'D')
        while win and win[0][0] < t0: lvl0[0] = win.popleft()[1]
        return real - max([lvl0[0]] + [h for _, h in win]) < np.log1p(-brk)
    def fill(t, nxt_entry):
        nonlocal op, nfill
        if mode is None or (nxt_entry is not None and nxt_entry == t): return
        mins = (t - t.astype('datetime64[D]')).astype('timedelta64[m]').astype(int)
        if mins >= fillcut or mins < 570 or brk_on(t): return
        while len([x for x in op if x > t]) < n:
            c = [s for s in skipped if s['x'] > t and s['e'] < t and (maxage is None or t - s['e'] <= np.timedelta64(int(maxage * 1440), 'm'))]
            best = None
            for s in c:
                ox = opx(s['tk'], t)
                if not ox == ox: continue
                un = ox / s['p0'] - 1
                if (cap is not None and un > cap) or (floor is not None and un < floor): continue
                key = s['ra'] if mode == 'ratio' else (-un if mode == 'cheap' else un)
                if best is None or key > best[0]: best = (key, s, ox)
            if best is None: return
            _, s, ox = best; skipped.remove(s); xpx = s['p0'] * (1 + s['r'] + C)
            op.append(s['x']); pend.append((s['x'], xpx / ox - 1 - C)); nfill += 1
    for i in range(len(Z)):
        e = E[i]
        exits = sorted(x for x in op if x <= e)                                 # slots that freed since the last entry
        for x in exits:
            realize(x); op = [y for y in op if y > x]; fill(x, e)
        realize(e); op = [y for y in op if y > e]
        if brk_on(e): continue
        if len(op) >= n: skipped.append(dict(e=e, x=X[i], r=R[i], tk=TK[i], ra=RA[i], p0=P0[i])); continue
        op.append(X[i]); pend.append((X[i], R[i]))
    realize(np.datetime64('2100-01-01'))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min(), nfill
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
print(f'{"rule":<62}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}   fills')
def show(nm, **kw):
    res = [sim(a, b, **kw) for a, b in P]
    print(f'  {nm:<60}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d, _ in res) + f'   {res[2][2]}', flush=True)
show('wait for a new flip (live)')
for mins in (15, 30, 45, 60):
    show(f'take a flip skipped <= {mins} min before the slot freed', mode='ratio', maxage=mins / 1440)
    show(f'  same, only if not up more than 1% since its signal', mode='ratio', maxage=mins / 1440, cap=.01)
