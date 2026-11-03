"""Nov 2 2026: slot mechanics around the low-vol finding (scan.py / vol_floor.py). DECLARED BEFORE RESULTS. lowvol = lowest within-year
fifth of 20d realized vol at the prior close (covers every trade); highvol = top third. Fixed budget, never above 100%, 8% breaker.
 M1 collision: low-vol flips go last when same-bar flips outnumber free slots (ratio otherwise)
 M2 rotate: book full + a non-low-vol flip arrives -> close the worst LOSING low-vol position at that bar's open (marked from 15m bars), take the flip
 M3 size tilt: weight 1.25/6 high-vol, 0.75/6 low-vol, 1/6 else; capped by idle cash
 M4 basket: drop low-vol flips and run 5 slots at 1/5 (M4b: drop + 6 slots, = F2q, for reference)
 Neighbour for M1/M2/M3: same with the IV-based low-vol flag (lowest IV fifth; IV missing -> not flagged).
 Pass: >= 70% of 40 matched runs beat live in BOTH periods and the IV neighbour >= 60%."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
OPT3 = open("/home/claude/fartpie/xs/opt_three.py").read(); exec(OPT3.split("X['h'] = np.where")[0].replace("print(", "(lambda *a, **k: None)("))
exec(OPT3[OPT3.index("HSH = np.array"):OPT3.index("pr = lambda s")])
X['yr'] = X.entry.dt.year; RV, BB = {}, {}
for tk in X.tk.unique():
    H = pd.read_parquet(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0], columns=['datetime', 'open', 'close'])
    H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); BB[tk] = (H.dt.values, H.open.values)
    c = H.groupby(H.dt.dt.normalize()).close.last(); RV[tk] = c.pct_change().rolling(20).std().shift(1)
X['rv'] = [RV[t].asof(d) for t, d in zip(X.tk, X.d)]
rvp = X.groupby('yr').rv.rank(pct=True); ivp = X.groupby('yr').iv.rank(pct=True)
LOW_RV = (rvp <= .2).fillna(False).values; HIGH_RV = (rvp > 2 / 3).fillna(False).values
LOW_IV = (ivp <= .2).fillna(False).values; HIGH_IV = (ivp > 2 / 3).fillna(False).values
EN = X.entry.values; EXv = X.exit.values; RE = X.ret.values; RA = X.ratio.values.astype(float); TKv = X.tk.values
def px_at(tk, t):
    tt, oo = BB[tk]; k = np.searchsorted(tt, t); return oo[min(k, len(oo) - 1)]
def simx(mask, low, high, mode, n=6):
    idx = np.where(mask)[0]
    key = -RA[idx] + (1e6 * low[idx] if mode == 'M1' else 0)
    o = idx[np.lexsort((TKv[idx], key, EN[idx]))]
    op = []; pend = {}; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for k in o:
        e = EN[k]
        for kk in [kk for kk, (xt, _, _) in pend.items() if xt <= e]:
            xt, r, w = pend.pop(kk); real += np.log1p(r * w); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q in pend]; gross = sum(pend[q][2] for q in op)
        w = 1 / n
        if mode == 'M3': w = (1.25 if high[k] else .75 if low[k] else 1) / 6
        w = min(w, 1 - gross)
        if len(op) >= n or w < .02:
            if mode == 'M2' and not low[k] and len(op) >= n:
                lows = [q for q in op if low[q]]
                if lows:
                    cur = {q: px_at(TKv[q], e) / px_at(TKv[q], EN[q]) - 1 - C2 for q in lows}; q = min(cur, key=cur.get)
                    if cur[q] < 0:
                        xt, r, ww = pend.pop(q); pend[q] = (e, cur[q], ww); op.remove(q)
                        xt2, r2, w2 = pend.pop(q); real += np.log1p(r2 * w2); hist.append((e, real)); win.append((e, real)); w = 1 / n
                    else: continue
                else: continue
            else: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        op.append(k); pend[k] = (EXv[k], RE[k], w)
    for xt, r, w in sorted(pend.values(), key=lambda z: z[0]): real += np.log1p(r * w); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def evm(mode, low, high, drop=None, n=6):
    base = np.ones(len(X), bool) if drop is None else ~drop
    return np.array([stat2(simx(base & ~dl, low, high, mode, n)) for dl in DROPS])
B0 = evm('live', LOW_RV, HIGH_RV); print('40 matched runs, 2019-21 / 2022-26'); line('live', B0)
for nm, mode, lo, hi, drop, n in (('M1 low-vol last in collisions', 'M1', LOW_RV, HIGH_RV, None, 6), ('   IV neighbour', 'M1', LOW_IV, HIGH_IV, None, 6),
                                  ('M2 rotate out losing low-vol', 'M2', LOW_RV, HIGH_RV, None, 6), ('   IV neighbour', 'M2', LOW_IV, HIGH_IV, None, 6),
                                  ('M3 size tilt 1.25 / 1 / 0.75', 'M3', LOW_RV, HIGH_RV, None, 6), ('   IV neighbour', 'M3', LOW_IV, HIGH_IV, None, 6),
                                  ('M4 drop low-vol, 5 slots @1/5', 'live', LOW_RV, HIGH_RV, LOW_RV, 5), ('M4b drop low-vol, 6 slots (=F2q)', 'live', LOW_RV, HIGH_RV, LOW_RV, 6),
                                  ('   IV neighbour, 5 slots', 'live', LOW_IV, HIGH_IV, LOW_IV, 5)):
    line(nm, evm(mode, lo, hi, drop, n), B0)
