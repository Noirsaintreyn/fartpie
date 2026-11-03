"""Nov 2 2026: slot mechanics around the low-vol finding (scan.py / vol_floor.py) + reserved slots. DECLARED BEFORE RESULTS.
lowvol = lowest within-year fifth of 20d realized vol at the prior close (covers every trade); highvol = top third.
Fixed budget, never above 100% invested, 8% breaker, ratio tie-break unless stated.
 M1 collision: low-vol flips go last when same-bar flips outnumber free slots
 M2 rotate: book full + a non-low-vol flip arrives -> close the worst LOSING low-vol position at that bar's open, take the flip
 M3 size tilt: weight 1.25/6 high-vol, 0.75/6 low-vol, 1/6 else; capped by idle cash
 M4 basket: drop low-vol flips, 5 slots at 1/5 (M4b: drop, 6 slots)
 M5 reserved slots: r of the 6 slots (r = 1, 2) take only HIGH-GRADE flips; ordinary flips may hold at most 6 - r.
    high-grade A = top-third realized vol;  B = strong push (entry-bar open vs prior close in ATR, top tercile) AND high reach
    (implied move / stop distance, top tercile; IV-covered trades only)
 Neighbour for M1-M3: same with the IV-based flags. Pass: >= 70% of 40 matched runs beat live in BOTH periods."""
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
q = lambda c: X.groupby('yr')[c].rank(pct=True)
rvp, ivp, mp, rp = q('rv'), q('iv'), q('margin'), q('reach')
LOW_RV = (rvp <= .2).fillna(False).values; HIGH_RV = (rvp > 2 / 3).fillna(False).values
LOW_IV = (ivp <= .2).fillna(False).values; HIGH_IV = (ivp > 2 / 3).fillna(False).values
GRADE_B = ((mp > 2 / 3) & (rp > 2 / 3)).fillna(False).values
print(f'flags: low-vol {LOW_RV.mean():.0%}, high-vol {HIGH_RV.mean():.0%}, grade B {GRADE_B.mean():.0%} of flips', flush=True)
EN = X.entry.values; EXv = X.exit.values; RE = X.ret.values; RA = X.ratio.values.astype(float); TKv = X.tk.values
def px_at(tk, t):
    tt, oo = BB[tk]; k = np.searchsorted(tt, t); return oo[min(k, len(oo) - 1)]
def simx(mask, low, high, mode, n=6, r=0):
    idx = np.where(mask)[0]; key = -RA[idx] + (1e6 * low[idx] if mode == 'M1' else 0)
    o = idx[np.lexsort((TKv[idx], key, EN[idx]))]
    pend = {}; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for k in o:
        e = EN[k]
        for kk in [kk for kk, (xt, _, _) in pend.items() if xt <= e]:
            xt, rr, w = pend.pop(kk); real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        gross = sum(v[2] for v in pend.values())
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        if mode == 'M5' and not high[k] and sum(1 for kk in pend if not high[kk]) >= n - r: continue
        if len(pend) >= n:
            if mode != 'M2' or low[k]: continue
            cur = {kk: px_at(TKv[kk], e) / px_at(TKv[kk], EN[kk]) - 1 - C2 for kk in pend if low[kk]}
            if not cur: continue
            kk = min(cur, key=cur.get)
            if cur[kk] >= 0: continue
            _, _, w0 = pend.pop(kk); real += np.log1p(cur[kk] * w0); hist.append((e, real)); win.append((e, real)); gross -= w0
        w = 1 / n if mode != 'M3' else (1.25 if high[k] else .75 if low[k] else 1) / 6
        w = min(w, 1 - gross)
        if w < .02: continue
        pend[k] = (EXv[k], RE[k], w)
    for xt, rr, w in sorted(pend.values(), key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def evm(mode, low, high, drop=None, n=6, r=0):
    base = np.ones(len(X), bool) if drop is None else ~drop
    return np.array([stat2(simx(base & ~dl, low, high, mode, n, r)) for dl in DROPS])
B0 = evm('live', LOW_RV, HIGH_RV); print('40 matched runs, 2019-21 / 2022-26'); line('live', B0)
for nm, mode, lo, hi, drop, n, r in (('M1 low-vol last in collisions', 'M1', LOW_RV, HIGH_RV, None, 6, 0), ('   IV neighbour', 'M1', LOW_IV, HIGH_IV, None, 6, 0),
                                     ('M2 rotate out losing low-vol', 'M2', LOW_RV, HIGH_RV, None, 6, 0), ('   IV neighbour', 'M2', LOW_IV, HIGH_IV, None, 6, 0),
                                     ('M3 size tilt 1.25 / 1 / 0.75', 'M3', LOW_RV, HIGH_RV, None, 6, 0), ('   IV neighbour', 'M3', LOW_IV, HIGH_IV, None, 6, 0),
                                     ('M4 drop low-vol, 5 slots @1/5', 'live', LOW_RV, HIGH_RV, LOW_RV, 5, 0), ('M4b drop low-vol, 6 slots', 'live', LOW_RV, HIGH_RV, LOW_RV, 6, 0),
                                     ('M5 reserve 1 slot for high-vol', 'M5', LOW_RV, HIGH_RV, None, 6, 1), ('M5 reserve 2 slots for high-vol', 'M5', LOW_RV, HIGH_RV, None, 6, 2),
                                     ('M5 reserve 1 slot for grade B', 'M5', LOW_RV, GRADE_B, None, 6, 1), ('M5 reserve 2 slots for grade B', 'M5', LOW_RV, GRADE_B, None, 6, 2)):
    line(nm, evm(mode, lo, hi, drop, n, r), B0)
