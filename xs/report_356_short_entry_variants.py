"""Nov 22 2026 report 356. Rey: is 'daily bull flip then a NEW daily bear flip (weekly bear)' the best way to enter the QQQ short?  Declared before running; exit identical for all (daily vector bull flip, or close > lowest low since entry + 3 x ATR14, next open), 2bp/side, one position.
 V0 frozen rule: enter after a daily bear flip made while the last completed weekly regime is bear.
 V1 V0 + JOIN: also enter at the next open when the WEEKLY flips bear on a week-end bar where the daily regime is already bear (arrive in a BEAR state without a fresh daily flip).
 V2 STATE: enter whenever (weekly bear AND daily bear) and flat, at the next open (re-enter after a trail exit while the state persists).
 V3 WEEKLY ONLY: enter on every weekly bear flip (any daily), exit on weekly bull flip or the trail.
Why V0 might be best: the daily bear flip is the only moment the vector stop is freshly placed; why not: V0 misses 'already bear' arrivals like now.  Reported: n, mean, win, short-only CAGR / DD, per-episode trades (2000-03, 2008, 2022-23, 2026)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine; import yfinance as yf
Q = yf.download('QQQ', start='1999-03-10', auto_adjust=True, progress=False); Q.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Q.columns]; Q = Q.dropna(); W = Q.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
rd, _, _, fd = run_state_machine(Q.reset_index(drop=True)); rw, _, _, fw = run_state_machine(W.reset_index(drop=True))
def atrs(df): c = df.close.values; t = np.maximum(df.high.values - df.low.values, np.maximum(np.abs(df.high.values - np.r_[c[0], c[:-1]]), np.abs(df.low.values - np.r_[c[0], c[:-1]]))); return pd.Series(t).ewm(alpha=1 / 14, adjust=False).mean().values
o, c, l = Q.open.values, Q.close.values, Q.low.values; A = atrs(Q); n = len(Q); idx = Q.index
wk_of_day = np.zeros(n, int)  # last COMPLETED weekly regime as of each daily close (weekly bar completes on its Friday close)
wi = W.index; k = np.searchsorted(wi.values, idx.values, side='left')  # weekly bar containing day
for i in range(n): wk_of_day[i] = rw[k[i] - 1] if (k[i] < len(wi) and wi[k[i]] != idx[i]) else (rw[k[i]] if k[i] < len(wi) else rw[-1]) if k[i] > 0 or wi[min(k[i], len(wi) - 1)] == idx[i] else 0
# weekly regime visible at the daily close: if the day IS the weekly bar's last day (Friday) its regime is complete
wk_flip_days = {}
for i, d in fw:
    if d == -1: wk_flip_days[int(idx.searchsorted(wi[i]))] = 1
dflip_bear = {i for i, d in fd if d == -1}; dflip_bull = {i for i, d in fd if d == 1}
def run(variant):
    rows = []; i = 1; pos = None
    while i < n - 1:
        sig = False
        if variant == 0: sig = i in dflip_bear and wk_of_day[i - 1] == -1 if False else (i in dflip_bear and (rw[np.searchsorted(wi.values, idx[i].to_datetime64(), side='left') - 1] == -1))
        elif variant == 1: sig = (i in dflip_bear and (rw[np.searchsorted(wi.values, idx[i].to_datetime64(), side='left') - 1] == -1)) or (i in wk_flip_days and rd[i] == -1)
        elif variant == 2: sig = wk_of_day[i] == -1 and rd[i] == -1
        elif variant == 3: sig = i in wk_flip_days
        if not sig: i += 1; continue
        e = i + 1; ml = np.inf; xi = None
        for t in range(e, n):
            ml = min(ml, l[t])
            if variant == 3 and (t in {int(idx.searchsorted(wi[j])) for j, d in fw if d == 1}): xi = min(t + 1, n - 1); break
            if variant != 3 and rd[t] == 1: xi = min(t + 1, n - 1); break
            if c[t] > ml + 3 * A[t]: xi = min(t + 1, n - 1); break
        if xi is None: xi = n - 1
        rows.append((idx[e], idx[xi], o[e], o[xi], -(o[xi] / o[e] - 1) - 4e-4)); i = xi
    return pd.DataFrame(rows, columns=['t0', 't1', 'p0', 'p1', 'ret'])
def st(E):
    pos = np.zeros(n)
    for a, b in zip(E.t0, E.t1): pos[idx.get_loc(a):idx.get_loc(b)] = -1
    r = (Q.open.shift(-1) / Q.open - 1).fillna(0).values; cst = np.abs(np.diff(np.r_[0, pos])) * 2e-4; s = pd.Series(pos * r - cst, index=idx); e = (1 + s).cumprod(); y = (idx[-1] - idx[0]).days / 365.25
    return f'CAGR {e.iloc[-1] ** (1 / y) - 1:5.1%}, DD {(e / e.cummax() - 1).min():6.1%}'
for v, nm in ((0, 'V0 frozen (new daily bear flip, weekly bear)'), (1, 'V1 V0 + join on weekly flip when daily already bear'), (2, 'V2 state entry (weekly bear & daily bear, re-enter)'), (3, 'V3 weekly bear flip only')):
    E = run(v); ep = lambda a, b: E[(E.t0 >= a) & (E.t0 < b)]
    print(f'{nm:55s} n {len(E):3d}, mean {E.ret.mean():+6.2%}, win {(E.ret > 0).mean():3.0%}, short-only {st(E)} | 2000-03 n{len(ep("2000","2004"))} {ep("2000","2004").ret.sum():+.1%} | 2008 n{len(ep("2007","2010"))} {ep("2007","2010").ret.sum():+.1%} | 2022-23 n{len(ep("2022","2024"))} {ep("2022","2024").ret.sum():+.1%} | 2025-26 n{len(ep("2025","2027"))} {ep("2025","2027").ret.sum():+.1%}')
