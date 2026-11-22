"""PRE-REGISTERED Nov 22 2026 (frozen): QQQ SHORT under the weekly-bear state, report 351B.  Nothing here may be changed after registration; trades after 2026-11-22 are the out-of-sample test.
RULE: daily QQQ bars (Yahoo, adjusted).  Signal = the DAILY vector (xs/vector_pine.run_state_machine) flips bear on a completed bar AND the last completed WEEKLY vector regime (weeks ending Friday) is bear.
 ENTRY: short at the next open.  EXIT (first of): the daily vector flips bull (exit at the next open) | a daily CLOSE above (lowest low since entry + 3 x ATR14) (exit at the next open).  Costs 2 bp per side.  One position, no pyramiding.
IN-SAMPLE (1999-2026-11-22): 12 trades, mean +8.2%, win 58%, short-only +2.9% CAGR / DD -24.6%; 2000-03 +87% (4), 2008 +4.7% (1), 2022-23 +9.2% (6, win 33%).  Pass criterion declared now: forward trades are judged on 'mean per trade > 0 and trail exit beats vector exit', but with <= 3 trades expected in any year this is a monitoring record, not a significance test.
USAGE: python xs/forward_qqq_short.py   -> prints status, appends one row per day to data/forward_qqq_short.csv (date, status, trail stop, last close)."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine; import yfinance as yf
Q = yf.download('QQQ', start='1999-03-10', auto_adjust=True, progress=False); Q.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Q.columns]; Q = Q.dropna()
W = Q.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def regat(df, dates):
    regs = run_state_machine(df.reset_index(drop=True))[0]; k = df.index.searchsorted(pd.DatetimeIndex(dates).normalize().values, side='left') - 1; return np.where(k >= 0, regs[np.clip(k, 0, len(regs) - 1)], 0)
def atrs(df): c = df.close.values; t = np.maximum(df.high.values - df.low.values, np.maximum(np.abs(df.high.values - np.r_[c[0], c[:-1]]), np.abs(df.low.values - np.r_[c[0], c[:-1]]))); return pd.Series(t).ewm(alpha=1 / 14, adjust=False).mean().values
regs, lv, _, fl = run_state_machine(Q.reset_index(drop=True)); wk = regat(W, Q.index); wk_now = int(run_state_machine(W.reset_index(drop=True))[0][-1]); o, c, l = Q.open.values, Q.close.values, Q.low.values; A = atrs(Q); n = len(Q); rows = []; status = 'FLAT (needs a daily bull flip then a new daily bear flip while the weekly is bear)'; stop = np.nan; last = pd.Timestamp(Q.index[-1]).date()
for i, d in fl:
    if d != -1 or wk[i] != -1: continue
    pending = i + 1 >= n
    if pending: status = f'PENDING ENTRY at next open (daily bear flip on {Q.index[i].date()}, weekly bear)'; break
    nxt = [j for j, dd in fl if dd == 1 and j > i]; j = nxt[0] if nxt else None; ml = np.inf; xi = None
    for t in range(i + 1, n):
        ml = min(ml, l[t]); stop_t = ml + 3 * A[t]
        if j is not None and t >= j: xi = min(j + 1, n - 1); break
        if c[t] > stop_t: xi = t + 1; break
    if xi is None or xi >= n:
        status = f'OPEN SHORT since {Q.index[i + 1].date()} open {o[i + 1]:.2f}'; stop = ml + 3 * A[n - 1]; rows.append((Q.index[i + 1], pd.NaT, o[i + 1], np.nan, np.nan)); continue
    rows.append((Q.index[i + 1], Q.index[xi], o[i + 1], o[xi], -(o[xi] / o[i + 1] - 1) - 4e-4))
E = pd.DataFrame(rows, columns=['entry', 'exit', 'p0', 'p1', 'ret']); print(E.tail(8).to_string()); print(f'trades {E.ret.notna().sum()}, mean {E.ret.mean():+.2%}, win {(E.ret > 0).mean():.0%}')
print(f'last bar {last} close {c[-1]:.2f} | daily regime {regs[-1]:+d} | weekly regime now {wk_now:+d} | status: {status}' + (f' | trail stop close-basis {stop:.2f}' if stop == stop else ''))
f = 'data/forward_qqq_short.csv'; row = pd.DataFrame([dict(date=str(last), close=round(float(c[-1]), 2), daily_regime=int(regs[-1]), weekly_regime=wk_now, status=status, trail=round(float(stop), 2) if stop == stop else '')])
old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(); pd.concat([old[old.date != str(last)] if len(old) else old, row]).to_csv(f, index=False)
