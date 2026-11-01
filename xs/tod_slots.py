"""Nov 1 2026: slots by time of day. Reserve R of the 6 slots for entries after time T (an entry before T needs more than R slots free);
or cap new entries per day before T. Plus avg trade by entry time."""
import os, glob, numpy as np, pandas as pd
from collections import deque
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = pd.read_parquet('/tmp/claude-0/ml/mtf_states.parquet').sort_values('entry').reset_index(drop=True)
TK = taken(L); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26'); mm = TK.entry.dt.hour * 60 + TK.entry.dt.minute
b = pd.cut(mm, [0, 570, 600, 630, 690, 780, 900], right=False, labels=['pre', '9:30', '10:00', '10:30-11:30', '11:30-13:00', '13:00+'])
print('avg trade % by entry time (trades the book takes)'); print((TK.groupby([b, 'h']).ret.agg(['size', 'mean']).unstack('h') * [1, 1, 100, 100]).round(2).to_string())
def simr(Z, a, b_, T, R=0, cap=None, n=6, brk=.08):
    Z = Z[(Z.entry >= a) & (Z.entry < b_)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; day = None; cnt = 0
    for e, x, r_ in zip(Z.entry.values, Z.exit.values, Z.ret.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]; te = pd.Timestamp(e); early = te.hour * 60 + te.minute < T
        if te.normalize() != day: day, cnt = te.normalize(), 0
        if len(op) >= n or (early and len(op) >= n - R) or (early and cap is not None and cnt >= cap): continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_)); cnt += early
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def sh(nm, **kw): print(f'  {nm:<44}' + ''.join(f'{c:>+8.1%} /{d:>4.0%}   ' for c, d in (simr(L, a, b_, **kw) for a, b_ in P)), flush=True)
print('\nbook (yearly return / max DD)'); sh('live', T=0)
for T in (600, 630, 690):
    for R in (1, 2, 3): sh(f'keep {R} slot(s) free for entries after {T // 60}:{T % 60:02d}', T=T, R=R)
for T in (600, 630):
    for c in (1, 2, 3): sh(f'max {c} new entries per day before {T // 60}:{T % 60:02d}', T=T, cap=c)
