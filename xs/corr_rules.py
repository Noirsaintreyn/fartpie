"""Oct 25 2026: correlation rules with a reason instead of a hand-picked chip list (full-S&P screener, ratio >= 1.5, live rules).
Correlation = 60-day correlation of daily returns, known at the prior close.
 C1  crowding: skip a flip if it is correlated > RHO with K or more open positions (same bet already on)
 C2  contagion skip: skip a flip if a position correlated > RHO with it closed at a LOSS in the last 3 days (its peer just failed)
 C3  contagion exit: when a position closes at a loss, also sell open positions correlated > RHO with it, at that moment's price
 C4  confirmation: when flips compete for slots, prefer one whose correlated peers (> RHO) are open and in profit
 C5  correlation-aware sizing: size 1.25x if a flip's average correlation with the open book < 0.3, 0.75x if > 0.6 (diversifiers bigger)"""
from collections import deque
import glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('\nP = [(np.datetime64')[0])
exec(open('xs/exit_lab2.py').read().split('BARS = {}')[1].split('VARS =')[0].replace('def bars', 'BARS = {}\ndef bars', 1))
T = T.sort_values('entry').reset_index(drop=True)
R = CL.pct_change(); PC = {}
def corr(a, b, d):
    if a == b: return 1.0
    k = (a, b) if a < b else (b, a)
    if k not in PC:
        if a not in R.columns or b not in R.columns: PC[k] = None
        else: PC[k] = R[k[0]].rolling(60, min_periods=40).corr(R[k[1]]).shift(1)
    s = PC[k]
    if s is None: return 0.0
    v = s.asof(d); return float(v) if v == v else 0.0
def px_at(tk, t):
    B = bars(tk)
    if B is None: return None
    k = np.searchsorted(B.dt.values, t)
    return float(B.open.values[min(k, len(B) - 1)])
def sim(Z, a, b, rule=None, rho=.7, K=1, n=6, brk=.08):
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    E, X, RT, TK, PX = Z.entry.values, Z.exit.values, Z.ret.values, Z.tk.values, Z.px.values
    pos = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; losers = []; i = 0; N = len(Z)
    def settle(upto):
        nonlocal real, lvl0
        while True:
            due = [p for p in pos if p['x'] <= upto]
            if not due: return
            p = min(due, key=lambda q: q['x']); pos.remove(p)
            real += np.log1p(p['r'] * p['w']); hist.append((p['x'], real)); win.append((p['x'], real))
            if p['r'] < 0:
                losers.append((p['x'], p['tk']))
                if rule == 'C3':                                                    # sell correlated open positions now
                    d = pd.Timestamp(p['x']).normalize()
                    for q in pos:
                        if q['x'] > p['x'] and corr(q['tk'], p['tk'], d) > rho and q['p0'] == q['p0']:
                            pr = px_at(q['tk'], p['x'])
                            if pr: q['r'] = pr / q['p0'] - 1 - 2e-4; q['x'] = p['x']
    while i < N:
        e = E[i]; j = i
        while j < N and E[j] == e: j += 1                                           # all flips at this bar
        settle(e)
        cand = list(range(i, j)); d = pd.Timestamp(e).normalize()
        if rule == 'C4':
            def score(k):
                peers = [q for q in pos if corr(TK[k], q['tk'], d) > rho]
                return -sum(1 for q in peers if (px_at(q['tk'], e) or 0) > q['p0'])
            cand.sort(key=score)
        for k in cand:
            if len(pos) >= n: break
            tk = TK[k]
            if hist:
                t0 = e - np.timedelta64(28, 'D')
                while win and win[0][0] < t0: lvl0 = win.popleft()[1]
                if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
            if rule == 'C1' and sum(1 for q in pos if corr(tk, q['tk'], d) > rho) >= K: continue
            if rule == 'C2' and any(t_ >= e - np.timedelta64(3, 'D') and corr(tk, l, d) > rho for t_, l in losers[-30:]): continue
            w = 1 / n
            if rule == 'C5' and pos:
                ac = np.mean([corr(tk, q['tk'], d) for q in pos]); w = (1 / n) * (1.25 if ac < .3 else .75 if ac > .6 else 1.0)
            pos.append(dict(tk=tk, x=X[k], r=RT[k], w=w, p0=PX[k]))
        i = j
    settle(np.datetime64('2100-01-01'))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); yy = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / yy) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, **kw): print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim(T, a, b, **kw) for a, b in P)), flush=True)
import os; PART = os.getenv('PART', 'ALL')
print(f'{"correlation rules":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live (this sim)')
if PART in ('ALL', 'A'):
    for rho in (.6, .7, .8):
        for K in (1, 2): show(f'C1 skip if corr > {rho} with {K}+ open positions', rule='C1', rho=rho, K=K)
    for rho in (.5, .6, .7): show(f'C2 skip if a corr > {rho} peer just lost', rule='C2', rho=rho)
if PART in ('ALL', 'B'):
    for rho in (.5, .6, .7): show(f'C3 sell corr > {rho} peers when one loses', rule='C3', rho=rho)
    for rho in (.5, .6): show(f'C4 prefer flips whose corr > {rho} peers are winning', rule='C4', rho=rho)
    show('C5 size by correlation with the open book', rule='C5')
