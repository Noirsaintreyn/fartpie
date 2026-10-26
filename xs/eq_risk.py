import numpy as np, pandas as pd, glob
from collections import deque
exec(open('/tmp/claude-0/ml/stopbook.py').read().split("def sim(")[0])
B = S[S['var'] == 'cap_none'].copy()
MED = float(B[B.entry < '2022-01-01'].sd.median())                      # median stop distance known by 2022 (no look-ahead into 2022-26)
raw = np.clip(MED / B.sd.clip(lower=.002), .5, 1.5); NORM = float(raw[B.entry < '2022-01-01'].mean())
print(f'median stop distance {MED:.2%}, normalizer {NORM:.3f}')
def sim(Z, a, b, n=6, brk=.08, wcol=None, cap=None):
    Z = Z[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; Wt = Z[wcol].values if wcol else np.ones(len(Z))
    for e, x, r_, ww in zip(Z.entry.values, Z.exit.values, Z.ret.values, Wt):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, w in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o[0] > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        w = ww / n
        if cap is not None: w = min(w, cap - sum(o[1] for o in op))
        if w <= .01: continue
        op.append((x, w)); pend.append((x, r_, w))
    for xt, rr, w in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
    eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    d = eq.pct_change().dropna(); return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, **kw): print(f'  {nm:<56}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(B, a, b, **kw) for a, b in P)), flush=True)
B['eq'] = np.clip(MED / B.sd.clip(lower=.002), .5, 1.5) / NORM
print(f'{"":58}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('current (equal slots)')
show('equal risk (median / stop, avg 1x)', wcol='eq')
show('equal risk + total exposure capped at 100%', wcol='eq', cap=1.0)
show('equal risk + total exposure capped at 110%', wcol='eq', cap=1.1)
for tgt in (.015, .02, .025):
    r2 = np.clip(tgt / B.sd.clip(lower=.002), .5, 1.5); B['e2'] = r2 / float(r2[B.entry < '2022-01-01'].mean()); B['e2raw'] = r2
    show(f'equal risk {tgt:.1%} / stop, avg 1x, capped at 100%', wcol='e2', cap=1.0)
    show(f'equal risk {tgt:.1%} / stop, raw weights, capped at 100%', wcol='e2raw', cap=1.0)
print('shrink only (never above 1 slot):')
for tgt in (.02, .03, .04):
    B['s'] = np.clip(tgt / B.sd.clip(lower=.002), .5, 1.0); show(f'size = min(1, {tgt:.0%} / stop distance), floor 0.5', wcol='s')
