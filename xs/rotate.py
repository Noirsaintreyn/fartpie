"""Nov 1 2026: rotation - when all 6 slots are full and a new flip arrives, drop the weakest open position for it (sold at that bar's open).
 worst<0        drop the open position with the lowest unrealized P&L, only if it is negative
 worst<-1%      same, only if it is below -1%
 lower ratio    drop the open position with the lowest options ratio, if the new flip's ratio is higher (and the old one is not up > +2%)
 stale          drop the oldest position if it is older than 3 days and up < +1%
40 matched perturbed runs vs live."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
OPN = {}
for tk in Ls.tk.unique():
    b = bars(tk)
    if b is not None: s = b[0]; OPN[tk] = (s.index.values, s.values)
def px_at(tk, t):
    ix, v = OPN[tk]; k = np.searchsorted(ix, t)
    return v[k] if k < len(v) else np.nan
def simr(Z, mode, n=6, brk=.08, C2=2e-4):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; pend = []; swaps = 0
    for tk, e, x, r_, ra in zip(Z.tk.values, Z.entry.values, Z.exit.values, Z.ret.values, Z.ratio.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, *_ in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q['x'] > e]
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        if len(op) >= n:
            if mode == 'live' or tk not in OPN: continue
            for q in op: q['u'] = px_at(q['tk'], e) / q['p0'] - 1
            cand = None
            if mode in ('worst<0', 'worst<-1%'):
                q = min(op, key=lambda q: q['u'] if q['u'] == q['u'] else 9); lim = 0 if mode == 'worst<0' else -.01
                if q['u'] == q['u'] and q['u'] < lim: cand = q
            elif mode == 'lower ratio':
                q = min(op, key=lambda q: q['ra'])
                if ra > q['ra'] and q['u'] == q['u'] and q['u'] < .02: cand = q
            elif mode == 'stale':
                q = min(op, key=lambda q: q['e'])
                if (e - q['e']) > np.timedelta64(3, 'D') and q['u'] == q['u'] and q['u'] < .01: cand = q
            if cand is None: continue
            op.remove(cand); pend = [p for p in pend if p[2] is not cand]; rr = cand['u'] - C2
            real += np.log1p(rr / n); hist.append((e, real)); win.append((e, real)); swaps += 1
        p0 = px_at(tk, e) if tk in OPN else np.nan
        q = dict(tk=tk, e=e, x=x, p0=p0 if p0 == p0 else 1.0, ra=ra); op.append(q); pend.append((x, r_, q))
    for xt, rr, _ in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    pend = [(a, b_) for a, b_, *_ in pend]
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last(), swaps
_old = simr
def simr2(Z, mode):                                                               # wrapper: pend tuples carry the position dict
    return _old(Z, mode)
rng = np.random.default_rng(53); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(mode):
    out, sw = [], []
    for dl in DR:
        eq, s = simr(Ls[~dl], mode); sw.append(s); out.append([v for cd in stat(dret(eq).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])])
    return np.array(out), int(np.median(sw))
def line(nm, a, b=None, sw=0):
    m = np.median(a[1:], axis=0); s = f'  {nm:<14} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}   swaps {sw}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
B0, _ = ev('live'); line('live', B0)
for mode in ('worst<0', 'worst<-1%', 'lower ratio', 'stale'):
    a, sw = ev(mode); line(mode, a, B0, sw)
