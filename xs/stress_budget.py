"""Nov 1 2026: stress sizing with a FIXED budget (never more than 100% invested, no margin).
 cash-up    stress trades sized 1.5/6 but only from idle cash (gross capped at 100%; if not enough cash, take what fits... or skip)
 reserve5   normal times max 5 slots (83%); in stress the 6th slot opens
 reserve7   normal 6 slots at 1/7 (86%); stress trades 1.5/7, gross capped at 100%
Stress = Rebound, jump-model bear, A+, or any of the three. 40 matched runs, return per drawdown vs live."""
exec(open('/home/claude/fartpie/xs/stress_size.py').read().split("rng = np.random.default_rng(41)")[0].replace("print(", "(lambda *a, **k: None)("))
ST2 = {'Rebound': ST['Rebound'], 'JM bear': ST['JM bear'], 'A+': ST['A+ (SPY downtrend)'], 'any of the 3': ST['Rebound'] | ST['JM bear'] | ST['A+ (SPY downtrend)']}
def simb(Z, stress, mode, brk=.08):
    o = np.lexsort((Z.tk.values, Z.h1.values, -Z.ratio.values, Z.entry.values)); E, X, R, Sx = Z.entry.values[o], Z.exit.values[o], Z.ret.values[o], stress[o]
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_, s in zip(E, X, R, Sx):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q[0] > e]; gross = sum(w for _, w in op); n = len(op)
        if mode == 'live': cap, w = 6, 1 / 6
        elif mode == 'cash-up': cap, w = 6, (1.5 / 6 if s else 1 / 6)
        elif mode == 'reserve5': cap, w = (6 if s else 5), 1 / 6
        elif mode == 'reserve7': cap, w = 6, (1.5 / 7 if s else 1 / 7)
        w = min(w, 1 - gross)
        if n >= cap or w < 1 / 12: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append((x, w)); pend.append((x, r_ * w))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
rng = np.random.default_rng(43); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(stress, mode): return np.array([[v for cd in stat(dret(simb(Ls[~dl], stress[~dl], mode)).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
B0 = ev(np.zeros(len(Ls), bool), 'live'); line('live (6 x 1/6)', B0)
for nm, s in ST2.items():
    print(f'\n{nm}  (on for {s.mean():.0%} of signals)')
    for mode in ('cash-up', 'reserve5', 'reserve7'): line(mode, ev(s, mode), B0)
