"""Nov 1 2026: in stress, bigger trades or a bigger basket? Stress definitions (prior close): dn = SPY daily vector bear & under 200d (A+),
Rebound (Garg et al.), Panic (Daniel-Moskowitz), JM-0 (jump-model bear), Fearful (VIX > 1y median), vote = 2+ of the five.
Modes while stress is on at entry:  size1.5 = 6 slots, 1.5/6 per trade (gross up to 150%)  |  basket9 = up to 9 slots at 1/6 (150%)
basket8 = 8 slots at 1/6 (133%)  |  basket9@1/9 = 9 slots at 1/9 (no extra gross, more names). Return, DD, return/DD; 40 matched runs."""
exec(open('/home/claude/fartpie/xs/regime_lit.py').read().split("FAV = {")[0].replace("print(", "(lambda *a, **k: None)("))
d = Ls.entry.dt.normalize()
mk2 = mk.reindex(d.values, method='ffill')
ST = {'A+ (SPY downtrend)': mk2.dn.fillna(False).astype(bool).values, 'Rebound': (Ls.A == 'Rebound').values, 'Panic': (Ls.B == 'Panic').values,
      'JM bear': (Ls.C == 'JM-0').values, 'Fearful': (Ls.D == 'Fearful').values}
ST['vote 2+'] = np.sum([v for v in ST.values()], axis=0) >= 2
def simd(Z, stress, mode, n=6, brk=.08):
    o = np.lexsort((Z.tk.values, Z.h1.values, -Z.ratio.values, Z.entry.values)); E, X, R, Sx = Z.entry.values[o], Z.exit.values[o], Z.ret.values[o], stress[o]
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_, s in zip(E, X, R, Sx):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]
        cap, w = n, 1 / n
        if s:
            if mode == 'size1.5': w = 1.5 / n
            elif mode == 'basket9': cap = 9
            elif mode == 'basket8': cap = 8
            elif mode == 'basket9@1/9': cap, w = 9, 1 / 9
        if len(op) >= cap: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_ * w))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
rng = np.random.default_rng(41); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def evalr(stress, mode):
    out = []
    for dl in DR:
        Z = Ls[~dl]; out.append([v for cd in stat(dret(simd(Z, stress[~dl], mode)).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])])
    return np.array(out)                                                          # cols: ret19, dd19, rd19, ret22, dd22, rd22
B0 = evalr(np.zeros(len(Ls), bool), 'none')
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<40} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('medians of 40 perturbed runs, 2019-21 / 2022-26'); line('live', B0)
for nm, s in ST.items():
    print(f'\n{nm}  (on for {s.mean():.0%} of signals)')
    for mode in ('size1.5', 'basket9', 'basket8', 'basket9@1/9'): line(mode, evalr(s, mode), B0)
