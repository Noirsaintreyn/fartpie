"""Nov 1 2026: is the SPY z-score short zone robust to the threshold? Shared slots and own 6-slot basket, absolute z thresholds."""
exec(open('/home/claude/fartpie/xs/baskets.py').read().split("L0 = dret(")[0])
L0 = dret(simw(Ls)); print(f'  {"live":<28}{fmt(stat(L0.cumsum()))}')
for c, ths in (('mz20', (1.5, 1.75, 2.0, 2.25, 2.5)), ('mz50', (1.0, 1.25, 1.5, 1.75, 2.0))):
    for t in ths:
        s = Sh[(Sh[c] > t).fillna(False)]; a = s.groupby(np.where(s.entry.dt.year <= 2021, 0, 1)).ret.mean()
        print(f'  SPY {c[1:]} > {t:<5} n {len(s):>5} avg {a.get(0, np.nan) * 100:+.2f}/{a.get(1, np.nan) * 100:+.2f}  shared: {fmt(stat(dret(simw(pd.concat([Ls, s]))).cumsum()))}   own6: {fmt(stat((L0 + dret(simw(s, n=6))).cumsum()))}', flush=True)
