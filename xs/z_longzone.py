"""Nov 1 2026: the long side of the zone idea - longs entered while SPY is over-extended averaged ~0. Skip long flips when SPY z20 / z50 is high
(thresholds around the value, to check it is not a single lucky cut). Exact runs, then 40 matched runs for the main candidates."""
exec(open('/home/claude/fartpie/xs/baskets.py').read().split("L0 = dret(")[0])
TK = taken(Ls).merge(Ls[['tk', 'entry', 'mz20', 'mz50']], on=['tk', 'entry'])
for c, ths in (('mz20', (1.5, 1.75, 2.0, 2.25)), ('mz50', (1.25, 1.5, 1.75, 2.0))):
    for t in ths:
        m = TK[c] > t; a = TK[m].groupby(np.where(TK[m].entry.dt.year <= 2021, 0, 1)).ret.agg(['size', 'mean'])
        print(f'  longs taken with SPY {c[1:]} > {t}: ' + '  '.join(f'{["19-21", "22-26"][i]} {a.loc[i, "mean"] * 100:+.2f}% (n {int(a.loc[i, "size"])})' for i in a.index) +
              f'   | book skipping them: {fmt(stat(dret(simw(Ls[~(Ls[c] > t).fillna(False)])).cumsum()))}', flush=True)
print(f'  live: {fmt(stat(dret(simw(Ls)).cumsum()))}   all longs taken avg ' + '  '.join(f'{v * 100:+.2f}%' for v in TK.groupby(TK.entry.dt.year <= 2021).ret.mean()[::-1]))
