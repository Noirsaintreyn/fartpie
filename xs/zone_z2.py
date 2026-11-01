"""Nov 1 2026: robustness of the two zone findings with the 40-run perturbed yardstick (same random 5% of flips dropped from both sides):
 (1) long filter: skip long flips when the stock is over-extended (z20 > 2 or z50 > 2)   (2) short overlay: short bear flips only when SPY z50 > 1.5 (or z20 > 2), 50% size."""
exec(open('/home/claude/fartpie/xs/zone_z.py').read().split("bins = [")[0])
rng = np.random.default_rng(11); NR = 40
def runs_long(mask):
    out = []
    for i in range(NR + 1):
        d = rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool); np.random.seed(i)
        out.append([c for c, _ in stat(simq(Ls[~d & mask]))])
    return np.array(out)
rng = np.random.default_rng(11); base = runs_long(np.ones(len(Ls), bool))
for nm, m in (('skip longs: stock z20 > 2', ~(Ls.sz20 > 2)), ('skip longs: stock z50 > 2', ~(Ls.sz50 > 2)), ('skip longs: z20 > 2 or z50 > 2', ~((Ls.sz20 > 2) | (Ls.sz50 > 2)))):
    rng = np.random.default_rng(11); r = runs_long(m.values); w = (r[1:] > base[1:]).mean(axis=0)
    print(f'  {nm:<34} exact {r[0, 0]:+.1%}/{r[0, 1]:+.1%}  median {np.median(r[1:, 0]):+.1%}/{np.median(r[1:, 1]):+.1%}  vs live median {np.median(base[1:, 0]):+.1%}/{np.median(base[1:, 1]):+.1%}  beats live {w[0]:.0%}/{w[1]:.0%}  (skips {(~m).mean():.1%})', flush=True)
idx = pd.bdate_range('2019-01-01', Ls.entry.max())
def ratio_runs(gate, w=.5):
    rng = np.random.default_rng(13); out = []
    for i in range(NR + 1):
        dl = rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool); ds = rng.random(len(Sh)) < .05 if i else np.zeros(len(Sh), bool)
        le = simq(Ls[~dl]).reindex(idx, method='ffill').fillna(0).diff().fillna(0)
        if gate is None: out.append([c / -d for c, d in stat(le.cumsum())]); continue
        se = simq(Sh[~ds & gate]).reindex(idx, method='ffill').fillna(0).diff().fillna(0)
        out.append([c / -d for c, d in stat((le + w * se).cumsum())])
    return np.array(out)
b = ratio_runs(None)
for nm, g in (('SPY z50 > 1.5', (Sh.mz50 > 1.5).values), ('SPY z20 > 2', (Sh.mz20 > 2).values)):
    r = ratio_runs(g); w = (r[1:] > b[1:]).mean(axis=0)
    print(f'  long + 50% shorts when {nm:<14} return/DD median {np.median(r[1:, 0]):.2f}/{np.median(r[1:, 1]):.2f} vs long alone {np.median(b[1:, 0]):.2f}/{np.median(b[1:, 1]):.2f}   better in {w[0]:.0%}/{w[1]:.0%} of matched runs', flush=True)
