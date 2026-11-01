"""Nov 1 2026: 40 matched perturbed runs for the two best basket setups: shorts (SPY z20 > 2) sharing the 6 long slots (no extra exposure),
and shorts (SPY z50 > 1.5) in their own 6-slot basket at 1/6 each (gross up to 200%). Metric: return per unit of drawdown."""
exec(open('/home/claude/fartpie/xs/baskets.py').read().split("L0 = dret(")[0])
rng = np.random.default_rng(21); R = {'live': [], 'share z20>2': [], 'own6 z50>1.5': [], 'share z50>1.5': []}
s20 = Sh[(Sh.mz20 > 2).fillna(False)]; s50 = Sh[(Sh.mz50 > 1.5).fillna(False)]
for i in range(41):
    dl = rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool); ds = rng.random(len(Sh)) < .05 if i else np.zeros(len(Sh), bool)
    l = Ls[~dl]; dsS = pd.Series(ds, Sh.index)
    for k, r in (('live', dret(simw(l))), ('share z20>2', dret(simw(pd.concat([l, s20[~dsS[s20.index]]])))),
                 ('own6 z50>1.5', dret(simw(l)) + dret(simw(s50[~dsS[s50.index]], n=6))), ('share z50>1.5', dret(simw(pd.concat([l, s50[~dsS[s50.index]]]))))):
        R[k].append([(c, d) for c, d in stat(r.cumsum())])
R = {k: np.array(v) for k, v in R.items()}
rd = lambda a: a[:, :, 0] / -a[:, :, 1]
for k in R:
    a = R[k]; print(f'  {k:<14} exact {a[0, 0, 0]:+.1%}/{a[0, 1, 0]:+.1%}  median return {np.median(a[1:, 0, 0]):+.1%}/{np.median(a[1:, 1, 0]):+.1%}  median DD {np.median(a[1:, 0, 1]):.0%}/{np.median(a[1:, 1, 1]):.0%}  return/DD {np.median(rd(a)[1:, 0]):.2f}/{np.median(rd(a)[1:, 1]):.2f}' +
          ('' if k == 'live' else f'   better return/DD than live in {(rd(a)[1:] > rd(R["live"])[1:]).mean(axis=0)[0]:.0%}/{(rd(a)[1:] > rd(R["live"])[1:]).mean(axis=0)[1]:.0%} of runs'), flush=True)
