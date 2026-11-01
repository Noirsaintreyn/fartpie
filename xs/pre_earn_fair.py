"""Nov 1 2026: the one 'bigger move' trait with per-trade edge in both periods - flips 2-3 trading days before earnings (+2.79% / +1.16% vs
+0.6% / +0.3%). Give them priority for slots, judged with 40 matched perturbed runs (return per drawdown and return)."""
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
Ls = Ls.reset_index(drop=True); L = Ls
src2 = open('/home/claude/fartpie/xs/pre_earn_flip.py').read(); exec(src2[src2.index('k = np.full'):src2.index("L['k'] = k")])
Ls['k'] = k
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<44} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
rng = np.random.default_rng(47); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
pr = Ls.ratio.groupby(Ls.entry.dt.year).rank(pct=True)
def ev(key): 
    Z0 = Ls.copy(); Z0['ratio'] = key.values
    return np.array([[v for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
B0 = ev(Ls.ratio); line('live (ratio)', B0)
for lo, hi in ((2, 3), (1, 3), (2, 5)):
    line(f'flips {lo}-{hi} days before earnings first', ev(pr + Ls.k.between(lo, hi) * 10), B0)
