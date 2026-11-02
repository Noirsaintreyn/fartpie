"""Nov 2 2026 follow-up: flips right after a burst of big overnight gaps (last 5 days' mean |gap| vs last 60) were the worst in both periods.
Skip them, at several fixed cut-offs (1.5x, 2x, 2.5x) so no single cut is picked. 40 matched runs."""
exec(open('/home/claude/fartpie/xs/ovn_vol.py').read().split("pr = lambda s")[0].replace("print(", "(lambda *a, **k: None)("))
rng = np.random.default_rng(107); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(Z0): return np.array([[v_ for cd in stat(dret(simw(Z0[~dl[Z0.index]])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in [pd.Series(d, Ls.index) for d in DR]])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<34} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
B0 = ev(Ls); line('live', B0)
for t in (1.5, 2.0, 2.5):
    m = Ls.on_abs5 > t; line(f'skip if recent gaps > {t}x normal ({m.mean():.1%})', ev(Ls[~m]), B0)
