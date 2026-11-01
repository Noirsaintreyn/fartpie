"""Nov 1 2026: shorts when SPY is stretched AND VIX is low (complacency) - a grid of neighbouring thresholds so no single cut is picked.
Avg short trade per period and the long book + own 6-slot short basket (exact run)."""
exec(open('/home/claude/fartpie/xs/regime_lit.py').read().split("FAV = {")[0].replace("print(", "(lambda *a, **k: None)("))
vp = vx.rolling(252).rank(pct=True).shift(1); Sh['vp'] = vp.reindex(Sh.entry.dt.normalize().values, method='ffill').values
L0 = dret(simw(Ls)); print(f'live {fmt(stat(L0.cumsum()))}')
for zt in (1.5, 1.75, 2.0, 2.25):
    for vq in (1.0, .5, .25):
        g = ((Sh.mz20 > zt) & (Sh.vp <= vq)).fillna(False); s = Sh[g]; a = s.groupby(s.entry.dt.year <= 2021).ret.mean()
        print(f'  SPY z20 > {zt:<5} VIX pct <= {vq:<4} n {len(s):>5}  avg {a.get(True, np.nan) * 100:+.2f}% / {a.get(False, np.nan) * 100:+.2f}%   long + own6 shorts: {fmt(stat((L0 + dret(simw(s, n=6))).cumsum()))}', flush=True)
