"""Nov 18 2026 report 289: NO TREE. One declared mechanism feature per model, fixed size tilt (x1.5 in the mechanism's good third, x0.5 in its bad third, x1 middle), thirds cut on train 2019-21.
Declared before results: RA: ATRP (calm) good=low | M1: absz good=low | V15a: absz good=low | R1: trend_eff good=low (consolidation) | ON+: gap good=high | S1: expansion good=high | RULES: none.
Overnight range already harvested elsewhere so not retested. Score = sum of pts*sz*w per trade, tilted / base, train | test, plus by-year test."""
import re
src = open('/home/claude/fartpie/xs/report_288_model_environments.py').read().split("rows = []")[0]
exec(src)
SPEC = {'RA': ('ATRP', -1), 'M1': ('absz', -1), 'V15a': ('absz', -1), 'R1': ('trend_eff', -1), 'ON+': ('gap', 1), 'S1': ('expansion', 1)}
W = {'RA': 1, 'M1': .3, 'R1': .5, 'ON+': .4, 'V15a': .3, 'S1': .2}
print('model  feat   | train base -> tilt (pts*w) | test base -> tilt | test by year tilt-base')
for nm, (c, sgn) in SPEC.items():
    T = L[nm]; T = T[T.sz > 0].copy() if nm != 'ON+' else T.copy(); T['d'] = pd.to_datetime(T.day); T = T[T.d <= '2026-10-31']
    if nm == 'S1' and 'reg' in T.columns: T = T[T.reg > 0]
    t = TC[c].reindex(T.d).values; good = (t == 0) if sgn < 0 else (t == 2); bad = (t == 2) if sgn < 0 else (t == 0)
    mult = np.where(good, 1.5, np.where(bad, .5, 1.0)); mult = np.where(np.isnan(t), 1.0, mult)
    T['b'] = T.pts.values * W[nm]; T['t'] = T.b * mult
    a = T[T.d.isin(F.index[tr])]; z = T[T.d.isin(F.index[te])]
    yy = z.groupby(z.d.dt.year)[['b', 't']].sum().round(0); print(f'{nm:5s} {c:9s} | {a.b.sum():7.0f} -> {a.t.sum():7.0f} | {z.b.sum():7.0f} -> {z.t.sum():7.0f} | ' + ' '.join(f"{y%100}:{r.t-r.b:+.0f}" for y, r in yy.iterrows()))
