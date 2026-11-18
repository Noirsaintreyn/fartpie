"""Nov 18 2026: 2026 forward test, price-only legs. Splices LSE/drive NQ 5m bars after 2026-01-30 into nq5_real (nq5_real_ext.parquet), reruns M1 noise momentum + R1 80% rule
(dte_engines) and reports pts/trade by year incl. 2026, split by QQQ net-GEX sign (gamma file runs to 2026-10-08)."""
import pandas as pd, numpy as np
_rp = pd.read_parquet
pd.read_parquet = lambda p, *a, **k: _rp(p.replace('nq5_real.parquet', 'nq5_real_ext.parquet'), *a, **k)
_e = open('/home/claude/fartpie/xs/dte_engines.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
DY = DY[DY.index != '2026-11-16']
gneg = DY.gex < 0
for nm, T in (('M1', m1()), ('R1', r1()), ('R1p', r1(True))):
    T = T[T.day.isin(DY.index) & (T.day != '2026-11-16')]; T['y'] = T.day.dt.year; T['g'] = T.day.map(gneg)
    for lab, S in (('all', T), ('negG', T[T.g == True]), ('posG', T[T.g == False])):
        a = S.groupby('y').pts.agg(['count', 'mean', 'sum']); print(nm, lab, ' '.join(f"{y%100:02d}:{r['count']:.0f}/{r['mean']:+.1f}" for y, r in a.iterrows()))
    q = T[T.day >= '2026-01-31']; m26 = q.groupby(q.day.dt.month).pts.sum().round(0).to_dict(); print(nm, '2026 post-Jan30 monthly pts', m26)
    T.to_parquet(f'/tmp/claude-0/ml/dte26_{nm}.parquet')
print('gex NaN share 2026:', DY.gex['2026'].isna().mean().round(2), 'last gex', DY.gex.dropna().index[-1])
