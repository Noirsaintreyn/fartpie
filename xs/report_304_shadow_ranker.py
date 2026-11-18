"""Nov 18 2026 report 304. Shadow ranker robustness (pre-set, no tuning): rank = brank - 2*(TREND+HI52+SIZE) - 5*(1-lowvol pct)  [MOM dropped: hurt in 300].
Reports per-year delta vs live, month-block bootstrap of the daily-return difference, and a LEAD test: index variance-ratio (low tercile) as a x1.25 size gate."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/report_303_vol_mechanism.py').read().split('# ---- (a)')[0])
exec('def sim' + open('/home/claude/fartpie/xs/report_303_vol_mechanism.py').read().split('def sim')[1].split("brank = B.brank")[0])
brank = B.brank.values + 1e-6 * B.ratio.rank(ascending=False).values; volr = B.vol20.rank(pct=True).values; n3 = B[['TREND', 'HI52', 'SIZE']].sum(axis=1).values
live = sim(brank)[0]; new = sim(brank - 2 * n3 - 5 * (1 - volr))[0]; t3 = sim(brank - 2 * n3)[0]
print('live      ', st2(live)); print('3sig K=2  ', st2(t3)); print('shadow    ', st2(new))
yr = lambda s: ((1 + s).groupby(s.index.year).prod() - 1) * 100; d = yr(new) - yr(live); print('per-year delta pts:', {int(k): round(v, 1) for k, v in d.items()}, '| improved', int((d > 0).sum()), 'of', len(d))
diff = (new - live); diff = diff[diff.index >= '2019-01-01']; mo = diff.groupby(diff.index.to_period('M')).sum(); rng = np.random.default_rng(1)
bs = np.array([rng.choice(mo.values, len(mo)).sum() for _ in range(5000)]); print(f'monthly sum of daily diff: mean {mo.mean()*100:+.2f}%/mo, bootstrap P(total<=0) = {(bs<=0).mean():.2f}')
for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
    m = mo[(mo.index.start_time >= a) & (mo.index.start_time < b)]; bs = np.array([rng.choice(m.values, len(m)).sum() for _ in range(5000)]); print(f'   {a[:4]}-{b[:4]}: {m.mean()*100:+.2f}%/mo  P(<=0) {(bs<=0).mean():.2f}')
g = B.idx_vr.quantile([1 / 3]).iloc[0] if False else B.loc[B.tr, 'idx_vr'].quantile(1 / 3)
lo = lambda r: 1.25 if r.idx_vr <= g else 1.0
print('LEAD: shadow + x1.25 size when index VR in low tercile ', st2(sim(brank - 2 * n3 - 5 * (1 - volr), lo)[0]))
print('2026 only (live vs shadow, %):', round(yr(live).get(2026, np.nan), 1), round(yr(new).get(2026, np.nan), 1))
d3 = (t3 - live); d3 = d3[d3.index >= '2019-01-01']; m3 = d3.groupby(d3.index.to_period('M')).sum(); print('3sig K=2 per-year delta:', {int(k): round(v, 1) for k, v in (yr(t3) - yr(live)).items()})
for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
    m = m3[(m3.index.start_time >= a) & (m3.index.start_time < b)]; bs = np.array([rng.choice(m.values, len(m)).sum() for _ in range(5000)]); print(f'   3sig {a[:4]}-{b[:4]}: {m.mean()*100:+.2f}%/mo  P(<=0) {(bs<=0).mean():.2f}')
