"""Oct 27 2026: one grade per signal from the edges that held up, display only. Points (all known at entry):
 +2 SPY downtrend (daily vector bear AND under its 200-day)      +1 / -1 conditions score top / bottom third (xs/score_ref.py, 6 inputs)
 +1 options ratio in its top third (2019-21 admitted cut)        -1 stretched entry (> 107% of today's HAR range used, xs/rangez)
Grades by total: >= 3 A+, 2 Better, 1 Good, 0 Iffy, <= -1 Bad. Checked on admitted trades, 2019-21 and 2022-26 separately."""
import numpy as np, pandas as pd, glob, json
ms = open('xs/market_signal.py').read(); exec(ms[:ms.index("PER = [('2016-06-01'")]); MK = OUT
exec(open('xs/edge_combo_oos.py').read().split("print('picked on 2019-21 only")[0])
R = json.load(open('live_data/score_ref.json'))
def pc(c, sg):
    ref = np.array(R['ref'][c]); v = L[c].values; p = np.searchsorted(ref, v) / len(ref); p = np.where(np.isnan(v.astype(float)), .5, p); return p if sg > 0 else 1 - p
L['score'] = sum(pc(c, s) for c, s in R['signs'].items()) / len(R['signs']); t1, t2 = R['cuts']
S = MK['SPY']; L['down'] = ((S.vec_d < 0) & (S.sma200 < 0)).astype(float).reindex(L.d.values, method='ffill').values > 0
RV = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/rv/*.parquet')); RV = RV[RV['var'] == 'live'][['tk', 'h1', 'entry', 'used']]
L = L.drop(columns=[c for c in ('used',) if c in L.columns]).merge(RV.drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left')
rq = L[L.inbook & (L.half == '19-21')].ratio.quantile(2 / 3)
L['pts'] = 2 * L.down + np.where(L.score > t2, 1, np.where(L.score < t1, -1, 0)) + (L.ratio >= rq) - (L.used > 1.07)
L['grade'] = pd.cut(L.pts, [-9, -.5, .5, 1.5, 2.5, 9], labels=['Bad', 'Iffy', 'Good', 'Better', 'A+'])
B = L[L.inbook]
g = B.groupby(['grade', 'half']).ret.agg(['mean', 'size', lambda r: (r > 0).mean(), lambda r: r.quantile(.95)]).unstack(); g.columns = [f'{a} {b}' for a, b in g.columns]
for c in g.columns:
    if 'mean' in c or '<lambda_1>' in c: g[c] *= 100
print('admitted trades by grade (avg trade %, count, win rate, 95th pct trade %):'); print(g.round(2).to_string())
print('\nshare of all signals by grade:', L.grade.value_counts(normalize=True).round(2).to_dict())
print('\nsimpler three-level grade:')
L['g3'] = np.where(L.down, 'A+ (SPY downtrend)', np.where(L.score > t2, 'B (conditions high)', np.where(L.score < t1, 'D (conditions low)', 'C (normal)')))
B = L[L.inbook]; g = B.groupby(['g3', 'half']).ret.agg(['mean', 'size', lambda r: (r > 0).mean()]).unstack(); g.columns = [f'{a} {b}' for a, b in g.columns]
for c in g.columns:
    if 'mean' in c: g[c] *= 100
print(g.round(2).to_string())
for y in range(2019, 2027):
    b = B[B.d.dt.year == y]; print(y, ' '.join(f"{k[:2]}:{b[b.g3 == k].ret.mean() * 100:+.2f}({(b.g3 == k).sum()})" for k in sorted(B.g3.unique())))
