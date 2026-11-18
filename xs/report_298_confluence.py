"""Nov 18 2026 report 298 (Rey: rank a flip higher when several signals align). DECLARED BEFORE RESULTS - five binary signals known before entry, thresholds from TRAIN (2019-22) only:
 S_ratio: ratio >= train median | S_ldv: cross-sectional $-volume percentile >= 2/3 | S_mom: 20d return >= train upper third | S_h1: 1h vector already bull at the flip | S_room: room to ratchet stop <= train median (tight stop)
SCORE = count (0-5). Tests: per-trade return by score (train | test), and 6-slot pick order = SCORE high first, ratio breaks ties (live = ratio only). Also SCORE>=3 as the entry filter (fewer, better trades?)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B = A.merge(W, on=['tk', 'entry']); B = B[(B.ratio >= 1.5) & (B.entry >= '2019-01-01') & (B.exit < END)].copy(); B['day'] = B.entry.dt.normalize(); B['h1'] = False; FE = []
for tk in B.tk.unique():
    Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); dv = Y.close * Y.volume
    f = pd.DataFrame({'ldv': np.log(dv.rolling(50).mean()), 'm20': np.log(Y.close).diff(20)}).shift(1); f['tk'] = tk; f['day'] = f.index; FE.append(f)
FE = pd.concat(FE); FE['ldvp'] = FE.groupby('day').ldv.rank(pct=True); B = B.merge(FE[['tk', 'day', 'ldvp', 'm20']], on=['tk', 'day'], how='left').dropna(subset=['ldvp', 'm20', 'room']); B['per'] = np.where(B.entry < '2023-01-01', 'train', 'test'); tr = B[B.per == 'train']
B['S_ratio'] = B.ratio >= tr.ratio.median(); B['S_ldv'] = B.ldvp >= 2 / 3; B['S_mom'] = B.m20 >= tr.m20.quantile(2 / 3); B['S_h1'] = B.s1 > .5; B['S_room'] = B.room <= tr.room.median()
B['score'] = B[['S_ratio', 'S_ldv', 'S_mom', 'S_h1', 'S_room']].sum(axis=1); print('trades', len(B)); print('\nper-trade mean return by SCORE (n): train | test')
r = B.groupby(['per', 'score']).ret.agg(['mean', 'size']); print('  ' + '\n  '.join(f"score {s}: train {r.loc[('train', s), 'mean']:+.2%} (n{int(r.loc[('train', s), 'size'])}) | test {r.loc[('test', s), 'mean']:+.2%} (n{int(r.loc[('test', s), 'size'])})" for s in range(6) if ('train', s) in r.index and ('test', s) in r.index))
for c in ('S_ratio', 'S_ldv', 'S_mom', 'S_h1', 'S_room'): rr = B.groupby(['per', c]).ret.mean(); print(f'  {c:8s} off/on: train {rr[("train", False)]:+.2%}/{rr[("train", True)]:+.2%} | test {rr[("test", False)]:+.2%}/{rr[("test", True)]:+.2%}')
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    return ' | '.join(out)
print('\n6-slot sim (CAGR/DD 2019-22 | 2023-26):')
for lab, X, key in (('ratio (live)', B, -B.ratio), ('SCORE high, ratio ties', B, -B.score * 1000 - B.ratio), ('SCORE>=3 only, ratio order', B[B.score >= 3], -B[B.score >= 3].ratio), ('SCORE>=2 only, SCORE order', B[B.score >= 2], -B[B.score >= 2].score * 1000 - B[B.score >= 2].ratio)):
    T0 = X.assign(_k=np.asarray(key)).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, sk_ = taken(); s_, w_ = daily(tk_, 6); print(f'  {lab:28s} {st2(s_)} | trades taken {len(tk_)}, avg slots used {w_.mean():.0%}', flush=True)
