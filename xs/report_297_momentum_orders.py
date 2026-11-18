"""Nov 18 2026 report 297 (Rey: 20-day or 5-day momentum). Pick order for simultaneous flips: prior 5d / 20d return (own and minus SPY), HIGH first (declared), plus short-term REVERSAL variant 5d LOW first as the opposite hypothesis. 6-slot sim, CAGR/DD 2019-22 | 2023-26; per-trade returns by third."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B = A.merge(W, on=['tk', 'entry']); B = B[(B.ratio >= 1.5) & (B.entry >= '2019-01-01') & (B.exit < END)].copy(); B['day'] = B.entry.dt.normalize(); B['h1'] = False; spy = np.log(J.SPY.copy()); spy.index = pd.DatetimeIndex(spy.index)
FE = []
for tk in B.tk.unique():
    Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); c = np.log(Y.close); sp = spy.reindex(Y.index).ffill()
    f = pd.DataFrame({'m5': c.diff(5), 'm20': c.diff(20), 'm60': c.diff(60), 'rs5': c.diff(5) - sp.diff(5), 'rs20': c.diff(20) - sp.diff(20)}).shift(1); f['tk'] = tk; f['day'] = f.index; FE.append(f)
B = B.merge(pd.concat(FE), on=['tk', 'day'], how='left').dropna(subset=['m5', 'm20', 'rs20']); B['per'] = np.where(B.entry < '2023-01-01', 'train', 'test'); print('trades', len(B))
print('per-trade mean return by third (train cuts): train | test')
for f in ('m5', 'm20', 'm60', 'rs5', 'rs20'):
    cut = B[B.per == 'train'][f].quantile([1 / 3, 2 / 3]).values; B['_t'] = np.where(B[f] <= cut[0], 'lo', np.where(B[f] <= cut[1], 'mid', 'hi')); r = B.groupby(['per', '_t']).ret.agg(['mean', 'size'])
    print(f'  {f:5s} ' + ' | '.join(f'{p}: ' + ' '.join(f"{t} {r.loc[(p, t), 'mean']:+.2%}" for t in ('lo', 'mid', 'hi')) for p in ('train', 'test')))
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    return ' | '.join(out)
g = B.groupby('entry'); print('\n6-slot pick order (CAGR/DD 2019-22 | 2023-26):')
for lab, key in (('ratio (live)', -B.ratio), ('5d mom high first', -B.m5), ('20d mom high first', -B.m20), ('60d mom high first', -B.m60), ('5d rel SPY high', -B.rs5), ('20d rel SPY high', -B.rs20), ('5d LOW first (reversal)', B.m5),
                 ('ratio + 20d ranks', g.ratio.rank(ascending=False) + g.m20.rank(ascending=False)), ('ratio + 5d ranks', g.ratio.rank(ascending=False) + g.m5.rank(ascending=False)), ('20d first, ratio ties', -B.m20.round(2) * 1 - 1e-6 * B.ratio)):
    T0 = B.assign(_k=np.asarray(key)).sort_values(['entry', '_k', 'tk'], kind='stable'); tk_, _ = taken(); s_, _ = daily(tk_, 6); print(f'  {lab:24s} {st2(s_)}', flush=True)
