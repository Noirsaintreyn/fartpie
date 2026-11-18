"""Report 295b: tests on the rebuilt flip table (xs/report_295_flip_quality.py). See that file for the declared features/directions. Train 2019-22, test 2023-26."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B = A.merge(W, on=['tk', 'entry'], how='inner'); B = B[(B.ratio >= 1.5) & (B.exit < END) & (B.entry >= '2019-01-01')].copy(); B['h1'] = False; print('trades', len(B))
B['per'] = np.where(B.entry < '2023-01-01', 'train', 'test')
DIR = {'room': -1, 'bearlen': 1, 's1': 1, 'rvol': 1, 'rngatr': 1, 'hi52': 1}; DIR_R = dict(DIR, ratio=1)
print('(a) mean return per trade by feature third (low/mid/high; cuts on train): train | test')
for f in DIR:
    cut = B[B.per == 'train'][f].quantile([1 / 3, 2 / 3]).values if f != 's1' else [.5, .5]
    B['_t'] = np.where(B[f] <= cut[0], 'lo', np.where(B[f] <= cut[1], 'mid', 'hi')) if f != 's1' else np.where(B[f] > .5, 'hi', 'lo')
    r = B.groupby(['per', '_t']).ret.agg(['mean', 'size']).round(4); print(f'  {f:8s} ' + ' | '.join(f"{p}: " + ' '.join(f"{t} {r.loc[(p, t), 'mean']:+.2%}(n{int(r.loc[(p, t), 'size'])})" for t in ('lo', 'mid', 'hi') if (p, t) in r.index) for p in ('train', 'test')))
base = B.copy()
def order(kind):
    X = base.copy()
    if kind == 'ratio': X['_k'] = -X.ratio
    elif kind.startswith('rs:'):
        fs = kind[3:].split('+'); g = X.groupby('entry'); X['_k'] = sum(g[f].rank(ascending=(DIR_R[f] < 0), na_option='bottom', method='average') for f in fs)
        X['_k'] = X._k - 1e-6 * X.ratio
    else: X['_k'] = -DIR[kind] * X[kind].fillna(-DIR[kind] * -1e9)
    return X.sort_values(['entry', '_k', 'tk'], kind='stable')
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    e = (1 + s).cumprod(); return ' | '.join(out) + f' | all {e.iloc[-1] ** (1 / ((s.index[-1] - s.index[0]).days / 365.25)) - 1:6.1%}'
print('\n(b) 6-slot sim, pick order for simultaneous flips | CAGR/DD 2019-22 | 2023-26 | all'); R = {}
for kind in ['ratio', 'room', 'bearlen', 's1', 'rvol', 'rngatr', 'hi52', 'rs:ratio+room', 'rs:ratio+rvol', 'rs:ratio+s1', 'rs:room+rvol+hi52']:
    T0 = order(kind); tk_, _ = taken(); s_, _ = daily(tk_, 6); R[kind] = s_; print(f'  {kind:18s} {st2(s_)}', flush=True)
g = base.groupby('entry'); base['n'] = g.tk.transform('size'); X = base[base.n >= 5]; print(f'\ncrowded bars (>=5 flips): {len(X)} trades. avg return by within-bar rank (1 = first under the declared direction):')
for f in DIR:
    X['rk'] = X.groupby('entry')[f].rank(ascending=(DIR_R[f] < 0), method='first', na_option='bottom'); print(f'  {f:8s} ' + ' '.join(f'{int(k)}:{v:+.2%}' for k, v in X.groupby(X.rk.clip(upper=6)).ret.mean().items()))
X['rk'] = X.groupby('entry').ratio.rank(ascending=False, method='first'); print(f'  ratio    ' + ' '.join(f'{int(k)}:{v:+.2%}' for k, v in X.groupby(X.rk.clip(upper=6)).ret.mean().items()))
