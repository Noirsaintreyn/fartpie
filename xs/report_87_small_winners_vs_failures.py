"""Nov 14 2026 WITHIN SMALL / LOW-PRICED FLIPS (close < $10 at the flip bar): what separates the big winners from the failures. Rey's point: the vector works on small stocks, the hard part is picking winners; sub-$2 percent moves are partly mechanical (low price = big %), so the within-group question is the real one. Declared before results: VOLX = flip-bar volume / prior 20d mean (H: high better); DEPTH = close / 250-day high - 1, i.e. how far below the year high (H: deeper base better, more room); DV = 20d median dollar volume (H: higher = more fuel, tradeable); BASE = days the 60-day range stayed within 25% of its low before the flip (H: longer coil better); R20 = 20-day return before the flip (H: not-yet-run better). Terciles within month. Outcomes: mean trade return, huge (>=+100%) rate, and 'win10' (>=+10%); AUC of each feature for win10 per era. PASS = top-minus-bottom tercile mean in the predicted direction in BOTH eras, pooled month-clustered t>=2, AUC>=0.55 both eras. Entry next open, exit open after opposite flip, 30bp (optimistic below $1M dollar volume)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-01-01']
    if len(B) < 400: continue
    o = B.open.values; h = B.high.values; l = B.low.values; c = B.close.values; v = B.volume.values.astype(float); idx = B.index; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index))
    dv = pd.Series(c * v).rolling(20).median().values; vm = pd.Series(v).rolling(20).mean().shift(1).values; hi250 = pd.Series(h).rolling(250).max().values; lo60 = pd.Series(l).rolling(60).min().values; hi60 = pd.Series(h).rolling(60).max().values; i = 260
    while i < n - 2:
        if reg[i] == 1 and reg[i - 1] != 1:
            j = i + 1
            while j < n and reg[j] != -1: j += 1
            if j + 1 < n and '2019-01-01' <= str(idx[i])[:10] < '2026-03-01' and 1 <= c[i] < 10 and dv[i] >= 5e5:
                base = 0
                for k in range(i - 1, max(i - 120, 0), -1):
                    if hi60[k] / lo60[k] - 1 <= .25: base += 1
                    else: break
                rows.append((tk, idx[i], o[j + 1] / o[i + 1] - 1 - 0.003, v[i] / vm[i], c[i] / hi250[i] - 1, dv[i], base, c[i] / c[i - 20] - 1, c[i]))
            i = j
        i += 1
F = pd.DataFrame(rows, columns=['tk', 'd', 'r', 'VOLX', 'DEPTH', 'DV', 'BASE', 'R20', 'px']).replace([np.inf, -np.inf], np.nan).dropna(); F['era'] = np.where(F.d < '2022-01-01', '2019-21', '2022-26Q1'); F['mo'] = F.d.dt.to_period('M'); F['win10'] = F.r >= .10; print('small flips', len(F), F.groupby('era').size().to_dict(), 'base rates huge', F.groupby('era').r.apply(lambda x: (x >= 1).mean()).round(3).to_dict(), 'win10', F.groupby('era').win10.mean().round(3).to_dict())
def st(x): return f'n {len(x):5d} mean {x.mean()*100:+6.1f}% med {x.median()*100:+5.1f}% huge {(x>=1).mean()*100:4.1f}% win10 {(x>=.1).mean()*100:4.1f}%'
def auc(s, y): 
    r = s.rank(); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
pred = {'VOLX': 1, 'DEPTH': -1, 'DV': 1, 'BASE': 1, 'R20': -1}
for f in pred: F[f + '_t'] = F.groupby('mo')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if len(s) >= 6 else np.nan)
for f, sg in pred.items():
    print(f'\n {f} (H: {"high" if sg>0 else "low"} better)'); res = {}
    for en in ('2019-21', '2022-26Q1', 'POOLED'):
        z = F if en == 'POOLED' else F[F.era == en]; a = z[z[f + '_t'] == 2].groupby('mo').r.mean(); b = z[z[f + '_t'] == 0].groupby('mo').r.mean(); d = (a - b).dropna(); t = d.mean() / (d.std() / np.sqrt(len(d))); dm = (z[z[f + '_t'] == 2].r.mean() - z[z[f + '_t'] == 0].r.mean()) * 100; au = auc(z[f], z.win10) if en != 'POOLED' else np.nan; res[en] = (dm, t, au)
        print(f'  {en:9s} T0 {st(z[z[f+"_t"]==0].r)} || T2 {st(z[z[f+"_t"]==2].r)} || top-bottom {dm:+.1f}pp t {t:+.1f}' + (f' AUC(win10) {au:.3f}' if en != 'POOLED' else ''))
    ok = all(res[e][0] * sg > 0 for e in ('2019-21', '2022-26Q1')) and res['POOLED'][1] * sg >= 2 and all((res[e][2] - .5) * sg >= .05 for e in ('2019-21', '2022-26Q1')); print('  ->', 'PASS' if ok else 'no pass')
F.to_pickle(X + 'small_flips.pkl')
