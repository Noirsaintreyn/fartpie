"""Nov 15 2026 STATE-MACHINE PATH ANALYSIS (Rey: analyze the state machine's states and how they relate to the model). Declared before results. Universe/flips = report_90 flips (daily wide, 2019-Mar 2026) with the frozen ridge-logistic score `lr` from report_90. For every in-position day t of every flip (day 0 = flip bar close) we record the machine's own state at the close: AGE = t - flip bar, DIST = (close - level)/ATR (how far above the ratchet), NR = number of ratchet raises of the level since the flip, and outcomes: FWD10 = open(t+11)/open(t+1) - 1 (hold-forward return, no costs), HAZ10 = bear flip occurs within the next 10 bars. Part 1 (state -> future): DIST buckets (<1.5, 1.5-3, 3-5, 5+ ATR) and NR buckets (0, 1-2, 3+) x AGE buckets (1-5, 6-20, 21+). H1: very stretched (DIST >= 5) has LOWER FWD10 than moderate DIST (mean reversion; take-profit state) in both eras. H2: higher NR has LOWER hazard (trend persistence). H3: low DIST (<1.5) has HIGHER hazard. PASS if direction holds in both eras. Part 2 (model -> path, TEST era 2022+ only): by frozen-score quintile within month: median max-DIST by day 10, share alive at day 20, mean NR at day 20 (alive flips), final trade return and what share of the total positive return the top-quintile carries. Cost-free path analysis; survivorship-flattered."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); fl = F.groupby('tk')
rows = []; pr = []
for tk, z in fl:
    B = D.get(tk)
    if B is None: continue
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']; idx = B.index; o = B.open.values; c = B.close.values; n = len(B); reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index))
    for d, lr, te in zip(z.d, z.lr, z.test):
        i = idx.searchsorted(d)
        if i >= n - 2 or idx[i] != d: continue
        nr = 0; j = i + 1
        while j < n and reg[j] != -1: j += 1
        ex = o[j + 1] / o[i + 1] - 1 if j + 1 < n else np.nan; mx = 0
        for t in range(i, min(j, i + 60)):
            if t > i and lev[t] > lev[t - 1]: nr += 1
            if t + 11 >= n: break
            dist = (c[t] - lev[t]) / atr[t] if atr[t] > 0 else np.nan; mx = max(mx, dist) if t - i <= 10 else mx
            rows.append((te, t - i, dist, nr, o[t + 11] / o[t + 1] - 1, float(j <= t + 10)))
        alive20 = (j - i) > 20; pr.append((te, d, lr, ex, mx, alive20, nr if alive20 else np.nan))
S = pd.DataFrame(rows, columns=['test', 'age', 'dist', 'nr', 'fwd10', 'haz']).dropna(subset=['dist']); S['era'] = np.where(S.test, '2022-26Q1', '2019-21'); print('state-days', len(S))
S['db'] = pd.cut(S.dist, [-9, 1.5, 3, 5, 99], labels=['<1.5', '1.5-3', '3-5', '5+']); S['nb'] = pd.cut(S.nr, [-1, 0, 2, 999], labels=['0', '1-2', '3+']); S['ab'] = pd.cut(S.age, [-1, 5, 20, 99], labels=['0-5', '6-20', '21+'])
cl = lambda x: x.clip(-.5, 1.0)
print('\nPART 1: state -> FWD10 (winsorized -50%..+100%) and 10-day bear-flip hazard')
for key, lab in (('db', 'DIST (ATRs above ratchet)'), ('nb', 'NR (ratchet raises since flip)')):
    print(f' by {lab}')
    for en in ('2019-21', '2022-26Q1'):
        z = S[S.era == en]; print(f'  {en:9s} ' + ' | '.join(f'{k}: n {len(g):6d} fwd10 {cl(g.fwd10).mean()*100:+5.1f}% haz {g.haz.mean()*100:4.1f}%' for k, g in z.groupby(key)))
print(' DIST x AGE (fwd10 / haz), both eras')
for en in ('2019-21', '2022-26Q1'):
    z = S[S.era == en]; print(f'  {en}: ' + ' | '.join(f'{a}/{b}: {cl(g.fwd10).mean()*100:+.1f}%/{g.haz.mean()*100:.0f}%' for (a, b), g in z.groupby(['ab', 'db'])))
P = pd.DataFrame(pr, columns=['test', 'd', 'lr', 'ret', 'maxdist', 'alive20', 'nr20']); T = P[P.test].copy(); T['mo'] = T.d.dt.to_period('M'); T['q'] = T.groupby('mo').lr.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) if len(s) >= 10 else np.nan)
print('\nPART 2: frozen score quintile -> path (TEST era 2022+), Q4 = highest score')
for q in range(5):
    g = T[T.q == q]; print(f' Q{q} n {len(g):5d} median max-DIST by day10 {g.maxdist.median():4.1f} | alive at day20 {g.alive20.mean()*100:4.1f}% | mean NR at day20 (alive) {g.nr20.mean():4.1f} | final trade mean {cl(g.ret).mean()*100:+5.1f}% median {g.ret.median()*100:+5.1f}%')
pos = T.ret.clip(lower=0); print(f' share of total positive return carried by Q4: {pos[T.q==4].sum()/pos.sum()*100:.0f}%  (equal share would be 20%); Spearman(score, maxdist) {T[["lr","maxdist"]].corr(method="spearman").iloc[0,1]:+.3f}, Spearman(score, final return) {T[["lr","ret"]].dropna().corr(method="spearman").iloc[0,1]:+.3f}')
