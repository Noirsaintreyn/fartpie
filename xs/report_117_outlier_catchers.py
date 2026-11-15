"""Nov 15 2026 WHAT CATCHES OUTLIERS (Rey: the stock rules live on outliers, 52-week-high distance may not matter; we need something that catches outliers). Declared before results. Target is the TAIL, not the mean: BIG = net trade return >= +30%, HUGE = >= +100%. Daily flips with every feature known at the flip-bar close: report_116 features (F1 close location, F2 flip-bar range/ATR, F3 gap, F4 close/252d high, F5 pre-flip volume trend, F6 above 200d SMA, F7 20d relative strength, F8 chop, F9 base length, F10 breadth, F11 close above level) + report_90 features (VOLX, PEERS, VR, EXT, R60, BLOW, LDV, LPX, frozen score lr) + market cap MCAP, turnover TURN, ATRP (daily ATR/price) from report_99's file (shares outstanding; MCAP has coverage gaps, reported). Metric per feature and era (TRAIN 2019-21, TEST 2022-26): AUC for BIG and for HUGE (0.5 = none; sign-agnostic because the direction is the question), and the BIG rate in the top vs bottom decile of the feature (cut on TRAIN). A feature is a CONSISTENT outlier catcher if |AUC-0.5| >= 0.04 in the SAME direction on BOTH eras for BIG in universe A (all flips) or B (price < $5). PART 2: combine the consistent features into a rank-sum (direction fixed from TRAIN) and judge on TEST only: share of all test BIG and HUGE flips captured in the top 10%/20% of the sum (vs 10%/20% by chance), precision (BIG rate) in the top 20% vs base rate, and mean/median net of that top 20%; universe B and A. Sample caveat: HUGE has ~250 train / ~200 test flips in A, ~40 / ~58 in B, so HUGE is reported for direction, not significance."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import rankdata
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; C = pd.read_pickle(X + 'confirm_features.pkl'); P = pd.read_pickle(X + 'physics_flips.pkl')[['tk', 'd', 'MCAP', 'TURN', 'ATRP']]; A = C.merge(P, on=['tk', 'd'], how='left'); COST = .003; A['n'] = A.r - COST; A['big'] = A.n >= .3; A['huge'] = A.n >= 1.0
print('flips', len(A), 'MCAP coverage', A.MCAP.notna().mean().round(3), 'BIG rate train/test', A[~A.test].big.mean().round(4), A[A.test].big.mean().round(4), 'HUGE', A[~A.test].huge.mean().round(4), A[A.test].huge.mean().round(4))
feats = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'VOLX', 'PEERS', 'VR', 'EXT', 'R60', 'BLOW', 'LDV', 'LPX', 'lr', 'MCAP', 'TURN', 'ATRP']; A['LMC'] = np.log(A.MCAP); A['LTURN'] = np.log(A.TURN.clip(lower=1e-9)); feats = [f if f not in ('MCAP', 'TURN') else {'MCAP': 'LMC', 'TURN': 'LTURN'}[f] for f in feats]
def auc(x, y):
    m = x.notna(); x = x[m]; y = y[m].astype(bool)
    if y.sum() < 5 or (~y).sum() < 5: return np.nan
    r = rankdata(x); return (r[y.values].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
cons = {}
for un, U in (('A all', A), ('B px<5', A[A.px < 5])):
    print(f'\n######## {un}: train n {(~U.test).sum()} test n {U.test.sum()} (BIG train {U[~U.test].big.sum()} test {U[U.test].big.sum()}; HUGE {U[~U.test].huge.sum()} / {U[U.test].huge.sum()}) ########')
    for f in feats:
        o = []; dirs = []; ok = True
        tr = U[~U.test]; q1, q9 = tr[f].quantile([.1, .9]).values
        for era, z in (('TR', U[~U.test]), ('TE', U[U.test])):
            a_b = auc(z[f], z.big); a_h = auc(z[f], z.huge); hi = z[z[f] >= q9].big.mean(); lo = z[z[f] <= q1].big.mean(); o.append(f'{era}: AUCbig {a_b:.3f} AUChuge {a_h:.3f} BIG top10% {hi*100:4.1f}% bot10% {lo*100:4.1f}%'); dirs.append(np.sign(a_b - .5)); ok &= abs(a_b - .5) >= .04
        ok &= dirs[0] == dirs[1]; print(f'  {f:7s} ' + ' | '.join(o) + (f'   <== CONSISTENT ({"+" if dirs[0] > 0 else "-"})' if ok else ''))
        if ok: cons.setdefault(un, {})[f] = dirs[0]
print('\nCONSISTENT outlier catchers:', cons)
def rank_sum(U, fd): return sum(d * rankdata(U[f].fillna(U[f].median())) / len(U) for f, d in fd.items())
for un, U in (('B px<5', A[A.px < 5]), ('A all', A)):
    fd = cons.get(un, {})
    if not fd: print(f'\n{un}: no consistent features'); continue
    tr = U[~U.test]; te = U[U.test].copy(); te['s'] = rank_sum(te, fd); print(f'\nPART 2 {un} (features {fd}) TEST: n {len(te)}, BIG base {te.big.mean()*100:.1f}%, HUGE base {te.huge.mean()*100:.2f}%, mean net {te.n.mean()*100:+.2f}%')
    for q in (.1, .2, .3):
        t = te[te.s >= te.s.quantile(1 - q)]; print(f'  top {int(q*100)}%: n {len(t):4d} BIG precision {t.big.mean()*100:5.1f}% (captures {t.big.sum()/te.big.sum()*100:4.1f}% of BIG, {t.huge.sum()/max(te.huge.sum(),1)*100:4.1f}% of HUGE) | mean {t.n.mean()*100:+6.2f}% med {t.n.median()*100:+6.2f}% ex-top3 {t.n.sort_values(ascending=False).iloc[3:].mean()*100:+6.2f}%')
A.to_pickle(X + 'outlier_features.pkl')
