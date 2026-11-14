"""Follow-up to report_75 (declared before results): no single feature cuts the bottom cleanly, so test the best a multi-feature model can do, strictly out of sample. Target LOSS10 = trade net return <= -10% (about 25% of screener-rule flips). Gradient boosting trained 2019-21 ONLY on the 16 report_75 features, evaluated 2022-26Q1: AUC for LOSS10; then cut the 20% of test flips with the highest predicted loss risk and report loss mass removed, HUGE winners removed, mean net before/after and total return per ORIGINAL trade (does cutting help the book, not just the average?). Two versions: all features, and without the volatility proxies (ATRP, STOP, STOPATR, LPX, RNG) to see whether it is just 'cut the quiet names'. Also a second target: LOW = net return in the worst third of the TRAIN distribution. Base = BV>=1.5 flips (the screener rule). Permutation importance on the test era."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score as AUC
from sklearn.inspection import permutation_importance
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['BREADTH'] = P.BREADTH.astype(float); S = P[P.BV >= 1.5].reset_index(drop=True)
FE = ['STOPATR', 'STOP', 'RNG', 'NFL', 'DSL', 'PLR', 'ATRP', 'ATREXP', 'DD52', 'POS52', 'R20', 'R120', 'BV', 'LDV', 'LPX', 'BREADTH']; NOVOL = [f for f in FE if f not in ('ATRP', 'STOP', 'STOPATR', 'LPX', 'RNG')]
tr = S[S.t < '2022']; te = S[S.t >= '2022']; third = tr.rn.quantile(1 / 3)
def run(cols, target, name):
    ytr = target(tr); yte = target(te); m = HGB(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=60, l2_regularization=1.0, random_state=0).fit(tr[cols], ytr); p = m.predict_proba(te[cols])[:, 1]; au = AUC(yte, p)
    cut = p >= np.quantile(p, .8); L = te.rn.clip(upper=0); lm = L[cut].sum() / L.sum() * 100; hg = te.HUGE[cut].sum() / te.HUGE.sum() * 100
    print(f'  {name:34s} test AUC {au:.3f} | cut 20%: loss mass {lm:4.1f}% HUGE {hg:4.1f}% | mean {te.rn.mean()*100:+.2f}% -> {te.rn[~cut].mean()*100:+.2f}% | total per original trade {te.rn.mean()*100:+.2f} -> {te.rn[~cut].sum()/len(te)*100:+.2f} | cut-group mean {te.rn[cut].mean()*100:+.2f}% median {te.rn[cut].median()*100:+.2f}%')
    return m, te[cols], yte
print(f'train n {len(tr)}, test n {len(te)}; LOSS10 rate train {(tr.rn<=-.10).mean()*100:.1f}% test {(te.rn<=-.10).mean()*100:.1f}%')
m, X, y = run(FE, lambda d: (d.rn <= -.10).astype(int), 'LOSS10, all features')
run(NOVOL, lambda d: (d.rn <= -.10).astype(int), 'LOSS10, no volatility proxies')
run(FE, lambda d: (d.rn <= third).astype(int), 'LOW third, all features')
run(NOVOL, lambda d: (d.rn <= third).astype(int), 'LOW third, no volatility proxies')
pi = permutation_importance(m, X, y, scoring='roc_auc', n_repeats=5, random_state=0); print('  permutation importance (LOSS10 all features, test AUC drop): ' + ', '.join(f'{f} {v:+.3f}' for f, v in sorted(zip(FE, pi.importances_mean), key=lambda z: -z[1])[:6]))
