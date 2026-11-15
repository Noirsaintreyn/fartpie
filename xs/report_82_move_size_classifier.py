"""CORRECTED LABEL (report_82 v1 used the full trough-to-peak move M, which contains the part already travelled at entry, L, so L/dte looked predictive mechanically: AUC 0.72 for L was partly by construction). Now MA = FORWARD move (entry to peak, hindsight) / ATR%. Everything else unchanged.
Nov 14 2026 THE CLASSIFIER QUESTION (post ideology change): which confirmed flips turn out to be LARGE RELATIVE TO THE LAG, in volatility units? Declared before results. Label per long flip: MA = available move (hindsight trough to peak, report_80) divided by the stock's ATR% at the flip = how many daily-vol units the leg travelled; BIGM = MA >= the 80th percentile of the TRAIN era (2019-21) (same cutoff applied to 2022-26Q1, so the rate may drift). Features (all observable at the flip bar; mechanism): volatility ATRP, STOPATR; liquidity LDV, LPX; participation BV; displacement L (already off the trough) and L/ATR, days since trough; prior run R20, R120, DD52, POS52, PLR (previous leg); chop NFL, DSL; vol expansion ATREXP; market BREADTH; insider NI. Part 1: univariate AUC for BIGM on TRAIN and TEST (a feature is a mechanism candidate only if the direction holds in both). Part 2: models trained on 2019-21 only (logistic on clipped standardized features; gradient boosting) tested on 2022-26Q1: AUC, then take the top 20% by predicted score and report net return per flip, median, HUGE recall, BIGM hit rate vs the same selection by single features (ATRP, BV, L, LDV) and random. Part 3: does BIGM-selection raise realised net returns, which is what pays? Base = all flips (31.8k) and the screener rule (BV>=1.5 is also a feature, so tested on all flips)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score as AUC
import os, pickle
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; AP = SC + 'legs_A.pkl'
if os.path.exists(AP): A = pd.read_pickle(AP); P = pd.read_pickle(SC + 'wide_flips_feat.pkl')
else:
    src = open('report_80_leg_anatomy.py').read(); exec(src.split("lg = lambda x: np.log1p(x)")[0]); A.to_pickle(AP)
P = P.reset_index(drop=True); P['rn'] = P.ret - 0.003; P['BREADTH'] = P.BREADTH.astype(float)
cols = ['ATRP', 'STOPATR', 'LDV', 'LPX', 'NFL', 'DSL', 'PLR', 'ATREXP', 'DD52', 'POS52', 'R20', 'R120', 'BREADTH', 'NI', 'rn']
A = A.reset_index(drop=True)
for c in cols: A[c] = P.loc[A.i, c].values
A['LA'] = A.L / A.ATRP; A['FM'] = (1 + A.M) / (1 + A.L) - 1; A['MA'] = A.FM / A.ATRP; tr = A.t < '2022'; cut = A.MA[tr].quantile(.8); A['BIGM'] = (A.MA >= cut).astype(int)
FE = ['ATRP', 'STOPATR', 'LDV', 'LPX', 'BV', 'L', 'LA', 'dte', 'R20', 'R120', 'DD52', 'POS52', 'PLR', 'NFL', 'DSL', 'ATREXP', 'BREADTH', 'NI']
print(f'legs {len(A)} | MA 80th pct (train) {cut:.1f} ATR units (median MA {A.MA.median():.1f}) | BIGM rate train {A.BIGM[tr].mean()*100:.1f}% test {A.BIGM[~tr].mean()*100:.1f}% | mean net of BIGM legs {A.rn[A.BIGM==1].mean()*100:+.1f}% vs others {A.rn[A.BIGM==0].mean()*100:+.1f}% (the label is hindsight: BIGM legs are not necessarily captured)')
print('\nPART 1: univariate AUC for BIGM (0.5 = nothing; sign must agree train/test)'); print(f'  {"feature":9s} {"train":>6s} {"test":>6s}  agree')
for f in FE:
    z = A[[f, 'BIGM', 't']].dropna(); a = AUC(z.BIGM[z.t < '2022'], z[f][z.t < '2022']); b = AUC(z.BIGM[z.t >= '2022'], z[f][z.t >= '2022']); print(f'  {f:9s} {a:6.3f} {b:6.3f}  {"yes" if (a-.5)*(b-.5) > 0 and min(abs(a-.5), abs(b-.5)) > .02 else ""}')
X = A[FE].copy(); lo, hi = X[tr].quantile(.01), X[tr].quantile(.99); Xc = X.clip(lo, hi, axis=1); mu, sd = Xc[tr].mean(), Xc[tr].std(); Z = ((Xc - mu) / sd).fillna(0)
te = ~tr; y = A.BIGM; mdl = {}
mdl['logistic'] = LR(C=0.5, max_iter=500).fit(Z[tr], y[tr]); pL = mdl['logistic'].predict_proba(Z[te])[:, 1]
mdl['boosting'] = HGB(max_iter=150, learning_rate=.05, max_depth=3, min_samples_leaf=80, l2_regularization=1., random_state=0).fit(X[tr], y[tr]); pG = mdl['boosting'].predict_proba(X[te])[:, 1]
T = A[te].copy(); rng = np.random.RandomState(0)
print('\nPART 2/3: trained on 2019-21, selection of the top 20% of 2022-26Q1 flips by each score'); print(f'  test n {len(T)} | base: net mean {T.rn.mean()*100:+.2f}% med {T.rn.median()*100:+.2f} | HUGE {int(T.HUGE.sum())} | BIGM rate {T.BIGM.mean()*100:.1f}%')
def sel(name, s, auc=True):
    s = pd.Series(np.asarray(s), index=T.index); k = s >= s.quantile(.8); z = T[k]
    print(f'  {name:26s} AUC {AUC(T.BIGM, s) if auc else float("nan"):.3f} | top20%: net mean {z.rn.mean()*100:+6.2f}% med {z.rn.median()*100:+5.2f} | HUGE recall {z.HUGE.sum()/T.HUGE.sum()*100:4.1f}% | BIGM hit {z.BIGM.mean()*100:4.1f}% | losers {(z.rn<0).mean()*100:.0f}%')
sel('logistic (16 features)', pL); sel('boosting', pG)
for f in ('ATRP', 'BV', 'L', 'LA', 'LDV', 'R20', 'NFL'): s = T[f].fillna(T[f].median()); sel(f'single: {f}' + (' (low=good)' if f in ('LDV',) else ''), -s if f in ('LDV',) else s)
sel('random', rng.rand(len(T)), auc=False)
co = pd.Series(mdl['logistic'].coef_[0], index=FE).sort_values(); print('\n  logistic standardized coefficients (train): ' + ', '.join(f'{k} {v:+.2f}' for k, v in co.items() if abs(v) >= 0.08))
