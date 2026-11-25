"""Report 494 (Nov 25 2026): Rey - 'each factor gets a rank by its score results and strength, adds up to one final score, give it a probability of confidence or A-E'.
DECLARED BEFORE RESULTS. Same population/outcome as 492 (sub-$5 live group, hit = PEAK>=+10%). Instead of hand +1/+2 points: a ridge-penalised LOGISTIC fit on TRAIN 2019-21 only over the 5 flags (HI atrp top third, FIRST, REC, EARLY30, NEGBAD); each coefficient = that factor's weight (strength x evidence, shrunk by n). Final score = intercept + sum of weights -> probability. Grades by TRAIN-probability quintile-ish cuts fixed before test: A >=75%, B 65-75, C 55-65, D 45-55, E <45 (declared round cuts). Judge on TEST 2022+: calibration (mean predicted vs actual reach per grade), monotone A>B>C>D>E on reach and mean, AUC vs the 492 points rule. Pass = test AUC >= 492's and calibration within ~10pts per grade with n>=15; else keep 492 points and show weights as descriptive."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_492_setup_grade.py').read(); exec(src[src.index("import numpy"):src.index("for cap in (5, 10):")])
def fit(X, y, lam=3.0):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1]); P = np.eye(len(w)) * lam; P[0, 0] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-X1 @ w)); H = X1.T @ (X1 * (p * (1 - p))[:, None]) + P; w = w + np.linalg.solve(H, X1.T @ (y - p) - P @ w)
    return w
def auc(s, y):
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
C = ['HI', 'FIRST', 'REC', 'EARLY30', 'NEGBAD']
S = O[O.px < 5].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut].copy(); ac = np.quantile(G[G.tr].ATRP, 2 / 3); G['HI'] = G.ATRP >= ac
G['pts'] = G.HI.astype(int) + G.FIRST.astype(int) + G.REC.astype(int) - G.EARLY30.astype(int) - 2 * G.NEGBAD.astype(int)
X = G[C].astype(float).values; y = G.hit.values.astype(float); tr = G.tr.values
w = fit(X[tr], y[tr]); G['logit'] = np.c_[np.ones(len(X)), X] @ w; G['p'] = 1 / (1 + np.exp(-G.logit))
print(f'train n {tr.sum()} test n {(~tr).sum()} | base train reach {y[tr].mean():.0%} test {y[~tr].mean():.0%}')
print('weights (log-odds; x4 ~ pts of reach): intercept %+.2f | ' % w[0] + ' | '.join(f'{c} {v:+.2f} (n flag train {int(G[tr][c].sum())})' for c, v in zip(C, w[1:])))
T = G[~G.tr]; print(f"TEST AUC weighted {auc(T.logit.values, T.hit.values.astype(int)):.3f} vs 492 points {auc(T.pts.values, T.hit.values.astype(int)):.3f}")
G['grade'] = np.select([G.p >= .75, G.p >= .65, G.p >= .55, G.p >= .45], list('ABCD'), 'E'); T = G[~G.tr]
f = lambda x: f"{int(x.hit.sum()):3d}/{len(x):3d} = {x.hit.mean():.0%} reach | pred {x.p.mean():.0%} | mean {x.r.mean():+6.1%} med {x.r.median():+.1%}" if len(x) else 'n 0'
for lab, E in (('ALL years', G), ('TEST 2022+', T)):
    print(lab); [print(f'  grade {g}: {f(E[E.grade == g])}') for g in 'ABCDE']
print('TEST era split reach by grade: ' + ' | '.join(f"{g}: 2022 {T[(T.grade==g)&(T.d.dt.year==2022)].hit.mean():.0%} (n {((T.grade==g)&(T.d.dt.year==2022)).sum()}) 23-26 {T[(T.grade==g)&(T.d.dt.year>=2023)].hit.mean():.0%} (n {((T.grade==g)&(T.d.dt.year>=2023)).sum()})" for g in 'ABCDE'))
json_w = dict(intercept=float(w[0]), **{c: float(v) for c, v in zip(C, w[1:])}); import json; json.dump(json_w, open('daily_grade_weights.json', 'w')); print(json_w)
