"""Nov 14 2026 PARTICIPANT CLASSIFIER (declared before results). Question: can a model that sees only the participant-structure of a flip (how much, how one-sided, how absorbed, who is trapped) tell the flips that become the tail (top decile of direction-adjusted trade return) from the fades? Same idea as the bull/bear SVM: classify the state at the flip, then act on the ranking. Universe: large-cap 15m flips (58 names, RTH, both directions, features in xs/participant_feats.py, W=26 bars/day). STRICT SPLIT: train 2019-21 only, test 2022+ untouched; also walk-forward by year (train on all earlier years, test each year 2022..2026). Models: single-feature baseline (BV), L2 logistic, RBF-SVM (subsample), gradient boosting. Metrics out of sample: AUC, top-quintile-score tail share vs base 10%, mean return top quintile vs all. Permutation importance (AUC drop) for the boosting model. Label threshold = 90th pct of train returns per direction."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/fartpie/xs')
from participant_feats import flips_for, FEATS
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AUC
import os
CACHE = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/part_large.pkl'
def load(tk):
    f = glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not f: return None
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').set_index('dt'); return H[['open', 'high', 'low', 'close', 'volume']]
if os.path.exists(CACHE): P = pd.read_pickle(CACHE)
else:
    T0 = pd.read_parquet('xs/ext/wide_T.parquet'); rows = []; closes = {}
    for tk in sorted(T0.tk.unique()):
        B = load(tk)
        if B is None: continue
        closes[tk] = B.close.groupby(B.index.normalize()).last(); rows += flips_for(B, 26, 6500, tk)
    P = pd.DataFrame(rows); C = pd.DataFrame(closes); r20 = C.pct_change(20); mk = pd.DataFrame({'MKT': r20.median(axis=1), 'DISP': r20.std(axis=1)}).shift(1); P = P.join(mk, on='day'); P.to_pickle(CACHE)
F = FEATS + ['MKT', 'DISP']; P = P.dropna(subset=F).sort_values('t').reset_index(drop=True); print('flips', len(P), 'names', P.tk.nunique())
def models():
    return {'BV only': None, 'logit': LR(C=0.1, max_iter=500), 'SVM-rbf': SVC(C=1.0, kernel='rbf', gamma='scale'), 'boost': HGB(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)}
def fit_score(name, m, Xtr, ytr, Xte, rng):
    if name == 'BV only': return Xte[:, F.index('BV')]
    sc = StandardScaler().fit(Xtr); a = sc.transform(Xtr); b = sc.transform(Xte)
    if name == 'SVM-rbf':
        k = rng.choice(len(a), min(12000, len(a)), replace=False); m.fit(a[k], ytr[k]); return m.decision_function(b)
    m.fit(a, ytr); return m.decision_function(b) if name == 'logit' else m.predict_proba(b)[:, 1]
def report(tag, tr, te, rng, imp=False):
    out = []
    for d, nm in ((1, 'LONG'), (-1, 'SHORT')):
        a = tr[tr.d == d]; b = te[te.d == d]; thr = a.ret.quantile(0.9); ya = (a.ret >= thr).values; yb = (b.ret >= thr).values; Xa = a[F].values; Xb = b[F].values
        for name, m in models().items():
            s = fit_score(name, m, Xa, ya, Xb, rng); q = s >= np.quantile(s, 0.8)
            out.append((nm, name, AUC(yb, s), yb[q].mean() * 100, yb.mean() * 100, b.ret.values[q].mean() * 100, b.ret.mean() * 100, len(b)))
            if imp and name == 'boost':
                base = AUC(yb, s); sc = StandardScaler().fit(Xa); drop = {}
                for k, f in enumerate(F):
                    d_ = []
                    for _ in range(3):
                        Xp = Xb.copy(); Xp[:, k] = rng.permutation(Xp[:, k]); d_.append(base - AUC(yb, m.predict_proba(sc.transform(Xp))[:, 1]))
                    drop[f] = np.mean(d_)
                print(f'  [{tag} {nm}] boost permutation importance (AUC drop):', ' '.join(f'{k} {v:+.3f}' for k, v in sorted(drop.items(), key=lambda z: -z[1])[:8]))
    print(f'\n{tag}: direction | model | test AUC | tail% in top-20% score (base) | mean ret % top-20% (all) | n')
    for r in out: print(f' {r[0]:5s} {r[1]:8s} AUC {r[2]:.3f} | tail {r[3]:4.1f}% ({r[4]:.1f}%) | ret {r[5]:+.2f}% ({r[6]:+.2f}%) | n {r[7]}')
rng = np.random.default_rng(0)
tr = P[(P.t >= '2019') & (P.t < '2022')]; te = P[P.t >= '2022']; print('train', len(tr), 'test', len(te))
report('STRICT train 2019-21 -> test 2022+', tr, te, rng, imp=True)
print('\nWALK-FORWARD boost AUC by test year (train = all earlier years from 2019)')
for y in range(2022, 2027):
    a = P[(P.t >= '2019') & (P.t < f'{y}')]; b = P[(P.t >= f'{y}') & (P.t < f'{y + 1}')]
    if len(b) < 500: continue
    line = f' {y}:'
    for d, nm in ((1, 'LONG'), (-1, 'SHORT')):
        aa = a[a.d == d]; bb = b[b.d == d]; thr = aa.ret.quantile(0.9); m = HGB(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0); s = fit_score('boost', m, aa[F].values, (aa.ret >= thr).values, bb[F].values, rng); yb = (bb.ret >= thr).values
        q = s >= np.quantile(s, 0.8); line += f' {nm} AUC {AUC(yb, s):.3f} BVauc {AUC(yb, bb.BV.values):.3f} top20 tail {yb[q].mean()*100:.1f}% ret {bb.ret.values[q].mean()*100:+.2f}% (all {bb.ret.mean()*100:+.2f}%) |'
    print(line)
