"""Nov 15 2026 IS FLIP DURATION MORE PREDICTABLE THAN FLIP RETURN? (follow-up to report_96: the frozen score tracked survival, 57% alive at day 20 in the top quintile vs 42% in the bottom, while rank-correlating NEGATIVELY with the final return). Declared before results. Same flips, same eight rank features as report_90, ridge logistic trained on 2019-21 ONLY, frozen, TEST 2022+. Targets: ALIVE20 = vector still bullish 20 bars after the flip bar, ALIVE60 = 60 bars. Report TEST AUC for each target, then top-vs-bottom quintile (within month) in mean trade return, median, and alive rate. H: duration AUC > the 0.563 AUC the same features get for win10 (return). PASS = ALIVE20 TEST AUC >= 0.60 and top-minus-bottom quintile mean return > 0 with month-clustered t >= 2. Cost-free path outcome plus the 30bp trade return already in the flips file."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); FE = ['VOLX', 'PEERS', 'VR', 'EXT', 'R60', 'BLOW', 'LDV', 'LPX']; K = ['k_' + f for f in FE]; dur = {}
for tk, z in F.groupby('tk'):
    B = D.get(tk)
    if B is None: continue
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']; idx = B.index; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index))
    for d in z.d:
        i = idx.searchsorted(d)
        if i >= n - 2 or idx[i] != d: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        dur[(tk, d)] = (j - i, j < n)
F['dur'] = [dur.get((t, d), (np.nan, False))[0] for t, d in zip(F.tk, F.d)]; F['closed'] = [dur.get((t, d), (np.nan, False))[1] for t, d in zip(F.tk, F.d)]; F = F.dropna(subset=['dur']); F['a20'] = F.dur > 20; F['a60'] = F.dur > 60; tr = F[~F.test]; te = F[F.test].copy()
print('flips', len(F), 'median duration (bars) train', tr.dur.median(), 'test', te.dur.median(), '| alive20 train/test', round(tr.a20.mean(), 3), round(te.a20.mean(), 3), '| alive60', round(tr.a60.mean(), 3), round(te.a60.mean(), 3))
for tgt in ('a20', 'a60', 'win10'):
    m = LogisticRegression(C=.5, max_iter=500).fit(tr[K], tr[tgt]); te['sc_' + tgt] = m.predict_proba(te[K])[:, 1]; print(f' target {tgt:5s} train AUC {roc_auc_score(tr[tgt], m.predict_proba(tr[K])[:,1]):.3f} TEST AUC {roc_auc_score(te[tgt], te["sc_"+tgt]):.3f} coefs ' + str(dict(zip(FE, np.round(m.coef_[0], 2)))))
print('single-feature TEST AUC for ALIVE20 (sign as is):', {f: round(roc_auc_score(te.a20, te['k_' + f]), 3) for f in FE})
for tgt in ('a20', 'a60'):
    te['q'] = te.groupby('mo')['sc_' + tgt].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) if len(s) >= 10 else np.nan); a = te[te.q == 4]; b = te[te.q == 0]; d = (a.groupby('mo').r.mean() - b.groupby('mo').r.mean()).dropna()
    print(f' {tgt} score quintiles: top Q alive20 {a.a20.mean()*100:.0f}% / bottom {b.a20.mean()*100:.0f}% | trade mean top {a.r.mean()*100:+.1f}% (med {a.r.median()*100:+.1f}) bottom {b.r.mean()*100:+.1f}% (med {b.r.median()*100:+.1f}) | spread {(a.r.mean()-b.r.mean())*100:+.1f}pp t {d.mean()/(d.std()/np.sqrt(len(d))):+.1f} | huge top {(a.r>=1).mean()*100:.1f}% bottom {(b.r>=1).mean()*100:.1f}%')
