"""Nov 15 2026 SURVIVAL MODEL WITH EARLY CONFIRMATION (follow-up to report_97). Question: does the machine's behaviour in the first 3 bars after a flip add predictive power for how long the state lasts, enough to justify entering at the open of day 4 (skipping the first 3 bars) instead of day 1? Declared before results. Flips alive at the close of day 3 (no bear flip in bars 1-3). Base features = report_90's eight rank features. EARLY features (all known at the close of day 3): D3 = (close3 - level3)/ATR3 (stretch above the ratchet), RET3 = close3/close0 - 1, VP3 = mean of volume/prior-20d-mean over days 1-3 (volume persistence), UP3 = number of up closes in days 1-3. Ridge logistic trained on 2019-21 survivors only, TEST 2022+. Targets: ALIVE20 (state still bullish 20 bars after the flip bar). H1: base+EARLY TEST AUC for ALIVE20 exceeds base-only by >= 0.03. H2: trading from the day-4 open (entry open(i+4), exit open after the bear flip, 30bp) with the top quintile of the base+EARLY score earns a mean at least equal to the day-1-entry top quintile of the base score on the same flips (so delay does not cost the tail). Also report the mean give-up: median price move from day-1 open to day-4 open on those flips. PASS = H1 and H2. TEST only 2022-26Q1; survivorship-flattered."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); FE = ['VOLX', 'PEERS', 'VR', 'EXT', 'R60', 'BLOW', 'LDV', 'LPX']; K = ['k_' + f for f in FE]; rows = []
for tk, z in F.groupby('tk'):
    B = D.get(tk)
    if B is None: continue
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']; idx = B.index; o = B.open.values; c = B.close.values; v = B.volume.values.astype(float); n = len(B); reg, lev, atr, _ = run_state_machine(B.assign(datetime=B.index)); vm = pd.Series(v).rolling(20).mean().shift(1).values
    for d in z.d:
        i = idx.searchsorted(d)
        if i >= n - 6 or idx[i] != d: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        if j <= i + 3 or j + 1 >= n: continue
        t = i + 3; rows.append((tk, d, j - i, (c[t] - lev[t]) / atr[t] if atr[t] > 0 else np.nan, c[t] / c[i] - 1, np.mean(v[i + 1:t + 1] / vm[i + 1:t + 1]), int(sum(c[k] > c[k - 1] for k in range(i + 1, t + 1))), o[j + 1] / o[i + 4] - 1 - .003, o[i + 4] / o[i + 1] - 1))
E_ = pd.DataFrame(rows, columns=['tk', 'd', 'dur', 'D3', 'RET3', 'VP3', 'UP3', 'r4', 'move14']); G = F.merge(E_, on=['tk', 'd']).replace([np.inf, -np.inf], np.nan).dropna(subset=['D3', 'RET3', 'VP3']); G['a20'] = G.dur > 20; EA = ['D3', 'RET3', 'VP3', 'UP3']
for f in EA: G['e_' + f] = G.groupby('mo')[f].rank(pct=True)
KE = K + ['e_' + f for f in EA]; tr = G[~G.test]; te = G[G.test].copy(); print('survivors-at-day-3 flips', len(G), 'train', len(tr), 'test', len(te), '| alive20 among survivors train/test', round(tr.a20.mean(), 3), round(te.a20.mean(), 3))
mb = LogisticRegression(C=.5, max_iter=500).fit(tr[K], tr.a20); me = LogisticRegression(C=.5, max_iter=500).fit(tr[KE], tr.a20); te['sb'] = mb.predict_proba(te[K])[:, 1]; te['se'] = me.predict_proba(te[KE])[:, 1]
ab, ae = roc_auc_score(te.a20, te.sb), roc_auc_score(te.a20, te.se); print(f' H1 TEST AUC ALIVE20 base-only {ab:.3f} base+EARLY {ae:.3f} (gain {ae-ab:+.3f}) | early coefs {dict(zip(EA, np.round(me.coef_[0][-4:], 2)))}  single early AUCs {({f: round(roc_auc_score(te.a20, te["e_"+f]), 3) for f in EA})}')
def top(s, col): te['q'] = te.groupby('mo')[s].transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=False) if len(x) >= 10 else np.nan); return te[te.q == 4]
a = top('sb', 'r'); b = top('se', 'r4'); bb = top('sb', 'r4'); print(f' H2 (same survivor flips) base-score top Q entering day 1: mean {a.r.mean()*100:+.1f}% med {a.r.median()*100:+.1f}% huge {(a.r>=1).mean()*100:.1f}% | base-score top Q entering day 4: mean {bb.r4.mean()*100:+.1f}% | base+EARLY top Q entering day 4: mean {b.r4.mean()*100:+.1f}% med {b.r4.median()*100:+.1f}% huge {(b.r4>=1).mean()*100:.1f}% alive20 {b.a20.mean()*100:.0f}% | all survivors day1 {te.r.mean()*100:+.1f}% day4 {te.r4.mean()*100:+.1f}%; median move day1->day4 open on survivors {te.move14.median()*100:+.1f}% (mean {te.move14.mean()*100:+.1f}%)')
