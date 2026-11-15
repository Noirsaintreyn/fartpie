"""Nov 15 2026 PHYSICS-STYLE PRESSURE / SUPPLY / SELF-EXCITATION MEASURES vs FLIP SIZE AND DURATION (Rey: 'test all we can'). Declared before results. Flips = report_90 flips (daily wide, 2019-Mar 2026), point-in-time shares outstanding from SEC cover pages (latest filing on or before the flip date, <= 400 days old). Outcomes per flip: R = trade return (30bp), PEAK = max high during the hold / entry open - 1 (size of the excursion, not the exit), DUR = bars until the bear flip. Measures at the flip-bar close: P1 = ATR/close * sqrt(VOLX) (square-root-law pressure: volatility times the root of volume relative to normal), VOLX alone, ATRP alone; TURN = flip-bar volume / shares outstanding (true turnover); MCAP = log(close*shares); BURST = number of days with volume >= 2x its 20d mean in the prior 20 days (attention history); VACF = lag-1 autocorrelation of log volume over the prior 20 days (self-excitation proxy). H1: P1 rank-correlates with PEAK more than VOLX or ATRP alone, same sign in both eras. H2: lower MCAP / higher TURN -> larger PEAK and higher huge-rate. H3: BURST and VACF positive with DUR. PASS per measure = month-clustered Spearman IC of the stated sign in BOTH eras with pooled |t| >= 2. Then: do the physics measures add to the report_90 feature set? Ridge logistic on TRAIN 2019-21 (base 8 rank features vs base + physics), TEST AUC for win10 and alive20. Survivorship-flattered; shares are current-listing CIK based."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr; from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); S = pd.read_pickle(X + 'shares_pit.pkl'); S['filed'] = pd.to_datetime(S.filed); S = S.sort_values('filed'); sh = {k: z[['filed', 'val']].values for k, z in S.groupby('tk')}
rows = []
for tk, z in F.groupby('tk'):
    B = D.get(tk)
    if B is None: continue
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']; idx = B.index; o = B.open.values; h = B.high.values; c = B.close.values; v = B.volume.values.astype(float); n = len(B); reg, _, atr, _ = run_state_machine(B.assign(datetime=B.index)); vm = pd.Series(v).rolling(20).mean().shift(1).values; lv = np.log(np.maximum(v, 1))
    for d in z.d:
        i = idx.searchsorted(d)
        if i >= n - 2 or idx[i] != d or i < 30: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        if j + 1 >= n: continue
        s = np.nan
        if tk in sh:
            a = sh[tk]; k = np.searchsorted(a[:, 0], np.datetime64(d)) - 1
            if k >= 0 and (np.datetime64(d) - a[k, 0]) / np.timedelta64(1, 'D') <= 400: s = float(a[k, 1])
        vr = v[i - 20:i] / vm[i - 20:i]; w = lv[i - 20:i]
        rows.append((tk, d, h[i + 1:j + 2].max() / o[i + 1] - 1, j - i, atr[i] / c[i], v[i] / vm[i], v[i] / s if s == s and s > 0 else np.nan, np.log(c[i] * s) if s == s and s > 0 else np.nan, float((vr >= 2).sum()), np.corrcoef(w[:-1], w[1:])[0, 1]))
P = pd.DataFrame(rows, columns=['tk', 'd', 'PEAK', 'DUR', 'ATRP', 'VOLX2', 'TURN', 'MCAP', 'BURST', 'VACF']); P['P1'] = P.ATRP * np.sqrt(P.VOLX2.clip(lower=0)); G = F.merge(P, on=['tk', 'd']).replace([np.inf, -np.inf], np.nan); G['era'] = np.where(G.test, '2022-26Q1', '2019-21'); print('flips', len(G), 'with shares', int(G.TURN.notna().sum()), '| median PEAK', round(G.PEAK.median(), 3), 'median DUR', G.DUR.median())
def ic(z, f, y): 
    r = z.groupby('mo').apply(lambda g: spearmanr(g[f], g[y], nan_policy='omit')[0] if g[[f, y]].dropna().shape[0] > 8 else np.nan).dropna(); return r.mean(), (r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 5 else np.nan)
print('\nMonth-clustered Spearman IC (t) of each measure with PEAK / DUR / R, by era (+ pooled)')
FEAT = ['P1', 'VOLX2', 'ATRP', 'TURN', 'MCAP', 'BURST', 'VACF']
for f in FEAT:
    line = f' {f:6s}'
    for y in ('PEAK', 'DUR', 'r'):
        parts = []
        for en in ('2019-21', '2022-26Q1', 'POOLED'):
            z = G if en == 'POOLED' else G[G.era == en]; m, t = ic(z, f, y); parts.append(f'{m:+.3f}({t:+.1f})')
        line += f' | {y:4s} ' + ' '.join(parts)
    print(line)
print('\nH2 huge-rate (>=+100%) by MCAP tercile and TURN tercile (within month), both eras:')
for f in ('MCAP', 'TURN'):
    G[f + '_t'] = G.groupby('mo')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False) if s.notna().sum() >= 9 else np.nan)
    for en in ('2019-21', '2022-26Q1'): z = G[G.era == en]; print(f'  {f:5s} {en:9s} ' + ' | '.join(f'T{k}: n {int((z[f+"_t"]==k).sum()):5d} huge {(z[z[f+"_t"]==k].r>=1).mean()*100:4.1f}% PEAKmed {z[z[f+"_t"]==k].PEAK.median()*100:5.1f}% mean r {z[z[f+"_t"]==k].r.mean()*100:+5.1f}%' for k in (0, 1, 2)))
# incremental value
PH = ['P1', 'TURN', 'MCAP', 'BURST', 'VACF']; H = G.dropna(subset=PH).copy()
for f in PH: H['p_' + f] = H.groupby('mo')[f].rank(pct=True)
K = ['k_' + f for f in ['VOLX', 'PEERS', 'VR', 'EXT', 'R60', 'BLOW', 'LDV', 'LPX']]; KP = K + ['p_' + f for f in PH]; H['win10'] = H.r >= .10; H['a20'] = H.DUR > 20; tr = H[~H.test]; te = H[H.test]
print(f'\nINCREMENTAL (train 2019-21 n {len(tr)}, test n {len(te)}): ridge logistic TEST AUC base vs base+physics')
for tgt in ('win10', 'a20'):
    a = LogisticRegression(C=.5, max_iter=500).fit(tr[K], tr[tgt]); b = LogisticRegression(C=.5, max_iter=500).fit(tr[KP], tr[tgt]); print(f'  {tgt:5s} base {roc_auc_score(te[tgt], a.predict_proba(te[K])[:,1]):.3f} base+physics {roc_auc_score(te[tgt], b.predict_proba(te[KP])[:,1]):.3f} | physics coefs {dict(zip(PH, np.round(b.coef_[0][-5:], 2)))}')
G.to_pickle(X + 'physics_flips.pkl')
