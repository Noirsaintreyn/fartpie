"""P(WIN) MODEL (Nov 11 2026, declared before results). Target = trade pts > 0. Features = the 67 from report_mix_screen (ICT, VP, OPT, ECON, TA, CAL, XA).
Walk-forward: for each test year Y (2016+), fit L2 logistic (C=.05, standardized, NaN->0) on all trades before Y; predict year Y. Tiers use quantiles of the TRAINING predictions (no look-ahead).
Audit: OOS AUC overall and by era; per-family alone and leave-one-family-out; calibration by decile. Saves OOS probability per trade for the chain run."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
s = open('/home/claude/fartpie/xs/report_mix_screen.py').read(); exec(s[:s.index('# ---- candidate conditions')])
ALLT = pd.concat([LEGS[k][0].assign(leg=k) for k in LEGS], ignore_index=True); ALLX = pd.concat([LEGS[k][1] for k in LEGS], ignore_index=True)
FAM = {c: fam[c] for c in ALLX.columns}; yr = pd.to_datetime(ALLT.day).dt.year.values; y = (ALLT.pts.values > 0).astype(int)
print(f'trades {len(ALLT)} base win rate {y.mean():.1%}; features {ALLX.shape[1]}', flush=True)
def wf(cols, C=.05, retq=False):
    X = ALLX[cols].values.astype(float); P = np.full(len(y), np.nan); Q = np.full((len(y), 2), np.nan)
    for Y in range(2016, 2027):
        tr, te = yr < Y, yr == Y
        if te.sum() == 0 or tr.sum() < 150: continue
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0); sd[sd == 0] = 1; Z = np.nan_to_num((X - mu) / sd)
        m = LogisticRegression(C=C, max_iter=500).fit(Z[tr], y[tr]); P[te] = m.predict_proba(Z[te])[:, 1]; ptr = m.predict_proba(Z[tr])[:, 1]; Q[te] = np.quantile(ptr, [1 / 3, 2 / 3])
    return (P, Q) if retq else P
def auc(P, mask=None):
    ok = ~np.isnan(P) & (np.ones(len(P), bool) if mask is None else mask); return roc_auc_score(y[ok], P[ok]) if ok.sum() > 50 and len(set(y[ok])) > 1 else np.nan
cols = list(ALLX.columns); P, Q = wf(cols, retq=True)
ok = ~np.isnan(P); era2 = yr >= 2022
print(f'\nALL FAMILIES walk-forward OOS: AUC {auc(P):.3f} (n {ok.sum()}) | 2016-21 {auc(P, ~era2):.3f} | 2022+ {auc(P, era2):.3f}', flush=True)
for Y in range(2016, 2027):
    m = yr == Y; print(f'   {Y}: AUC {auc(P, m):.3f} n {int((m & ok).sum())}', end='')
print()
# placebo: shuffle y within year-leg, AUC distribution
rng = np.random.default_rng(3); nul = []
for it in range(40):
    y0 = y.copy()
    for Y in np.unique(yr):
        m = np.where(yr == Y)[0]; y0[m] = rng.permutation(y0[m])
    ysave = y; y = y0; nul.append(auc(wf(cols))); y = ysave
nul = np.array(nul); print(f'placebo AUC (40 label shuffles): mean {nul.mean():.3f} sd {nul.std():.3f} max {nul.max():.3f} | observed z = {(auc(P) - nul.mean()) / nul.std():.1f}', flush=True)
print('\n--- AUDIT by family: OOS AUC alone | AUC with the family left out (all-families AUC %.3f) ---' % auc(P))
for f in ['ICT', 'VP', 'OPT', 'ECON', 'TA', 'CAL', 'XA']:
    fc = [c for c in cols if FAM[c] == f]; rest = [c for c in cols if FAM[c] != f]
    print(f'  {f:5} ({len(fc):2d} features): alone {auc(wf(fc)):.3f} | left out {auc(wf(rest)):.3f}', flush=True)
print('\n--- AUDIT by single feature (OOS AUC alone, top 15 by |AUC-0.5|) ---')
sa = sorted(((c, auc(wf([c]))) for c in cols), key=lambda t: -abs(np.nan_to_num(t[1], nan=.5) - .5))
for c, a in sa[:15]: print(f'  {c:16} {FAM[c]:5} {a:.3f}')
print('\n--- calibration (OOS), probability deciles ---')
d = pd.DataFrame({'p': P, 'w': y, 'pts': ALLT.pts.values})[ok]; d['q'] = pd.qcut(d.p, 10, labels=False)
print(d.groupby('q').agg(p=('p', 'mean'), win=('w', 'mean'), pts=('pts', 'mean'), n=('w', 'size')).round(3).to_string())
lo, hi = Q[:, 0], Q[:, 1]; tier = np.where(np.isnan(P), np.nan, np.where(P >= hi, 1, np.where(P <= lo, -1, 0)))
print('\ntiers (thresholds from training predictions): bottom/mid/top  win rate | mean pts | n')
for t, nm in ((-1, 'bottom'), (0, 'mid'), (1, 'top')):
    m = tier == t; print(f'  {nm:6} {y[m].mean():.1%} | {ALLT.pts.values[m].mean():+.1f} | {int(m.sum())}   (2022+: {y[m & era2].mean():.1%} | {ALLT.pts.values[m & era2].mean():+.1f} | {int((m & era2).sum())})')
pickle.dump(dict(rows=ALLT[['leg', 'day', 'm']].copy(), P=P, tier=tier, pbar=float(np.nanmean(y))), open(SC + 'pwin.pkl', 'wb')); print('saved')
