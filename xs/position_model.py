"""Nov 4 2026: can the open position's own state rate its odds from here? DECLARED BEFORE RESULTS. For every live-rule trade (17,447, 54 names with bars) and every session close while it is still
open (checkpoints 1..30 sessions after entry): features known at that close:
  age (sessions), pnl (close / entry - 1), peak (best high so far), dd (peak - pnl), pnl and dd in ATR units (daily ATR14 / price), atr_pct, track (1h), options ratio, regime (calm bull / bull+stress / bear calm / bear+stress),
  days to next earnings, SPY 20d trend.  Targets from that close to the trade's actual exit: remaining return, P(remaining > 0), P(remaining >= +5%).
Train on trades entered 2019-21, test on 2022-26 (untouched). Compare with an age-only baseline. Report: AUC, calibration by predicted-probability third, rank correlation with remaining return.
Output: model + feature spec saved to live_data/position_model.pkl only if it beats the age-only baseline out of sample."""
exec(open('xs/ema_common.py').read())
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
Xc = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet').set_index(['tk', 'entry', 'h1'])[['regime', 'dnext', 'spy20']]
rows = []
for i, r in XL.iterrows():
    B = BAR.get(r.tk); j = None if B is None else B['pos'].get(r.entry)
    if B is None or j is None or np.ndim(j) or j < 1: continue
    j = int(j); je = B['pos'].get(r.exit); jx = int(je) if (je is not None and not np.ndim(je)) else None
    if jx is None: continue
    px = B['o'][j]; m = B['m']; sess_end = [b for b in range(j, jx) if m[b] == 945]               # last RTH 15m bar of each session (15:45) while open
    atr = B['d_atr'][j - 1] / B['d_c'][j - 1] if B['d_c'][j - 1] else np.nan; pk = 0.0
    try: ctx = Xc.loc[(r.tk, r.entry, bool(r.h1))]; reg, dn, s20 = ctx.regime, float(ctx.dnext), float(ctx.spy20)
    except Exception: reg, dn, s20 = None, np.nan, np.nan
    last_b = j
    for k, b in enumerate(sess_end, 1):
        pk = max(pk, B['h'][last_b:b + 1].max() / px - 1); last_b = b + 1; c = B['c'][b]; pnl = c / px - 1
        if k > 30: break
        rem = (1 + r.ret) / (1 + pnl) - 1
        rows.append((r.tk, r.entry, k, pnl, pk, pk - pnl, atr, int(r.h1), r.ratio, reg, dn - k if dn == dn else np.nan, s20, rem))
D = pd.DataFrame(rows, columns=['tk', 'entry', 'age', 'pnl', 'peak', 'dd', 'atr', 'h1', 'ratio', 'regime', 'dnext', 'spy20', 'rem'])
D['pnl_atr'] = D.pnl / D.atr; D['dd_atr'] = D.dd / D.atr
for g in ('calm bull', 'bull+stress', 'bear calm', 'bear+stress'): D['r_' + g] = (D.regime == g).astype(float)
D['y'] = (D.rem > 0).astype(int); D['y5'] = (D.rem >= .05).astype(int); D['train'] = D.entry.dt.year <= 2021
FE = ['age', 'pnl', 'peak', 'dd', 'atr', 'pnl_atr', 'dd_atr', 'h1', 'ratio', 'dnext', 'spy20'] + [c for c in D if c.startswith('r_')]; BASE = ['age']
print('checkpoint rows', len(D), ' train', D.train.sum(), ' test', (~D.train).sum(), '  base rate P(rem>0): train', round(D[D.train].y.mean(), 3), 'test', round(D[~D.train].y.mean(), 3))
res = {}
for nm, cols in (('age only', BASE), ('age + pnl + peak + dd', ['age', 'pnl', 'peak', 'dd']), ('full state', FE)):
    tr, te = D[D.train], D[~D.train]; clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=.06, max_iter=150, min_samples_leaf=200, random_state=0).fit(tr[cols], tr.y)
    reg_ = HistGradientBoostingRegressor(max_depth=3, learning_rate=.06, max_iter=150, min_samples_leaf=200, random_state=0).fit(tr[cols], tr.rem.clip(-.3, .5))
    p = clf.predict_proba(te[cols])[:, 1]; er = reg_.predict(te[cols]); auc = roc_auc_score(te.y, p); ic = spearmanr(er, te.rem)[0]
    print(f'{nm:<26} test AUC(P rem>0) {auc:.3f}   rank corr of expected remaining return {ic:+.3f}'); res[nm] = (clf, reg_, cols, p, er)
clf, reg_, cols, p, er = res['full state']; te = D[~D.train].copy(); te['p'] = p; te['er'] = er; q = pd.qcut(te.p.rank(method='first'), 3, labels=['weak', 'middle', 'strong'])
print('\nOUT-OF-SAMPLE (2022-26) by predicted-probability third:')
for k, g in te.groupby(q): print(f'  {k:<7} P(green from here) predicted {g.p.mean():.0%} actual {g.y.mean():.0%} | mean remaining {g.rem.mean()*100:+.2f}% | reaches +5% {g.y5.mean():.0%} | n {len(g):,}')
te['bucket'] = pd.cut(te.age, [0, 1, 3, 7, 30]); print('\nwithin age buckets, strong vs weak third (actual P(green), mean remaining):')
for b, g in te.groupby('bucket', observed=True):
    qq = pd.qcut(g.p.rank(method='first'), 3, labels=['weak', 'middle', 'strong']); print(f'  age {b}:', {k: f'{x.y.mean():.0%} / {x.rem.mean()*100:+.1f}%' for k, x in g.groupby(qq)})
base_auc = roc_auc_score(te.y, res['age only'][3]); full_auc = roc_auc_score(te.y, p); print(f'\nlift over age-only: AUC {base_auc:.3f} -> {full_auc:.3f}')
if full_auc - base_auc >= .02:
    tr = D[D.train]; ptr = clf.predict_proba(tr[cols])[:, 1]; cut = (float(np.quantile(ptr, 1 / 3)), float(np.quantile(ptr, 2 / 3)))
    pickle.dump(dict(clf=clf, reg=reg_, cols=cols, cuts=cut, auc=full_auc, base_auc=base_auc), open('live_data/position_model.pkl', 'wb')); print('saved live_data/position_model.pkl, tercile cuts', cut)
else: print('NOT saved: no out-of-sample lift over age alone')
