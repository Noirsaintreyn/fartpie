"""Nov 8 2026: WHY NOT JUST CLASSIFY GOOD FLIPS? Head-to-head, out of sample. DECLARED BEFORE RESULTS.
Every RTH 15m vector flip. Train on 2018-21 only, test on 2022 - Oct 2026 (no retuning). Targets: 'good' = trend-hold result > 0 (TH) and 75 / 50 result > 0 (F75).
Inputs: the 9 composite checks as 0/1 plus their continuous versions (RVOL, 5-day vol, ATR expansion, GEX, VWAP z, VWAP distance, minute, 21-day z, 50-day trend,
prior-day Yang-Zhang, value / range location). Models: logistic regression (L2), shallow gradient boosting (depth 2, 150 trees), and the plain COUNT of checks.
Report on the test years: AUC (0.50 = coin flip), win rate and points per flip by predicted third, and the book with contracts = the model's tier (0 / 1 / 2 / 3 by
predicted thirds and top decile) vs the count (checks - 3). Also: how many flips there are to learn from, and how far the 2018-21 relationships held."""
_t = open('xs/nq_composite_th.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
p50, q50 = run1(50.0); X['th50'] = p50
chk = pd.DataFrame({'gex': (X.gex < 0), 'vol': (X.rv5 >= cut['rv5']), 'rvol': (X.rvol >= cut['rvol']), 'atr': (X.comp >= cut['comp']), 'vp': locok, 'early': early,
                    'range': X.rng.isin(['discount half', 'beyond trade-side extreme']), 'fade': fade, 'ruleA': X.ruleA == 1}).astype(float)
cont = pd.DataFrame({'rvol_c': np.log(X.rvol.clip(.05)), 'rv5_c': X.rv5, 'comp_c': np.log(X.comp.clip(.05)), 'gex_c': np.sign(X.gex) * np.log1p(np.abs(X.gex) / 1e8), 'vz': X.vz.clip(-4, 6),
                     'vwd': X.vwd.clip(-4, 6), 'me': X.me, 'z21': X.z21.clip(-4, 4), 'yz1': X.yz1, 'inval': (X['value'] == 'value, trade side of POC').astype(float),
                     'prem': (X.rng == 'premium half').astype(float)})
FEAT = pd.concat([chk, cont], axis=1).fillna(0.0); tr = (X.day < '2022-01-01').values; te = ~tr
mu, sd = FEAT[tr].mean(), FEAT[tr].std().replace(0, 1); Z = ((FEAT - mu) / sd).values
if __name__ == '__main__':
    print(f'flips to learn from: {tr.sum()} (2018-21), test {te.sum()} (2022-26); good flips (TH > 0) {np.mean(X.th50[tr] > 0):.0%} / {np.mean(X.th50[te] > 0):.0%}; '
          f'(F75 > 0) {np.mean(X.f75[tr] > 0):.0%} / {np.mean(X.f75[te] > 0):.0%}', flush=True)
    print('\nwhich single checks held their sign from 2018-21 to 2022-26 (TH points, with minus without):', flush=True)
    for c in chk.columns:
        a = X.th50[tr & (chk[c] == 1).values].mean() - X.th50[tr & (chk[c] == 0).values].mean(); b = X.th50[te & (chk[c] == 1).values].mean() - X.th50[te & (chk[c] == 0).values].mean()
        print(f'  {c:<6} {a:+6.1f} -> {b:+6.1f} {"held" if np.sign(a) == np.sign(b) else "FLIPPED"}', flush=True)
    for tgt, ycol in (('TH > 0', 'th50'), ('F75 > 0', 'f75')):
        y = (X[ycol] > 0).astype(int).values; pts = X[ycol].values
        M = {'logistic': LogisticRegression(C=.2, max_iter=3000).fit(Z[tr], y[tr]).predict_proba(Z)[:, 1],
             'boosting': GradientBoostingClassifier(max_depth=2, n_estimators=150, learning_rate=.05, subsample=.8, random_state=0).fit(Z[tr], y[tr]).predict_proba(Z)[:, 1],
             'count of checks': SCt.astype(float) + np.random.RandomState(0).rand(len(X)) * 1e-6}
        print(f'\nTARGET {tgt} (points = {ycol})   AUC train / test | test by predicted third (cut on train): win, points per flip  low | mid | high | top 10%', flush=True)
        for nm, s in M.items():
            lo, hi, t10 = np.quantile(s[tr], [1 / 3, 2 / 3, .9]); b = np.select([s <= lo, s <= hi], [0, 1], 2)
            out = ' | '.join(f'{np.mean(y[te & (b == v)]):3.0%} {np.nanmean(pts[te & (b == v)]):+6.1f} (n{(te & (b == v)).sum():3d})' for v in (0, 1, 2))
            print(f'  {nm:<16} {roc_auc_score(y[tr], s[tr]):.3f} / {roc_auc_score(y[te], s[te]):.3f} | {out} | {np.mean(y[te & (s >= t10)]):3.0%} {np.nanmean(pts[te & (s >= t10)]):+6.1f} (n{(te & (s >= t10)).sum():3d})', flush=True)
            if ycol == 'th50':
                tier = np.select([s <= lo, s <= hi, s < t10], [0.0, 1.0, 2.0], 3.0) if nm != 'count of checks' else comp_sz
                Pd, Mo, cum, T = book(np.full(len(X), 50.0), tier, '2022-01-01', '2027-01-01')
                print(f'  {"":<16} book 2022-26 (contracts by tier): {Pd.mean():+.2f} pts/day, avg {T.sz.mean():.2f} contracts, Sharpe {Pd.mean() / Pd.std() * np.sqrt(252):.2f}, max DD {(cum - cum.cummax()).min():+.0f}', flush=True)
