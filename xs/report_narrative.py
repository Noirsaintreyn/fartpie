"""ICT-style NARRATIVE: combine the individual concepts (sweep, structure, displacement, FVG, trend/chop, momentum, session story) into one read per trade and test whether confluence has edge beyond single features. Nov 11 2026.
A) feature correlation matrix (redundancy) + Spearman vs pts. B) declared narrative count (6 pre-declared ICT-logic components) vs outcome. C) 2-way AND screen (all pairs of binary components, era-consistent only; expect false positives). D) walk-forward models (ridge, depth-2 boosted trees; train on years<Y, test Y, 2018+), top vs bottom tercile of predicted score. All features known at entry."""
import numpy as np, pandas as pd, itertools, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
from sklearn.linear_model import Ridge; from sklearn.ensemble import HistGradientBoostingRegressor
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
def load(nm):
    a = pd.read_pickle(SC + f'ict_{nm}.pkl').set_index(['day', 'm']); b = pd.read_pickle(SC + f'po3htf_{nm}.pkl').set_index(['day', 'm']); c = pd.read_pickle(SC + f'po3sess_{nm}.pkl').set_index(['day', 'm'])
    a = a[~a.index.duplicated()]; F = a.join(b[['PD', 'PS', 'PW', 'PM', 'HTF', 'n_with']], how='left').join(c[['AL', 'AN', 'LN', 'al']], how='left').reset_index()
    F['yr'] = F.day.dt.year; F['era'] = np.where(F.yr < 2022, 'A', 'B'); return F
NUM = ['fvg_with', 'disp', 'mss', 'pd_disc', 'sess_disc', 'open_mid', 'open_830', 'open_930', 'gap', 'prior_day', 'prior_loc', 'smt', 'rsi', 'adx', 'bbr', 'er', 'dol', 'PD', 'PS', 'PW', 'PM', 'HTF', 'n_with', 'AL', 'AN', 'LN']
for nm in ('V15a', 'RA'):
    F = load(nm).dropna(subset=['AL', 'n_with']); F = F.reset_index(drop=True)
    print(f'\n######## {nm}: {len(F)} trades, mean {F.pts.mean():+.1f}, win {np.mean(F.pts>0):.0%}')
    # A
    C = F[NUM].corr(method='spearman'); sp = F[NUM].apply(lambda s: s.corr(F.pts, method='spearman')).sort_values(key=abs, ascending=False)
    print('-- A. Spearman with pts (top 8):', ', '.join(f'{k} {v:+.2f}' for k, v in sp.head(8).items()))
    hi = [(a, b, C.loc[a, b]) for a, b in itertools.combinations(NUM, 2) if abs(C.loc[a, b]) >= .45]
    print('   redundant pairs |rho|>=.45:', ', '.join(f'{a}~{b} {r:+.2f}' for a, b, r in sorted(hi, key=lambda t: -abs(t[2]))[:14]))
    # B declared narrative count
    adx_lo = F[F.era == 'A'].adx.quantile(1/3); rsi_md = F[F.era == 'A'].rsi.median()
    comp = pd.DataFrame({'sweep_with': F.n_with >= 1, 'MSS_with': F.mss == 1, 'displacement_with': F.disp == 1, 'FVG_with': F.fvg_with == 1, 'trending(ADX>low3rd)': F.adx > adx_lo, 'momentum(RSI>med)': F.rsi > rsi_md})
    F['score'] = comp.sum(axis=1)
    print('-- B. declared narrative count (sweep, MSS, displacement, FVG, ADX not low, RSI>median):')
    for s in sorted(F.score.unique()):
        g = F[F.score == s]; a, b = g[g.era == 'A'].pts, g[g.era == 'B'].pts
        print(f'   score {int(s)}: n {len(g):4d} win {np.mean(g.pts>0):3.0%} mean {g.pts.mean():+6.1f} | A n{len(a):3d} {a.mean() if len(a) else np.nan:+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean() if len(b) else np.nan:+6.1f} ({tstat(b):+.1f})')
    lo_, hi_ = F[F.score <= 2].pts, F[F.score >= 5].pts
    print(f'   score<=2 n {len(lo_)} mean {lo_.mean():+.1f} | score>=5 n {len(hi_)} mean {hi_.mean():+.1f} | A diff {F[(F.score>=5)&(F.era=="A")].pts.mean()-F[(F.score<=2)&(F.era=="A")].pts.mean():+.1f}  B diff {F[(F.score>=5)&(F.era=="B")].pts.mean()-F[(F.score<=2)&(F.era=="B")].pts.mean():+.1f}')
    # C two-way AND
    comp2 = comp.copy(); comp2['London_swept_Asia_with'] = F.AL == 1; comp2['Asia_held'] = F.al == 0; comp2['wk/mo_swept_with'] = F.HTF == 1; comp2['buy_above_930open'] = F.open_930 == -1
    rows = []
    for a, b in itertools.combinations(comp2.columns, 2):
        m = comp2[a] & comp2[b]; g = F[m]; ga, gb = g[g.era == 'A'].pts, g[g.era == 'B'].pts
        if len(ga) >= 25 and len(gb) >= 25 and np.sign(ga.mean()) == np.sign(gb.mean()):
            rows.append((a, b, len(g), g.pts.mean(), ga.mean(), tstat(ga), gb.mean(), tstat(gb), min(abs(tstat(ga)), abs(tstat(gb)))))
    rows.sort(key=lambda r: -r[-1]); npairs = len(list(itertools.combinations(comp2.columns, 2)))
    print(f'-- C. 2-way AND confluence ({npairs} pairs tested; showing era-consistent, n>=25 per era, best by weaker-era t):')
    for r in rows[:6]: print(f'   {r[0]} & {r[1]}: n {r[2]} mean {r[3]:+.1f} | A {r[4]:+.1f} (t {r[5]:+.1f}) | B {r[6]:+.1f} (t {r[7]:+.1f})')
    # D walk-forward
    X = F[NUM].fillna(0).values; y = F.pts.values; yrs = F.yr.values; out = []
    for mdl, name in ((lambda: Ridge(alpha=300.0), 'ridge'), (lambda: HistGradientBoostingRegressor(max_depth=2, max_iter=60, learning_rate=.05, min_samples_leaf=25), 'boost d2')):
        pred = np.full(len(F), np.nan)
        for Y in range(2018, 2027):
            tr, te = yrs < Y, yrs == Y
            if tr.sum() < 150 or te.sum() == 0: continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; m_ = mdl().fit((X[tr] - mu) / sd, y[tr]); pred[te] = m_.predict((X[te] - mu) / sd)
        ok = ~np.isnan(pred); q = pd.qcut(pd.Series(pred[ok]).rank(method='first'), 3, labels=['bottom', 'mid', 'top']); yy = pd.Series(y[ok]).reset_index(drop=True)
        r = {k: yy[q == k] for k in ('bottom', 'mid', 'top')}; d = r['top'].mean() - r['bottom'].mean()
        se = np.sqrt(r['top'].var() / len(r['top']) + r['bottom'].var() / len(r['bottom']))
        print(f'-- D. walk-forward {name:9}: n {ok.sum()} | bottom third {r["bottom"].mean():+6.1f} mid {r["mid"].mean():+6.1f} top third {r["top"].mean():+6.1f} | top-bottom {d:+.1f} (t {d/se:+.1f}) | top win {np.mean(r["top"]>0):.0%} bottom win {np.mean(r["bottom"]>0):.0%}')
    F.to_pickle(SC + f'narr_{nm}.pkl')
