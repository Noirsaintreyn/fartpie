"""Nov 10 2026: FEED ALL KNOWN CONTEXT INTO A WALK-FORWARD GBM ON THE NOISE ENGINE'S TRADES; WHERE IS THE EDGE. DECLARED BEFORE RESULTS.
Context (all known at 09:30): HAR forecast of today's log range (HARf), HAR parts, Yang-Zhang 20d rank (QQQ), VIX level & 5d change, QQQ ATM IV, GEX pct, DEX pct, vanna, charm, vanna flow, prior-day volume-profile / TPO features from noise_context (gp, dpoc, dtpoc, vaw, pocpos, sp, bal, open vs prior VA). Plus the trade-time features from adv_models M3.
Tests: (1) walk-forward GBM (refit each Jan) win-probability filter, context only / trade-time only / both; (2) permutation importance (gain) pooled; (3) does context predict the DAY's range (size) vs the trade's PAYOFF: corr of each feature with |pts| and with pts, both periods."""
import sys, io, contextlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
exec(open(f'{SP}/ctxhead.py').read())
from sklearn.linear_model import LinearRegression
from arch import arch_model
M = pd.DataFrame({'HAR_d': X.d, 'HAR_w': X.w, 'HAR_m': X.m}); M['YZr'] = YZr.shift(1).reindex(ix, method='ffill').values
M['VIX'] = np.log(vx.shift(1).reindex(ix, method='ffill')).values; M['VIXd5'] = np.log(vx).diff(5).shift(1).reindex(ix, method='ffill').values; M['IV'] = np.log(G1.atm_iv).shift(1).reindex(ix, method='ffill').values
M['y'] = lr.values; fitm = M.dropna().index < '2020-01-01'; D = M.dropna(subset=['HAR_d','HAR_w','HAR_m','y']); hf = LinearRegression().fit(D[D.index<'2020-01-01'][['HAR_d','HAR_w','HAR_m']], D[D.index<'2020-01-01'].y); M['HARf'] = np.nan; M.loc[D.index,'HARf'] = hf.predict(D[['HAR_d','HAR_w','HAR_m']]); M = M.drop(columns='y')
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
Fg = pd.DataFrame(index=G1.index); Fg['gex'] = OWN.reindex(Fg.index); Fg['dex'] = G1.call_dex + G1.put_dex; Fg['vanna'] = G2.net_vanna.reindex(Fg.index); Fg['charm'] = G2.net_charm.reindex(Fg.index); Fg['iv5'] = G1.atm_iv.diff(5); Fg['vflow'] = np.sign(Fg.vanna) * -np.sign(Fg.iv5); Fg['gexp'] = pct(Fg.gex); Fg['dexp'] = pct(Fg.dex)
Fg = Fg[['gexp','dexp','vanna','charm','vflow']].shift(1).reindex(ix, method='ffill'); M = M.join(Fg)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import noise_context as NC
VP = NC.F.copy(); VP.index = pd.to_datetime(VP.index).normalize(); vpc = [c for c in VP.columns if c not in ('gex',) and VP[c].dtype != object][:14]; M = M.join(VP[vpc], rsuffix='_vp')

import lightgbm as lgb
G = pd.read_pickle(f'{SP}/G_noise.pkl'); G['day'] = pd.to_datetime(G.day).dt.normalize(); G = G.join(M, on='day'); CXC = [c for c in M.columns]; TRC = [c for c in G.columns if c not in CXC + ['day','pts']]
G = G[G.day >= '2018-02-01'].reset_index(drop=True); yr = G.day.dt.year.values; print(len(G), 'trades; context cols', len(CXC), 'trade cols', len(TRC), flush=True)
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
def ev(label, keep):
    out = []
    for nm, f in (('18-21', yr < 2022), ('22-26', yr >= 2022)):
        d = G[f & keep]; s = d.groupby('day').pts.sum().reindex(pd.DatetimeIndex(sorted(G.day[f].unique()))).fillna(0); out.append(f'{nm}: {s.mean():+5.2f}/d (t {s.mean()/(s.std()/np.sqrt(len(s))):+5.2f}, m/s {s.mean()/s.std():.3f}, n {len(d)})')
    print(f'  {label:<34} ' + ' | '.join(out), flush=True)
imp = {}
for lab, cols in (('trade-time only', TRC), ('context only', CXC), ('both', TRC + CXC)):
    keep = np.zeros(len(G), bool); gain = pd.Series(0.0, index=cols)
    for Y in range(2018, 2027):
        tr = yr < Y; te = yr == Y
        if te.sum() == 0 or tr.sum() < 400: continue
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=8, min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=5.0, verbose=-1, importance_type='gain').fit(G.loc[tr, cols], G.pts[tr] > 0)
        keep[te] = m.predict_proba(G.loc[te, cols])[:, 1] >= np.quantile(m.predict_proba(G.loc[tr, cols])[:, 1], .5); gain += pd.Series(m.feature_importances_, index=cols)
    imp[lab] = (gain / gain.sum()).sort_values(ascending=False); ev(f'GBM filter, {lab}', keep)
ev('no filter (all noise trades)', np.ones(len(G), bool))
print('\nTop gain importance (both):'); print(imp['both'].head(12).round(3).to_string())
print('\n=== context vs SIZE (|pts|) and vs PAYOFF (pts): spearman, 18-21 | 22-26 ===')
for c in CXC:
    r = []
    for f in (yr < 2022, yr >= 2022):
        d = G[f][[c, 'pts']].dropna(); r.append((d[c].corr(d.pts.abs(), method='spearman'), d[c].corr(d.pts, method='spearman')) if len(d) > 100 else (np.nan, np.nan))
    print(f'  {c:<8} size {r[0][0]:+.2f} | {r[1][0]:+.2f}   payoff {r[0][1]:+.2f} | {r[1][1]:+.2f}')
