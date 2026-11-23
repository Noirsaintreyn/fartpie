"""Nov 22 2026 report 366. Rey: (A) 'a model using all current market context, replayed over history, that says the statistical likelihood a vector flip is strong or not'; (B) 'a volatility z-score so we can trade oversold/overbought better, because the regular z-score misfires a lot'.  Levels are out.
Declared before running.
A. CONTEXT REPLAY (walk-forward, no look-ahead): for each test year Y in 2021..2026 fit on all flips that EXITED before Jan 1 of Y, predict the flips entered in Y.  Features = the 114 pre-entry context features (stock, sector/peer, market, clock; X359 minus the same-day market return).  Two models: (1) L2 logistic (C=0.01, standardised, clipped +-5); (2) ANALOG: PCA(10) of the standardised features and the 300 nearest past flips (Euclidean) -> share of them that were strong.  'Strong' = trade return > +5% (the tail that carries the profit); also P(win).  Pass: pooled out-of-sample AUC > 0.55 for 'strong' AND top-decile mean trade return above the all-flip mean with t >= 2 AND the same sign in at least 5 of 6 years.  Also a calibration table (predicted vs realised by decile).
B. VOLATILITY Z-SCORE.  Move = 5-day log return r5 ending at the prior close.  Three z = r5 / (sigma x sqrt5) with sigma measured on days BEFORE the 5-day window so the move does not inflate its own denominator:  z_reg (std of daily returns, 20d), z_yz (Yang-Zhang 20d), z_har (HAR blend 0.3 daily + 0.4 weekly + 0.3 monthly Parkinson variance; the fixed weights used in report 358).  Test 1, ALL stock-days: misfire = oversold (z < -2) followed by a further 5-day decline; compare P(further decline), mean next-5d return, and how often |z| > 2 (normal = 4.6%).  Test 2, AT FLIPS: bull flips with z < -2 on the prior close (oversold) and z > +2 (extended) vs the rest, mean trade return train 2019-21 | test 2022-26, week-clustered t.  Declared direction: oversold at the flip -> BETTER, extended -> WORSE, and z_yz/z_har should misfire less than z_reg.  Pass: same sign in both halves, |t| >= 2 in at least one."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry); b['exit'] = pd.to_datetime(b.exit); b['day'] = pd.to_datetime(b.day).astype('datetime64[ns]')
X = pd.read_parquet(SP + 'X359.parquet'); feats = [c for c in X.columns if c not in ('ret', 'tr', 'b_mkt')]
M = pd.concat([b[['tk', 'entry', 'exit', 'day', 'ret']], X[feats].reset_index(drop=True)], axis=1); M = M[M.ret.notna()].reset_index(drop=True); M['yr'] = M.entry.dt.year; M['big'] = (M.ret > .05).astype(int); M['win'] = (M.ret > 0).astype(int); M['wk'] = M.entry.dt.to_period('W').astype(str)
print('== A. CONTEXT REPLAY, walk-forward (%d flips, %d features) ==' % (len(M), len(feats)), flush=True)
rows = []
for Y in range(2021, 2027):
    tr = M[M.exit < f'{Y}-01-01']; te = M[M.yr == Y]
    if len(te) < 50: continue
    med = tr[feats].median(); sc = StandardScaler().fit(tr[feats].fillna(med)); f = lambda D: np.clip(sc.transform(D[feats].fillna(med)), -5, 5); Xt, Xe = f(tr), f(te)
    lg = LogisticRegression(C=.01, max_iter=2000).fit(Xt, tr.big); p1 = lg.predict_proba(Xe)[:, 1]; lw = LogisticRegression(C=.01, max_iter=2000).fit(Xt, tr.win); pw = lw.predict_proba(Xe)[:, 1]
    pc = PCA(10, random_state=0).fit(Xt); nn = NearestNeighbors(n_neighbors=300).fit(pc.transform(Xt)); _, ix = nn.kneighbors(pc.transform(Xe)); p2 = tr.big.values[ix].mean(1)
    rows.append(te.assign(p_lg=p1, p_win=pw, p_an=p2))
P = pd.concat(rows)
def rep(col, lab):
    aucs = [roc_auc_score(g.big, g[col]) for _, g in P.groupby('yr')]; auc = roc_auc_score(P.big, P[col]); top = P[P[col] >= P[col].quantile(.9)]; bot = P[P[col] <= P[col].quantile(.1)]
    t = (top.ret.mean() - P.ret.mean()) / (top.ret.std() / np.sqrt(len(top))); pos = sum(g[g[col] >= g[col].quantile(.9)].ret.mean() > g.ret.mean() for _, g in P.groupby('yr'))
    print('%-22s pooled OOS AUC(strong) %.3f | by year %s | top decile mean %+.2f%% (all %+.2f%%, t %.1f), bottom decile %+.2f%% | years top>all: %d/6 | P(strong) top %.1f%% vs all %.1f%%' % (lab, auc, ' '.join('%.2f' % a for a in aucs), top.ret.mean() * 100, P.ret.mean() * 100, t, bot.ret.mean() * 100, pos, top.big.mean() * 100, P.big.mean() * 100), flush=True)
rep('p_lg', 'logistic (strong)'); rep('p_an', 'analog 300-NN (strong)')
print('P(win) model: pooled AUC %.3f' % roc_auc_score(P.win, P.p_win))
print('calibration, logistic P(strong), deciles (predicted -> realised, mean trade ret):')
P['dec'] = pd.qcut(P.p_lg.rank(method='first'), 10, labels=False); cal = P.groupby('dec').agg(pred=('p_lg', 'mean'), real=('big', 'mean'), ret=('ret', 'mean'), n=('ret', 'size')); print((cal * [100, 100, 100, 1]).round(1).to_string(), flush=True)
print('\n== B. VOLATILITY Z-SCORES ==')
D = []
for tk in sorted(b.tk.unique()):
    fl = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(fl); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    H = H[(H.index.hour * 60 + H.index.minute >= 570) & (H.index.hour * 60 + H.index.minute < 960)]; g = H.groupby(H.index.normalize())
    d = pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()}); d = d[(d.o > 0) & (d.h >= d.l)]
    r = np.log(d.c).diff(); r[r.abs() > .5] = np.nan; n = 20; k = .34 / (1.34 + (n + 1) / (n - 1))
    on = np.log(d.o / d.c.shift()); oc = np.log(d.c / d.o); rs = np.log(d.h / d.c) * np.log(d.h / d.o) + np.log(d.l / d.c) * np.log(d.l / d.o)
    yz = np.sqrt((on.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()).clip(lower=1e-10)); rv = (np.log(d.h / d.l) ** 2) / (4 * np.log(2))
    har = np.sqrt(.3 * rv + .4 * rv.rolling(5).mean() + .3 * rv.rolling(22).mean()); reg = r.rolling(20).std(); r5 = np.log(d.c / d.c.shift(5))
    o = pd.DataFrame({'tk': tk, 'date': d.index, 'z_reg': r5 / (reg.shift(5) * np.sqrt(5)), 'z_yz': r5 / (yz.shift(5) * np.sqrt(5)), 'z_har': r5 / (har.shift(5) * np.sqrt(5)), 'fwd5': np.log(d.c.shift(-5) / d.c)}); D.append(o.dropna())
A = pd.concat(D); A['date'] = pd.to_datetime(A.date).astype('datetime64[ns]'); A = A[A.date >= '2019-01-01']; A['half'] = np.where(A.date < '2022-01-01', 'train', 'test'); A['wk'] = A.date.dt.to_period('W').astype(str)
print('Test 1, all stock-days (%d): oversold z<-2 -> further 5d decline?' % len(A))
for z in ('z_reg', 'z_yz', 'z_har'):
    out = []
    for h in ('train', 'test'):
        S = A[A.half == h]; o = S[S[z] < -2]; ob = S[S[z] > 2]; m = sm.OLS(o.fwd5.values, np.ones(len(o))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(o.wk)[0]})
        out.append('|z|>2 %.1f%% (n_os %d) P(fall) %.1f%% mean %+.2f%% (t %+.1f)' % ((S[z].abs() > 2).mean() * 100, len(o), (o.fwd5 < 0).mean() * 100, o.fwd5.mean() * 100, m.tvalues[0]))
    print('%-6s TRAIN %s | TEST %s' % (z, out[0], out[1]), flush=True)
print('\nTest 2, at bull flips (z on the prior close): oversold / extended vs the rest, mean trade return')
F = M.assign(pd_=M.day - pd.Timedelta(days=1)).sort_values('pd_'); F = pd.merge_asof(F, A.sort_values('date')[['tk', 'date', 'z_reg', 'z_yz', 'z_har']], left_on='pd_', right_on='date', by='tk', direction='backward', tolerance=pd.Timedelta(days=5)); F['half'] = np.where(F.entry < '2022-01-01', 'train', 'test')
for z in ('z_reg', 'z_yz', 'z_har'):
    for lab, mask in (('oversold z<-2', lambda s: s[z] < -2), ('extended z>+2', lambda s: s[z] > 2)):
        out = []
        for h in ('train', 'test'):
            S = F[(F.half == h) & F[z].notna()]; T = S[mask(S)]; Rr = S[~mask(S)]
            if len(T) < 15: out.append('n=%d (too few)' % len(T)); continue
            Zz = pd.concat([T.assign(x=1.0), Rr.assign(x=0.0)]); m = sm.OLS(Zz.ret.values, sm.add_constant(Zz.x.values)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Zz.wk)[0]}); out.append('%+.2f%% (t %+.1f) n=%d, win %.0f%% vs %.0f%%' % (m.params[1] * 100, m.tvalues[1], len(T), T.win.mean() * 100, Rr.win.mean() * 100))
        print('%-6s %-14s TRAIN %-42s TEST %s' % (z, lab, out[0], out[1]), flush=True)
