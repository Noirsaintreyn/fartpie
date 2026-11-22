"""Nov 22 2026 report 361. Rey: use advanced statistics (bootstrap, extreme-value theory, ruin/first-passage, hierarchical shrinkage) on the live book, and look for anything that identifies bigger/better trades.
Declared before running:
 A. BLOCK BOOTSTRAP: live-order 6-slot book daily returns, 21-day blocks, 5000 resamples of the 2019-26 path -> 90% range for CAGR, max DD, Sharpe; P(CAGR<0), P(DD worse than -30%/-40%).  Also month-clustered bootstrap of the trade mean (12,320 flips).
 B. EXTREME VALUES: Hill tail index of positive trades (top 5%/10%), GPD shape above the 90th pct, profit share of the top 1/5/10% of trades, mean trade with the top 1%/5% removed.
 C. EMPIRICAL-BAYES NAME SHRINKAGE: key = (n*mean_i + 50*m0)/(n+50) from the name's own PRIOR closed trades (exit < this entry), m0 = all prior closed trades; high-first pick order in the book (no look-ahead), vs live and the 360 random-order null (95th pct CAGR 23.0%).  Also train->test rank correlation of name means.
 D. BIG-WINNER MODEL: L2 logistic on the 114 pre-entry features, label ret>+5%, fit on flips entered before 2021-12 (exit <2022), judged ONLY on flips entered 2022+: AUC, top-decile lift, and the book run with the model score as pick order in 2022+ (live order before).  Pass: test AUC>0.55, top-decile mean ret > base by t>2, test-half book CAGR > live 17.5%.
THEORY: C: some names' flips pay more (liquidity, trend persistence) and shrinkage stops one lucky history dominating; D: whatever predicts the +5% tail is the one thing worth a slot.  Expected: A/B describe risk; C/D most likely fail like 354-360."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
X = pd.read_parquet(SP + 'X359.parquet'); b = pd.read_parquet(SP + 'b302.parquet').reset_index(drop=True); b['entry'] = pd.to_datetime(b.entry); X['tk'] = b.tk.values; X['entry'] = b.entry.values
X = X.rename(columns={c: 'f_' + c for c in X.columns if c not in ('ret', 'tr', 'tk', 'entry')})
feats = [c for c in X.columns if c not in ('ret', 'tr', 'tk', 'entry', 'f_b_mkt')]
B0 = B0.merge(X.drop(columns=['ret', 'tr']), on=['tk', 'entry'], how='left')
def liveorder(Bs): return Bs.assign(_k=Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable')
order = liveorder; TK, _ = sim_swap(B0, None); z = spread(TK); z = z[z.index >= '2019-01-01']
def stats_of(a): return cagr(a), mdd(a), sharpe(a.values)
c0, d0, s0 = stats_of(z); print('LIVE book: CAGR %.1f%% DD %.1f%% Sharpe %.2f, %d trades' % (c0 * 100, d0 * 100, s0, len(TK)))
print('\n== A. BLOCK BOOTSTRAP (21-day blocks, 5000 resamples) ==')
v = z.values; n = len(v); rng = np.random.default_rng(1); L = 21; res = []
for _ in range(5000):
    st = rng.integers(0, n - L, size=n // L + 1); s = np.concatenate([v[i:i + L] for i in st])[:n]; e = np.cumprod(1 + s); y = n / 252
    res.append((e[-1] ** (1 / y) - 1, (e / np.maximum.accumulate(e) - 1).min(), s.mean() / s.std() * np.sqrt(252)))
R = np.array(res); q = lambda a: np.percentile(a, [5, 50, 95])
print('CAGR 5/50/95: %.1f / %.1f / %.1f %%' % tuple(q(R[:, 0]) * 100)); print('maxDD 5/50/95: %.1f / %.1f / %.1f %%' % tuple(q(R[:, 1]) * 100)); print('Sharpe 5/50/95: %.2f / %.2f / %.2f' % tuple(q(R[:, 2])))
print('P(CAGR<0) %.1f%% | P(DD<-30%%) %.1f%% | P(DD<-40%%) %.1f%%' % ((R[:, 0] < 0).mean() * 100, (R[:, 1] < -.3).mean() * 100, (R[:, 1] < -.4).mean() * 100))
B0['ym'] = B0.entry.dt.to_period('M'); g = B0.groupby('ym').ret.agg(['sum', 'count']); mu = B0.ret.mean(); bm = []
for _ in range(5000): k = rng.integers(0, len(g), len(g)); bm.append(g['sum'].values[k].sum() / g['count'].values[k].sum())
bm = np.array(bm); print('trade mean %.3f%%, month-clustered bootstrap 90%% range %.3f%% to %.3f%%, P(mean<=0) %.2f%%' % (mu * 100, np.percentile(bm, 5) * 100, np.percentile(bm, 95) * 100, (bm <= 0).mean() * 100))
print('\n== B. EXTREME VALUES ==')
r = B0.ret.values; pos = np.sort(r[r > 0])[::-1]
for f in (.05, .10):
    k = int(len(pos) * f); h = 1 / np.mean(np.log(pos[:k] / pos[k])); print('Hill tail index alpha (top %d%% of winners, k=%d): %.2f' % (f * 100, k, h))
u = np.quantile(r, .9); ex = r[r > u] - u; c, loc, sc = stats.genpareto.fit(ex, floc=0); print('GPD above 90th pct (%.2f%%): shape xi %.2f (xi>0 = heavy tail; xi>=0.5 = infinite variance), scale %.2f%%' % (u * 100, c, sc * 100))
gp = np.sort(r[r > 0])[::-1]; tot = gp.sum()
for f in (.01, .05, .10): print('top %d%% of all trades = %.0f%% of gross profit' % (f * 100, gp[:int(len(r) * f)].sum() / tot * 100))
srt = np.sort(r)
for f in (.01, .05): print('mean trade with top %d%% removed: %+.3f%% (full %+.3f%%)' % (f * 100, srt[:int(len(r) * (1 - f))].mean() * 100, mu * 100))
print('\n== C. EMPIRICAL-BAYES NAME SHRINKAGE ==')
B0 = B0.sort_values('entry').reset_index(drop=True); sh = np.zeros(len(B0)); ex_ = B0.exit.values; en = B0.entry.values; rt = B0.ret.values; tk = B0.tk.values
srtx = np.argsort(ex_); ex_s = ex_[srtx]
for i in range(len(B0)):
    m = np.searchsorted(ex_s, en[i]); ids = srtx[:m]
    if m < 30: sh[i] = 0; continue
    m0 = rt[ids].mean(); same = rt[ids][tk[ids] == tk[i]]; sh[i] = (same.sum() + 50 * m0) / (len(same) + 50)
B0['f_shrunk'] = sh
def order_f(col, hi=True):
    def o(Bs): return Bs.assign(_k=(-Bs[col].values if hi else Bs[col].values)).sort_values(['entry', '_k', 'tk'], kind='stable')
    return o
def half_stats(TK):
    zz = spread(TK); zz = zz[zz.index >= '2019-01-01']; return cagr(zz), mdd(zz), sharpe(zz.values), cagr(zz[zz.index < '2022-01-01']), cagr(zz[zz.index >= '2022-01-01'])
order = order_f('f_shrunk'); TKc, _ = sim_swap(B0, None); hc = half_stats(TKc); print('shrunk-name-mean HIGH first: CAGR %.1f%% DD %.1f%% Sharpe %.2f | train %.1f test %.1f   (live 23.6 / -23.5 / 2.60 | 33.8 17.5; null p95 CAGR 23.0)' % (hc[0] * 100, hc[1] * 100, hc[2], hc[3] * 100, hc[4] * 100))
nm = B0.assign(per=np.where(B0.entry < '2022-01-01', 'a', 'b')).groupby(['tk', 'per']).ret.mean().unstack().dropna(); print('name mean return 2019-21 vs 2022-26: Spearman %.2f (n=%d names, p=%.3f)' % (*stats.spearmanr(nm.a, nm.b), len(nm)) if False else 'name mean return 2019-21 vs 2022-26: Spearman rho %.2f, p %.3f, n=%d names' % (stats.spearmanr(nm.a, nm.b)[0], stats.spearmanr(nm.a, nm.b)[1], len(nm)))
print('\n== D. BIG-WINNER MODEL (train: entered <2021-12 and exit <2022; test: entered 2022+) ==')
tr = B0[(B0.entry < '2021-12-01') & (B0.exit < '2022-01-01')]; te = B0[B0.entry >= '2022-01-01'].copy()
med = tr[feats].median(); Xt = tr[feats].fillna(med).values; sc = StandardScaler().fit(Xt); Xt = np.clip(sc.transform(Xt), -5, 5); Xe = np.clip(sc.transform(te[feats].fillna(med).values), -5, 5)
yt = (tr.ret > .05).values; ye = (te.ret > .05).values
for C in (.003, .01, .05):
    m = LogisticRegression(C=C, max_iter=2000).fit(Xt, yt); p = m.predict_proba(Xe)[:, 1]; auc = roc_auc_score(ye, p); top = te.ret.values[p >= np.quantile(p, .9)]
    t = (top.mean() - te.ret.mean()) / (top.std() / np.sqrt(len(top))); print('C=%.3f: train AUC %.3f, TEST AUC %.3f | test top-decile mean ret %+.2f%% vs all %+.2f%% (t=%.1f), P(+5%%) top decile %.1f%% vs %.1f%%' % (C, roc_auc_score(yt, m.predict_proba(Xt)[:, 1]), auc, top.mean() * 100, te.ret.mean() * 100, t, (top > .05).mean() * 100, ye.mean() * 100))
m = LogisticRegression(C=.01, max_iter=2000).fit(Xt, yt); coef = pd.Series(m.coef_[0], index=feats).sort_values(); print('largest +/- coefficients (C=.01):', coef.tail(5).round(3).to_dict(), coef.head(5).round(3).to_dict())
B0['f_score'] = np.nan; B0.loc[te.index, 'f_score'] = m.predict_proba(Xe)[:, 1]
def order_d(Bs):
    k = np.where(Bs.entry.values >= np.datetime64('2022-01-01'), -Bs.f_score.fillna(0).values, Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values); return Bs.assign(_k=k).sort_values(['entry', '_k', 'tk'], kind='stable')
order = order_d; TKd, _ = sim_swap(B0, None); hd = half_stats(TKd); print('book with model score as 2022+ pick order: test CAGR %.1f%% (live 17.5) | full CAGR %.1f%% DD %.1f%% Sharpe %.2f' % (hd[4] * 100, hd[0] * 100, hd[1] * 100, hd[2]))
