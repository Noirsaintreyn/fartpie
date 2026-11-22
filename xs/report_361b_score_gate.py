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
print('\n== D. BIG-WINNER MODEL (train: entered <2021-12 and exit <2022; test: entered 2022+) ==')
tr = B0[(B0.entry < '2021-12-01') & (B0.exit < '2022-01-01')]; te = B0[B0.entry >= '2022-01-01'].copy()
med = tr[feats].median(); Xt = tr[feats].fillna(med).values; sc = StandardScaler().fit(Xt); Xt = np.clip(sc.transform(Xt), -5, 5); Xe = np.clip(sc.transform(te[feats].fillna(med).values), -5, 5)
yt = (tr.ret > .05).values; ye = (te.ret > .05).values
for C in (.003, .01, .05):
    m = LogisticRegression(C=C, max_iter=2000).fit(Xt, yt); p = m.predict_proba(Xe)[:, 1]; auc = roc_auc_score(ye, p); top = te.ret.values[p >= np.quantile(p, .9)]
    t = (top.mean() - te.ret.mean()) / (top.std() / np.sqrt(len(top))); print('C=%.3f: train AUC %.3f, TEST AUC %.3f | test top-decile mean ret %+.2f%% vs all %+.2f%% (t=%.1f), P(+5%%) top decile %.1f%% vs %.1f%%' % (C, roc_auc_score(yt, m.predict_proba(Xt)[:, 1]), auc, top.mean() * 100, te.ret.mean() * 100, t, (top > .05).mean() * 100, ye.mean() * 100))
m = LogisticRegression(C=.01, max_iter=2000).fit(Xt, yt); coef = pd.Series(m.coef_[0], index=feats).sort_values(); print('largest +/- coefficients (C=.01):', coef.tail(5).round(3).to_dict(), coef.head(5).round(3).to_dict())

B0['f_score'] = np.nan; B0.loc[te.index, 'f_score'] = m.predict_proba(Xe)[:, 1]
def half_stats(TK):
    zz = spread(TK); zz = zz[zz.index >= '2019-01-01']; return cagr(zz), mdd(zz), sharpe(zz.values), cagr(zz[zz.index < '2022-01-01']), cagr(zz[zz.index >= '2022-01-01'])
def order_d(Bs):
    k = np.where(Bs.entry.values >= np.datetime64('2022-01-01'), -Bs.f_score.fillna(0).values, Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values); return Bs.assign(_k=k).sort_values(['entry', '_k', 'tk'], kind='stable')
print('\n== D2. USE THE SCORE AS A GATE / MIXED ORDER (2022+ only; before 2022 live order, all flips) ==')
def run(gate_q=None, mix=False):
    thr = B0.f_score.quantile(gate_q) if gate_q else None
    def o(Bs):
        live = Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values
        if mix: live = np.where(Bs.entry.values >= np.datetime64('2022-01-01'), Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values + 0*Bs.f_score.fillna(0).values, live)
        return Bs.assign(_k=live).sort_values(['entry', '_k', 'tk'], kind='stable')
    global order; order = o
    Bx = B0 if thr is None else B0[(B0.entry < '2022-01-01') | (B0.f_score >= thr)].reset_index(drop=True)
    TKx, _ = sim_swap(Bx, None); h = half_stats(TKx); return h, len(TKx)
for q in (None, .2, .4, .5, .6, .8):
    h, nt = run(q); print('gate: keep score >= %s quantile: trades %d, test CAGR %.1f%%, full CAGR %.1f%% DD %.1f%% Sharpe %.2f' % (q, nt, h[4] * 100, h[0] * 100, h[1] * 100, h[2]))
tk_live = pd.DataFrame(TK, columns=['tk','entry','exit','ret','ratio']); tl = tk_live[tk_live.entry >= '2022-01-01']; print('live book 2022+ trades:', len(tl), 'mean ret %.3f%%' % (tl.ret.mean()*100), '| all test flips mean %.3f%%' % (te.ret.mean()*100))
order = order_d; TKd, _ = sim_swap(B0, None); td = pd.DataFrame(TKd, columns=['tk','entry','exit','ret','ratio']); td = td[td.entry >= '2022-01-01']; print('score-order book 2022+ trades:', len(td), 'mean ret %.3f%%' % (td.ret.mean()*100), 'mean score %.3f vs live-book mean score %.3f' % (B0.set_index(['tk','entry']).f_score.reindex(pd.MultiIndex.from_frame(td[['tk','entry']])).mean(), B0.set_index(['tk','entry']).f_score.reindex(pd.MultiIndex.from_frame(tl[['tk','entry']])).mean()))
