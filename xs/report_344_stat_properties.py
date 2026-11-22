"""Nov 22 2026 report 344. Rey: break the system down into its statistical properties (the way auction-market theory uses a Gaussian) and see if that helps.  Declared before running.
A. SHAPE of each model's trade P&L (8 NQ models, legs287_ext, 2018+): skew, excess kurtosis, Jarque-Bera, Hill tail index alpha of the winners (top 10%) and of the losers, Student-t df vs Gaussian log-lik (on standardised trades), top-5% profit share.  Question: is a Gaussian summary (mean/sd/Sharpe) a fair description, i.e. is alpha > 2 (finite variance)?
B. CLUSTERING of the stack's daily P&L: ACF of |P&L| and signed P&L at lags 1,5,10,20 with Ljung-Box p, train vs test.  THEORY: trend days cluster (volatility/persistence clustering in NQ: range forecasts work, report 253-270); if big stack days cluster, yesterday's information predicts tomorrow's P&L magnitude.
C. PREDICTABILITY (known at the prior close): next-day stack P&L (pts) by terciles of (i) trailing 5-day stack P&L, (ii) trailing 5-day sum of |stack P&L|, (iii) trailing 5-day NQ jump count (Lee-Mykland J>3 on 5m RTH bars, BPV of the previous 12 bars), (iv) trailing 5-day mean jump share RJ of daily RV.  Cuts on TRAIN 2019-21; test 2018 + 2022-26.
D. AUCTION-THEORY GAUSSIAN: daily z = (close - open) / ADR14 of NQ RTH: Gaussian vs Student-t vs 2-component Gaussian mixture log-lik (balanced bell + trend component), share/means of the components, tail mass beyond |z|>1,2 vs normal.
Why it should fail: P&L of an 8-model stack that is already a trend-day engine is heavy-tailed by construction; clustering of |P&L| is volatility clustering that range models already capture; sizing tests (heavier days after heavy days) were null before."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; L = pickle.load(open(SP + 'legs287_ext.pkl', 'rb')); M = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1']
T = pd.concat([L[k][['day', 'pts']].assign(model=k) for k in M]); T['day'] = pd.to_datetime(T.day).dt.normalize(); T = T[T.day >= '2018-01-01'].sort_values('day')
def hill(x, q):
    x = np.sort(x[x > 0])[::-1]; k = max(int(len(x) * q), 10); return 1 / np.mean(np.log(x[:k] / x[k])) if len(x) > k else np.nan
print('A. SHAPE per model (pts per trade)')
print(f'{"model":6s}{"n":>6s}{"mean":>7s}{"sd":>7s}{"skew":>6s}{"exkurt":>7s}{"JB p":>8s}{"alphaW":>7s}{"alphaL":>7s}{"t df":>6s}{"dLL t-G":>9s}{"top5%":>7s}')
for m in M + ['ALL']:
    x = (T if m == 'ALL' else T[T.model == m]).pts.values; z = (x - x.mean()) / x.std(); df, lo, sc = stats.t.fit(z, f0=None); llt = stats.t.logpdf(z, df, lo, sc).sum(); llg = stats.norm.logpdf(z).sum(); k = int(len(x) * .05); top = np.sort(x)[::-1][:k].sum() / x.sum()
    print(f'{m:6s}{len(x):6d}{x.mean():7.2f}{x.std():7.1f}{stats.skew(x):6.1f}{stats.kurtosis(x):7.1f}{stats.jarque_bera(x)[1]:8.3f}{hill(x, .10):7.2f}{hill(-x, .10):7.2f}{df:6.1f}{llt - llg:9.0f}{top:7.0%}')
cal = pd.bdate_range('2018-01-01', T.day.max()); D = T.groupby('day').pts.sum().reindex(cal).fillna(0)
print('\nB. CLUSTERING of daily stack P&L (ACF, Ljung-Box p at lag 20)')
from statsmodels.stats.diagnostic import acorr_ljungbox
for lab, s in (('all', D), ('train 2019-21', D['2019':'2021']), ('test 2018+22-26', pd.concat([D['2018'], D['2022':]]))):
    a = s.abs(); r = ' '.join(f'{k}:{a.autocorr(k):+.2f}' for k in (1, 5, 10, 20)); sg = ' '.join(f'{k}:{s.autocorr(k):+.2f}' for k in (1, 5)); print(f'   {lab:16s} |P&L| ACF {r} | LB p {acorr_ljungbox(a, lags=[20])["lb_pvalue"].iloc[0]:.4f} || signed ACF {sg} | LB p {acorr_ljungbox(s, lags=[20])["lb_pvalue"].iloc[0]:.3f}')
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); N = N[N.day >= '2017-06-01']
rows = {}
for d, x in N.groupby('day'):
    c = x.sort_values('m').close.values; o = x.sort_values('m').open.iloc[0]
    if len(c) < 70: continue
    r = np.diff(np.log(c)); a = np.abs(r); bp = (np.pi / 2) * a[1:] * a[:-1]; bpv_roll = pd.Series(np.r_[np.nan, a[:-1]] * np.r_[np.nan, np.nan, a[:-2]]).rolling(12).mean().shift(1).values * (np.pi / 2); J = r / np.sqrt(bpv_roll)
    rv = (r ** 2).sum(); bv = bp.sum(); rows[d] = dict(o=o, c=c[-1], h=x.high.max(), l=x.low.min(), nj=int(np.nansum(np.abs(J) > 3)), rj=max(0, rv - bv) / rv)
F = pd.DataFrame(rows).T.astype(float); F['adr'] = (F.h - F.l).rolling(14).mean().shift(1); F['z'] = (F.c - F.o) / F.adr
pnl = D.reindex(F.index).fillna(0); G = pd.DataFrame({'next': pnl.shift(-1), 'p5': pnl.rolling(5).sum(), 'a5': pnl.abs().rolling(5).sum(), 'j5': F.nj.rolling(5).sum(), 'rj5': F.rj.rolling(5).mean()}).dropna(); G = G[G.index >= '2018-01-01']
tr = (G.index >= '2019-01-01') & (G.index < '2022-01-01'); te = ~tr
def f(m): s = G.loc[m, 'next']; return f'{s.mean():+6.1f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)})'
print(f'\nC. next-day stack P&L (pts) by terciles of the prior 5 days; unconditional train {f(tr)} | test {f(te)}')
for col, lab in (('p5', 'trailing 5d stack P&L'), ('a5', 'trailing 5d sum |P&L|'), ('j5', 'trailing 5d NQ jump count'), ('rj5', 'trailing 5d jump share RJ')):
    q = G.loc[tr, col].quantile([1 / 3, 2 / 3]).values; Lw, Md, Hg = G[col] <= q[0], (G[col] > q[0]) & (G[col] < q[1]), G[col] >= q[1]; print(f'   {lab:26s} cuts {q.round(2)}\n      train low {f(tr & Lw)} | mid {f(tr & Md)} | high {f(tr & Hg)}\n      test  low {f(te & Lw)} | mid {f(te & Md)} | high {f(te & Hg)}')
print(f'\nD. daily NQ RTH z = (close-open)/ADR14, n={F.z.notna().sum()}: sd {F.z.std():.2f}, skew {stats.skew(F.z.dropna()):+.2f}, ex-kurt {stats.kurtosis(F.z.dropna()):+.2f}; mass |z|>1: {np.mean(np.abs(F.z.dropna()) > 1):.1%} (Gaussian with same sd {2 * (1 - stats.norm.cdf(1 / F.z.std())):.1%}), |z|>2: {np.mean(np.abs(F.z.dropna()) > 2):.1%} ({2 * (1 - stats.norm.cdf(2 / F.z.std())):.1%})')
z = F.z.dropna().values; df, lo, sc = stats.t.fit(z); print(f'   loglik: Gaussian {stats.norm.logpdf(z, z.mean(), z.std()).sum():.0f} | Student-t(df {df:.1f}) {stats.t.logpdf(z, df, lo, sc).sum():.0f}')
try:
    from sklearn.mixture import GaussianMixture
    for k in (2, 3):
        g = GaussianMixture(k, n_init=5, random_state=1).fit(z.reshape(-1, 1)); print(f'   {k}-comp mixture loglik {g.score(z.reshape(-1, 1)) * len(z):.0f}, BIC {g.bic(z.reshape(-1, 1)):.0f}; weights {g.weights_.round(2)}, means {g.means_.ravel().round(2)}, sd {np.sqrt(g.covariances_.ravel()).round(2)}')
    print(f'   single Gaussian BIC {-2 * stats.norm.logpdf(z, z.mean(), z.std()).sum() + 2 * np.log(len(z)):.0f}')
except Exception as e: print('   sklearn n/a', e)
