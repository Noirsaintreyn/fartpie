"""Nov 22 2026 report 365. Rey: 'jump diffusion model?'  Declared before running.
(1) MERTON FIT: daily close-to-close returns of the 54 flip names, pooled, per half (train 2019-21, test 2022-26): r = diffusion N(mu, s^2) + Poisson(lam) jumps ~ N(mj, sj^2); MLE with up to 4 jumps/day.  Reports lam (jumps/yr), mj, sj, s and the share of variance from jumps.  Purpose: how much of the return distribution is jumps vs smooth drift.
(2) HYPOTHESIS (mechanism: a vector flip is a regime change that raises the arrival rate of UPWARD jumps, so the flip's edge is jump timing, not drift): a daily return > +3 trailing-60d sigmas (prior days only) is an UP-JUMP.  For every flip, y = 1 if an up-jump occurs on any of the next 5 trading days after the flip day.  Baseline = that name's share of all 5-day windows with an up-jump in the same half.  Declared direction: flips > baseline in BOTH halves, |t| >= 2 in at least one (week-clustered).  Also the same for DOWN-jumps (declared: not higher, a flip should not raise crash risk) and for 10-day windows.
Where this would matter: if flips raise jump intensity, the flip is an event-time clock for jump risk, which suggests holding through jumps (no profit target) and sizing by jump intensity, not by volatility."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from scipy.stats import norm, poisson
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
B = pd.read_parquet(SP + 'b302.parquet'); B['entry'] = pd.to_datetime(B.entry); B['day'] = pd.to_datetime(B.day).astype('datetime64[ns]')
R = {}
for tk in sorted(B.tk.unique()):
    f = sorted(glob.glob(U + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    c = H.close.groupby(H.index.normalize()).last(); r = np.log(c).diff().dropna(); r = r[r.abs() < .5]; R[tk] = r
def merton(x):
    x = x[np.isfinite(x)]
    def nll(p):
        mu, ls, lam, mj, lsj = p; s = np.exp(ls); sj = np.exp(lsj); lam = np.exp(lam); L = 0
        for k in range(5): L = L + poisson.pmf(k, lam) * norm.pdf(x, mu + k * mj, np.sqrt(s * s + k * sj * sj))
        return -np.log(L + 1e-300).sum()
    best = None
    for l0 in (-3.5, -2.5):
        o = minimize(nll, [x.mean(), np.log(x.std() * .7), l0, 0.0, np.log(x.std() * 3)], method='Nelder-Mead', options=dict(maxiter=1500, xatol=1e-6, fatol=1e-6))
        if best is None or o.fun < best.fun: best = o
    mu, ls, lam, mj, lsj = best.x; return mu, np.exp(ls), np.exp(lam), mj, np.exp(lsj)
print('== 1. Merton jump-diffusion fit, pooled daily returns of the 54 names (standardised by each name\'s own half-sample std so levels do not mix) ==')
for lab, a, b in (('train 2019-21', '2019-01-01', '2021-12-31'), ('test 2022-26', '2022-01-01', '2026-12-31')):
    xs_ = []
    for tk, r in R.items():
        z = r[(r.index >= a) & (r.index <= b)]; xs_.append((z - z.mean()) / z.std())
    x = pd.concat(xs_).values; mu, s, lam, mj, sj = merton(x); tot = x.var(); jv = lam * (mj ** 2 + sj ** 2)
    print('%s: n %d | diffusion sd %.2f (of 1.00 total), jumps/yr %.1f (lam/day %.3f), mean jump %+.2f sd, jump sd %.2f sd | jump share of variance %.0f%%' % (lab, len(x), s, lam * 252, lam, mj, sj, jv / (s * s + jv) * 100), flush=True)
print('\n== 2. do flips raise the arrival rate of up-jumps? ==')
rows = []
for tk, r in R.items():
    sd = r.rolling(60, min_periods=40).std().shift(1); up = (r > 3 * sd).astype(float); dn = (r < -3 * sd).astype(float); idx = r.index
    for H_ in (5, 10):
        fu = pd.Series([up.values[i + 1:i + 1 + H_].max() if i + H_ < len(up) else np.nan for i in range(len(up))], index=idx); fd = pd.Series([dn.values[i + 1:i + 1 + H_].max() if i + H_ < len(dn) else np.nan for i in range(len(dn))], index=idx)
        rows.append(pd.DataFrame({'tk': tk, 'date': idx, 'H': H_, 'up': fu.values, 'dn': fd.values}))
W = pd.concat(rows); W['date'] = pd.to_datetime(W.date).astype('datetime64[ns]'); W['half'] = np.where(W.date < '2022-01-01', 'train', 'test'); W = W[W.date >= '2019-01-01']
base = W.groupby(['tk', 'half', 'H'])[['up', 'dn']].mean().rename(columns={'up': 'bup', 'dn': 'bdn'}).reset_index()
Fl = B[['tk', 'day', 'entry', 'ret']].copy(); Fl['wk'] = Fl.entry.dt.to_period('W').astype(str)
J = Fl.merge(W, left_on=['tk', 'day'], right_on=['tk', 'date']).merge(base, on=['tk', 'half', 'H']).dropna(subset=['up'])
print('%-4s %-6s %-9s %-26s %-26s' % ('H', 'type', '', 'TRAIN flips vs baseline (t)', 'TEST flips vs baseline (t)'))
for H_ in (5, 10):
    for typ, y, b in (('up', 'up', 'bup'), ('down', 'dn', 'bdn')):
        out = []
        for h in ('train', 'test'):
            D_ = J[(J.H == H_) & (J.half == h)]; d = D_[y] - D_[b]; m = sm.OLS(d.values, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D_.wk)[0]}); out.append('%.1f%% vs %.1f%% (%+.1f pts, t %+.1f, n %d)' % (D_[y].mean() * 100, D_[b].mean() * 100, d.mean() * 100, m.tvalues[0], len(D_)))
        print('%-4d %-6s %-9s %-26s %-26s' % (H_, typ, '', out[0], out[1]), flush=True)
