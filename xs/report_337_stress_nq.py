"""Nov 18 2026 report 337. FORMAL STRESS TEST 1/3: the NQ prop algo (8 models: RA, M1, R1, ON+, ON-, RULES, V15a, S1; NZ/CH/FIG-V excluded), 1 contract per model, legs287_ext through Nov 13 2026 (before prop-account rules).  Battery (declared before running; modelled on the checks in the profile Rey was shown, applied where the data allow):
 S1 Sharpe + PSR (P true SR > 0, skew/kurt aware) + DEFLATED Sharpe for N trials in {50, 200, 1000, 5000} (assumption: skill-less trials have an annualised SR sd of 0.5; our real trial count is unlogged, so a grid is shown).
 S2 stationary block bootstrap (mean block 20 days, 5,000): P(SR<0), 5th-95th pct of annual pts, 95th pct max drawdown (pts).  Reshuffling only tests sequence luck.
 S3 sign-flip permutation (10,000): is the mean P&L distinguishable from a coin-flipped direction with the same trade magnitudes (per trade)?
 S4 cost stress: extra pts per trade round trip (0.5 / 1 / 2 / 4 pts) -> annual pts, Sharpe, years positive.
 S5 concentration: remove the top 1 / 2 / 5% of TRADES (and of days) -> annual pts left; leave-one-year-out; yearly table.
 S6 halves: 2018-2022 vs 2023-2026 Sharpe.
 Not testable here (stated, not skipped silently): look-ahead replacement and cross-platform match (needs the signal code, not the leg tables), parameter-neighbourhood / CPCV-PBO (no variant matrix stored), late-entry (no intrabar paths in the leg tables).  Late-entry is tested on the ORB engine in report 333/334 style only."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from stress_lib import *
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; L = pickle.load(open(SP + 'legs287_ext.pkl', 'rb')); M = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1']
T = pd.concat([L[k][['day', 'pts']].assign(model=k) for k in M]); T['day'] = pd.to_datetime(T.day).dt.normalize(); T = T[T.day >= '2018-01-01'].sort_values('day')
D = T.groupby('day').pts.sum(); cal = pd.bdate_range(D.index.min(), D.index.max()); D = D.reindex(cal).fillna(0); yrs = (D.index[-1] - D.index[0]).days / 365.25
print(f'{len(T)} trades, {len(D)} business days {D.index.min().date()} -> {D.index.max().date()}; total {D.sum():.0f} pts = {D.sum() / yrs:.0f} pts/yr; pts/trade {T.pts.mean():+.2f}; trades/yr {len(T) / yrs:.0f}')
x = D.values; sr = sharpe(x); print(f'\nS1 Sharpe (daily pts, 252) {sr:.2f} | PSR(0) {psr(x):.3f}')
for n in (50, 200, 1000, 5000): p, s0 = dsr(x, n); print(f'   DSR with N={n:5d} trials: {p:.3f}  (expected max skill-less SR {s0:.2f})')
B = stationary_bootstrap(x, 5000); bs = np.array([sharpe(x[i]) for i in B]); ann = np.array([x[i].sum() / yrs for i in B]); dd = np.array([maxdd_cum(np.cumsum(x[i])) for i in B])
print(f'\nS2 block bootstrap: P(SR<0) {np.mean(bs < 0):.4f} | SR 5-95% {np.percentile(bs, 5):.2f}-{np.percentile(bs, 95):.2f} | pts/yr 5-95% {np.percentile(ann, 5):.0f}-{np.percentile(ann, 95):.0f} | max DD actual {maxdd_cum(np.cumsum(x)):.0f} pts, bootstrap median {np.median(dd):.0f}, 5% worst {np.percentile(dd, 5):.0f}')
print(f'\nS3 sign-flip permutation on trades (per-trade pts): p = {signflip_p(T.pts.values):.4f}; on days: p = {signflip_p(x):.4f}')
print('\nS4 cost stress (extra pts per trade):')
for c in (0, .5, 1, 2, 4):
    y = (T.assign(p=T.pts - c).groupby('day').p.sum().reindex(cal).fillna(0)); yy = y.groupby(y.index.year).sum(); print(f'   +{c} pts: {y.sum() / yrs:6.0f} pts/yr, SR {sharpe(y.values):.2f}, years positive {int((yy > 0).sum())}/{len(yy)}, pts/trade {T.pts.mean() - c:+.2f}')
print('\nS5 concentration: remove the top share of TRADES / DAYS')
for q in (.01, .02, .05):
    k = int(len(T) * q); r = T.pts.sort_values(ascending=False).iloc[k:].sum(); kd = int(len(D) * q); rd = D.sort_values(ascending=False).iloc[kd:].sum()
    print(f'   top {q:.0%}: trades removed {k} -> {r / yrs:6.0f} pts/yr ({r / T.pts.sum():.0%} of total); days removed {kd} -> {rd / yrs:6.0f} pts/yr ({rd / D.sum():.0%})')
y = D.groupby(D.index.year).sum(); print('   by year:', y.round(0).astype(int).to_dict()); print('   leave-one-year-out pts/yr:', {int(k): round((D.sum() - v) / (yrs - 1)) for k, v in y.items()})
print('   by model total pts / trades / pts per trade:', {m: (round(T[T.model == m].pts.sum()), int((T.model == m).sum()), round(T[T.model == m].pts.mean(), 2)) for m in M})
a, b = D[D.index < '2023-01-01'], D[D.index >= '2023-01-01']; print(f'\nS6 halves: 2018-22 SR {sharpe(a.values):.2f}, {a.sum() / ((a.index[-1] - a.index[0]).days / 365.25):.0f} pts/yr | 2023-26 SR {sharpe(b.values):.2f}, {b.sum() / ((b.index[-1] - b.index[0]).days / 365.25):.0f} pts/yr')
