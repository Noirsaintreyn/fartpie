"""Nov 15 2026 PORTFOLIO RETURNS AND MAX DRAWDOWN: stacked-score model vs the unfiltered flip book. Declared before results. Daily wide universe flips (report_90 file), entry next open, exit open after the bear flip, 30bp round trip ($1-10 sensitivity at 100bp). Portfolio: max 20 concurrent positions, fixed notional 5% of STARTING capital each (additive, no compounding, so a few winners cannot dominate through compounding), marked to the daily close, positions beyond 20 are skipped (first come; same-day ties random for unfiltered books, by frozen score for score books). Scenarios (TEST period 2022-01..2026-03, the frozen window, plus 2019-21 for reference): CURRENT-LIKE = all flips; CURRENT-LIKE+VOL = flips with flip-bar volume >= 1.5x (the screener's volume rule); SCORE = frozen ridge-logistic score (trained 2019-21) above the 80th percentile of the TRAIN-period score distribution (no look-ahead, fixed cutoff); $1-10 = price below $10 at the flip bar; $1-10 & SCORE. Metrics: total return (% of starting capital), per year, max drawdown (percentage points of starting capital, peak to trough on the daily equity curve), Sharpe (daily pnl, annualized), calmar (annual return / maxDD), trades, avg positions, worst month. Survivorship-flattered, small-cap costs understated, capacity not modeled."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); F = pd.read_pickle(X + 'stack_flips.pkl'); cut = F[~F.test].lr.quantile(.8); print('train 80th pct score cutoff', round(cut, 4)); paths = []
for tk, z in F.groupby('tk'):
    B = D.get(tk)
    if B is None: continue
    B = B[~B.index.duplicated()]; B = B[B.index >= '2018-06-01']; idx = B.index; o = B.open.values; c = B.close.values; n = len(B); reg, _, _, _ = run_state_machine(B.assign(datetime=B.index))
    for d, lr, px, vx, te in zip(z.d, z.lr, z.px, z.VOLX, z.test):
        i = idx.searchsorted(d)
        if i >= n - 2 or idx[i] != d: continue
        j = i + 1
        while j < n and reg[j] != -1: j += 1
        if j + 1 >= n: continue
        e, x = i + 1, j + 1; v = np.r_[c[e:x] / o[e], o[x] / o[e]]; paths.append((idx[e], idx[x], v, lr, px, vx, te, tk))
print('paths', len(paths)); alld = pd.DatetimeIndex(sorted(set(D['AAPL'].index) if 'AAPL' in D else set())) if False else None
dates = pd.DatetimeIndex(sorted({d for b in D.values() for d in b.index[b.index >= '2018-06-01']}))
def sim(sel, start, end, rng, order='random', f=.05, N=20, cost=.003, cost_small=None):
    P = [p for p in paths if sel(p) and start <= p[0] < end]; P.sort(key=lambda p: (p[0], -p[3] if order == 'score' else rng.random())); dd = dates[(dates >= start) & (dates < end)]; pos = {dd[k]: k for k in range(len(dd))}; pnl = np.zeros(len(dd)); act = np.zeros(len(dd)); open_ends = []; taken = 0
    for p in P:
        e, x, v, lr, px, vx, te, tk = p
        if e not in pos: continue
        k0 = pos[e]; open_ends = [t for t in open_ends if t > k0]
        if len(open_ends) >= N: continue
        if x not in pos: kx = len(dd) - 1
        else: kx = pos[x]
        vals = f * v[:kx - k0 + 1]; prev = np.r_[f, vals[:-1]]; d_ = vals - prev; d_[-1] -= f * (cost_small if (cost_small and px < 10) else cost); pnl[k0:k0 + len(d_)] += d_; act[k0:kx + 1] += 1; open_ends.append(kx); taken += 1
    eq = np.cumsum(pnl); peak = np.maximum.accumulate(np.r_[0, eq])[1:]; dd_pts = (eq - peak).min(); yrs = len(dd) / 252; s = pd.Series(pnl, index=dd); mo = s.groupby(s.index.to_period('M')).sum(); yr = s.groupby(s.index.year).sum()
    return dict(total=eq[-1] * 100, annual=eq[-1] / yrs * 100, mdd=dd_pts * 100, sharpe=pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan, trades=taken, avgpos=act.mean(), worstmo=mo.min() * 100, years={k: round(v * 100, 1) for k, v in yr.items()})
S = {'CURRENT-LIKE (all flips)': (lambda p: True, 'random'), 'CURRENT-LIKE + vol>=1.5x': (lambda p: p[5] >= 1.5, 'random'), 'SCORE top 20% (frozen cutoff)': (lambda p: p[3] >= cut, 'score'), '$1-10 only': (lambda p: p[4] < 10, 'random'), '$1-10 & SCORE': (lambda p: p[4] < 10 and p[3] >= cut, 'score')}
def agg(sel, order, a, b, cs=None):
    seeds = range(20) if order == 'random' else range(1); R = [sim(sel, pd.Timestamp(a), pd.Timestamp(b), np.random.default_rng(sd), order, cost_small=cs) for sd in seeds]; g = lambda k: np.array([r[k] for r in R]); return R[0], g
for lab, (a, b) in (('TEST 2022-01 to 2026-02 (frozen window)', ('2022-01-01', '2026-03-01')), ('REFERENCE 2019-21 (in-sample for the score)', ('2019-01-01', '2022-01-01'))):
    print(f'\n== {lab} == max 20 positions, 5% of starting capital each; random-order books are averaged over 20 random tie-break draws (min..max shown); costs 30bp, small-caps 100bp in the 2nd line')
    for nm, (sel, order) in S.items():
        r0, g = agg(sel, order, a, b); _, g2 = agg(sel, order, a, b, .01); rng_ = lambda x: f'{x.min():+.0f}..{x.max():+.0f}' if len(x) > 1 else ''
        print(f' {nm:30s} total {g("total").mean():+7.1f}% ({rng_(g("total"))}) annual {g("annual").mean():+6.1f}% maxDD {g("mdd").mean():+6.1f}pts ({rng_(g("mdd"))}) sharpe {np.nanmean(g("sharpe")):+.2f} calmar {g("annual").mean()/abs(g("mdd").mean()):+.2f} trades {g("trades").mean():4.0f} avgpos {g("avgpos").mean():4.1f} worst month {g("worstmo").mean():+.1f}%')
        print(f' {"":30s} @100bp small caps: total {g2("total").mean():+7.1f}% maxDD {g2("mdd").mean():+6.1f}pts sharpe {np.nanmean(g2("sharpe")):+.2f} | by year (draw 0) {r0["years"]}')
