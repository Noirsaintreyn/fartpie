"""Nov 15 2026 SLOT DESIGN FOR THE SUB-$5 BOOK (Rey: 'work on slot design'). Declared before results. Candidate pools (TEST 2022-01..2026-02, entry next open, exit open after the bear flip, additive P&L as a share of STARTING capital, 5% per position, 30bp and 100bp): OS = report_118 outlier score (train-ECDF ranks of ATRP, EXT, turnover; cut = train 80th pct) AND price < $5; FZ = frozen daily score top 20% AND price < $5. The problem: first-come fills slots with whatever fires first, so the ranking is wasted when slots are scarce. POLICIES, all with N slots (a slot frees at the exit open): P0 FIRST-COME (baseline; ties on the same day broken by score; must reproduce report_118 within noise); P1 QUEUE-K: a flip that finds no free slot waits up to K trading days; when a slot frees, the highest-score waiting flip that is still in its bull state enters at the PRIOR CLOSE (proxy for the next open) - K = 3, 5, 10; P2 REPLACE-m: if full and a new flip scores at least m (in OS units, range 0-3) above the lowest-score open position, close that position at today's value (cost paid) and enter the new one - m = 0.3, 0.6; P3 TIER RESERVE: N=10 slots, 5 reserved for flips in the top 10% of OS (cut = train 90th pct), 5 open to any pool flip. Metrics: total return, annualised, max drawdown (pts of starting capital), Sharpe, trades, average positions, worst month, per-year; N = 5, 10, 20. Lead bar: a policy beats P0 on Sharpe AND total at the same N and cost AND is not worse in more than one of the five calendar years. Survivorship-biased; thin fills unmodeled; the queue entry price is a proxy; one test window."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_100_portfolio_dd.py').read(); exec(src[:src.index("S = {'CURRENT-LIKE")])
Of = pd.read_pickle(X + 'outlier_features.pkl'); Of['LTURN'] = np.log(Of.TURN.clip(lower=1e-9)); tr = Of[~Of.test]; ok = Of.dropna(subset=['ATRP', 'EXT', 'LTURN']).copy()
def ecdf(col): s = np.sort(tr[col].dropna().values); return lambda x: np.searchsorted(s, x, side='right') / len(s)
ok['OS'] = ecdf('ATRP')(ok.ATRP.values) + ecdf('EXT')(ok.EXT.values) + ecdf('LTURN')(ok.LTURN.values); cutOS = ok[~ok.test].OS.quantile(.8); cut90 = ok[~ok.test].OS.quantile(.9); D_ = {tk: q.index for tk, q in D.items()}; key = {}
for tk, g in ok.groupby('tk'):
    ix = D_.get(tk)
    for d, os_ in zip(g.d, g.OS):
        k = ix.searchsorted(d)
        if k + 1 < len(ix): key[(tk, ix[k + 1])] = os_
P2 = [(p[0], p[1], p[2], key[(p[7], p[0])], p[4], p[5], p[6], p[7]) for p in paths if (p[7], p[0]) in key]
a, b = pd.Timestamp('2022-01-01'), pd.Timestamp('2026-03-01'); dd = dates[(dates >= a) & (dates < b)]; pos = {d: k for k, d in enumerate(dd)}; T = len(dd)
def build(PS, sel):
    C = []
    for p in PS:
        if not sel(p) or not (a <= p[0] < b) or p[0] not in pos: continue
        e = pos[p[0]]; L = len(p[2]); C.append(dict(e=e, x=min(e + L - 1, T - 1), v=p[2], sc=p[3], ok=(e + L - 1) <= T - 1, tk=p[7]))
    return C
def simp(C, N, f, cost, policy, K=0, m=0., resv=0, resv_cut=None):
    by = {}
    for c in C: by.setdefault(c['e'], []).append(c)
    pnl = np.zeros(T); act = np.zeros(T); open_ = []; wait = []; trades = 0
    def enter(c, t):
        s = t - c['e']; base = 1.0 if s == 0 else c['v'][s - 1]; L = len(c['v']); xend = min(c['e'] + L - 1, T - 1); prev = 1.0
        for j in range(s, L):
            idx = c['e'] + j
            if idx >= T: break
            r = c['v'][j] / base; pnl[idx] += f * (r - prev); prev = r; act[idx] += 1
        if c['e'] + L - 1 < T: pnl[xend] -= f * cost
        return dict(c=c, exit=xend, t0=t)
    for t in range(T):
        open_ = [o for o in open_ if o['exit'] > t]
        new = sorted(by.get(t, []), key=lambda c: -c['sc'])
        if policy == 'queue': wait = [c for c in wait if c['e'] + len(c['v']) - 1 > t and t - c['e'] <= K and True] + new
        else: wait = new
        wait = sorted(wait, key=lambda c: -c['sc'])
        for c in list(wait):
            free = N - len(open_)
            hi = resv_cut is not None and c['sc'] >= resv_cut
            if policy == 'reserve': 
                nhi = sum(1 for o in open_ if o['c']['sc'] >= resv_cut); nlo = len(open_) - nhi; ok_ = (hi and (nhi < N - 0)) or ((not hi) and nlo < N - resv and len(open_) < N)
                if hi and len(open_) >= N: ok_ = False
                if not hi and len(open_) >= N - max(0, resv - nhi): ok_ = False
                if ok_: open_.append(enter(c, t)); trades += 1; wait.remove(c)
                continue
            if free > 0: open_.append(enter(c, t)); trades += 1; wait.remove(c)
            elif policy == 'replace' and open_:
                w = min(open_, key=lambda o: o['c']['sc'])
                if c['sc'] - w['c']['sc'] >= m and w['exit'] > t:
                    cur = w['c']['v'][min(t - w['c']['e'], len(w['c']['v']) - 1)]; prevr = w['c']['v'][min(t - w['c']['e'] - 1, len(w['c']['v']) - 1)] if t - w['c']['e'] >= 1 else 1.0
                    # remove its future P&L beyond today and book the exit cost today
                    L = len(w['c']['v']); base = 1.0 if w['t0'] == w['c']['e'] else w['c']['v'][w['t0'] - w['c']['e'] - 1]; prev = (w['c']['v'][t - w['c']['e']] / base)
                    for j in range(t - w['c']['e'] + 1, L):
                        idx = w['c']['e'] + j
                        if idx >= T: break
                        pnl[idx] -= f * (w['c']['v'][j] / base - (w['c']['v'][j - 1] / base if j - 1 >= w['t0'] - w['c']['e'] else 1.0)); act[idx] -= 1
                    if w['c']['e'] + L - 1 < T: pnl[min(w['c']['e'] + L - 1, T - 1)] += f * cost
                    pnl[t] -= f * cost; open_.remove(w); open_.append(enter(c, t)); trades += 1; wait.remove(c)
        if policy != 'queue': wait = []
    eq = np.cumsum(pnl) * 100; dly = np.diff(np.r_[0, eq]); mdd = (np.maximum.accumulate(eq) - eq).max(); yrs = T / 252; mon = pd.Series(pnl * 100, index=dd).groupby(dd.to_period('M')).sum()
    return dict(total=eq[-1], ann=eq[-1] / yrs, mdd=-mdd, sh=dly.mean() / dly.std() * np.sqrt(252) if dly.std() > 0 else np.nan, trades=trades, avgpos=act.mean(), wm=mon.min(), yr={y: round(g.sum(), 1) for y, g in pd.Series(pnl * 100, index=dd).groupby(dd.year)})
pools = {'OS & px<5': (P2, lambda p: p[3] >= cutOS and p[4] < 5, cut90), 'FZ & px<5': (paths, lambda p: p[3] >= cut and p[4] < 5, None)}
for pn, (PS, sel, c90) in pools.items():
    C = build(PS, sel); print(f'\n########## POOL {pn}: candidates {len(C)} ##########')
    for cost in (.003, .01):
        print(f'\n -- cost {cost*1e4:.0f}bp, 5% per position, TEST 2022-01..2026-02 --')
        for N in (5, 10, 20):
            pol = [('P0 first-come', dict(policy='fcfs')), ('P1 queue K=3', dict(policy='queue', K=3)), ('P1 queue K=5', dict(policy='queue', K=5)), ('P1 queue K=10', dict(policy='queue', K=10)), ('P2 replace m=0.3', dict(policy='replace', m=.3)), ('P2 replace m=0.6', dict(policy='replace', m=.6))]
            if c90 is not None and N == 10: pol.append(('P3 reserve 5 for top10%', dict(policy='reserve', resv=5, resv_cut=c90)))
            for nm, kw in pol:
                R = simp(C, N, .05, cost, **kw); print(f'  N={N:2d} {nm:24s} total {R["total"]:+7.1f}% ann {R["ann"]:+6.1f}% maxDD {R["mdd"]:+7.1f} sharpe {R["sh"]:+.2f} trades {R["trades"]:4d} avgpos {R["avgpos"]:4.1f} worstmo {R["wm"]:+6.1f}% | {R["yr"]}')
