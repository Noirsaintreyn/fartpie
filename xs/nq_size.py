"""Nov 4 2026: position size vs the prop eval for the NQ NY AM trade (150/75, S0 and S0 + SPY 20d up). $/pt: 1 NQ 20, 5 MNQ 10, 3 MNQ 6, 2 MNQ 4, 1 MNQ 2.
Same eval rules as nq_prop_of.py. Per size: pass rate, median sessions to pass, mean sessions per attempt, expected sessions per pass = mean attempt / pass rate
(restart after a fail), expected fees per pass at $49 a month (21 sessions), vector vs coin-flip direction (10 draws). Micro costs: 1.2 pt round trip vs 1.0 for NQ."""
_s = open('xs/nq_ctx.py').read(); exec(_s[:_s.index("for T, S in ((100, 50), (150, 75)):")])
def prop2(R, usd, target=3000, mll=2000, dll=1000):
    v = R.groupby('day').pts.sum().reindex(days).fillna(0).values * usd; res = []
    for s in range(len(v)):
        bal, peak, floor, best = 0.0, 0.0, -mll, 0.0; out = None
        for k in range(s, len(v)):
            day = max(v[k], -dll); bal += day; best = max(best, day)
            if bal <= floor: out = ('fail', k - s + 1); break
            if bal >= target and best <= .5 * bal: out = ('pass', k - s + 1); break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        if out: res.append(out)
    Rr = pd.DataFrame(res, columns=['r', 'n']); pr = (Rr.r == 'pass').mean()
    return pr, Rr[Rr.r == 'pass'].n.median(), Rr.n.mean(), Rr.n.mean() / pr if pr else np.inf
rng = np.random.default_rng(5)
for nm, msk in (('S0 (negative gamma)', np.ones(len(C2), bool)), ('S0 + SPY 20d up', C2.MKT20 == 'SPY 20d up')):
    sub = C2[msk].reset_index(drop=True); print(f'\n== {nm}: {len(sub)} trades, 150/75', flush=True)
    for usd, lab, cost in ((20, '1 NQ', 1.0), (10, '5 MNQ', 1.2), (6, '3 MNQ', 1.2), (4, '2 MNQ', 1.2), (2, '1 MNQ', 1.2)):
        R = run_q(sub, 150, 75); R['pts'] = R.pts + 1.0 - cost; pr, mp, ma, ep = prop2(R, usd); cf = []
        for _ in range(10):
            Z = sub.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); Rc = run_q(Z, 150, 75); Rc['pts'] = Rc.pts + 1.0 - cost; cf.append(prop2(Rc, usd)[0])
        print(f'  {lab:<6} ${usd:>2}/pt  pass {pr:.0%} (coin-flip {np.median(cf):.0%})  median {mp:.0f} sessions to pass | expected {ep:.0f} sessions per pass incl. restarts = {ep / 21:.1f} months, fees ~${ep / 21 * 49:,.0f}', flush=True)
