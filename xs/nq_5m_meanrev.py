"""Nov 4 2026: 5m vector flips taken only in the MEAN-REVERTING direction (Rey: NQ mean-reverts intraday, so a lower-timeframe vector inside a reversion gate). DECLARED BEFORE RESULTS.
5m QQQ flips, signal bars closing 09:30-11:30, both ways, <= 2 a day, 75/50, flat 12:00, 1 pt cost. Skip a thin 60-day path; size 1 / 0.5 / 0.3 NQ by HAR high + VWAP stretched (Rule A sizing).
Gates: none | Z-MR long only when QQQ's prior-close 20d z < 0, short only when z > 0 | TO-VWAP long only below session VWAP, short only above (trading toward VWAP) | both.
Per trade by period (2018-21 / 22-23 / 24-25 / 2026) + coin-flip direction; Lucid 50K ($88) and Topstep sims incl. 2026."""
_lt = open('xs/nq_ltf_vector.py').read(); exec(_lt[:_lt.rindex("if __name__ == '__main__':")])
C5f = SETS['F5 5m flip'].copy(); C5f['z'] = prior(XQ.qz, C5f.day)
C5f = C5f[~C5f.THIN.astype(bool)].copy()
G = {'5m, no gate': np.ones(len(C5f), bool), '5m, Z-MR': (np.sign(-C5f.z) == C5f.dir).values | C5f.z.isna().values,
     '5m, toward VWAP': (C5f.vwapd < 0).values, '5m, Z-MR + toward VWAP': ((np.sign(-C5f.z) == C5f.dir).values | C5f.z.isna().values) & (C5f.vwapd < 0).values}
rng = np.random.default_rng(13); PERS = ('18-21', '22-23', '24-25', '2026'); DS = {}
for nm, m in G.items():
    Cc = C5f[m].reset_index(drop=True); R = run_l(Cc); R['per'] = np.select([R.day < P0, R.day < '2024-01-01', R.day < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
    cf = np.median([run_l(Cc.assign(dir=rng.choice([-1, 1], len(Cc)))).net.mean() for _ in range(10)])
    print(f'  {nm:<26} trades {len(R)} ({len(R) / 105:.1f}/mo) win {R.net.gt(0).mean():.0%} | ' + ' / '.join(f'{p} {R[R.per == p].net.mean():+5.1f} (n{(R.per == p).sum()})' for p in PERS) + f' | coin-flip {cf:+.1f}', flush=True)
    DS[nm] = build_day(Cc)
SZ2['C5'] = lambda th: (1.0, .5, .3)[min(th[1] + th[2], 2)]; SZE2['C5'] = SZ2['C5']
def job8(a):
    global DAY
    nm, fn = a; DAY = DS[nm]; r = job2((fn, 'C5', 'C5')); f = F[fn]
    y = career(f, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), NS, SZE2['C5'], SZ2['C5']); full = career(f, 0, NS, SZE2['C5'], SZ2['C5'])
    return dict(r, var=nm, y26=y['net'], y26p=y['evals'], full=full['net'])
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in DS for fn in ('LUCID FLEX', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job8, jobs)
    print('\nFIRM SIMS', flush=True)
    for r in RES:
        print(f'  {r["var"]:<26} {r["firm"]:<10} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m median {r["k_med"]:+6,.0f} | 2022-Oct26 net {r["full"]:+7,.0f} | 2026 net {r["y26"]:+5,.0f} ({r["y26p"]} evals)', flush=True)
