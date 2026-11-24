"""Nov 24 2026 report 444. Rey: pass rate isn't the point, payouts are; evals are cheap, so accept a lower pass rate. Separate leverage for EVAL (G_e) and FUNDED (G_f), flat ratio-gated flip book, Fantasy rules (report 443 machinery).
DECLARED BEFORE RESULTS. Grid: tier High (8% DD / 4% daily, fee $155 on 10K) and Med (6% / 3%, $115); G_e in {1,2,3,4}, G_f in {0.5,1}. Starts: every 10th trading day 2018-09..2026-03.
Per attempt record: day passed, day of first payout (funded cycle +5% -> payout 4.5% of start), payouts in the first 180 days after passing, breach in funded.
Report (Rey's format): share PASSED by 30/60/90 days, share with FIRST PAYOUT by 60/90/120 days (from start), breached, still running; $ per attempt on 10K = 450*payouts - fee; ROI = net/fee; by half 18-21 | 22+.
Objective (declared): maximize net per attempt in 22+ starts subject to first payout by 90 days >= 25% of attempts and net > 0 in BOTH halves."""
src = open('xs/report_443_fantasy_perps_sim.py').read(); exec(src[:src.index("TIER = {")])
import numpy as np, pandas as pd
TIER = {'High': (.08, .04, 155), 'Med': (.06, .03, 115)}
tds = pd.DatetimeIndex(sorted(set(days))); starts = [tds[i] for i in range(0, len(tds), 10) if pd.Timestamp('2018-09-01') <= tds[i] <= pd.Timestamp('2026-03-31')]
rows = []
for tier, (dd, dl, fee) in TIER.items():
    for Ge in (1, 2, 3, 4):
        for Gf in (.5, 1):
            for sd in starts:
                si = int(np.searchsorted(grid, np.datetime64(sd))); t0 = pd.Timestamp(grid[si]); o, ei = cycle(si, Ge, dd, dl, .10); tp = (pd.Timestamp(grid[ei]) - t0).days if o == 'pass' else np.nan
                fp = np.nan; pays = 0; fb = False
                if o == 'pass':
                    cur = ei + 1
                    while cur < NG - 5:
                        o2, e2 = cycle(cur, Gf, dd, dl, .05); td = (pd.Timestamp(grid[e2]) - t0).days
                        if o2 == 'pass':
                            if td - tp <= 180: pays += 1
                            if np.isnan(fp): fp = td
                            cur = e2 + 1
                        else: fb = o2 == 'breach'; break
                rows.append(dict(tier=tier, Ge=Ge, Gf=Gf, start=sd, out=o, tpass=tp, tfirst=fp, pays=pays, fb=fb, fee=fee))
        print(tier, Ge, 'done', flush=True)
R = pd.DataFrame(rows); R['half'] = np.where(R.start.dt.year <= 2021, '18-21', '22+'); R.to_pickle(SP + 'r444.pkl')
print('\ntier Ge Gf | passed<=30/60/90d | 1st payout<=60/90/120d | breach(eval) % | net$/attempt 10K all | 18-21 | 22+ | ROI 22+')
for (tier, Ge, Gf), g in R.groupby(['tier', 'Ge', 'Gf']):
    a = lambda x, c, d: (x[c] <= d).mean() * 100; net = lambda x: (x.pays * 450 - x.fee).mean(); h1, h2 = g[g.half == '18-21'], g[g.half == '22+']
    print(f"{tier:4s} {Ge:2d} {Gf:3.1f} | {a(g,'tpass',30):3.0f} {a(g,'tpass',60):3.0f} {a(g,'tpass',90):3.0f} | {a(g,'tfirst',60):3.0f} {a(g,'tfirst',90):3.0f} {a(g,'tfirst',120):3.0f} | {(g.out=='breach').mean()*100:3.0f} | {net(g):+6.0f} | {net(h1):+6.0f} | {net(h2):+6.0f} | {net(h2)/g.fee.iloc[0]:+4.1f}x")
