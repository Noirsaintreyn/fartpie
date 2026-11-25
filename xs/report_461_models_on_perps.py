"""Nov 24 2026 report 461. Rey: do the signal models (NEG, P/C VOL, SKEW, CHARM) help the perps prop (Fantasy rules, reports 443/444 machinery)? And do they ADD trades?
DECLARED BEFORE RESULTS. They cannot add trades: every model fires only on a flip that is already in the gated 15m book (G20 ratio gate); they SELECT. So the test is selection: same names (those with option features), 15m long flips, 6 slots, exit vector flip.
Arms (all on the SAME names so the comparison is fair): BASE = all gated flips; NEG; PCV (put/call vol top third); SKW; CHM; ANY = NEG|PCV|SKW|CHM.
Rules: report 444 (High tier 8% static DD / 4% daily loss / +10% target, fee $155 per 10K, funded +5% cycles paying 4.5%, costs 0.06%/fill + 10.95%/yr financing). Grid Ge in {1,2}, Gf = 1; starts every 10th trading day 2018-09..2026-03.
Output per arm: trades per month, passed by 30/60/90d, first payout by 90/120d, breach, net $/attempt on 10K (all | 18-21 | 22+). Bar (declared): beats BASE net in 22+ AND first payout by 90d >= 25%, else no help."""
src = open('xs/report_443_fantasy_perps_sim.py').read(); exec(src[:src.index("TIER = {")])
import numpy as np, pandas as pd
exec(open('xs/report_457_optfeat_flips.py').read().split("NAMES = dict")[0].split('"""')[2])
Fx = FE.reset_index(); Fx['day'] = pd.to_datetime(Fx.day)
Lx = L.copy(); Lx['day'] = pd.to_datetime(Lx.entry).dt.normalize(); Lx = Lx.merge(Fx, on=['tk', 'day'], how='left'); Lx = Lx[Lx.tk.isin(TK)].copy()
Lx['NEG'] = Lx.gex < 0; Lx['PCV'] = Lx.r_pc_vol >= 2 / 3; Lx['SKW'] = Lx.r_skew >= 2 / 3; Lx['CHM'] = Lx.r_charm >= 2 / 3; Lx = Lx[Lx.gex.notna()]
Lx['ANY'] = Lx.NEG | Lx.PCV.fillna(False) | Lx.SKW.fillna(False) | Lx.CHM.fillna(False); Lx['BASE'] = True
print('trades', len(L), 'with features', len(Lx), flush=True)
TP0, ent0, ex0 = TP, ent_i, ex_i; idx_of = {i: k for k, i in enumerate(L.index)}
months = (grid[-1] - grid[0]).astype('timedelta64[D]').astype(float) / 30.4
TIER = {'High': (.08, .04, 155)}
tds = pd.DatetimeIndex(sorted(set(days))); starts = [tds[i] for i in range(0, len(tds), 10) if pd.Timestamp('2018-09-01') <= tds[i] <= pd.Timestamp('2026-03-31')]
rows = []; cnt = {}
for arm in ('BASE', 'NEG', 'PCV', 'SKW', 'CHM', 'ANY'):
    sel = Lx[Lx[arm].fillna(False)].index.values; sel = np.array(sorted(sel)); cnt[arm] = len(sel) / months
    TP = [TP0[k] for k in sel]; ent_i = ent0[sel]; ex_i = ex0[sel]
    for Ge in (1, 2):
        for sd in starts:
            si = int(np.searchsorted(grid, np.datetime64(sd))); t0 = pd.Timestamp(grid[si]); o, ei = cycle(si, Ge, .08, .04, .10); tp = (pd.Timestamp(grid[ei]) - t0).days if o == 'pass' else np.nan
            fp = np.nan; pays = 0
            if o == 'pass':
                cur = ei + 1
                while cur < NG - 5:
                    o2, e2 = cycle(cur, 1, .08, .04, .05); td = (pd.Timestamp(grid[e2]) - t0).days
                    if o2 == 'pass':
                        if td - tp <= 180: pays += 1
                        if np.isnan(fp): fp = td
                        cur = e2 + 1
                    else: break
            rows.append(dict(arm=arm, Ge=Ge, start=sd, out=o, tpass=tp, tfirst=fp, pays=pays))
    print(arm, 'done', flush=True)
R = pd.DataFrame(rows); R['half'] = np.where(R.start.dt.year <= 2021, '18-21', '22+')
print('\narm Ge | trades/mo | passed<=30/60/90d | 1st payout<=90/120d | breach% | net$/attempt 10K all | 18-21 | 22+')
for (arm, Ge), g in R.groupby(['arm', 'Ge'], sort=False):
    a = lambda x, c, d: (x[c] <= d).mean() * 100; net = lambda x: (x.pays * 450 - 155).mean()
    print(f"{arm:4s} {Ge} | {cnt[arm]:5.1f} | {a(g,'tpass',30):3.0f} {a(g,'tpass',60):3.0f} {a(g,'tpass',90):3.0f} | {a(g,'tfirst',90):3.0f} {a(g,'tfirst',120):3.0f} | {(g.out=='breach').mean()*100:3.0f} | {net(g):+6.0f} | {net(g[g.half=='18-21']):+6.0f} | {net(g[g.half=='22+']):+6.0f}")
