"""Report 535 (Nov 28 2026): Rey: add PCV/SKW/CHM (+NEG) to the perps prop model to get MORE trades. Report 461 showed the flags SELECT inside the G20 gated book (fewer trades, slower to +10%, no help). Here the gate is RELAXED so the flags can ADD trades.
DECLARED BEFORE RESULTS. Same 23 names, 15m long flips, vector exit, 6 slots, report-444 High tier, starts every 10th trading day 2018-09..2026-03, Ge in {1,2}, Gf=1 (identical machinery to 461). Flip set = ALL flips (no ratio gate) with gate flags: G20 (20d median options ratio >= 2, the book), G15 (>= 1.5, the live filter). Flags NEG (net gamma<0), PCV/SKW/CHM (own 252d rank top third) from prior close; nfl = number of flags.
Arms: BASE = G20 (= 461 BASE). U_ANY = G20 | (G15 & nfl>=1). U_2 = G20 | (G15 & nfl>=2). U_NOG2 = G20 | (nfl>=2, no ratio gate). F_ONLY2 = G15 & nfl>=2 (flags replace the G20 gate). 
Output as 461 plus trades/month. Declared bar: MORE trades than BASE AND (beats BASE net/attempt in 22+ AND first payout by 90d >= 25%) else the extra trades do not pay. Selection effects are removed by construction (BASE is a subset of every U arm).
"""
SPp = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
s440 = open('xs/report_440_neggamma_short_and_2025.py').read()
s440 = s440.replace(' or not gate[fb - 1]: continue', ': continue').replace('neg=bool(gprev[fb - 1] < 0),', 'neg=bool(gprev[fb - 1] < 0), G20=bool(gate[fb - 1]), G15=bool(gate15[fb - 1]),')
s440 = s440.replace('    for dirn, bucket in', '    gate15 = G15[tk].reindex(days, method="ffill").values.astype(float) == 1 if tk in G15 else np.zeros(n, bool)\n    for dirn, bucket in', 1)
s440 = s440.replace('names = sorted(', 'G15 = Rr.rolling(20, min_periods=10).median().shift(1) >= 1.5\nnames = sorted(', 1)
assert 'gate15' in s440 and 'G15 = Rr' in s440
open(SPp + 'p440.py', 'w').write(s440)
src = open('xs/report_443_fantasy_perps_sim.py').read().replace("xs/report_440_neggamma_short_and_2025.py", SPp + "p440.py")
exec(src[:src.index("TIER = {")])
import numpy as np, pandas as pd
exec(open('xs/report_457_optfeat_flips.py').read().split("NAMES = dict")[0].split('"""')[2])
Fx = FE.reset_index(); Fx['day'] = pd.to_datetime(Fx.day)
Lx = L.copy(); Lx['day'] = pd.to_datetime(Lx.entry).dt.normalize(); Lx = Lx.merge(Fx, on=['tk', 'day'], how='left'); Lx = Lx[Lx.tk.isin(TK)].copy()
Lx['NEG'] = Lx.gex < 0; Lx['PCV'] = Lx.r_pc_vol >= 2 / 3; Lx['SKW'] = Lx.r_skew >= 2 / 3; Lx['CHM'] = Lx.r_charm >= 2 / 3; Lx = Lx[Lx.gex.notna()]
Lx['ANY'] = Lx.NEG | Lx.PCV.fillna(False) | Lx.SKW.fillna(False) | Lx.CHM.fillna(False); Lx['nfl'] = Lx.NEG.astype(int) + Lx.PCV.fillna(False).astype(int) + Lx.SKW.fillna(False).astype(int) + Lx.CHM.fillna(False).astype(int); Lx['BASE'] = Lx.G20; Lx['U_ANY'] = Lx.G20 | (Lx.G15 & (Lx.nfl >= 1)); Lx['U_2'] = Lx.G20 | (Lx.G15 & (Lx.nfl >= 2)); Lx['U_NOG2'] = Lx.G20 | (Lx.nfl >= 2); Lx['F_ONLY2'] = Lx.G15 & (Lx.nfl >= 2)
print('trades', len(L), 'with features', len(Lx), flush=True)
TP0, ent0, ex0 = TP, ent_i, ex_i; idx_of = {i: k for k, i in enumerate(L.index)}
months = (grid[-1] - grid[0]).astype('timedelta64[D]').astype(float) / 30.4
TIER = {'High': (.08, .04, 155)}
tds = pd.DatetimeIndex(sorted(set(days))); starts = [tds[i] for i in range(0, len(tds), 10) if pd.Timestamp('2018-09-01') <= tds[i] <= pd.Timestamp('2026-03-31')]
rows = []; cnt = {}
for arm in ('BASE', 'U_ANY', 'U_2', 'U_NOG2', 'F_ONLY2'):
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
    print(f"{arm:8s} {Ge} | {cnt[arm]:5.1f} | {a(g,'tpass',30):3.0f} {a(g,'tpass',60):3.0f} {a(g,'tpass',90):3.0f} | {a(g,'tfirst',90):3.0f} {a(g,'tfirst',120):3.0f} | {(g.out=='breach').mean()*100:3.0f} | {net(g):+6.0f} | {net(g[g.half=='18-21']):+6.0f} | {net(g[g.half=='22+']):+6.0f}")
