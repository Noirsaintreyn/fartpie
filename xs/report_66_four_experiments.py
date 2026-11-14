"""Nov 14 2026 FOUR EXPERIMENTS ON THE TAIL (declared before results). Data: wide daily flips with features (report_64: 31,814 long flips, 2019-2026Q1, survivorship-biased), SEC SIC industry codes, market state built from the vector itself (breadth = share of names in bull regime; equal-weight index from the same universe, close-only synthetic OHLC so its vector is approximate). Strict eras 2019-21 / 2022-26Q1, 30bp, month-clustered t. 
EXP1 INDUSTRY: HUGE (>=+100%) rate and mean net return by SIC group; hint = group share of huge winners exceeds its share of flips in BOTH eras. 
EXP2 MARKET-STATE GATES (pre-declared, five only): G1 index in vector-bull; G2 index bull flipped <=40 days ago; G3 breadth THRUST (min breadth in prior 20d <=0.30 and breadth now >= min+0.20); G4 WASHOUT (equal-weight index drawdown >= 15%); G5 index in vector-bear. 
EXP3 VOLATILITY SIZING/FILTER: ATR% quintiles, top-quintile-only by year, and vol-tilted vs inverse-vol vs equal weighting of per-flip returns. 
EXP4 BOOK BY YEAR (report_63 mechanics: K=10 slots, 10% each, same-day ties by highest flip-bar volume, 30bp, realized at exit): volume-only vs volume+G1 gate vs volume+insider vs volume+ATR top-quintile vs volume+ATR top quintile+G1."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['yr'] = P.t.dt.year; M = pd.read_pickle(SC + 'market_state.pkl'); sic = pd.read_csv(SC + 'sic.csv')
E1 = (P.t >= '2019') & (P.t < '2022'); E2 = P.t >= '2022'
def mc(S, m, base):
    s = S[m & base]; r = S[~m & base]; a = s.groupby(s.t.dt.to_period('M')).rn.mean(); b = r.groupby(r.t.dt.to_period('M')).rn.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
# ---------------- EXP1
def grp(s):
    s = int(s) if pd.notna(s) else -1
    if 2833 <= s <= 2836 or s in (8731, 2834): return 'BIOPHARMA'
    if s in (3841, 3842, 3844, 3845, 3826, 3829, 8071, 8090, 8093, 8000, 8011, 8050, 8060): return 'MEDDEV/HEALTHSVC'
    if 7370 <= s <= 7379: return 'SOFTWARE/IT'
    if 3570 <= s <= 3579 or 3660 <= s <= 3679 or s in (3690, 3695, 3559): return 'HARDWARE/SEMIS'
    if 1300 <= s <= 1389 or 2900 <= s <= 2999 or s in (4922, 4923, 4924, 5171, 5172): return 'ENERGY'
    if 1000 <= s <= 1499: return 'MINING/METALS'
    if 3711 <= s <= 3716 or 3790 <= s <= 3799 or s in (5500, 5511): return 'AUTOS/EV'
    if 6000 <= s <= 6799: return 'FINANCIAL/REIT'
    if 5200 <= s <= 5999 or 7000 <= s <= 7099 or 5800 <= s <= 5899: return 'RETAIL/CONSUMER'
    if 4400 <= s <= 4799: return 'TRANSPORT/AIR'
    if 2000 <= s <= 3999: return 'INDUSTRIAL/MFG'
    return 'OTHER/SERVICES'
sic['grp'] = sic.sic.map(grp); P = P.merge(sic[['tk', 'grp', 'desc']], on='tk', how='left'); P['grp'] = P.grp.fillna('OTHER/SERVICES')
print('EXP1 INDUSTRY: group | flips share % (19-21 / 22+) | HUGE share % (19-21 / 22+) | HUGE rate % (19-21 / 22+) | mean net % (19-21 / 22+)  [* = huge share > flip share in both eras]')
for g, x in P.groupby('grp'):
    r = []
    for e in (E1, E2):
        S = P[e]; xs = S[S.grp == g]; r.append((len(xs) / len(S) * 100, xs.HUGE.sum() / max(S.HUGE.sum(), 1) * 100, xs.HUGE.mean() * 100, xs.rn.mean() * 100))
    star = '*' if r[0][1] > r[0][0] * 1.15 and r[1][1] > r[1][0] * 1.15 else ' '
    print(f' {g:18s} flips {r[0][0]:4.1f}/{r[1][0]:4.1f} | huge {r[0][1]:4.1f}/{r[1][1]:4.1f} | rate {r[0][2]:4.2f}/{r[1][2]:4.2f} | mean {r[0][3]:+6.1f}/{r[1][3]:+6.1f} {star}')
# ---------------- EXP2
M2 = M.copy(); br = M2.breadth; M2['min20'] = br.rolling(20).min().shift(1)
M2['G1'] = M2.idx_reg > 0; M2['G2'] = (M2.idx_reg > 0) & (M2.idx_days_in_reg <= 40); M2['G3'] = (M2.min20 <= 0.30) & (br - M2.min20 >= 0.20); M2['G4'] = M2.idx_dd <= -0.15; M2['G5'] = M2.idx_reg < 0
for g in ('G1', 'G2', 'G3', 'G4', 'G5'): P[g] = M2[g].reindex(P.d0).fillna(False).values
P['VQ'] = np.digitize(P.ATRP, P.ATRP.quantile([.2, .4, .6, .8]).values); print('\nEXP2 MARKET-STATE GATES: gate on vs off | mean net %, median, huge %, n | clustered t on-vs-off   (bases: all flips, BV>=1.5)')
for en, e in (('2019-21', E1), ('2022-26Q1', E2)):
    S = P[e].reset_index(drop=True); print(f' --- {en}')
    for bn, base in (('all', pd.Series(True, index=S.index)), ('BV>=1.5', S.BV >= 1.5)):
        for g, lab in (('G1', 'index vector-bull'), ('G2', 'index bull <=40d old'), ('G3', 'breadth thrust'), ('G4', 'index drawdown>=15%'), ('G5', 'index vector-bear')):
            on = S[S[g] & base]; off = S[~S[g] & base]
            if len(on) < 20: print(f'   {bn:8s} {lab:22s} n {len(on)}'); continue
            print(f'   {bn:8s} {lab:22s} on {on.rn.mean()*100:+6.1f}% med {on.rn.median()*100:+5.1f} huge {on.HUGE.mean()*100:4.1f}% n {len(on):5d} | off {off.rn.mean()*100:+6.1f}% med {off.rn.median()*100:+5.1f} huge {off.HUGE.mean()*100:4.1f}% n {len(off):5d} | t {mc(S, S[g], base):+.1f}')
# ---------------- EXP3
print('\nEXP3 VOLATILITY: top-ATR%-quintile flips by year (all flips | with BV>=1.5): n, mean net, median, huge%  vs all flips that year')
for y, g in P.groupby('yr'):
    q = g[g.VQ == 4]; qb = q[q.BV >= 1.5]
    print(f'  {y}: Q5 n {len(q):4d} mean {q.rn.mean()*100:+6.1f}% med {q.rn.median()*100:+5.1f} huge {q.HUGE.mean()*100:4.1f}% | Q5&BV n {len(qb):4d} mean {qb.rn.mean()*100:+6.1f}% | all flips mean {g.rn.mean()*100:+5.1f}%')
print(' weighted per-flip mean net % by era (equal | weight ~ ATR% | weight ~ 1/ATR%):')
for en, e in (('2019-21', E1), ('2022-26Q1', E2)):
    S = P[e]; w1 = np.ones(len(S)); w2 = S.ATRP.values; w3 = 1 / S.ATRP.values
    print(f'  {en}: ' + ' | '.join(f'{n} {np.average(S.rn, weights=w)*100:+.2f}' for n, w in (('equal', w1), ('~ATR%', w2), ('~1/ATR%', w3))))
# ---------------- EXP4
P['ent'] = P.t.dt.normalize(); P['ext'] = [e + pd.offsets.BDay(int(h)) for e, h in zip(P.ent, P.hold)]
def book(S, K=10):
    S = S.sort_values(['ent', 'BV'], ascending=[True, False]); opn = []; pnl = []; taken = 0
    for r in S.itertuples():
        keep = []
        for (x, w, rr) in opn:
            if x > r.ent: keep.append((x, w, rr))
            else: pnl.append((x, w * rr))
        opn = keep
        if len(opn) < K: opn.append((r.ext, 1 / K, r.rn)); taken += 1
    for (x, w, rr) in opn: pnl.append((x, w * rr))
    return pnl, taken
def annual(pnl):
    s = pd.DataFrame(pnl, columns=['d', 'c']).sort_values('d'); eq = 1.0; out = {}; yr = None; start = 1.0
    for d, c in zip(s.d, s.c):
        if d.year != yr:
            if yr is not None: out[yr] = eq / start - 1
            yr = d.year; start = eq
        eq *= (1 + c)
    out[yr] = eq / start - 1; return out
base = P[P.BV >= 1.5]; top2 = P[(P.NI >= 1) & (P.BV >= 1.5)].rn.nlargest(2).index
V = {'volume only': base, 'vol + G1 index bull': base[base.G1], 'vol + insider': base[base.NI >= 1], 'vol + ATR top-quintile': base[base.VQ == 4], 'vol + ATRtop + G1': base[(base.VQ == 4) & base.G1], 'vol + insider + G1': base[(base.NI >= 1) & base.G1]}
res = {k: annual(book(S)[0]) for k, S in V.items()}; yrs = list(range(2019, 2027))
print('\nEXP4 BOOK BY YEAR (K=10, realized at exit, 2026 = Jan-Mar)'); print(' year | ' + ' | '.join(f'{k:22s}' for k in V))
for y in yrs: print(f' {y} | ' + ' | '.join(f'{res[k].get(y, float("nan"))*100:+21.1f}%' for k in V))
for k in V:
    a = res[k]; g = np.prod([1 + a.get(y, 0) for y in yrs]); print(f' {k:22s} signals {len(V[k]):5d} compounded {(g-1)*100:+7.0f}% | worst yr {min(a.values())*100:+6.1f}% | pos yrs {sum(v>0 for v in a.values())}/{len(a)}')
