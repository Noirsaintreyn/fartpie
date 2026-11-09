"""Nov 9 2026: WHICH SETUP KEEPS ~$5K A MONTH MOST CONSISTENTLY, WHERE DO ITS LOSSES COME FROM, AND SHOULD WE SKIP / DE-RISK / HAND OFF THOSE ENVIRONMENTS? DECLARED BEFORE RESULTS.
Fleet (xs/fleet_firms.py policy): Lucid Flex, 5 accounts, one eval a day while under 5 and the system's trailing 20-day P&L is positive, cash starts at 20 fees, withdraw ASAP.
Setups: S0 profit algo only (U3 + YZ) | S1 + old qualifier (first noise signal +7.5 / 40) | S2 + rules C | S3 + rules C + TPO (current) | S4 + rules D |
        S5 = S3 with funded sizing profit x1.25, rules x1.5 | S6 = S3 + V15a (vector all day, off in top-third gamma, 3 micros) | S7 rules C + TPO alone.
Consistency (fleet started Jan 2022, the honest window): average and median month, months >= $2.5k, losing months, worst 3 months, longest run of months with no payout.
Loss attribution (the setup picked): each engine's points per trade (1 contract) by environment at the prior close - QQQ gamma sign, gamma tercile, Yang-Zhang 20d tercile,
IV tercile, DEX tercile, NQ vs its 50-day average, FOMC / big 8:30 days; Rule A flips also by side, with / against DEX, open vs prior value, gap with / against.
RULE (declared): an engine is skipped or halved in a cell only if it LOST per trade in BOTH 2018-21 and 2022-25 with >= 15 trades in each; 'hand off' = in those cells,
the other engines that made money there in both periods. Each candidate is then run through the fleet (2022 start) next to the setup as is."""
_fm = open('/home/claude/fartpie/xs/fleet_firms.py').read(); exec(_fm[:_fm.rindex("if __name__ == '__main__':")])
_ra = open('/home/claude/fartpie/xs/rules_algo2.py').read()
E_ = entries(); hm = {}
for d in days:
    for i in gi[d]: hm[h2x[i]] = m[i]
def touch2(j, jend, dr, S_):
    pts, jx = touch(j, jend, dr, S_); e = Xo[j]; seg = Xl[j:jx + 1] if dr > 0 else Xh[j:jx + 1]
    return pts, (max(min(((seg - e) * dr).min(), 0.0), -S_) if pts > -S_ - 1 else -S_) - 1.0
Q40 = pd.DataFrame([(d, hm[j]) + touch2(j, end_x[d], dr, 40) for d, j, dr in E_['NOISE']], columns=['day', 'm', 'pts', 'mae'])
def rules_take(nm, extra):
    cols = [x if x != 'tadr' else f'tadr{nm}' for x in FEAT] + extra; DS2 = DS.copy()
    mdl_ = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=.05, min_samples_leaf=300, l2_regularization=1.0, random_state=0).fit(DS2.loc[tr, cols], DS2.loc[tr, f'y{nm}'])
    DS2['_p'] = mdl_.predict_proba(DS2[cols])[:, 1]
    def dly(msk, cut):
        z_ = DS2[msk & (DS2._p >= cut)]; return z_.sort_values(['day', 'tmin', '_p'], ascending=[True, True, False]).drop_duplicates('day')
    rs = [(cut, dly(va, cut)[f'usd{nm}'].mean(), len(dly(va, cut))) for cut in np.round(np.arange(.40, .97, .01), 2)]; rs = [r for r in rs if r[2] >= 40]
    gd = [r for r in rs if r[1] >= 10]; ct = min(gd)[0] if gd else max(rs, key=lambda r: r[1])[0]
    q = dly(DS2.yr >= 2014, ct); q = q[~(q.day.isin(FOMC) & (q.entry_m >= 810))]
    return q.assign(m=q.entry_m, pts=q[f'pts{nm}'], mae=q[f'mae{nm}'], sz=q[f'mic{nm}'] / 10, ce=1.0, cf=1.0, dir=q.dir, tmin=q.tmin)[['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf', 'dir', 'tmin']]
RCb = rules_take('C', []); RDb = rules_take('D', [])
V = run_sig2('s_vector'); V15a = V[~hig(V)]
ENG = {'RA': (RAs, 1.0, RULE['RA']), 'M1': (M1gg, .3, RULE['M1']), 'R1': (R1g, .5, RULE['R1']), 'ON+': (OBpos, .4, RULE['ON+']), 'ON-': (OBneg, .1, RULE['ON-'])}
def setup(extra=(), pm=1.0, profit=True):
    sp = [(T_, u * pm, yz) for T_, u, yz in ENG.values()] if profit else []
    return sp + list(extra)
SET = {'S0 profit only': setup(), 'S1 + old qualifier': setup([(Q40, 1.0, None)]), 'S2 + rules C': setup([(RCb, 1.0, None)]), 'S3 + rules C+TPO (now)': setup([(RC, 1.0, None)]),
       'S4 + rules D': setup([(RDb, 1.0, None)]), 'S6 S3 + V15a': setup([(RC, 1.0, None), (V15a, .3, None)]), 'S7 rules C+TPO alone': setup([(RC, 1.0, None)], profit=False)}
def fleet(spec_e, spec_f, fn, zn, k0, budget=20):
    global LE, LF
    LE, LF = lists(spec_e, 'e'), lists(spec_f, 'f'); R = FIRM[fn]; z = SZ[zn]; fee = R['fee'][z['i']]; TRJ = {}
    pnl = pd.Series([sum(net * 2 * max(1, round(10 * th[0] * z['scale'])) for th, net, mae, p in zz) for zz in LE]); hot = (pnl.rolling(20).sum().shift(1) > 0).values
    cash = budget * fee; act = []; pq = []; P = {}; Cn = {}; Fe = {}
    for k in range(k0, NS):
        mo = D_[k].to_period('M')
        while pq and pq[0][0] == k:
            kk, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; Cn[mo] = Cn.get(mo, 0) + 1
        act = [e for e in act if e >= k]
        if len(act) < 5 and cash >= fee and hot[k]:
            if k not in TRJ:
                r, k2 = eval_run(R, z, k, NS)
                if r != 'pass': TRJ[k] = (k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_run(R, z, k2 + 1, NS); TRJ[k] = (NS if fr == 'open' else k3, pays)
            end, pays = TRJ[k]; cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
            for kk, x in pays: heapq.heappush(pq, (kk, x))
    months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
    N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C_ = pd.Series({mo: Cn.get(mo, 0) for mo in months})
    run = mx = 0
    for v in C_.values: run = run + 1 if v == 0 else 0; mx = max(mx, run)
    return dict(avg=N.mean(), med=N.median(), ge25=np.mean(N >= 2500), lose=np.mean(N < 0), w3=N.rolling(3).sum().min(), dry=mx, sd=N.std(), N=N)
def line(nm, r): return (f'  {nm:<34} avg ${r["avg"]:6,.0f} | median ${r["med"]:6,.0f} | months >= $2.5k {r["ge25"]:4.0%} | losing {r["lose"]:4.0%} | worst 3 mo ${r["w3"]:7,.0f} | '
                         f'longest no-payout run {r["dry"]:2d} mo | sd ${r["sd"]:6,.0f}')
if __name__ == '__main__':
    LOCK['on'] = False; k22 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01')))
    RES = {}
    for zn in ('50K', '150K'):
        print(f'\n=== 1) Lucid Flex {zn} x 5, fleet started Jan 2022 (net of eval fees, per month) ===', flush=True)
        for nm, sp in SET.items():
            r = fleet(sp, sp, 'Lucid Flex', zn, k22); RES[(zn, nm)] = r; print(line(nm, r), flush=True)
        sp = SET['S3 + rules C+TPO (now)']; spf = setup([(RC, 1.5, None)], pm=1.25); r = fleet(sp, spf, 'Lucid Flex', zn, k22); RES[(zn, 'S5')] = r; print(line('S5 S3, funded p x1.25 r x1.5', r), flush=True)
    # ---- 2) loss attribution ----
    print('\n=== 2) where the points come from and go: points per trade at 1 contract (n), 2018-21 | 2022-25 ===', flush=True)
    ma50 = DY.c.rolling(50).mean().shift(1); trend = np.sign(DY.c.shift(1) - ma50)
    def env(d):
        d = pd.Timestamp(d); g = DY.gex.get(d, np.nan); g2 = DY.g2p.get(d, np.nan); yz = DY.yz20p.get(d, np.nan); iv = fget(d, 'ivp'); dx = fget(d, 'dexp'); tr_ = trend.get(d, np.nan)
        t3_ = lambda v: 'low' if v <= 1 / 3 else ('high' if v >= 2 / 3 else 'mid') if np.isfinite(v) else 'na'
        nw = NW.loc[d] if d in NW.index else None
        return {'gamma sign': 'neg' if g < 0 else ('pos' if g > 0 else 'na'), 'gamma tercile': t3_(g2), 'YZ20 tercile': t3_(yz), 'IV tercile': t3_(iv), 'DEX tercile': t3_(dx),
                'NQ vs 50d avg': 'above' if tr_ > 0 else ('below' if tr_ < 0 else 'na'),
                'day type': 'FOMC' if nw is not None and nw.fomc == 1 else ('big 8:30' if nw is not None and nw.r830 >= 3 else 'other')}
    ALL = {**{k: v[0] for k, v in ENG.items()}, 'RULES C+TPO': RC, 'V15a': V15a}
    TAB = []
    for en, T_ in ALL.items():
        T_ = T_[(T_.day >= '2018-02-01')].copy()
        if en == 'RA': T_ = T_[T_.sz > 0]
        E = pd.DataFrame([env(d) for d in T_.day], index=T_.index); T_ = T_.join(E); T_['per'] = np.where(T_.day.dt.year >= 2022, 'b', 'a')
        if 'dir' in T_: T_['side'] = np.where(T_.dir > 0, 'long', 'short')
        if en == 'RA':
            T_['DEX with/against'] = np.where(T_.dexag, 'with', 'against')
            T_['open vs prior value'] = [('above VAH' if DY.o[d] > VA[days[di_[d] - 1]][2] else ('below VAL' if DY.o[d] < VA[days[di_[d] - 1]][0] else 'inside')) for d in T_.day]
            T_['gap with trade'] = np.where((DY.o.reindex(T_.day).values - DY.c.shift(1).reindex(T_.day).values) * T_.dir.values > 0, 'with', 'against')
        a, b = T_[T_.per == 'a'].pts, T_[T_.per == 'b'].pts
        print(f'\n  {en}: {len(T_)} trades | 2018-21 {a.mean():+.2f} ({len(a)}) | 2022-25 {b.mean():+.2f} ({len(b)})', flush=True)
        vars_ = list(E.columns) + [c_ for c_ in ('side', 'DEX with/against', 'open vs prior value', 'gap with trade') if c_ in T_]
        for v in vars_:
            parts = []
            for lv, g_ in T_.groupby(v):
                ga, gb = g_[g_.per == 'a'].pts, g_[g_.per == 'b'].pts; parts.append(f'{lv}: {ga.mean():+5.1f} ({len(ga)}) | {gb.mean():+5.1f} ({len(gb)})')
                TAB.append((en, v, lv, ga.mean(), len(ga), gb.mean(), len(gb)))
            print(f'    {v:<20} ' + '   '.join(parts), flush=True)
    TB = pd.DataFrame(TAB, columns=['eng', 'var', 'lv', 'a', 'na', 'b', 'nb'])
    BAD = TB[(TB.a < 0) & (TB.b < 0) & (TB.na >= 15) & (TB.nb >= 15) & (TB.lv != 'na')]
    print('\n=== 3) cells where an engine lost in BOTH periods (>= 15 trades each) - and who made money there in both ===', flush=True)
    for r_ in BAD.itertuples():
        helpers = TB[(TB['var'] == r_.var) & (TB.lv == r_.lv) & (TB.eng != r_.eng) & (TB.a > 0) & (TB.b > 0) & (TB.na >= 15) & (TB.nb >= 15)]
        print(f'  {r_.eng:<12} {r_.var} = {r_.lv}: {r_.a:+.1f} ({r_.na}) | {r_.b:+.1f} ({r_.nb})   -> made money there: '
              + (', '.join(f'{h.eng} {h.a:+.1f}/{h.b:+.1f}' for h in helpers.itertuples()) or 'none'), flush=True)
    # ---- 4) skip / halve those cells, run through the fleet ----
    def filt(T_, en, mult):
        T_ = T_.copy(); E = pd.DataFrame([env(d) for d in T_.day], index=T_.index)
        if en == 'RA':
            E['DEX with/against'] = np.where(T_.dexag, 'with', 'against'); E['side'] = np.where(T_.dir > 0, 'long', 'short')
            E['open vs prior value'] = [('above VAH' if DY.o[d] > VA[days[di_[d] - 1]][2] else ('below VAL' if DY.o[d] < VA[days[di_[d] - 1]][0] else 'inside')) if di_.get(d, 0) > 0 else 'na' for d in T_.day]
            E['gap with trade'] = np.where((DY.o.reindex(T_.day).values - DY.c.shift(1).reindex(T_.day).values) * T_.dir.values > 0, 'with', 'against')
        if 'dir' in T_ and 'side' not in E: E['side'] = np.where(T_.dir > 0, 'long', 'short')
        hit = np.zeros(len(T_), bool)
        for r_ in BAD[BAD.eng == en].itertuples():
            if r_.var in E: hit |= (E[r_.var] == r_.lv).values
        if 'sz' not in T_: T_['sz'], T_['ce'], T_['cf'] = 1.0, 1.0, 1.0
        T_.loc[hit, 'sz'] = T_.loc[hit, 'sz'] * mult; return T_, hit.sum()
    if len(BAD):
        for mult, lab in ((0.0, 'skip'), (0.5, 'half size')):
            sp = []; info = []
            for en, (T_, u, yz) in ENG.items():
                T2, nh = filt(T_, en, mult); sp.append((T2, u, yz)); info.append(f'{en} {nh}')
            T2, nh = filt(RC, 'RULES C+TPO', mult); sp.append((T2, 1.0, None)); info.append(f'rules {nh}')
            for zn in ('50K', '150K'):
                r = fleet(sp, sp, 'Lucid Flex', zn, k22); print(line(f'{zn} S3, {lab} the bad cells', r) + f'   [{", ".join(info)} trades touched]', flush=True)
                print(line(f'{zn} S3 as is', RES[(zn, 'S3 + rules C+TPO (now)')]), flush=True)
