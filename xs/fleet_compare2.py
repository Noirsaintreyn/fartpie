"""Nov 9 2026: FAIR SETUP COMPARISON + WHAT BLOWS THE ACCOUNTS. DECLARED BEFORE RESULTS.
xs/fleet_compare.py ranked setups with ONE fleet started Jan 2022 on 20 fees - whether it survived its first weeks decided most rows (luck, not the setup).
Here: a 5-account Lucid Flex fleet (same policy) started at the first trading day of EVERY month Jan 2022 - Dec 2023 (24 starts), each run to Jan 2026, budget 40 fees
(so ruin does not decide the comparison); pooled months from all starts. Also the share of starts that survive on 20 fees (first payout before the cash runs out).
Setups as before (S0-S7) + S3x = S3 with the overnight-after-negative-gamma leg skipped where it lost in both periods (low IV tercile, DEX middle tercile, big 8:30 days).
Blowup attribution (S3, Flex 50K funded accounts started every 5th day 2022 - Jun 2025): each engine's P&L over the 10 trading days that end in a blown account,
and over all funded days, and on the worst 5% of days."""
_fc = open('/home/claude/fartpie/xs/fleet_compare.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
def onx(T_):
    T_ = T_.copy(); T_['sz'], T_['ce'], T_['cf'] = 1.0, 1.0, 1.0
    bad = [fget(d, 'ivp') <= 1 / 3 or (1 / 3 < fget(d, 'dexp') < 2 / 3) or (d in NW.index and NW.r830.get(d, 0) >= 3) for d in T_.day]
    T_.loc[np.array(bad), 'sz'] = 0.0; return T_
SET['S3x S3, ON- skipped in its bad cells'] = [(RAs, 1.0, RULE['RA']), (M1gg, .3, RULE['M1']), (R1g, .5, RULE['R1']), (OBpos, .4, RULE['ON+']), (onx(OBneg), .1, RULE['ON-']), (RC, 1.0, None)]
SET['S5 S3, funded p x1.25 r x1.5'] = None
def multi(sp_e, sp_f, zn, budgets=(40, 20)):
    global LE, LF
    LE, LF = lists(sp_e, 'e'), lists(sp_f, 'f'); R = FIRM['Lucid Flex']; z = SZ[zn]; fee = R['fee'][z['i']]; TRJ = {}
    pnl = pd.Series([sum(net * 2 * max(1, round(10 * th[0] * z['scale'])) for th, net, mae, p in zz) for zz in LE]); hot = (pnl.rolling(20).sum().shift(1) > 0).values
    def traj(k):
        if k not in TRJ:
            r, k2 = eval_run(R, z, k, NS)
            if r != 'pass': TRJ[k] = (k2 if r == 'fail' else NS, [])
            else: fr, k3, pays = fund_run(R, z, k2 + 1, NS); TRJ[k] = (NS if fr == 'open' else k3, pays)
        return TRJ[k]
    starts = [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2022-01', '2023-12', freq='M')]; out = {}
    for budget in budgets:
        allN, surv, dry = [], 0, []
        for k0 in starts:
            cash = budget * fee; act = []; pq = []; P = {}; Fe = {}; Cn = {}
            for k in range(k0, NS):
                mo = D_[k].to_period('M')
                while pq and pq[0][0] == k:
                    kk, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; Cn[mo] = Cn.get(mo, 0) + 1
                act = [e for e in act if e >= k]
                if len(act) < 5 and cash >= fee and hot[k]:
                    end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                    for kk, x in pays: heapq.heappush(pq, (kk, x))
            months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
            N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C_ = pd.Series({mo: Cn.get(mo, 0) for mo in months})
            allN.append(N.values); surv += C_.sum() > 0; run = mx = 0
            for v in C_.values: run = run + 1 if v == 0 else 0; mx = max(mx, run)
            dry.append(mx)
        A = np.concatenate(allN); out[budget] = dict(avg=A.mean(), med=np.median(A), ge25=np.mean(A >= 2500), ge5=np.mean(A >= 5000), lose=np.mean(A < 0), surv=surv / len(starts), dry=np.median(dry))
    return out
if __name__ == '__main__':
    LOCK['on'] = False
    for zn in ('50K', '150K'):
        print(f'\n=== Lucid Flex {zn} x 5: fleets started every month Jan 2022 - Dec 2023, all their months pooled (net of eval fees) ===', flush=True)
        print(f'  {"setup":<38} | budget 40 fees: avg / median month, months >= $2.5k, >= $5k, losing, median longest no-payout run | starts that survive on 20 fees', flush=True)
        for nm, sp in SET.items():
            if sp is None: o_ = multi(SET['S3 + rules C+TPO (now)'], setup([(RC, 1.5, None)], pm=1.25), zn)
            else: o_ = multi(sp, sp, zn)
            a = o_[40]; b = o_[20]
            print(f'  {nm:<38} | ${a["avg"]:6,.0f} / ${a["med"]:6,.0f}, {a["ge25"]:4.0%}, {a["ge5"]:4.0%}, losing {a["lose"]:4.0%}, dry {a["dry"]:4.1f} mo | {b["surv"]:4.0%}', flush=True)
    # ---- blowup attribution (S3, 50K) ----
    print('\n=== what blows the funded accounts (S3, Flex 50K, funded accounts started every 5th day 2022 - Jun 2025) ===', flush=True)
    sp = SET['S3 + rules C+TPO (now)']; LE, LF = lists(sp, 'e'), lists(sp, 'f'); R = FIRM['Lucid Flex']; z = SZ['50K']
    names = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES']; EL = {nm: lists([s_], 'f') for nm, s_ in zip(names, sp)}
    ED = pd.DataFrame({nm: [sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in L_] for nm, L_ in EL.items()}, index=D_)
    lost_k = []; fund_days = set()
    for s in range(k22, k25, 5):
        fr, k3, pays = fund_run(R, z, s, NS); end = NS - 1 if fr == 'open' else k3; fund_days.update(range(s, end + 1))
        if fr == 'lost': lost_k.append(k3)
    W = pd.concat([ED.iloc[max(0, k - 9):k + 1].sum() for k in lost_k], axis=1).T if lost_k else pd.DataFrame()
    FD = ED.iloc[sorted(fund_days)]; tot = FD.sum(axis=1); worst = FD[tot <= tot.quantile(.05)]
    print(f'  {len(lost_k)} blown accounts. $ by engine (1 account, 50K size):', flush=True)
    print(f'  {"":<36}' + ''.join(f'{n_:>9}' for n_ in names) + f'{"total":>10}', flush=True)
    for lab, row in (('all funded days, $ per day', FD.mean()), ('worst 5% of days, $ per day', worst.mean()), ('10 days before a blowup, $ total', W.mean() if len(W) else FD.mean() * 0)):
        print(f'  {lab:<36}' + ''.join(f'{v:9,.0f}' for v in row.values) + f'{row.sum():10,.0f}', flush=True)
    print(f'  share of blowup windows where the engine lost money: ' + ', '.join(f'{n_} {np.mean(W[n_] < 0):.0%}' for n_ in names), flush=True)
