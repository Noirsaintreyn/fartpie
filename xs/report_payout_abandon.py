"""CAPPED SIM - Lucid contract limits applied (declared before results). FLEET OF 10 EVALS / 5 FUNDED WITH A HELD RESERVE (Nov 10 2026, declared before results): up to 10 accounts held (evals + passed-but-waiting), at most 5 trading funded at once, passed accounts beyond 5 wait in a reserve queue and go funded when a slot opens (ASSUMES Lucid lets a passed account wait unactivated - check). Buy 2 evals per payout up to 10 total, then replace whenever an account fails or goes funded-dead. Compare with the 5-slot plan. RAMP-UP POLICY (Nov 10 2026, declared before results): start with ONE Lucid 50K eval, after each payout buy 2 more (up to 5 slots), once 5 are full buy a replacement whenever one falls out. Room rule, cost 0.5 pt. Copied setup from report_room_cost05.py. Earlier: Nov 10 2026: COST 0.5 PT INSTEAD OF 1.0 (+0.5 pt added back on every trade taken) and a gentler overnight-range variant (wide x1.25 only), same report as xs/report_t4_e2.py. Earlier: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
Per eval (started every 2nd trading day; 2018-21 and 2022-25 starts), timing (start -> pass, PASS -> FIRST PAYOUT on the funded account, between payouts),
one account at a time from 2018, the 5-account fleet from 2018 (payouts a month, months with 0 / >= 2, $ a month, losing, worst, by year) and 24 starts 2022-23."""
_t = open('/home/claude/fartpie/xs/trail_tiers.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
if __name__ == '__main__':
    V = run_sig2('s_vector'); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0)].copy()
    REF = V[V.day >= '2018-02-01'].join(feats(V[V.day >= '2018-02-01'])); q_ = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30; cut = q_.quantile(1 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    base = []
    for en, (T_, u, yz) in zip(NAMES, variant(True, relm)):
        T_ = T_.copy()
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    import pandas as _pd
    _Dn = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/nq_days.pkl'); _on = _Dn.OH - _Dn.OL; _WD = (_on > _on.shift(1).rolling(20).quantile(2 / 3))
    def _ow(T_, en):
        T_ = T_.copy()
        if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
            d = _pd.to_datetime(T_.day); w = _WD.reindex(d).values; ok = np.asarray(d <= _pd.Timestamp('2026-01-30'))
            T_['sz'] = T_.sz.values * np.where(ok, np.where(w == True, 1.5, 0.5), 1.0)
        return T_
    def _c05(T_): T_ = T_.copy(); T_['pts'] = T_.pts.values + .5 * (T_.sz.values > 0); return T_
    base_c = [(_c05(T_), u, yz) for T_, u, yz in base]
    def _w125(T_, en):
        T_ = _c05(T_)
        if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
            d = _pd.to_datetime(T_.day); w = _WD.reindex(d).values; ok = np.asarray(d <= _pd.Timestamp('2026-01-30')); T_['sz'] = T_.sz.values * np.where(ok & (w == True), 1.25, 1.0)
        return T_
    base_w = [(_w125(T_, en), u, yz) for en, (T_, u, yz) in zip(NAMES, base)]
    LOCK['on'] = False; import os; TIER = os.environ.get('TIER', '50K'); A = ACCX[TIER]; fee = A['fee']; E2, F0 = EP['E2 harder de-risk'], FP['F0 x1']
    END = {'RA': 720, 'V15a': 720, 'M1': 960, 'R1': 960, 'RULES': 960, 'ON+': 'N', 'ON-': 'N'}
    def lists_cap(spec, phase):
        L = [[] for _ in range(NS)]
        for (T_, u, yzrule), en in zip(spec, NAMES):
            if u <= 0: continue
            for r_ in T_.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * r_.sz * (r_.ce if phase == 'e' else r_.cf)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m if en not in ('ON+', 'ON-') else -1, ((min(uu, 1.75),), r_.pts, r_.mae, (r_.m, END[en]))))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    def cap_eval(A): return 40 if TIER == '50K' else 100
    def cap_fund(A, prof):
        if TIER == '50K': return 20 if prof < 1000 else 30 if prof < 2000 else 40
        return 40 if prof < 1000 else 50 if prof < 2000 else 60 if prof < 3000 else 80 if prof < 4500 else 100
    def run_day_c(L, k, st, mll, scale, mult, cap):
        cum = 0.0; bal = st['bal']; opn = []                                   # (window end, contracts) of trades still open
        for th, net, mae, p in L[k]:
            if cum <= -1000 * scale: break
            n = max(1, int(round(10 * th[0] * scale * mult)))
            if cap is not None:
                m0, we = p; night = (we == 'N')
                opn = [(w, c) for w, c in opn if (w == 'N') == night and (night or w > m0)]
                room = cap - sum(c for _, c in opn)
                if room <= 0: continue
                n = min(n, room); opn.append((we, n))
            mm = 2.0 * n
            if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
            cum += net * mm
        return cum, False
    def EV(L, A, s, end, pol, capped):
        st = dict(bal=0.0, floor=-A['mll'], peak=0.0); best = 0.0
        for k in range(s, end):
            cu = (st['bal'] - st['floor']) / A['mll']; pr = st['bal'] / A['target']
            pnl, br = run_day_c(L, k, st, A['mll'], A['scale'], pol(cu, pr), cap_eval(A) if capped else None)
            if br: return 'fail', k
            st['bal'] += pnl; best = max(best, pnl)
            if st['bal'] >= A['target'] and best <= .5 * st['bal']: return 'pass', k
            if st['floor'] < 100: st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
            if ABX is not None and (k - s + 1) % 21 == 0 and st['bal'] < ABX: return 'fail', k
        return 'open', end
    def FU(L, A, s, end, pol, capped):
        st = dict(bal=0.0, floor=-A['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
        for k in range(s, end):
            cu = (st['bal'] - st['floor']) / A['mll']
            pnl, br = run_day_c(L, k, st, A['mll'], A['scale'], pol(cu, len(pays)), cap_fund(A, st['bal']) if capped else None)
            if br: return 'lost', k, pays
            st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
            if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
            if wins >= 5 and b > 0 and (not pays or since > 0) and b >= BTH:
                amt = min(.5 * b, A['cap'], FLAT or 1e9)
                if amt >= 500:
                    pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                    if len(pays) >= 5: return 'graduated', k, pays
        return 'open', end, pays
    LE_, LF_ = lists_cap(base_c, 'e'), lists_cap(base_c, 'f'); CAP = False
    def sim(k0, per_payout=2, max_acc=10, max_fund=5, stagger=0):
        global CAP
        due = []; cash = 0.0; oop = 0.0; ev = []; res = []; fu = []; pq = []; P = {}; Fe = {}; NP = {}; full_k = None; resc = 0; full = False; pend = []
        def buy(k):
            nonlocal cash, oop
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            r, k2 = EV(LE_, A, k, NS, E2, CAP); ev.append((k2 if r in ('pass', 'fail') else NS, r)); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
            for i_ in range(1, max(0, int(np.ceil(max(k2 - k + 1, 1) / 21)) - 1) + 1):
                if k + 21 * i_ < NS: heapq.heappush(due, (k + 21 * i_, REN))
        def go_funded(k, st):
            fr, k3, pays = FU(LF_, A, st, NS, F0, CAP); fu.append(NS if fr == 'open' else k3)
            for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        buy(k0)
        for k in range(k0 + 1, NS):
            mo = D_[k].to_period('M'); got = 0
            while due and due[0][0] <= k:
                _, amt_ = heapq.heappop(due); cash -= amt_; Fe[mo] = Fe.get(mo, 0) + amt_
                if cash < 0: oop += -cash; cash = 0.0
            while pq and pq[0][0] == k:
                kk_, x = heapq.heappop(pq); cash += x; got += 1; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
            fu = [e for e in fu if e >= k]
            done = [e for e in ev if e[0] <= k]; ev = [e for e in ev if e[0] > k]
            for _, r in done:
                if r == 'pass': res.append(1)
            while res and len(fu) < max_fund: res.pop(0); go_funded(k, k + 1)
            nacc = len(ev) + len(res) + len(fu)
            if full:
                while len(ev) + len(res) + len(fu) < max_acc and cash >= fee: buy(k)
            else:
                for _ in range(got * per_payout):
                    if len(ev) + len(res) + len(fu) < max_acc and cash >= fee: buy(k)
                if not (ev or res or fu) and not pq: buy(k); resc += 1
                if len(ev) + len(res) + len(fu) >= max_acc: full = True; full_k = k
        months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
        N_ = pd.Series({m: P.get(m, 0.0) - Fe.get(m, 0.0) for m in months}); C_ = pd.Series({m: NP.get(m, 0) for m in months})
        return N_, C_, oop, full_k, resc, P
    starts = [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2018-02', '2024-06', freq='M')]
    print(f'\n=== {TIER}: UNCAPPED vs LUCID CONTRACT CAPS (eval cap {cap_eval(A)} MNQ assumed; funded by account profit) | {len(starts)} monthly starts, room rule, cost 0.5 pt ===')
    print('  concurrency approximated by engine windows (RA / V15a to 12:00, M1 / R1 / RULES to close, overnight legs together); trades over the cap are cut to the room left or skipped')
    print('  plan | year-1 net | year-2 net / month | payouts / month yr2 | months with 0 payouts (yr2) | out-of-pocket', flush=True)
    BTH = 0; FLAT = None; CAP = True; REN = float(os.environ.get('RENEW', 100 if TIER == '50K' else 244))
    print(f'ABANDON RULE (Nov 10 2026, declared before results): at each 21-trading-day mark an unfinished eval is renewed only if its balance is at or above the threshold, otherwise it is dropped and counts as failed (no more fees). RENEWAL ASSUMPTION: first month ${fee}, each further 21-trading-day month of an unfinished eval costs ${REN:.0f} (charged when due; a blown or passed eval stops paying)')
    k22, k25 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))); EVC = {}
    def ev1(s):
        if s not in EVC: EVC[s] = EV(LE_, A, s, NS, E2, True)
        return EVC[s]
    ABX = None
    POL = tuple((f'B=4000, {"renew always" if x is None else "renew only if bal >= $%d" % x}', 4000, x) for x in (None, 0, 500, 1000, 1500, 2000))
    print(f'##### {TIER}: FULL STATS WITH EVAL RENEWALS, Lucid caps, room rule, cost 0.5, fee ${fee} #####')
    print('--- ONE ACCOUNT (eval started every 2nd day; funded = F0) ---')
    for nm, B_, ABX in POL:
        BTH = B_; EVC.clear()
        for pl, S_ in (('2018-21 starts', range(0, k22, 2)), ('2022-25 starts', range(k22, k25, 2))):
            R_ = []
            for s in S_:
                r, k2 = ev1(s)
                if r != 'pass': R_.append(None); continue
                fr, k3, pays = FU(LF_, A, k2 + 1, NS, F0, True); R_.append((k2 - s + 1, fr, k3 - k2, pays, k2))
            n = len(R_); P_ = [x for x in R_ if x]; allp = [p for x in P_ for _, p in x[3]]; fund = P_
            np_ = np.array([len(x[3]) for x in P_]) if P_ else np.array([0]); gaps = [b[0] - a[0] for x in P_ for a, b in zip(x[3][:-1], x[3][1:])]
            cost_ev = sum(fee + REN * max(0, int(np.ceil(max(ev1(s)[1] - s + 1, 1) / 21)) - 1) for s in S_); net_ev = (sum(allp) - cost_ev) / n; avg_ren = (cost_ev - fee * n) / n
            print(f'  {nm:<18} {pl}: pass {len(P_) / n:.0%} | funded paid >=1: {np.mean(np_ >= 1):.0%} of funded (all evals {sum(np_ >= 1) / n:.0%}) | >=2: {np.mean(np_ >= 2):.0%} | >=3: {np.mean(np_ >= 3):.0%} | payouts per funded {np_.mean():.2f} per eval {np_.sum() / n:.2f} | avg payout ${np.mean(allp):,.0f} | net per eval after renewals ${net_ev:,.0f} (renewal fees ${avg_ren:,.0f}/eval) | blown before 1st payout {np.mean([(x[1] == "lost") and not x[3] for x in P_]):.0%} | blown ever {np.mean([x[1] == "lost" for x in P_]):.0%} | days: pass {np.median([x[0] for x in P_]):.0f}, funded->1st {np.median([x[3][0][0] - x[4] for x in P_ if x[3]]):.0f}, between {np.median(gaps) if gaps else float("nan"):.0f}', flush=True)
    print('--- 10-ACCOUNT FLEET (3 per payout, 5 funded, held reserve), 77 monthly starts Feb 2018 - Jun 2024 ---')
    for nm, B_, ABX in POL:
        BTH = B_; EVC.clear(); Z = [sim(k0, per_payout=3) for k0 in starts]; ok = [z for z in Z if len(z[0]) >= 24]; st_ = [k for k, z in zip(starts, Z) if len(z[0]) >= 24]
        y1 = np.array([z[0].iloc[:12].sum() for z in Z]); y2 = np.array([z[0].iloc[12:24].mean() for z in ok]); cn = np.concatenate([z[1].iloc[12:24].values for z in ok]); mn = np.concatenate([z[0].iloc[12:24].values for z in ok])
        early = np.array([D_[k] < pd.Timestamp('2022-01-01') for k in st_])
        print(f'  {nm:<18} year-1 net: mean ${y1.mean():,.0f}, median ${np.median(y1):,.0f}, worst start ${y1.min():,.0f} | out-of-pocket ${np.mean([z[2] for z in Z]):,.0f} | months to a full fleet {np.nanmean([(z[3] - k0) / 21 for z, k0 in zip(Z, starts) if z[3] is not None]):.1f} | rescue re-buys {np.mean([z[4] for z in Z]):.1f}')
        print(f'  {"":<18} year-2 net/month: mean ${y2.mean():,.0f}, median ${np.median(y2):,.0f}, p10 ${np.percentile(y2, 10):,.0f}, p90 ${np.percentile(y2, 90):,.0f} | starts 2018-21 ${y2[early].mean():,.0f} vs 2022-24 ${y2[~early].mean():,.0f}')
        print(f'  {"":<18} year-2 monthly: net p10 ${np.percentile(mn, 10):,.0f} / p50 ${np.percentile(mn, 50):,.0f} / p90 ${np.percentile(mn, 90):,.0f}, worst ${mn.min():,.0f}, negative months {np.mean(mn < 0):.0%} | payouts/month {cn.mean():.2f}: 0 {np.mean(cn == 0):.0%}, 1 {np.mean(cn == 1):.0%}, 2 {np.mean(cn == 2):.0%}, 3 {np.mean(cn == 3):.0%}, 4+ {np.mean(cn >= 4):.0%}', flush=True)
