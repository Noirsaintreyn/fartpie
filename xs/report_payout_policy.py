"""FLEET OF 10 EVALS / 5 FUNDED WITH A HELD RESERVE (Nov 10 2026, declared before results): up to 10 accounts held (evals + passed-but-waiting), at most 5 trading funded at once, passed accounts beyond 5 wait in a reserve queue and go funded when a slot opens (ASSUMES Lucid lets a passed account wait unactivated - check). Buy 2 evals per payout up to 10 total, then replace whenever an account fails or goes funded-dead. Compare with the 5-slot plan. RAMP-UP POLICY (Nov 10 2026, declared before results): start with ONE Lucid 50K eval, after each payout buy 2 more (up to 5 slots), once 5 are full buy a replacement whenever one falls out. Room rule, cost 0.5 pt. Copied setup from report_room_cost05.py. Earlier: Nov 10 2026: COST 0.5 PT INSTEAD OF 1.0 (+0.5 pt added back on every trade taken) and a gentler overnight-range variant (wide x1.25 only), same report as xs/report_t4_e2.py. Earlier: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
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
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']; E0, E2, F0 = EP['E0 x1'], EP['E2 harder de-risk'], FP['F0 x1']
    LOCK['on'] = False; import os; TIER = os.environ.get('TIER', '50K'); A = ACCX[TIER]; fee = A['fee']; E2, F0 = EP['E2 harder de-risk'], FP['F0 x1']
    LE_, LF_ = lists(base_c, 'e'), lists(base_c, 'f'); TRJ = {}
    def traj(k):
        if k not in TRJ:
            r, k2 = eval_p(LE_, A, k, NS, E2)
            if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
            else: fr, k3, pays = fund_p(LF_, A, k2 + 1, NS, F0); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
        return TRJ[k]
    def sim(k0, per_payout=2, max_acc=10, max_fund=5, stagger=0):
        cash = 0.0; oop = 0.0; ev = []; res = []; fu = []; pq = []; P = {}; Fe = {}; NP = {}; full_k = None; resc = 0; full = False; pend = []
        def buy(k):
            nonlocal cash, oop
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            r, k2 = eval_p(LE_, A, k, NS, E2); ev.append((k2 if r in ('pass', 'fail') else NS, r)); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
        def go_funded(k, st):
            fr, k3, pays = fund_p(LF_, A, st, NS, F0); fu.append(NS if fr == 'open' else k3)
            for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        buy(k0)
        for k in range(k0 + 1, NS):
            mo = D_[k].to_period('M'); got = 0
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
    # ---- PAYOUT-REQUEST POLICY SWEEP (Nov 10 2026, declared before results). Rey's target: ~4 payouts a month at ~$2k each.
    # Why accounts die: after a payout the floor is fixed at +100 and the balance drops by the payout, so cushion = balance - 100 is often well under the $2,000 MLL. Waiting until the balance is B before asking (payout = min(.5 b, cap), or flat $F) leaves a bigger cushion, so the funded account should live for more payouts.
    # Policies: base (ask as soon as 5 qualifying days and >= $500) | B = 2500 / 3000 / 4000 (min balance to ask) | flat $2,000 at B = 4000. Everything else = report_fleet10 (room rule, caps NOT applied here, cost 0.5, 77 monthly starts, 10 accounts / 5 funded / 3 per payout).
    import functools
    def make_fund(B, flat=None):
        def fp(L, A_, s, end, pol):
            st = dict(bal=0.0, floor=-A_['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
            for k in range(s, end):
                cu = (st['bal'] - st['floor']) / A_['mll']
                pnl, br = run_day_x(L, k, st, A_['mll'], 100.0, A_['scale'], pol(cu, len(pays)), 1000)
                if br: return 'lost', k, pays
                st['bal'] += pnl; since += pnl; wins += pnl >= A_['qday']; b = st['bal']
                if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A_['mll']), 100.0)
                if wins >= 5 and b > 0 and (not pays or since > 0) and b >= B:
                    amt = min(.5 * b, A_['cap'], flat if flat else 1e9)
                    if amt >= 500:
                        pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                        if len(pays) >= 5: return 'graduated', k, pays
            return 'open', end, pays
        return fp
    starts = [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2018-02', '2024-06', freq='M')]
    print(f'\n=== {TIER}: PAYOUT-REQUEST POLICY, {len(starts)} monthly starts, year-2 steady state ===')
    for lab, B, fl in (('ask asap (base)', 0, None), ('wait for B=2500', 2500, None), ('wait for B=3000', 3000, None), ('wait for B=4000', 4000, None), ('B=4000, flat $2,000', 4000, 2000), ('B=5000', 5000, None)):
        fund_p = make_fund(B, fl); Z = []
        for k0 in starts:
            N_, C_, oop, fk, resc, P = sim(k0, per_payout=3)
            Z.append((N_.iloc[:12].sum(), N_.iloc[12:24].mean() if len(N_) >= 24 else np.nan, C_.iloc[12:24].mean() if len(C_) >= 24 else np.nan, np.mean(C_.iloc[12:24] == 0) if len(C_) >= 24 else np.nan, sum(P.values()) / max(sum(C_.values), 1) if False else np.nan))
        Z = np.array(Z, float)
        print(f'  {lab:<22} year-1 ${np.nanmean(Z[:, 0]):>7,.0f} | yr2 ${np.nanmean(Z[:, 1]):>6,.0f}/mo | payouts/mo {np.nanmean(Z[:, 2]):.2f} | months with 0 payouts {np.nanmean(Z[:, 3]):.0%}', flush=True)
