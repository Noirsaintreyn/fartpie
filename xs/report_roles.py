"""ACCOUNT ROLES (declared before results): instead of copy-trading one signal on every account, give accounts different engine mixes / sizes so deaths and payout days decorrelate. FLEET OF 10 EVALS / 5 FUNDED WITH A HELD RESERVE (Nov 10 2026, declared before results): up to 10 accounts held (evals + passed-but-waiting), at most 5 trading funded at once, passed accounts beyond 5 wait in a reserve queue and go funded when a slot opens (ASSUMES Lucid lets a passed account wait unactivated - check). Buy 2 evals per payout up to 10 total, then replace whenever an account fails or goes funded-dead. Compare with the 5-slot plan. RAMP-UP POLICY (Nov 10 2026, declared before results): start with ONE Lucid 50K eval, after each payout buy 2 more (up to 5 slots), once 5 are full buy a replacement whenever one falls out. Room rule, cost 0.5 pt. Copied setup from report_room_cost05.py. Earlier: Nov 10 2026: COST 0.5 PT INSTEAD OF 1.0 (+0.5 pt added back on every trade taken) and a gentler overnight-range variant (wide x1.25 only), same report as xs/report_t4_e2.py. Earlier: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
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
    GRP = {'ON': ('ON+', 'ON-', 'R1'), 'AM': ('RA', 'M1'), 'RL': ('RULES', 'V15a')}
    def role(keep=None, mult=1.0):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base_c):
            T_ = T_.copy(); T_['sz'] = T_.sz.values * mult * (1.0 if keep is None or en in keep else 0.0); out.append((T_, u, yz))
        return out
    PLANS = {'ALL x1 (copy-trade, old)': [(None, 1.0)], 'size ladder 0.7 / 1.0 / 1.4': [(None, .7), (None, 1.0), (None, 1.4)], 'all x1.4': [(None, 1.4)], 'all x0.7': [(None, .7)],
             'engine split: overnight | morning | rules (3 roles)': [(GRP['ON'], 1.0), (GRP['AM'], 1.0), (GRP['RL'], 1.0)], 'engine split x1.5 each': [(GRP['ON'], 1.5), (GRP['AM'], 1.5), (GRP['RL'], 1.5)],
             'mixed: all x1 / overnight+morning x1.3 / rules+morning x1.3': [(None, 1.0), (GRP['ON'] + GRP['AM'], 1.3), (GRP['RL'] + GRP['AM'], 1.3)]}
    def make(plan):
        R = []
        for keep, m in plan:
            sp = role(keep, m); LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); R.append((LE_, LF_))
        return R
    def sim(k0, RL, per_payout=3, max_acc=10, max_fund=5):
        cash = 0.0; oop = 0.0; ev = []; res = []; fu = []; pq = []; P = {}; Fe = {}; NP = {}; ctr = [0]; full = False
        def buy(k):
            nonlocal cash, oop
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            ri = ctr[0] % len(RL); ctr[0] += 1; r, k2 = eval_p(RL[ri][0], A, k, NS, E2); ev.append((k2 if r in ('pass', 'fail') else NS, r, ri)); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
        def go_funded(k, st, ri):
            fr, k3, pays = fund_p(RL[ri][1], A, st, NS, F0); fu.append(NS if fr == 'open' else k3)
            for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        buy(k0)
        for k in range(k0 + 1, NS):
            mo = D_[k].to_period('M'); got = 0
            while pq and pq[0][0] == k:
                kk_, x = heapq.heappop(pq); cash += x; got += 1; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
            fu = [e for e in fu if e >= k]; done = [e for e in ev if e[0] <= k]; ev = [e for e in ev if e[0] > k]
            for _, r, ri in done:
                if r == 'pass': res.append(ri)
            while res and len(fu) < max_fund: go_funded(k, k + 1, res.pop(0))
            if full:
                while len(ev) + len(res) + len(fu) < max_acc and cash >= fee: buy(k)
            else:
                for _ in range(got * per_payout):
                    if len(ev) + len(res) + len(fu) < max_acc and cash >= fee: buy(k)
                if not (ev or res or fu) and not pq: buy(k)
                if len(ev) + len(res) + len(fu) >= max_acc: full = True
        months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
        return pd.Series({m: P.get(m, 0.0) - Fe.get(m, 0.0) for m in months}), pd.Series({m: NP.get(m, 0) for m in months}), oop
    starts = [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2018-02', '2024-06', freq='M')]
    print(f'\n=== {TIER}: ACCOUNT ROLES IN A 10-ACCOUNT FLEET (3 per payout, 5 funded, reserve) | {len(starts)} monthly starts, room rule, cost 0.5 pt ===')
    print('  plan | year-1 net | year-2 net / month | payouts / month yr2 | months with 0 payouts (yr2) | worst yr2 month (avg) | out-of-pocket', flush=True)
    for nm, plan in PLANS.items():
        RL = make(plan); Z = [sim(k0, RL) for k0 in starts]; ok = [z for z in Z if len(z[0]) >= 24]
        y1 = np.mean([z[0].iloc[:12].sum() for z in Z]); y2 = np.mean([z[0].iloc[12:24].mean() for z in ok]); pm = np.mean([z[1].iloc[12:24].mean() for z in ok]); zero = np.mean([(z[1].iloc[12:24] == 0).mean() for z in ok]); wm = np.mean([z[0].iloc[12:24].min() for z in ok])
        print(f'  {nm:<60} | ${y1:>7,.0f} | ${y2:>6,.0f} | {pm:4.2f} | {zero:4.0%} | ${wm:>7,.0f} | ${np.mean([z[2] for z in Z]):,.0f}', flush=True)
