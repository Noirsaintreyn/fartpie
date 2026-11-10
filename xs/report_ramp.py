"""RAMP-UP POLICY (Nov 10 2026, declared before results): start with ONE Lucid 50K eval, after each payout buy 2 more (up to 5 slots), once 5 are full buy a replacement whenever one falls out. Room rule, cost 0.5 pt. Copied setup from report_room_cost05.py. Earlier: Nov 10 2026: COST 0.5 PT INSTEAD OF 1.0 (+0.5 pt added back on every trade taken) and a gentler overnight-range variant (wide x1.25 only), same report as xs/report_t4_e2.py. Earlier: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
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
    def sim(k0, per_payout=2, slots=5):
        cash = 0.0; oop = 0.0; act = []; pq = []; full = False; P = {}; Fe = {}; NP = {}; full_k = None; resc = 0
        def buy(k):
            nonlocal cash, oop
            r, end, pays, _ = traj(k)
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            act.append(end); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
            for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        buy(k0)
        for k in range(k0 + 1, NS):
            mo = D_[k].to_period('M'); got = 0
            while pq and pq[0][0] == k:
                kk_, x = heapq.heappop(pq); cash += x; got += 1; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
            act = [e for e in act if e >= k]
            if full:
                while len(act) < slots and cash >= fee: buy(k)
            else:
                for _ in range(got * per_payout):
                    if len(act) < slots and cash >= fee: buy(k)
                if not act and not pq: buy(k); resc += 1                      # everything died with no payout coming: rebuy 1 out of pocket
                if len(act) >= slots: full = True; full_k = k
        months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
        N_ = pd.Series({m: P.get(m, 0.0) - Fe.get(m, 0.0) for m in months}); C_ = pd.Series({m: NP.get(m, 0) for m in months})
        return N_, C_, oop, full_k, resc, P
    starts = [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2018-02', '2024-06', freq='M')]
    R = []
    for k0 in starts:
        N_, C_, oop, fk, resc, P = sim(k0); first = next((i for i, m in enumerate(N_.index) if P.get(m, 0) > 0), None)
        R.append(dict(start=D_[k0], oop=oop, full_mo=None if fk is None else (fk - k0) / 21, resc=resc, first_mo=None if first is None else first,
                      y1=N_.iloc[:12].sum(), y2=N_.iloc[12:24].mean() if len(N_) >= 24 else np.nan, m13_24=C_.iloc[12:24].mean() if len(C_) >= 24 else np.nan, tot=N_.sum(), months=len(N_)))
    R = pd.DataFrame(R); print(f'\n=== {TIER}: ONE ACCOUNT FIRST, +2 PER PAYOUT, FULL AT 5, THEN REPLACE EVERY DROP ({len(R)} monthly starts Feb 2018 - Jun 2024, room rule, cost 0.5 pt) ===')
    print(f'  reached 5 slots: {R.full_mo.notna().mean():.0%} of starts; months to fill (median / 25-75%): {R.full_mo.median():.1f} ({R.full_mo.quantile(.25):.1f}-{R.full_mo.quantile(.75):.1f})')
    print(f'  first payout (months after start): median {R.first_mo.median():.0f}, 90th pct {R.first_mo.quantile(.9):.0f}; no payout in first 6 months: {np.mean(R.first_mo.isna() | (R.first_mo >= 6)):.0%}')
    print(f'  out-of-pocket fees ($88 each): average ${R.oop.mean():,.0f}, median ${R.oop.median():,.0f}, worst ${R.oop.max():,.0f}; starts needing a rescue re-buy: {np.mean(R.resc > 0):.0%}')
    print(f'  net in year 1: average ${R.y1.mean():,.0f}, median ${R.y1.median():,.0f}, negative in {np.mean(R.y1 < 0):.0%} of starts')
    print(f'  net per month in year 2: average ${R.y2.mean():,.0f}, median ${R.y2.median():,.0f}; payouts a month in year 2: {R.m13_24.mean():.2f}')
    print(f'  net per month over the whole run: average ${(R.tot / R.months).mean():,.0f}')
    by = R.groupby(R.start.dt.year).agg(n=('oop', 'size'), y1=('y1', 'mean'), y2=('y2', 'mean'), pay=('m13_24', 'mean'))
    print('  by start year: ' + ' | '.join(f'{y}: yr1 ${r.y1:,.0f}, yr2 ${r.y2 * 12 if r.y2 == r.y2 else float("nan"):,.0f}/yr' for y, r in by.iterrows()))
    for lab, ps in (('+3 per payout', 3), ('+1 per payout', 1)):
        Z = [sim(k0, per_payout=ps) for k0 in starts]; yy = np.array([z[0].iloc[12:24].mean() for z in Z if len(z[0]) >= 24]); fm = [(z[3] - k0) / 21 for z, k0 in zip(Z, starts) if z[3] is not None]
        print(f'  {lab}: year-2 net / month ${yy.mean():,.0f}, filled 5 in {len(fm) / len(Z):.0%} of starts, median {np.median(fm):.1f} months')
