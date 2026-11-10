"""Nov 10 2026: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
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
    base_ow = [(_ow(T_, en), u, yz) for en, (T_, u, yz) in zip(NAMES, base)]
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']; E0, E2, F0 = EP['E0 x1'], EP['E2 harder de-risk'], FP['F0 x1']
    k22, k25 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), int(np.searchsorted(D_, pd.Timestamp('2025-07-01')))
    for lab, sp, ep in (('CURRENT SYSTEM', base, E0), ('+ ROOM RULE (eval size x (room / $2,000)^2, min 0.3)', base, E2), ('+ ROOM RULE + OVERNIGHT-RANGE SIZING (daytime engines x1.5 wide / x0.5 narrow Globex range)', base_ow, E2)):
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_p(LE_, A, k, NS, ep)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
                else: fr, k3, pays = fund_p(LF_, A, k2 + 1, NS, F0); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
            return TRJ[k]
        print(f'\n=== {lab} ===', flush=True)
        for pl, S_ in (('2018-21 starts', range(0, k22, 2)), ('2022-25 starts', range(k22, k25, 2))):
            tr_ = [(s, traj(s)) for s in S_]; pas = np.mean([t[0] == 'pass' for _, t in tr_]); paid = np.mean([len(t[2]) > 0 for _, t in tr_]); ppe = np.mean([len(t[2]) for _, t in tr_])
            allp = [x for _, t in tr_ for _, x in t[2]]; usd = np.mean([sum(x for _, x in t[2]) for _, t in tr_]) - fee
            d_pass = [t[3] - s + 1 for s, t in tr_ if t[0] == 'pass']; d_fp = [t[2][0][0] - t[3] for s, t in tr_ if t[2]]; d_first = [t[2][0][0] - s + 1 for s, t in tr_ if t[2]]
            gaps = [b[0] - a[0] for _, t in tr_ for a, b in zip(t[2][:-1], t[2][1:])]
            print(f'  {pl}: pass {pas:.0%} | paid at least once {paid:.0%} ({paid / pas:.0%} of funded) | payouts per eval {ppe:.2f} ({1 / ppe:.2f} evals per payout) | avg payout ${np.mean(allp):,.0f} | '
                  f'net ${usd:,.0f} per eval', flush=True)
            print(f'      trading days (median, middle half): start -> pass {np.median(d_pass):.0f} ({np.percentile(d_pass, 25):.0f}-{np.percentile(d_pass, 75):.0f}) | FUNDED -> FIRST PAYOUT '
                  f'{np.median(d_fp):.0f} ({np.percentile(d_fp, 25):.0f}-{np.percentile(d_fp, 75):.0f}) | start -> first payout {np.median(d_first):.0f} | between payouts {np.median(gaps):.0f} '
                  f'({np.percentile(gaps, 25):.0f}-{np.percentile(gaps, 75):.0f})', flush=True)
        s = 0; chain = []
        while s < NS - 1:
            r, end, pays, _ = traj(s); chain.append(pays); s = end + 1
        run_ = mx = 0
        for p_ in chain: run_ = run_ + 1 if not p_ else 0; mx = max(mx, run_)
        print(f'  one account at a time 2018-26: {len(chain)} evals, {sum(len(p_) for p_ in chain)} payouts, longest run without a payout {mx} evals', flush=True)
        allv = []
        for k0 in [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2022-01', '2023-12', freq='M')] + [0]:
            cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; NP = {}
            for k in range(k0, NS):
                mo = D_[k].to_period('M')
                while pq and pq[0][0] == k:
                    kk_, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
                act = [e for e in act if e >= k]
                if len(act) < 5 and cash >= fee and hot[k]:
                    r, end, pays, _ = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                    for kk_, x in pays: heapq.heappush(pq, (kk_, x))
            months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M'); N_ = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months})
            C_ = pd.Series({mo: NP.get(mo, 0) for mo in months})
            if k0 == 0: N18, C18 = N_, C_
            else: allv.append((N_.values, C_.values))
        av = np.concatenate([a for a, _ in allv]); ac = np.concatenate([b for _, b in allv])
        print(f'  5-account fleet, 24 starts 2022-23: ${av.mean():,.0f} avg / ${np.median(av):,.0f} median a month, losing {np.mean(av < 0):.0%} | {ac.mean():.2f} payouts a month, '
              f'0 in {np.mean(ac == 0):.0%}, >= 2 in {np.mean(ac >= 2):.0%} of months', flush=True)
        print(f'  5-account fleet from 2018: ${N18.mean():,.0f}/mo, losing {np.mean(N18 < 0):.0%}, worst ${N18.min():,.0f} | {C18.mean():.2f} payouts a month, 0 in {np.mean(C18 == 0):.0%}, '
              f'>= 2 in {np.mean(C18 >= 2):.0%} of months', flush=True)
        print('    by year: ' + ' | '.join(f'{y}: {C18[C18.index.year == y].mean():.1f}/mo ${N18[N18.index.year == y].mean() / 1000:.1f}k' for y in range(2018, 2026)), flush=True)
