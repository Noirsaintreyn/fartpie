"""Nov 9 2026: V2 vs V2 + SKIP SLOW FLIPS (Rule A and V15a; 30-min move into the flip < 0.225 ADR) - full Lucid Flex 50K report incl. HOW OFTEN PAYOUTS COME.
Same sims as xs/flex50.py / xs/flip_char2.py. Per eval: an eval started every 2nd trading day Feb 2018 - Jun 2025, run until fail / blow / 5th payout.
Timing: trading days start -> pass, start -> 1st payout, between payouts on a funded account; payouts per funded month.
One account at a time from Feb 2018 (next eval the day after one ends). 5-account fleet from Feb 2018 (buy an eval when < 5 active and the trailing 20-day P&L > 0,
40 fees of starting cash, payouts recycled): payouts a month, months with 0 / >= 2 payouts, $ a month after fees, worst month, by year; and the 24-start fleet (2022-23)."""
_fc = open('/home/claude/fartpie/xs/flip_char.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
if __name__ == '__main__':
    V = run_sig2('s_vector'); V = V[(V.day >= '2018-02-01')].copy(); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0) & (RAs.day >= '2018-02-01')].copy()
    REF = V.join(feats(V)); cut = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30.quantile(1 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']; base = variant(True, relm)
    def modify(keys):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base):
            T_ = T_.copy()
            if en in keys: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
            out.append((T_, u, yz))
        return out
    for lab, sp in (('V2 as is', base), ('V2 + skip slow flips (Rule A + V15a)', modify({'RA', 'V15a'}))):
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_x(LE_, A, k, NS, 1000)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
                else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
            return TRJ[k]
        ST = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)); tr_ = [(s, traj(s)) for s in ST]
        pas = np.mean([t[0] == 'pass' for _, t in tr_]); paid = np.mean([len(t[2]) > 0 for _, t in tr_]); ppe = np.mean([len(t[2]) for _, t in tr_])
        allp = [x for _, t in tr_ for _, x in t[2]]; usd = np.mean([sum(x for _, x in t[2]) for _, t in tr_])
        d_pass = [t[3] - s + 1 for s, t in tr_ if t[0] == 'pass']; d_first = [t[2][0][0] - s + 1 for s, t in tr_ if t[2]]
        gaps = [b[0] - a[0] for _, t in tr_ for a, b in zip(t[2][:-1], t[2][1:])]
        fdays = sum(t[1] - t[3] for _, t in tr_ if t[0] == 'pass'); fpay = sum(len(t[2]) for _, t in tr_ if t[0] == 'pass')
        print(f'\n=== {lab} ===', flush=True)
        print(f'  per eval: pass {pas:.0%} | paid at least once {paid:.0%} ({paid / pas:.0%} of funded) | payouts per eval {ppe:.2f} = {1 / ppe:.2f} evals per payout | '
              f'avg payout ${np.mean(allp):,.0f} | paid per eval ${usd:,.0f} (${usd - fee:,.0f} after the fee)', flush=True)
        print(f'  timing (trading days): start -> pass median {np.median(d_pass):.0f} | start -> 1st payout median {np.median(d_first):.0f} (25-75%: {np.percentile(d_first, 25):.0f}-{np.percentile(d_first, 75):.0f})'
              f' | between payouts median {np.median(gaps):.0f} (25-75%: {np.percentile(gaps, 25):.0f}-{np.percentile(gaps, 75):.0f}) | payouts per funded month {fpay / fdays * 21:.2f}', flush=True)
        s = 0; chain = []
        while s < NS - 1:
            r, end, pays, _ = traj(s); chain.append((s, pays)); s = end + 1
        npay = sum(len(p_) for _, p_ in chain); run = mx = 0
        for _, p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
        months = pd.period_range(D_[0].to_period('M'), D_[NS - 2].to_period('M'), freq='M'); PC = pd.Series(0, index=months)
        for _, p_ in chain:
            for k_, x in p_: PC[D_[k_].to_period('M')] += 1
        print(f'  one account at a time (Feb 2018 - Jan 2026): {len(chain)} evals, {npay} payouts = {len(chain) / max(npay, 1):.2f} evals per payout, longest run without a payout {mx} evals, '
              f'months with a payout {np.mean(PC > 0):.0%}, payouts a month {PC.mean():.2f}', flush=True)
        hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
        cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; NP = {}
        for k in range(0, NS):
            mo = D_[k].to_period('M')
            while pq and pq[0][0] == k:
                kk_, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
            act = [e for e in act if e >= k]
            if len(act) < 5 and cash >= fee and hot[k]:
                r, end, pays, _ = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                for kk_, x in pays: heapq.heappush(pq, (kk_, x))
        N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C = pd.Series({mo: NP.get(mo, 0) for mo in months})
        yrs = ' '.join(f'{y}: {C[C.index.year == y].mean():.1f}/mo ${N[N.index.year == y].mean() / 1000:.1f}k' for y in sorted(set(months.year)) if y < 2026)
        print(f'  5-account fleet from Feb 2018: payouts a month {C.mean():.2f} | months with 0 payouts {np.mean(C == 0):.0%}, with >= 2 {np.mean(C >= 2):.0%} | '
              f'net ${N.mean():,.0f}/mo, worst month ${N.min():,.0f}, losing months {np.mean(N < 0):.0%}', flush=True)
        print(f'    by year: {yrs}', flush=True)
        fl = multi_x({'x': sp}, '50K')
        print(f'  fleet, 24 monthly starts 2022-23: avg ${fl["avg"]:,.0f} / median ${fl["med"]:,.0f} a month, >= $2.5k {fl["ge25"]:.0%}, losing {fl["lose"]:.0%}, dry {fl["dry"]:.1f} mo', flush=True)
