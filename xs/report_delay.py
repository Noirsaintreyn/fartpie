"""DELAYED ENTRY ON BAND CONFIRMATION (RA) + ONE-ACCOUNT STATS, 50K, Lucid caps. Rule (declared before results): bad states skipped (RA, V15a); RA flips that start INSIDE the noise area are not taken at the flip but at the open after the first 5m close beyond the band in the trade direction, up to the 11:30 bar, then the normal RA exit (stop 50 / target 75 / flat 12:00); none by 11:30 = no trade. V15a unchanged (its exit rules are not reproduced here). ORB / NOISE-BAND STATE SIZING ON THE VECTOR ENGINES (RA, V15a) - declared before results, rule fixed from the early period (RA 2018-21, V15a 2014-21): skip flips that enter with price already beyond the opposite OR (15m) or the opposite noise band (both were worse in the early period too); good = beyond the noise band or beyond the OR in the trade direction. Lucid contract caps applied. CAPPED SIM - Lucid contract limits applied. FLEET OF 10 EVALS / 5 FUNDED WITH A HELD RESERVE (Nov 10 2026, declared before results): up to 10 accounts held (evals + passed-but-waiting), at most 5 trading funded at once, passed accounts beyond 5 wait in a reserve queue and go funded when a slot opens (ASSUMES Lucid lets a passed account wait unactivated - check). Buy 2 evals per payout up to 10 total, then replace whenever an account fails or goes funded-dead. Compare with the 5-slot plan. RAMP-UP POLICY (Nov 10 2026, declared before results): start with ONE Lucid 50K eval, after each payout buy 2 more (up to 5 slots), once 5 are full buy a replacement whenever one falls out. Room rule, cost 0.5 pt. Copied setup from report_room_cost05.py. Earlier: Nov 10 2026: COST 0.5 PT INSTEAD OF 1.0 (+0.5 pt added back on every trade taken) and a gentler overnight-range variant (wide x1.25 only), same report as xs/report_t4_e2.py. Earlier: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
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
        return 'open', end
    def FU(L, A, s, end, pol, capped):
        st = dict(bal=0.0, floor=-A['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
        for k in range(s, end):
            cu = (st['bal'] - st['floor']) / A['mll']
            pnl, br = run_day_c(L, k, st, A['mll'], A['scale'], pol(cu, len(pays)), cap_fund(A, st['bal']) if capped else None)
            if br: return 'lost', k, pays
            st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
            if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
            if wins >= 5 and b > 0 and (not pays or since > 0):
                amt = min(.5 * b, A['cap'])
                if amt >= 500:
                    pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                    if len(pays) >= 5: return 'graduated', k, pays
        return 'open', end, pays
    def sim(k0, per_payout=2, max_acc=10, max_fund=5, stagger=0):
        global CAP
        cash = 0.0; oop = 0.0; ev = []; res = []; fu = []; pq = []; P = {}; Fe = {}; NP = {}; full_k = None; resc = 0; full = False; pend = []
        def buy(k):
            nonlocal cash, oop
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            r, k2 = EV(LE_, A, k, NS, E2, CAP); ev.append((k2 if r in ('pass', 'fail') else NS, r)); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
        def go_funded(k, st):
            fr, k3, pays = FU(LF_, A, st, NS, F0, CAP); fu.append(NS if fr == 'open' else k3)
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
    _o = open('xs/orb_noise_vector.py').read(); NS_ = {}; exec(_o[:_o.index("OUT, NAMES = pickle.load")], NS_); Gd, SIGd, Od, PCd, stf = NS_['G'], NS_['SIG'], NS_['O'], NS_['PC'], NS_['state']
    XS = pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/orb_noise_vector.pkl')
    ST = {(r.eng, r.day, r.m): ('bad' if (r.orb == 'against' or r.nb == 'opposite') else ('good' if (r.nb == 'beyond' or r.orb == 'with') else 'mid')) for r in XS.itertuples()}
    DIRS = {(r.eng, r.day, r.m): r.dir for r in XS.itertuples()}
    def ra_path(day, m0, dr, cost):
        g = Gd[day]; ep = g.o.get(m0, np.nan)
        if not np.isfinite(ep): return None
        worst = 0.0
        for mm, b in g[(g.index >= m0) & (g.index < 720)].iterrows():
            lo, hi = (b.l - ep) * dr, (b.h - ep) * dr
            if dr < 0: lo, hi = (ep - b.h), (ep - b.l)
            if lo <= -50: return -50.0 - cost, min(worst, -50.0) - cost
            if hi >= 75: return 75.0 - cost, worst - cost
            worst = min(worst, lo)
        ex = g[g.index < 720]; px = ex.c.iloc[-1]
        return (px - ep) * dr - cost, worst - cost
    def delayed(day, m0, dr, cost):
        g = Gd[day]; sg = SIGd.loc[day]; o = Od[day]; pc = PCd[day] if np.isfinite(PCd[day]) else o
        for mm in range(m0, 695, 5):
            if mm not in g.index or not np.isfinite(sg.get(mm, np.nan)): continue
            ub, lb = max(o, pc) * (1 + sg[mm]), min(o, pc) * (1 - sg[mm]); c_ = g.c[mm]
            if (c_ > ub if dr > 0 else c_ < lb):
                if mm + 5 >= 720: return None
                r = ra_path(day, mm + 5, dr, cost)
                return None if r is None else (mm + 5, r[0], r[1])
        return None
    # reproduction check of the RA exit logic against the pipeline table (cost 1.0 + 0.5 added back = 0.5 here)
    _ra = dict(zip(NAMES, base_c))['RA'][0]; _ra = _ra[_ra.sz > 0]; ok_ = []
    for r in _ra.itertuples():
        dr = DIRS.get(('RA', pd.Timestamp(r.day).normalize(), int(r.m)))
        if dr is None: continue
        p = ra_path(pd.Timestamp(r.day).normalize(), int(r.m), dr, 0.5)
        if p is not None: ok_.append((p[0], r.pts))
    ok_ = np.array(ok_); print(f'  RA exit logic vs pipeline table: {len(ok_)} trades, exact (within 0.6 pt) {np.mean(np.abs(ok_[:, 0] - ok_[:, 1]) < 0.6):.0%}, mean sim {ok_[:, 0].mean():+.2f} vs table {ok_[:, 1].mean():+.2f}', flush=True)
    def mk(skip_bad, delay):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base_c):
            T_ = T_.copy()
            if en in ('RA', 'V15a') and (skip_bad or delay):
                keys = [(en, pd.Timestamp(dd).normalize(), int(mm_)) for dd, mm_ in zip(T_.day, T_.m)]; stt = [ST.get(k, 'mid') for k in keys]
                f = np.ones(len(T_)); mnew = T_.m.values.copy(); pnew = T_.pts.values.copy(); anew = T_.mae.values.copy()
                for j, (k, x) in enumerate(zip(keys, stt)):
                    if skip_bad and x == 'bad': f[j] = 0.0
                    elif delay and en == 'RA' and x == 'mid' and T_.sz.values[j] > 0:
                        r = delayed(k[1], k[2], DIRS[k], 0.5) if k in DIRS else None
                        if r is None: f[j] = 0.0
                        else: mnew[j], pnew[j], anew[j] = r
                T_['sz'] = T_.sz.values * f; T_['m'] = mnew; T_['pts'] = pnew; T_['mae'] = anew
            out.append((T_, u, yz))
        return out
    def sim(k0, per_payout=2, max_acc=10, max_fund=5, stagger=0):
        global CAP
        cash = 0.0; oop = 0.0; ev = []; res = []; fu = []; pq = []; P = {}; Fe = {}; NP = {}; full_k = None; resc = 0; full = False; pend = []
        def buy(k):
            nonlocal cash, oop
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            r, k2 = EV(LE_, A, k, NS, E2, CAP); ev.append((k2 if r in ('pass', 'fail') else NS, r)); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
        def go_funded(k, st):
            fr, k3, pays = FU(LF_, A, st, NS, F0, CAP); fu.append(NS if fr == 'open' else k3)
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
    XS = pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/orb_noise_vector.pkl')
    ST = {(r.eng, r.day, r.m): ('bad' if (r.orb == 'against' or r.nb == 'opposite') else ('good' if (r.nb == 'beyond' or r.orb == 'with') else 'mid')) for r in XS.itertuples()}
    def mk(skip_bad, good_mult):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base_c):
            T_ = T_.copy()
            if en in ('RA', 'V15a'):
                stt = [ST.get((en, pd.Timestamp(dd).normalize(), int(mm_)), 'mid') for dd, mm_ in zip(T_.day, T_.m)]
                f = np.array([0.0 if (x == 'bad' and skip_bad) else (good_mult if x == 'good' else 1.0) for x in stt]); T_['sz'] = T_.sz.values * f
            out.append((T_, u, yz))
        return out
    for en in ('RA', 'V15a'):
        t = dict(zip(NAMES, base_c))[en][0]; t = t[t.sz > 0]; stt = pd.Series([ST.get((en, pd.Timestamp(dd).normalize(), int(mm_)), 'none') for dd, mm_ in zip(t.day, t.m)], index=t.index)
        print(f'  {en} trades in the pipeline: ' + ' | '.join(f'{k}: n {int((stt == k).sum())}, {t.pts[stt == k].mean():+.1f} pts' for k in ('good', 'mid', 'bad', 'none')))
    def sim(k0, per_payout=2, max_acc=10, max_fund=5, stagger=0):
        global CAP
        cash = 0.0; oop = 0.0; ev = []; res = []; fu = []; pq = []; P = {}; Fe = {}; NP = {}; full_k = None; resc = 0; full = False; pend = []
        def buy(k):
            nonlocal cash, oop
            if cash >= fee: cash -= fee
            else: oop += fee - cash; cash = 0.0
            r, k2 = EV(LE_, A, k, NS, E2, CAP); ev.append((k2 if r in ('pass', 'fail') else NS, r)); mo = D_[k].to_period('M'); Fe[mo] = Fe.get(mo, 0) + fee
        def go_funded(k, st):
            fr, k3, pays = FU(LF_, A, st, NS, F0, CAP); fu.append(NS if fr == 'open' else k3)
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
    k22, k25 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), int(np.searchsorted(D_, pd.Timestamp('2025-07-01')))
    def one_acct(LE_, LF_):
        TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = EV(LE_, A, k, NS, E2, True)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
                else: fr, k3, pays = FU(LF_, A, k2 + 1, NS, F0, True); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
            return TRJ[k]
        res = []
        for pl, S_ in (('2018-21 starts', range(0, k22, 2)), ('2022-25 starts', range(k22, k25, 2))):
            tr_ = [(s, traj(s)) for s in S_]; pas = np.mean([t[0] == 'pass' for _, t in tr_]); paid = np.mean([len(t[2]) > 0 for _, t in tr_]); ppe = np.mean([len(t[2]) for _, t in tr_])
            allp = [x for _, t in tr_ for _, x in t[2]]; net = np.mean([sum(x for _, x in t[2]) for _, t in tr_]) - fee
            dp = np.median([t[3] - s + 1 for s, t in tr_ if t[0] == 'pass']); dfp = np.median([t[2][0][0] - t[3] for s, t in tr_ if t[2]]); dfirst = np.median([t[2][0][0] - s + 1 for s, t in tr_ if t[2]])
            gaps = [b[0] - a[0] for _, t in tr_ for a, b in zip(t[2][:-1], t[2][1:])]
            res.append(f'    {pl}: pass {pas:.0%} | paid once {paid:.0%} | payouts/eval {ppe:.2f} ({1 / ppe:.2f} evals per payout) | avg payout ${np.mean(allp):,.0f} | net ${net:,.0f}/eval | days: start->pass {dp:.0f}, funded->1st payout {dfp:.0f}, start->1st payout {dfirst:.0f}, between {np.median(gaps):.0f}')
        s = 0; chain = []
        while s < NS - 1:
            r, end, pays, _ = traj(s); chain.append(pays); s = end + 1
        run_ = mx = 0
        for p_ in chain: run_ = run_ + 1 if not p_ else 0; mx = max(mx, run_)
        res.append(f'    one account at a time 2018-26: {len(chain)} evals, {sum(len(p_) for p_ in chain)} payouts, longest run without a payout {mx} evals')
        return res
    print(f'\n=== 50K, ONE ACCOUNT (per eval started every 2nd trading day), Lucid caps, room rule, cost 0.5 pt ===', flush=True)
    SPS = {}
    for nm, (sk, dl) in (('A. base (no state)', (False, False)), ('B. skip bad states', (True, False)), ('C. B + delayed entry on band confirmation (RA)', (True, True))):
        sp = mk(sk, dl); SPS[nm] = sp; LE_, LF_ = lists_cap(sp, 'e'), lists_cap(sp, 'f'); print(f'  {nm}'); print('\n'.join(one_acct(LE_, LF_)), flush=True)
    print('\n=== SIDES: trades taken (sz > 0), count and avg pts by direction ===')
    for nm, sp in SPS.items():
        row = []
        for en, (T_, u, yz) in zip(NAMES, sp):
            if en not in ('RA', 'V15a'): continue
            t = T_[T_.sz > 0]; dd = [DIRS.get((en, pd.Timestamp(a).normalize(), int(b)), np.nan) for a, b in zip(t.day, t.m)] if nm.startswith(('A', 'B')) else None
            if dd is None: dd = [np.nan] * len(t)
            row.append((en, len(t)))
        print(f'  {nm}: ' + ', '.join(f'{e} {n}' for e, n in row))
    for en in ('RA', 'V15a'):
        X_ = XS[XS.eng == en]; print(f'  {en} base trades by side (states known): long {int((X_.dir > 0).sum())} / short {int((X_.dir < 0).sum())}; avg pts long {X_.pts[X_.dir > 0].mean():+.1f} short {X_.pts[X_.dir < 0].mean():+.1f}; bad-state skips: long {int(((X_.dir > 0) & ((X_.orb == "against") | (X_.nb == "opposite"))).sum())}, short {int(((X_.dir < 0) & ((X_.orb == "against") | (X_.nb == "opposite"))).sum())}')
    Xr = XS[(XS.eng == 'RA')]; mid = Xr[(Xr.nb == 'inside') & (Xr.orb != 'with') & (Xr.orb != 'against')]
    print(f'  RA flips inside the band: long {int((mid.dir > 0).sum())} / short {int((mid.dir < 0).sum())}')
