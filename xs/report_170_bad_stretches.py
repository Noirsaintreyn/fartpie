"""Report 170 (where the bad stretches come from; copy of 166b/169 harness): IFVG sized by EMA-trend score (tiers declared in report 166), on top of the deploy config (S1 regime-ON 0.2 ct). Header of 162: S1 premium fade (report 161 trades, MNQ-scaled 0.1-0.3 ct) added to variant I + IFVG 0.5 ct in the Lucid 50K sim. Eval 2018-21 starts show S1 inactive-regime effects; 2022-25 is the regime-on window. Header of 146: IFVG (report 144) added as an extra leg to variant I in the Lucid 50K sim (same harness as report_18b; chain replaced by IFVG; idle-days-only = IFVG trades on days the core legs do not trade). Original header: (batch 2, FVG-2R from report_18): CHAIN (ICT/ICC sweep->MSS+FVG->limit, xs/report_17_icc_chain.py) ADDED AS AN 8TH LEG at 0.5 contract (5 MNQ) to the current best (variant I), vs variant I alone. 50K one account, Lucid caps, eval and funded, same sim as report_gap. Declared before results.
NZ ROLE ACCOUNT with a tight stop and a daily loss limit (STOP / LDL env vars, defaults 25 / 40; Nov 10 2026, declared before results). Same sim as report_noise_only.
NOISE-AREA MOMENTUM ALONE ON A 50K ACCOUNT (NZ only, no S6): NOISE-AREA MOMENTUM AS A NEW ENGINE (ET clock fixed - the CSV is Central time) ADDED TO S6 + ROOM RULE, 50K one-account stats, Lucid caps. Declared before results: engine NZ = noise-area momentum (rules in xs/nq_noise_onwide.py: sigma from 14 prior sessions, decisions at HH:00/HH:30 10:00-15:30, exit when the close crosses max(UB, VWAP) / min(LB, VWAP), reverse on the opposite band, hard stop 40, flat 15:55, cost 0.5 pt), size u = 0.5 (5 MNQ at sz 1), no filters. Earlier: DELAYED ENTRY ON BAND CONFIRMATION (RA) + ONE-ACCOUNT STATS. Rule (declared before results): bad states skipped (RA, V15a); RA flips that start INSIDE the noise area are not taken at the flip but at the open after the first 5m close beyond the band in the trade direction, up to the 11:30 bar, then the normal RA exit (stop 50 / target 75 / flat 12:00); none by 11:30 = no trade. V15a unchanged (its exit rules are not reproduced here). ORB / NOISE-BAND STATE SIZING ON THE VECTOR ENGINES (RA, V15a) - declared before results, rule fixed from the early period (RA 2018-21, V15a 2014-21): skip flips that enter with price already beyond the opposite OR (15m) or the opposite noise band (both were worse in the early period too); good = beyond the noise band or beyond the OR in the trade direction. Lucid contract caps applied. CAPPED SIM - Lucid contract limits applied. FLEET OF 10 EVALS / 5 FUNDED WITH A HELD RESERVE (Nov 10 2026, declared before results): up to 10 accounts held (evals + passed-but-waiting), at most 5 trading funded at once, passed accounts beyond 5 wait in a reserve queue and go funded when a slot opens (ASSUMES Lucid lets a passed account wait unactivated - check). Buy 2 evals per payout up to 10 total, then replace whenever an account fails or goes funded-dead. Compare with the 5-slot plan. RAMP-UP POLICY (Nov 10 2026, declared before results): start with ONE Lucid 50K eval, after each payout buy 2 more (up to 5 slots), once 5 are full buy a replacement whenever one falls out. Room rule, cost 0.5 pt. Copied setup from report_room_cost05.py. Earlier: Nov 10 2026: COST 0.5 PT INSTEAD OF 1.0 (+0.5 pt added back on every trade taken) and a gentler overnight-range variant (wide x1.25 only), same report as xs/report_t4_e2.py. Earlier: SAME REPORT AS xs/report_t4_e2.py, with the Rule A trail variant replaced by OVERNIGHT-RANGE SIZING (xs/nq_edge_hunt.py flag, applied to the daytime engines RA / M1 / R1 / RULES / V15a; the overnight legs untouched). Original docstring: FULL LUCID FLEX 50K REPORT - current system vs + room rule (eval E2) vs + room rule + Rule A 3x-ATR trail (T4). Funded sizing as now (F0).
Per eval (started every 2nd trading day; 2018-21 and 2022-25 starts), timing (start -> pass, PASS -> FIRST PAYOUT on the funded account, between payouts),
one account at a time from 2018, the 5-account fleet from 2018 (payouts a month, months with 0 / >= 2, $ a month, losing, worst, by year) and 24 starts 2022-23."""
_t = open('/home/claude/fartpie/xs/trail_tiers.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
import os; STOP = float(os.environ.get('STOP', 25)); LDL = float(os.environ.get('LDL', 40))
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
    _o = open('xs/orb_noise_vector.py').read(); NS_ = {}; exec(_o[:_o.index("OUT, NAMES = pickle.load")], NS_); Gd, SIGd, Od, PCd = NS_['G'], NS_['SIG'], NS_['O'], NS_['PC']
    def noise_day(day, cost=0.5, stop=40.0):
        g = Gd[day]; sg = SIGd.loc[day]; o = Od[day]; pc = PCd[day] if np.isfinite(PCd[day]) else o; rows = []; pos = 0; e = None; em = None; worst = 0.0; pend = None
        idx = list(g.index)
        for j, t in enumerate(idx):
            b = g.loc[t]
            if pend is not None: pos, e, em, worst = pend, b.o, t, 0.0; pend = None
            if pos != 0:
                lo = (b.l - e) * pos if pos > 0 else (e - b.h); worst = min(worst, lo)
                if lo <= -stop: rows.append((day, em, pos, -stop - cost, worst - cost)); pos = 0; continue
            if j == len(idx) - 1:
                if pos != 0: rows.append((day, em, pos, (b.c - e) * pos - cost, worst - cost)); pos = 0
                break
            if (t + 5) % 30 != 0 or t + 5 < 600 or t + 5 > 930 or not np.isfinite(sg.get(t, np.nan)): continue
            ub, lb = max(o, pc) * (1 + sg[t]), min(o, pc) * (1 - sg[t]); c_ = b.c; vw = b.vw; nxt = g.o[idx[j + 1]]
            if pos == 1 and c_ < max(ub, vw): rows.append((day, em, 1, nxt - e - cost, worst - cost)); pos = 0
            elif pos == -1 and c_ > min(lb, vw): rows.append((day, em, -1, e - nxt - cost, worst - cost)); pos = 0
            if pos == 0 and c_ > ub: pend = 1
            elif pos == 0 and c_ < lb: pend = -1
        return rows
    allr = []
    for dd in sorted(Gd):
        if dd < pd.Timestamp('2014-02-01') or dd > pd.Timestamp('2026-01-30'): continue
        run_ = 0.0
        for r_ in sorted(noise_day(dd, stop=STOP), key=lambda r: r[1]):
            if run_ <= -LDL: break
            run_ += r_[3]; allr.append(r_)
    NZ = pd.DataFrame(allr, columns=['day', 'm', 'dir', 'pts', 'mae']); NZ['sz'] = 1.0; NZ['ce'] = 1.0; NZ['cf'] = 1.0
    pdp = NZ.groupby('day').pts.sum().reindex(sorted(d_ for d_ in Gd if pd.Timestamp('2014-02-01') <= d_ <= pd.Timestamp('2026-01-30'))).fillna(0)
    print(f'  NZ engine: {len(NZ)} trades; pts/day (1 contract) 2014-21 {pdp[pdp.index < "2022-01-01"].mean():+.2f} | 2022-26 {pdp[pdp.index >= "2022-01-01"].mean():+.2f}', flush=True)
    for sd, nm in ((1, 'long'), (-1, 'short')):
        z = NZ[NZ.dir == sd]; print(f'    {nm}: {len(z)} trades, avg {z.pts.mean():+.2f} pts | 2014-21 {z.pts[z.day < "2022-01-01"].mean():+.2f} | 2022-26 {z.pts[z.day >= "2022-01-01"].mean():+.2f}')
    NAMES = list(NAMES) + ['NZ', 'CH', 'S1']; END['NZ'] = 960; END['CH'] = 955; END['S1'] = 955
    S6 = list(base_c); Z0 = [(T_, 0.0, yz) for T_, u, yz in S6]; NZl = NZ[NZ.dir > 0].copy()
    NZl = NZ[NZ.dir > 0].copy()
    import sys as _s; _s.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as _Z
    def _state(d, m):
        d = pd.Timestamp(d)
        if d not in _Z.A or d not in _Z.SI: return 0
        jb = int((m - 570) // 5) - 1; o, h, l, c, v, vw = _Z.A[d][0]
        if jb < 1 or jb >= len(c) or np.isnan(c[jb]): return 0
        sg = _Z.SIG[_Z.SI[d]]
        if not np.isfinite(sg[jb]): return 0
        op = _Z.OPEN[d]; pc = _Z.PC[d] if np.isfinite(_Z.PC[d]) else op; ub, lb = max(op, pc) * (1 + sg[jb]), min(op, pc) * (1 - sg[jb])
        return 1 if c[jb] > ub else -1 if c[jb] < lb else 0
    def band_boost(spec, f_with, f_in):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-2]):
            if en in ('RA', 'V15a') and 'dir' in T_.columns:
                T_ = T_.copy(); b = np.array([_state(d, m) for d, m in zip(T_.day, T_.m)]); w = (b * T_.dir.values > 0)
                T_['sz'] = T_.sz.values * np.where(w, f_with, np.where(b == 0, f_in, 1.0))
            out.append((T_, u, yz))
        return out
    NZf = NZ.copy(); NZf['ce'] = 0.0; NZfl = NZf[NZf.dir > 0].copy()
    _gp = (_Dn.gap.abs() / _Dn.rng20.shift(1)); _gr = _gp.rolling(252, min_periods=120).rank(pct=True)
    def gapmul(spec, lo, hi):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-2]):
            if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
                T_ = T_.copy(); r = _gr.reindex(_pd.to_datetime(T_.day)).values
                T_['sz'] = T_.sz.values * np.where(r <= .33, lo, np.where(r >= .67, hi, 1.0))
            out.append((T_, u, yz))
        return out
    _X = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/s1_trades162.pkl')
    def _mk2(g): X = pd.DataFrame(dict(day=_pd.to_datetime(g.day), m=g.m.astype(float), dir=g.dir, pts=g.pts, mae=g.mae)); X['sz'] = 1.0; X['ce'] = 1.0; X['cf'] = 1.0; return X
    S1R = _mk2(_X[_X.reg > 0])
    _C = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ifvg_score166.pkl')
    def _mk(g, f): X = pd.DataFrame(dict(day=pd.to_datetime(g.d), m=g.m.astype(float), dir=g.dr, pts=g.pts, mae=g.mae)); X['sz'] = f(g.score.values) / 0.5; X['ce'] = 1.0; X['cf'] = 1.0; return X
    print('  IFVG trades with score:', len(_C))
    CH0 = _mk(_C, lambda s: np.full(len(s), .5)); CH3 = _mk(_C, lambda s: np.where(s >= 4, .75, np.where(s >= 2, .5, .25))); CH2 = _mk(_C, lambda s: np.where(s >= 3, .75, .25)); CH2b = _mk(_C, lambda s: np.where(s >= 3, 1.0, .25)); CH2c = _mk(_C, lambda s: np.where(s >= 3, .75, .5))
    _I = gapmul(S6, 0.5, 1.0) + [(NZ, 0.0, None)]
    SPS = {'DEPLOY: I + IFVG 0.5 flat + S1 regime 0.2': _I + [(CH0, 0.5, None), (S1R, 0.2, None)]}
    # daily correlation of NZ with the rest of the system
    base_daily = pd.Series(0.0, index=pdp.index)
    for en, (T_, u, yz) in zip(NAMES[:-2], S6):
        t = T_[T_.sz > 0]; base_daily = base_daily.add((t.pts * t.sz * u).groupby(pd.to_datetime(t.day).dt.normalize().values).sum().reindex(pdp.index).fillna(0), fill_value=0)
    print(f'  daily correlation of NZ pts with the S6 system pts: {np.corrcoef(pdp.values, base_daily.values)[0, 1]:+.2f}', flush=True)
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

    def chain_stats(LE_, LF_):
        TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = EV(LE_, A, k, NS, E2, True)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
                else: fr, k3, pays = FU(LF_, A, k2 + 1, NS, F0, True); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
            return TRJ[k]
        FEE_ACC, FEE_MO = 90.0, 100.0
        print(f'  eval fee in sim = ${fee}; Rey: ${FEE_ACC:.0f} per account + ${FEE_MO:.0f}/month while holding one')
        for pl, S_ in (('2018-21 starts', range(0, k22, 3)), ('2022-25 starts', range(k22, k25, 3))):
            gaps = []; costs = []; accts = []
            for s0 in S_:
                s = s0; t_prev = s0; n_acc = 0; cost = 0.0
                while s < NS - 1:
                    r, end, pays, k2 = traj(s); n_acc += 1; cost += FEE_ACC
                    for (kd, amt) in pays:
                        cost += FEE_MO * (kd - t_prev) / 21.0; gaps.append(kd - t_prev); costs.append(cost); accts.append(n_acc); t_prev = kd; n_acc = 0; cost = 0.0
                    # months while holding an account that ends without further payouts
                    cost += FEE_MO * ((end if end < NS else NS - 1) - max(t_prev, s)) / 21.0 if not pays else 0.0
                    if pays: cost = FEE_MO * (end - pays[-1][0]) / 21.0 if end > pays[-1][0] else 0.0
                    if not pays: t_prev = end
                    s = end + 1
            g = np.array(gaps); c = np.array(costs); a_ = np.array(accts)
            print(f'    {pl}: payout gaps (trading days, incl. start->1st payout): median {np.median(g):.0f}, p90 {np.percentile(g,90):.0f}, max {g.max():.0f} (~{g.max()/21:.1f} months) | accounts bought between payouts: median {np.median(a_):.0f}, max {a_.max():.0f} | fees paid between payouts: median ${np.median(c):,.0f}, p90 ${np.percentile(c,90):,.0f}, max ${c.max():,.0f}')
    print(f'\n=== SKIP (analysis only) 50K, ONE ACCOUNT (per eval started every 2nd trading day), Lucid caps, room rule, cost 0.5 pt ===', flush=True)

    # ================= report 170: where do the bad stretches come from =================
    import pickle as _pk
    legs = [(en, T_, u) for en, (T_, u, yz) in zip(NAMES[:-3], gapmul(S6, 0.5, 1.0)) if u > 0] + [('IFVG', CH0, 0.5), ('S1', S1R, 0.2)]
    daily = {}
    for en, T_, u in legs:
        t_ = T_[T_.sz > 0]; daily[en] = (t_.pts * t_.sz * u * (t_.ce if 'ce' in t_ else 1.0)).groupby(_pd.to_datetime(t_.day).dt.normalize().values).sum()
    DL = _pd.DataFrame(daily).fillna(0.0); DL = DL[(DL.index >= '2018-02-01') & (DL.index <= '2026-01-30')]; DL['SYS'] = DL.sum(axis=1)
    _b = _pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); _b['d'] = _b.dt.dt.normalize(); _b['m'] = _b.dt.dt.hour * 60 + _b.dt.dt.minute
    _r = _b[(_b.m >= 570) & (_b.m <= 955)].groupby('d').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); _r['tr'] = _r.h - _r.l
    E = _pd.DataFrame(index=_r.index); E['ret'] = _r.c.diff(); E['atrp'] = _r.tr.rolling(14).mean() / _r.c; E['ret20'] = _r.c / _r.c.shift(20) - 1
    E['er20'] = (_r.c - _r.c.shift(20)).abs() / _r.c.diff().abs().rolling(20).sum(); E['dd250'] = _r.c / _r.c.rolling(250, min_periods=60).max() - 1; E['below200'] = (_r.c < _r.c.rolling(200, min_periods=100).mean()).astype(float)
    E['gapabs'] = ((_r.o - _r.c.shift()).abs() / _r.tr.rolling(14).mean()).rolling(20).mean(); E['rv5'] = _r.tr.rolling(5).mean() / _r.tr.rolling(60).mean()
    Eh = E.shift(1)  # known before the day
    DL = DL.join(E[['ret']]); print(f'\nSYSTEM daily pts (NQ-pt x contract equiv): days {len(DL)}, mean {DL.SYS.mean():+.2f}, std {DL.SYS.std():.1f}; beta to NQ daily change {np.polyfit(DL.ret.fillna(0), DL.SYS, 1)[0]:+.3f}, corr {np.corrcoef(DL.ret.fillna(0), DL.SYS)[0,1]:+.2f}')
    for en in [c for c in DL.columns if c not in ('SYS', 'ret')]: print(f'   {en:6s} total {DL[en].sum():+8.0f}  corr to NQ chg {np.corrcoef(DL.ret.fillna(0), DL[en])[0,1]:+.2f}  daily corr to rest {np.corrcoef(DL[en], DL.SYS - DL[en])[0,1]:+.2f}')
    # 20-day blocks
    B = DL.SYS.rolling(20).sum(); blk = DL.index[19::20]; Bk = _pd.DataFrame({'pnl': [DL.SYS[max(0, DL.index.get_loc(d) - 19):DL.index.get_loc(d) + 1].sum() for d in blk], 'start': [DL.index[DL.index.get_loc(d) - 19] for d in blk]}).set_index('start')
    Bk = Bk.join(Eh, how='left'); q = Bk.pnl.quantile(.2); Bk['bad'] = Bk.pnl <= q
    print(f'\n20-day blocks: {len(Bk)}, bad = bottom 20% (pnl <= {q:+.1f}); mean pnl good {Bk[~Bk.bad].pnl.mean():+.1f} bad {Bk[Bk.bad].pnl.mean():+.1f}; share of blocks with pnl<0: {(Bk.pnl<0).mean():.0%}')
    for c in ['atrp', 'ret20', 'er20', 'dd250', 'below200', 'gapabs', 'rv5']:
        g, bd = Bk[~Bk.bad][c].dropna(), Bk[Bk.bad][c].dropna(); print(f'   env at block start {c:8s} good {g.mean():+.3f}  bad {bd.mean():+.3f}  (n {len(g)}/{len(bd)}, t {(bd.mean()-g.mean())/np.sqrt(bd.var()/len(bd)+g.var()/len(g)):+.1f})')
    print('   P&L by environment (block mean pnl; blocks): ')
    for c in ['atrp', 'er20', 'ret20', 'dd250']:
        x = Bk.dropna(subset=[c]); qs = x[c].quantile([1/3, 2/3]).values; lab = np.where(x[c] <= qs[0], 'low', np.where(x[c] <= qs[1], 'mid', 'high')); print(f'     {c:6s} ' + '  '.join(f'{l}: {x.pnl[lab==l].mean():+6.1f} (neg {100*(x.pnl[lab==l]<0).mean():.0f}%, n{(lab==l).sum()})' for l in ('low', 'mid', 'high')))
    x = Bk.dropna(subset=['below200']); print('     below 200d SMA: ' + '  '.join(f'{int(v)}: {x.pnl[x.below200==v].mean():+6.1f} (neg {100*(x.pnl[x.below200==v]<0).mean():.0f}%, n{(x.below200==v).sum()})' for v in (0.0, 1.0)))
    # by leg in bad blocks
    print('\n   leg P&L in the bad blocks vs good (avg per 20d block):')
    bs = Bk.index[Bk.bad]; gs = Bk.index[~Bk.bad]
    def blockleg(en, starts): return np.mean([DL[en][DL.index.get_loc(d):DL.index.get_loc(d) + 20].sum() for d in starts])
    for en in [c for c in DL.columns if c not in ('SYS', 'ret')]: print(f'     {en:6s} bad {blockleg(en, bs):+7.1f}  good {blockleg(en, gs):+7.1f}')
    # worst 5 blocks
    print('\n   worst 8 blocks (start, pnl, atrp, ret20, er20, dd250):')
    for d, r in Bk.sort_values('pnl').head(8).iterrows(): print(f'     {d.date()} {r.pnl:+7.1f}  atrp {r.atrp*100:4.2f}%  ret20 {r.ret20*100:+5.1f}%  er20 {r.er20:.2f}  dd250 {r.dd250*100:+5.1f}%  worst leg {min([c for c in DL.columns if c not in ("SYS","ret")], key=lambda en: DL[en][DL.index.get_loc(d):DL.index.get_loc(d)+20].sum())}')
    Bk.to_pickle("/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/blocks170.pkl"); DL.to_pickle("/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/daily170.pkl")
