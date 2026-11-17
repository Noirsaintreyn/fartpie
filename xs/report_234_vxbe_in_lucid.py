"""Report 234 (break-even at +0.5R, target 1R version of the 1H 1:1 leg in the Lucid sim; derived from 231/226)\nReport 226 (payout targets measured, no rebuild: payouts a month, longest gap between payouts and fees spent inside a gap, for 1-5 accounts started 15 trading days apart, each replaced when it fails or graduates). Derived from report 225b (combinations of the levers that improved both eras in report 225; chosen after seeing both eras, so in-sample). Report 225: payout-convexity levers on the REAL algo in the Lucid 50K sim (baseline = COMBO without FIG-V, eval policy E2 = clip(c^2,.3,1), funded F0 x1, daily loss stop $1,000). DECLARED before results. One lever at a time vs the same baseline; a lever counts only if net $/eval improves in BOTH start eras (2018-21 and 2022-25) without a lower paid-once share.
A SIZE MAP: all sizes x m, m in .5 .75 1.25 1.5 2 (both phases); then eval-only and funded-only x .5 / 1.5 / 2 (the expectation separates: pass rate x funded payouts).
B EVAL POLICY when behind (loss per failed eval is capped at the fee, so pressing near the floor may beat de-risking): E0 flat; PRESS x1.5 when cushion < .5; BARBELL = E2 above .3 cushion, x1.5 below it; INVERSE = clip(1.6 - c, .8, 1.5).
C FUNDED POLICY: F1 clip(c,.4,1); PROTECT = x.5 once the balance is >= $4,000 (the $2,000 payout is already maxed); POSTPAY = x clip(c,.3,1) after the first payout only (floor is then fixed just under the balance).
D PAYOUT TIMING: take a payout only when it is >= $1,000 / >= $2,000 (default $500).
E DAY LOCKS: funded, stop for the day once the day is +$150 / +$300 (bank the qualifying day); eval, stop for the day at +$1,500 (a bigger day cannot count past the 50% rule); daily loss stop $500 / $1,500 instead of $1,000.
Original harness docstring follows. Report 194: DOLLAR-RISK SIZING (Rey Nov 15: target $ loss per trade, contracts = floor($ / (stop pts x $2)), min 1 micro) on the V15a, M1 and RA legs (stops known from their rules: .15 x ADR for V15a and M1, 50 pts for RA), vs the fixed sizes, Lucid 50K, FIG-V removed. Harness 190. Original: Report 190: Lucid 50K with FIG-V (IFVG) re-simulated on 1m fill-resolved bars (report 189 fill-bar audit) vs as originally simulated vs removed. Harness 188. Original: Report 188: STATE-MACHINE TAKE PROFIT on IFVG + fixed-dollar-risk sizing, on the report-177 COMBO (harness 181). DECLARED BEFORE RESULTS (Rey Nov 15: down -> target more / trail aggressively; up -> take profit right at the goal; risk in dollars). Per-trade IFVG outcomes at 1R/2R/3R come from report 168 (same entries and stops). State rules: EVAL down = cushion (balance above trailing floor / MLL) < 0.4 -> 3R; near goal = balance >= 60% of the eval target -> 1R; else 2R. FUNDED down = cushion < 0.4 -> 3R; near payout = 3+ of the 5 qualifying days done -> 1R; else 2R. Variants: full machine, near-goal only, down only, fixed R1, fixed R3. Fixed-dollar-risk sizing: IFVG size x clip(median risk / trade risk, 0.5, 1.5), S1 same. MAE is kept as the R2 path value (conservative for R1). Original header: Report 181 (session-gate tilts on the 177 combo; harness 177) Report 177: S1 overnight-VWAP gate, RA 40d IPDA premium/discount gate, V15a high-LVR gate in the Lucid one-account sim (deploy baseline). Declared before results. (harness: report 175) Report 166b: IFVG sized by EMA-trend score (tiers declared in report 166), on top of the deploy config (S1 regime-ON 0.2 ct). Header of 162: S1 premium fade (report 161 trades, MNQ-scaled 0.1-0.3 ct) added to variant I + IFVG 0.5 ct in the Lucid 50K sim. Eval 2018-21 starts show S1 inactive-regime effects; 2022-25 is the regime-on window. Header of 146: IFVG (report 144) added as an extra leg to variant I in the Lucid 50K sim (same harness as report_18b; chain replaced by IFVG; idle-days-only = IFVG trades on days the core legs do not trade). Original header: (batch 2, FVG-2R from report_18): CHAIN (ICT/ICC sweep->MSS+FVG->limit, xs/report_17_icc_chain.py) ADDED AS AN 8TH LEG at 0.5 contract (5 MNQ) to the current best (variant I), vs variant I alone. 50K one account, Lucid caps, eval and funded, same sim as report_gap. Declared before results.
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
    DSTOP=[1000.0]; DLOCK={'e':0,'f':0}; PAYMIN=[500.0]; POL={}
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
                if uu > 0: L[k].append((r_.m if en not in ('ON+', 'ON-') else -1, ((min(uu, 1.75),) + ((r_.R1, r_.R3) if hasattr(r_, 'R1') else ()), r_.pts, r_.mae, (r_.m, END[en]))))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    GS = dict(cu=1.0, pr=0.0, ph='e', wins=0); TPCFG = ['R2']
    def TPM():
        c = TPCFG[0]
        if c == 'R1': return 1
        if c == 'R3': return 3
        if c == 'R2': return 2
        down = GS['cu'] < .4; near = (GS['pr'] >= .6) if GS['ph'] == 'e' else (GS['wins'] >= 3)
        if c == 'full': return 3 if down else 1 if near else 2
        if c == 'near': return 1 if near else 2
        if c == 'down': return 3 if down else 2
        return 2
    def cap_eval(A): return 40 if TIER == '50K' else 100
    def cap_fund(A, prof):
        if TIER == '50K': return 20 if prof < 1000 else 30 if prof < 2000 else 40
        return 40 if prof < 1000 else 50 if prof < 2000 else 60 if prof < 3000 else 80 if prof < 4500 else 100
    def run_day_c(L, k, st, mll, scale, mult, cap):
        cum = 0.0; bal = st['bal']; opn = []                                   # (window end, contracts) of trades still open
        for th, net, mae, p in L[k]:
            if cum <= -DSTOP[0] * scale: break
            if DLOCK[GS.get('ph','e')] and cum >= DLOCK[GS.get('ph','e')]: break
            n = max(1, int(round(10 * th[0] * scale * mult)))
            if cap is not None:
                m0, we = p; night = (we == 'N')
                opn = [(w, c) for w, c in opn if (w == 'N') == night and (night or w > m0)]
                room = cap - sum(c for _, c in opn)
                if room <= 0: continue
                n = min(n, room); opn.append((we, n))
            if len(th) > 2: md = TPM(); net = th[1] if md == 1 else th[2] if md == 3 else net
            mm = 2.0 * n
            if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
            cum += net * mm
        return cum, False
    def EV(L, A, s, end, pol, capped):
        st = dict(bal=0.0, floor=-A['mll'], peak=0.0); best = 0.0
        for k in range(s, end):
            cu = (st['bal'] - st['floor']) / A['mll']; pr = st['bal'] / A['target']
            GS.update(cu=cu, pr=pr, ph='e', wins=0)
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
            GS.update(cu=cu, pr=0, ph='f', wins=wins, bal=st['bal'], npay=len(pays))
            pnl, br = run_day_c(L, k, st, A['mll'], A['scale'], pol(cu, len(pays)), cap_fund(A, st['bal']) if capped else None)
            if br: return 'lost', k, pays
            st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
            if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
            if wins >= 5 and b > 0 and (not pays or since > 0):
                amt = min(.5 * b, A['cap'])
                if amt >= PAYMIN[0]:
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
    NAMES = list(NAMES) + ['NZ', 'CH', 'S1', 'UN']; END['UN'] = 955; END['NZ'] = 960; END['CH'] = 955; END['S1'] = 955
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
        for (T_, u, yz), en in zip(spec, NAMES[:-3]):
            if en in ('RA', 'V15a') and 'dir' in T_.columns:
                T_ = T_.copy(); b = np.array([_state(d, m) for d, m in zip(T_.day, T_.m)]); w = (b * T_.dir.values > 0)
                T_['sz'] = T_.sz.values * np.where(w, f_with, np.where(b == 0, f_in, 1.0))
            out.append((T_, u, yz))
        return out
    NZf = NZ.copy(); NZf['ce'] = 0.0; NZfl = NZf[NZf.dir > 0].copy()
    _gp = (_Dn.gap.abs() / _Dn.rng20.shift(1)); _gr = _gp.rolling(252, min_periods=120).rank(pct=True)
    def gapmul(spec, lo, hi):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-3]):
            if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
                T_ = T_.copy(); r = _gr.reindex(_pd.to_datetime(T_.day)).values
                T_['sz'] = T_.sz.values * np.where(r <= .33, lo, np.where(r >= .67, hi, 1.0))
            out.append((T_, u, yz))
        return out
    _X = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/s1_trades162.pkl'); _X['day'] = _pd.to_datetime(_X.day)
    _g = _pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date').shift(1)
    _w = _pd.read_csv('/mnt/user-data/uploads/fartpie/gex_walls_nq.csv', parse_dates=['date']).set_index('date').shift(1)
    _bb = _pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); _bb['d'] = _bb.dt.dt.normalize(); _bb['m'] = _bb.dt.dt.hour * 60 + _bb.dt.dt.minute
    _rr = _bb[(_bb.m >= 570) & (_bb.m <= 955)].groupby('d').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min')); _atr = (_rr.h - _rr.l).rolling(14).mean().shift(1)
    q1 = _g.net_vanna[(_g.index >= '2022-05-01') & (_g.index <= '2023-12-31')].quantile(1/3)
    def feats(df, entcol, drcol, daycol):
        d = _pd.to_datetime(df[daycol]); F = _pd.DataFrame(index=df.index); ent = df[entcol].values if entcol in df else d.map(_rr.o).values
        F['lowv'] = (d.map(_g.net_vanna) <= q1).values & d.map(_g.net_vanna).notna().values; cw = d.map(_w.call_wall_nq).values; pw = d.map(_w.put_wall_nq).values; at = d.map(_atr).values; dr = df[drcol].values
        tw = np.where(dr > 0, cw, pw); F['room'] = dr * (tw - ent) / at; F['near'] = ((np.abs(ent - cw) / at <= .5) | (np.abs(ent - pw) / at <= .5)); F['have'] = d.map(_w.call_wall_nq).notna().values & (d >= '2022-05-01').values
        return F
    FS = feats(_X.assign(dr_=-1.0), 'entry_none', 'dr_', 'day'); _X['up'] = (FS.lowv | (FS.room > .5)) & FS.have
    _C = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/uni_ifvg.pkl'); _C = _C[_C.model == 'IFVG'].copy(); FI = feats(_C, 'ent', 'dr', 'd'); _C['near'] = (FI.near & FI.have).values
    print('  S1 up-lever days (low vanna or room>0.5ATR) in regime-ON trades:', int(((_X.reg > 0) & _X.up).sum()), 'of', int((_X.reg > 0).sum()), '| IFVG near-wall trades:', int(_C.near.sum()), 'of', len(_C))
    def _mk2(g, f): X = pd.DataFrame(dict(day=g.day, m=g.m.astype(float), dir=g.dir, pts=g.pts, mae=g.mae)); X['sz'] = f(g); X['ce'] = 1.0; X['cf'] = 1.0; return X[X.sz > 0]
    S1_flat = _mk2(_X, lambda g: np.where(g.reg > 0, 1.0, 0.0)); S1_up = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(g.up, 1.5, 1.0), 0.0)); S1_up2 = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(g.up, 2.0, 1.0), 0.0))
    def _mk(g, f): X = pd.DataFrame(dict(day=pd.to_datetime(g.d), m=g.m.astype(float), dir=g.dr, pts=g.pts, mae=g.mae)); X['sz'] = f(g); X['ce'] = 1.0; X['cf'] = 1.0; return X
    CH0 = _mk(_C, lambda g: np.ones(len(g))); CHn = _mk(_C, lambda g: np.where(g.near, 1.5, 1.0)); CHn2 = _mk(_C, lambda g: np.where(g.near, 2.0, 1.0))
    _I = gapmul(S6, 0.5, 1.0) + [(NZ, 0.0, None)]; UNI = _mk(_pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/uni_ifvg.pkl').query("model=='UNI'"), lambda g: np.ones(len(g)))
    # ---- report 177 additions (declared before results): S1 overnight-VWAP gate, RA 40d IPDA gate, V15a LVR gate
    _bb['tp'] = (_bb.high + _bb.low + _bb.close) / 3 * _bb.volume
    _on = _bb[(_bb.m >= 1080) | (_bb.m < 570)].copy(); _on['d2'] = (_on.dt + _pd.to_timedelta(np.where(_on.m >= 1080, 1, 0), unit='D')).dt.normalize()
    _ov = _on.groupby('d2').tp.sum() / _on.groupby('d2').volume.sum(); _op = _rr.o
    _dd = _pd.to_datetime(_X.day); _ovx = _dd.map(_ov).values; _opx = _dd.map(_op).values; _okv = np.isfinite(_ovx) & np.isfinite(_opx); _below = _okv & (_opx <= _ovx)
    _X['below'] = _below; _X['okv'] = _okv
    S1_g = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(g.okv & ~g.below, 0.0, 1.0), 0.0))
    S1_t = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(g.below, 1.5, 1.0), 0.0))
    S1_t2 = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(g.below, 1.5, 0.5), 0.0))
    print('  S1 regime-ON trades: below/at ON VWAP', int(((_X.reg > 0) & _X.below).sum()), 'above', int(((_X.reg > 0) & _X.okv & ~_X.below).sum()), 'no data', int(((_X.reg > 0) & ~_X.okv).sum()))
    _h40 = _rr.h.rolling(40).max().shift(1); _l40 = _rr.l.rolling(40).min().shift(1); _mid40 = (_h40 + _l40) / 2
    def ra_gate(spec, mult):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-3]):
            if en == 'RA' and 'dir' in T_.columns:
                T_ = T_.copy(); d = _pd.to_datetime(T_.day); mid = d.map(_mid40).values; op = d.map(_op).values; ok = np.isfinite(mid) & np.isfinite(op)
                disc = op < mid; al = (T_.dir.values > 0) == disc; T_['sz'] = T_.sz.values * np.where(ok & ~al, mult, 1.0)
            elif en == 'RA': print('  RA has no dir column:', list(T_.columns))
            out.append((T_, u, yz))
        return out
    sys.path.insert(0, '/home/claude/fartpie/xs'); from feats138 import dfeat
    _dm = _pd.DataFrame({dd: {'h': np.nanmax(_Z.A[dd][0][1]), 'l': np.nanmin(_Z.A[dd][0][2]), 'c': _Z.A[dd][0][3][~np.isnan(_Z.A[dd][0][3])][-1], 'v': np.nansum(_Z.A[dd][0][4])} for dd in _Z.ALLD}).T; _dm.index = _pd.DatetimeIndex(_dm.index)
    _LVR = dfeat(_dm.astype(float)).LVR
    _vt = [T_ for (T_, u, yz), en in zip(S6, NAMES[:-3]) if en == 'V15a'][0]; _lv = _vt.day.map(lambda x: _LVR.asof(_pd.Timestamp(x))); _lvcut = _lv[_pd.to_datetime(_vt.day) < '2022-01-01'].quantile(2 / 3); print('  V15a LVR train 2/3 cut', round(float(_lvcut), 3), 'high-LVR trades', int((_lv > _lvcut).sum()), 'of', len(_vt))
    def v_gate(spec, mult):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-3]):
            if en == 'V15a':
                T_ = T_.copy(); lv = T_.day.map(lambda x: _LVR.asof(_pd.Timestamp(x))).values; T_['sz'] = T_.sz.values * np.where(lv > _lvcut, mult, 1.0)
            out.append((T_, u, yz))
        return out
    # ---- report 181: session gates as size tilts (declared before results): IFVG x1.5 when ACCD (overnight acceptance on the side of the open) in train-top tercile; S1 x1.5 when London range/ATR in train-top tercile
    _on2 = _bb[(_bb.m >= 1080) | (_bb.m < 565)].copy(); _on2['sd'] = (_on2.dt + _pd.to_timedelta(np.where(_on2.m >= 1080, 1, 0), unit='D')).dt.normalize()
    _R = _bb[(_bb.m >= 570) & (_bb.m <= 955)].groupby('d').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); _pdh = _R.h.shift(1); _pdl = _R.l.shift(1); _pc = _R.c.shift(1)
    _lon = _bb[(_bb.m >= 120) & (_bb.m < 300)].groupby('d').apply(lambda g: g.high.max() - g.low.min()); LONR = (_lon.reindex(_R.index) / _atr.reindex(_R.index))
    _accd = {}
    for _g_, _gg in _on2.groupby('sd'):
        if _g_ in _pdh.index and np.isfinite(_pdh[_g_]) and len(_gg) >= 20 and _g_ in _R.index:
            up = (_gg.close > _pdh[_g_]).mean(); dn = (_gg.close < _pdl[_g_]).mean(); _accd[_g_] = up if _R.o[_g_] > _pc[_g_] else dn
    ACCD = _pd.Series(_accd)
    _dI = _pd.to_datetime(_C.d).dt.normalize(); _aI = _dI.map(ACCD).values; _cutA = np.nanquantile(_aI[(_dI < '2022-01-01').values], 2 / 3); _hiA = np.isfinite(_aI) & (_aI > _cutA)
    _dS = _pd.to_datetime(_X.day).dt.normalize(); _lS = _dS.map(LONR).values; _cutL = np.nanquantile(_lS[(_dS < '2022-01-01').values & (_X.reg > 0).values], 2 / 3); _hiL = np.isfinite(_lS) & (_lS > _cutL)
    print('  ACCD cut', round(float(_cutA), 3), 'IFVG high-acceptance trades', int(_hiA.sum()), 'of', len(_C), '| LON_R cut', round(float(_cutL), 3), 'S1 regime-ON high-London trades', int(((_X.reg > 0).values & _hiL).sum()), 'of', int((_X.reg > 0).sum()))
    CHa = _mk(_C, lambda g: np.where(_hiA, 1.5, 1.0)); CHa2 = _mk(_C, lambda g: np.where(_hiA, 1.5, 0.75))
    _bw = np.asarray(_X.below)
    S1_c = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(_bw, 1.5, 0.5) * np.where(_hiL, 1.5, 1.0), 0.0))
    S1_c2 = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(_bw, 1.5, 0.5) * np.where(_hiL, 1.5, 0.67), 0.0))
    S1_d = _mk2(_X, lambda g: np.where(g.reg > 0, np.where(_hiL, 1.5, 1.0), 0.0))
    _CB = v_gate(ra_gate(_I, 0.5), 0.0)
    _T168 = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/tp168.pkl'); print('  tp168 rows', len(_T168), 'IFVG rows', len(_C), 'date match', bool((_pd.to_datetime(_T168.d).values == _pd.to_datetime(_C.d).values).all()), 'R2 vs pts max diff', float(np.abs(_T168.R2.values - _C.pts.values).max()))
    _K = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ifvg_1m189.pkl'); _K = _K.dropna(subset=['pts1m']).copy(); _K['d'] = _pd.to_datetime(_K.d)
    CHc = _mk(_K.rename(columns={'dr': 'dr'}).assign(pts=_K.pts1m.values, mae=_K.mae1m.values), lambda g: np.ones(len(g)))
    CHs = CHc.copy(); CHs['pts'] = CHs.pts.values - 0.5 * (_K.stopped.values == True)
    print('  corrected FIG-V trades', len(CHc), 'original', len(CH0))
    def dollar_size(spec, Rd, legs=('V15a', 'M1', 'RA')):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-3]):
            if en in legs and u > 0:
                T_ = T_.copy(); d_ = _pd.to_datetime(T_.day); stop = d_.map(DY.adr).values * .15 if en in ('V15a', 'M1') else np.full(len(T_), 50.0)
                n = np.maximum(1, np.floor(Rd / (stop * 2.0))); T_['sz'] = np.where(T_.sz.values > 0, n / (10.0 * u), 0.0)
            out.append((T_, u, yz))
        return out
    _base = _CB + [(CHc, 0.0, None), (S1_t2, 0.2, None)]
    SPS = {'BASELINE: COMBO without FIG-V (fixed sizes)': _base}
    for Rd in (100, 150, 200, 300, 400):
        SPS[f'$-RISK ${Rd} per trade on V15a/M1/RA (rest as before)'] = dollar_size(_CB, Rd) + [(CHc, 0.0, None), (S1_t2, 0.2, None)]
    _dd = _pd.to_datetime(_vt.day); _st = _dd.map(DY.adr).values * .15; print('  V15a stop (0.15 ADR) median', round(float(np.nanmedian(_st)), 1), 'pts; contracts at $100:', np.unique(np.maximum(1, np.floor(100 / (_st[np.isfinite(_st)] * 2))), return_counts=True))
    TPS = {}
    def one_acct(LE_, LF_):
        TRJ = {}
        def traj(k):
            if k not in TRJ:
                r, k2 = EV(LE_, A, k, NS, POL.get('e',E2), True)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [], None)
                else: fr, k3, pays = FU(LF_, A, k2 + 1, NS, POL.get('f',F0), True); TRJ[k] = (r, NS if fr == 'open' else k3, pays, k2)
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
        tot=sum(x for p_ in chain for _,x in p_); yrs=(D_[NS-1]-D_[0]).days/365.25; res.append(f'    one account at a time 2018-26: {len(chain)} evals, {sum(len(p_) for p_ in chain)} payouts, ${tot:,.0f} paid, fees ${len(chain)*fee:,.0f}, net ${(tot-len(chain)*fee)/yrs:,.0f} a year, {sum(len(p_) for p_ in chain)/yrs/12:.2f} payouts a month, longest run without a payout {mx} evals')
        _c=[]; _s=0
        while _s < NS - 1:
            r, end, pays, _ = traj(_s); _c += [k for k,_x in pays]; _s = end + 1
        _g=sorted([((D_[b]-D_[a]).days, str(D_[a].date()), str(D_[b].date())) for a,b in zip(_c[:-1],_c[1:])],reverse=True)[:4]; res.append('    longest gaps (days, from, to): '+'; '.join(f'{g} d {a} -> {b}' for g,a,b in _g)); _y={}
        for k in _c: _y[D_[k].year]=_y.get(D_[k].year,0)+1
        res.append('    payouts by year: '+' '.join(f'{y}:{n}' for y,n in sorted(_y.items())))
        for N_ in (1,):
            ev_=[]; st_=[]
            for sl in range(N_):
                s=sl*15
                while s < NS - 1:
                    r, end, pays, _ = traj(s); st_.append(s); ev_ += [k for k,_x in pays]; s = end + 1
            ev_=sorted(ev_); st_=np.array(sorted(st_)); mo=(D_[NS-1]-D_[0]).days/30.44; pts_=[0]+ev_; gd=[(D_[b]-D_[a]).days for a,b in zip(pts_[:-1],pts_[1:])]; fs=[fee*int(((st_>a)&(st_<=b)).sum()) for a,b in zip(pts_[:-1],pts_[1:])]
            gd2=gd[1:] if len(gd)>1 else gd; fs2=fs[1:] if len(fs)>1 else fs; ym=set((D_[k].year,D_[k].month) for k in ev_); allm=[(y,m_) for y in range(2018,2026) for m_ in range(1,13) if (y,m_)>=(2018,7)]
            res.append(f'    fleet of {N_}: {len(ev_)/mo:.2f} payouts a month | longest gap {max(gd2)} days (95th pct {np.percentile(gd2,95):.0f}, median {np.median(gd2):.0f}) | start to first payout {gd[0]} days | most fees inside one gap ${max(fs2)} | months with no payout {np.mean([m_ not in ym for m_ in allm])*100:.0f}%')
        return res
    from vx_trades import vx
    VXT = vx(be=0.5); print('  VX trades', len(VXT), flush=True)
    NAMES[10] = 'VX'; END['VX'] = 955
    def vxleg(mask, N):
        X = VXT[mask(VXT)] if mask else VXT
        T_ = pd.DataFrame(dict(day=X.day.values, m=X.m.values, dir=X['dir'].values, pts=X.pts.values, mae=X.mae.values)); T_['sz'] = 1.0; T_['ce'] = 1.0; T_['cf'] = 1.0
        return (T_, N / 10.0, None)
    e2=EP['E2 harder de-risk']; clip=lambda x,a,b: float(np.clip(x,a,b)); TPCFG[0]='R2'
    SPV = [('main only', _base)]
    for N_ in (1, 2):
        SPV.append((f'+ VX 1H 1:1 all, {N_} MNQ', _base + [vxleg(None, N_)]))
        SPV.append((f'+ VX 1H 1:1 neg-gamma only, {N_} MNQ', _base + [vxleg(lambda d: d.neg.values, N_)]))
    POLS = [('payout >= $500 (default)', {}, 500.0), ('FULL payout only (>= $2,000)', {}, 2000.0)]
    for pn, pol, pm in POLS:
        print(f'\n=== {pn} ===', flush=True)
        for nm, sp in SPV:
            LE_, LF_ = lists_cap(sp, 'e'), lists_cap(sp, 'f')
            POL.clear(); POL.update(pol); PAYMIN[0] = pm; DLOCK['e'] = 0; DLOCK['f'] = 0; DSTOP[0] = 1000.0
            print(f'  {nm}'); print('\n'.join(one_acct(LE_, LF_)), flush=True)
