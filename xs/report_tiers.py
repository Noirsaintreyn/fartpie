"""NZ ROLE ACCOUNT with a tight stop and a daily loss limit (STOP / LDL env vars, defaults 25 / 40; Nov 10 2026, declared before results). Same sim as report_noise_only.
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
    NAMES = list(NAMES) + ['NZ']; END['NZ'] = 960
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
        for (T_, u, yz), en in zip(spec, NAMES[:-1]):
            if en in ('RA', 'V15a') and 'dir' in T_.columns:
                T_ = T_.copy(); b = np.array([_state(d, m) for d, m in zip(T_.day, T_.m)]); w = (b * T_.dir.values > 0)
                T_['sz'] = T_.sz.values * np.where(w, f_with, np.where(b == 0, f_in, 1.0))
            out.append((T_, u, yz))
        return out
    NZf = NZ.copy(); NZf['ce'] = 0.0; NZfl = NZf[NZf.dir > 0].copy()
    _gp = (_Dn.gap.abs() / _Dn.rng20.shift(1)); _gr = _gp.rolling(252, min_periods=120).rank(pct=True)
    def gapmul(spec, lo, hi):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-1]):
            if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
                T_ = T_.copy(); r = _gr.reindex(_pd.to_datetime(T_.day)).values
                T_['sz'] = T_.sz.values * np.where(r <= .33, lo, np.where(r >= .67, hi, 1.0))
            out.append((T_, u, yz))
        return out
    _tier = _pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/dt_tier.pkl')
    def dtmul(spec, up, dn):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES[:-1]):
            if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
                T_ = T_.copy(); r = _tier.reindex(_pd.to_datetime(T_.day)).fillna(0).values
                T_['sz'] = T_.sz.values * np.where(r > 0, up, np.where(r < 0, dn, 1.0))
            out.append((T_, u, yz))
        return out
    SPS = {'A. S6 + room rule (current)': S6, 'I. current best: x0.5 bottom-third gap': gapmul(S6, 0.5, 1.0) + [(NZ, 0.0, None)], 'P. I + day-type x1.0/x0.5 (cut low only)': dtmul(gapmul(S6, 0.5, 1.0), 1.0, 0.5) + [(NZ, 0.0, None)]}
    # daily correlation of NZ with the rest of the system
    base_daily = pd.Series(0.0, index=pdp.index)
    for en, (T_, u, yz) in zip(NAMES[:-1], S6):
        t = T_[T_.sz > 0]; base_daily = base_daily.add((t.pts * t.sz * u).groupby(pd.to_datetime(t.day).dt.normalize().values).sum().reindex(pdp.index).fillna(0), fill_value=0)
    print(f'  daily correlation of NZ pts with the S6 system pts: {np.corrcoef(pdp.values, base_daily.values)[0, 1]:+.2f}', flush=True)
    # ------------------------------------------------------------------ WHAT STOPS US / WHAT TAKES SO LONG (Flex 50K, variant I) -------------------------------
    LE_, LF_ = lists_cap(SPS['I. current best: x0.5 bottom-third gap'], 'e'), lists_cap(SPS['I. current best: x0.5 bottom-third gap'], 'f')
    k18 = int(np.searchsorted(D_, pd.Timestamp('2018-02-01'))); k25_ = int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))); k22b = int(np.searchsorted(D_, pd.Timestamp('2022-01-01')))
    def rd(L, k, st, mll, scale, mult, cap, dstop=True):
        cum = 0.0; bal = st['bal']; opn = []; ntr = 0; stopped = False; capped_n = 0
        for th, net, mae, p in L[k]:
            if dstop and cum <= -1000 * scale: stopped = True; break
            n = max(1, int(round(10 * th[0] * scale * mult)))
            if cap is not None:
                m0, we = p; night = (we == 'N'); opn = [(w, c) for w, c in opn if (w == 'N') == night and (night or w > m0)]; room = cap - sum(c for _, c in opn)
                if room <= 0: capped_n += 1; continue
                if n > room: capped_n += 1
                n = min(n, room); opn.append((we, n))
            mm = 2.0 * n
            if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True, ntr, stopped, capped_n, cum + mae * mm
            cum += net * mm; ntr += 1
        return cum, False, ntr, stopped, capped_n, 0.0
    def evd(L, s, end, pol, capped=True, dstop=True, consist=True):
        st = dict(bal=0.0, floor=-A['mll'], peak=0.0); best = 0.0; D = dict(days=0, trade_days=0, stopdays=0, derisk_days=0, capdays=0, blocked=0, worst=0.0, maxbal=0.0)
        for k in range(s, end):
            cu = (st['bal'] - st['floor']) / A['mll']; pr = st['bal'] / A['target']; m = pol(cu, pr)
            pnl, br, ntr, sp, cn, _ = rd(L, k, st, A['mll'], A['scale'], m, cap_eval(A) if capped else None, dstop)
            D['days'] += 1; D['trade_days'] += ntr > 0; D['stopdays'] += sp; D['derisk_days'] += (m < 0.999); D['capdays'] += cn > 0; D['worst'] = min(D['worst'], pnl)
            if br: D.update(res='fail', end=k, pnl_fail=pnl, bal=st['bal'], floor=st['floor']); return D
            st['bal'] += pnl; best = max(best, pnl); D['maxbal'] = max(D['maxbal'], st['bal'])
            if st['bal'] >= A['target']:
                if (not consist) or best <= .5 * st['bal']: D.update(res='pass', end=k, bal=st['bal']); return D
                D['blocked'] += 1
            if st['floor'] < 100: st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
        D.update(res='open', end=end, bal=st['bal']); return D
    one = lambda cu, pr: 1.0
    def study(nm, **kw):
        pol = kw.pop('pol', E2); rows = []
        for s_ in range(k18, k25_, 2):
            D = evd(LE_, s_, NS, pol, **kw); D['dpass'] = D['end'] - s_ + 1; rows.append(D)
        R = pd.DataFrame(rows); p = R[R.res == 'pass']
        print(f'  {nm:34} pass {np.mean(R.res=="pass"):4.0%} | fail {np.mean(R.res=="fail"):4.0%} | open {np.mean(R.res=="open"):3.0%} | days to pass median {p.dpass.median():3.0f} mean {p.dpass.mean():3.0f} p90 {p.dpass.quantile(.9):3.0f} | de-risk days/eval {R.derisk_days.mean():4.1f} | daily-stop days {R.stopdays.mean():3.1f} | cap-bound days {R.capdays.mean():3.1f} | blocked by consistency {p.blocked.mean():3.1f}d')
        return R
    # ------------------------------------------------------------------ GOOD / BAD / TERRIBLE: days and trades (variant Z = skip small-gap days), NQ pts unit-weighted -------------------------------
    ZSP = gapmul(S6, 0.0, 1.0) + [(NZ, 0.0, None)]; LEz = lists_cap(ZSP, 'e'); sc = A['scale']
    print('trade-frame columns:', list(ZSP[0][0].columns), flush=True)
    ENV = pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/env_daily.pkl')
    # ---------- trades table
    rows = []
    for (T_, u, yz), en in zip(ZSP, NAMES):
        if u <= 0: continue
        t = T_[T_.sz > 0]
        for r_ in t.itertuples():
            w = u * r_.sz * r_.ce; w = min(w, 1.75)
            rows.append((pd.Timestamp(r_.day), en, r_.m, r_.pts * w, r_.mae * w, w, getattr(r_, 'dir', getattr(r_, 'side', np.nan))))
    TR = pd.DataFrame(rows, columns=['day', 'leg', 'm', 'pts', 'mae', 'w', 'side']); TR = TR[(TR.day >= '2018-02-01') & (TR.day <= '2026-01-30')]
    TR['half'] = np.where(TR.day.dt.year < 2022, 'A', 'B'); TR = TR.join(ENV[['gap_rel', 'vxn_pct', 'rng_prev_rel', 'trend_eff10', 'dow']], on='day')
    # ---------- day table
    D = TR.groupby('day').agg(pts=('pts', 'sum'), n=('pts', 'size'), worst=('mae', 'min')).join(ENV[['gap_rel', 'vxn_pct', 'rng_prev_rel', 'trend_eff10', 'dow', 'absmove', 'R_day', 'half']], how='left')
    D = D.dropna(subset=['gap_rel']); D['half'] = np.where(D.index.year < 2022, 'A', 'B')
    q = D.pts.quantile([.05, .25, .60, .90]).values
    D['tier'] = np.select([D.pts <= q[0], D.pts <= q[1], D.pts <= q[2], D.pts <= q[3]], ['1 TERRIBLE (bottom 5%)', '2 BAD (5-25%)', '3 FLAT (25-60%)', '4 GOOD (60-90%)'], '5 GREAT (top 10%)')
    tot = D.pts.sum()
    print(f'\n=== DAYS: {len(D)} traded days | mean {D.pts.mean():+.2f} pts | tier cuts (pts/day): {q.round(1)} ===')
    for tn, g in D.groupby('tier'):
        a, b = g[g.half == 'A'], g[g.half == 'B']
        print(f'{tn:24} n {len(g):4d} | mean {g.pts.mean():+7.2f} | share of all P&L {g.pts.sum()/tot:+6.0%} | trades/day {g.n.mean():.1f} | gap_rel {g.gap_rel.mean():.2f} vxn_pct {g.vxn_pct.mean():.2f} prev_rng_rel {g.rng_prev_rel.mean():.2f} trend_eff {g.trend_eff10.mean():.2f} |move|/rng20 {g.absmove.mean():.2f} | halves: share of days A {len(a)/len(D[D.half=="A"]):.0%} B {len(b)/len(D[D.half=="B"]):.0%}')
    top = D.pts.sort_values(ascending=False); print(f'  concentration: best 10 days = {top.head(10).sum()/tot:.0%} of total P&L | best 5% of days = {top.head(int(len(D)*.05)).sum()/tot:.0%} | worst 5% of days = {top.tail(int(len(D)*.05)).sum()/tot:.0%} | without the best 20 days total = {(tot-top.head(20).sum())/tot:+.0%}')
    print('  P(TERRIBLE day) and P(GREAT day) by feature third (cuts from 2018-21):')
    A_ = D[D.half == 'A']
    for f in ['gap_rel', 'vxn_pct', 'rng_prev_rel', 'trend_eff10']:
        c1, c2 = A_[f].quantile([1 / 3, 2 / 3]).values; out = []
        for nm, m in (('low', D[f] <= c1), ('mid', (D[f] > c1) & (D[f] <= c2)), ('high', D[f] > c2)):
            g = D[m]; ga, gb = g[g.half == 'A'], g[g.half == 'B']
            out.append(f'{nm}: terrible {np.mean(g.tier.str[0]=="1"):4.0%} (A {np.mean(ga.tier.str[0]=="1"):3.0%}/B {np.mean(gb.tier.str[0]=="1"):3.0%}) great {np.mean(g.tier.str[0]=="5"):4.0%} (A {np.mean(ga.tier.str[0]=="5"):3.0%}/B {np.mean(gb.tier.str[0]=="5"):3.0%}) mean {g.pts.mean():+.1f}')
        print(f'   {f:13} ' + ' | '.join(out))
    print('   by weekday terrible%: ' + ' '.join(f'{["Mon","Tue","Wed","Thu","Fri"][d]} {np.mean(D[D.dow==d].tier.str[0]=="1"):.0%}' for d in range(5)))
    cum = D.pts.cumsum(); dd = (cum - cum.cummax()); print(f'  max drawdown {dd.min():.0f} pts ({(dd.min()):.0f} on a total of {tot:.0f}) | worst day {D.pts.min():.0f} | worst 5-day sum {D.pts.rolling(5).sum().min():.0f} | worst 20-day {D.pts.rolling(20).sum().min():.0f} | longest losing streak {max((len(list(g)) for k_, g in __import__("itertools").groupby(D.pts < 0) if k_), default=0)} days')
    # ---------- trades
    TR['pu'] = TR.pts / TR.w
    qt = TR.pts.quantile([.05, .40, .70, .95]).values
    TR['tier'] = np.select([TR.pts <= qt[0], TR.pts <= qt[1], TR.pts <= qt[2], TR.pts <= qt[3]], ['1 TERRIBLE (bottom 5%)', '2 BAD (5-40%)', '3 SCRATCH (40-70%)', '4 GOOD (70-95%)'], '5 GREAT (top 5%)')
    tt = TR.pts.sum(); print(f'\n=== TRADES: {len(TR)} (unit-weighted pts) | mean {TR.pts.mean():+.2f} | win rate {np.mean(TR.pts>0):.0%} | tier cuts {qt.round(1)} ===')
    for tn, g in TR.groupby('tier'):
        print(f'{tn:24} n {len(g):5d} | mean {g.pts.mean():+7.2f} | share of P&L {g.pts.sum()/tt:+6.0%} | avg MAE {g.mae.mean():6.1f} | avg entry minute {g.m.mean()/60:4.1f}h | gap_rel {g.gap_rel.mean():.2f}')
    print(f'  best 5% of trades = {TR.pts.nlargest(int(len(TR)*.05)).sum()/tt:.0%} of P&L | worst 5% = {TR.pts.nsmallest(int(len(TR)*.05)).sum()/tt:.0%} | median trade {TR.pts.median():+.2f}')
    print('\n  BY LEG:')
    for lg, g in TR.groupby('leg'):
        w, l = g[g.pts > 0], g[g.pts <= 0]
        print(f'   {lg:6} n {len(g):5d} win {np.mean(g.pts>0):4.0%} avg win {w.pts.mean():+6.2f} avg loss {l.pts.mean():+6.2f} expectancy {g.pts.mean():+5.2f} | share of P&L {g.pts.sum()/tt:+5.0%} | terrible-rate {np.mean(g.tier.str[0]=="1"):4.0%} | 18-21 {g[g.half=="A"].pts.mean():+5.2f} 22-26 {g[g.half=="B"].pts.mean():+5.2f}')
    TR['hr'] = (TR.m // 60).astype(int)
    print('\n  BY ENTRY HOUR (ET minutes//60): ' + ' | '.join(f'{h}h n {len(g)} mean {g.pts.mean():+.2f} terr {np.mean(g.tier.str[0]=="1"):.0%}' for h, g in TR[TR.leg != 'ON+'][TR.leg != 'ON-'].groupby('hr') if len(g) > 80))
    if TR.side.notna().any(): print('  BY SIDE: ' + ' | '.join(f'{s_}: n {len(g)} mean {g.pts.mean():+.2f} win {np.mean(g.pts>0):.0%}' for s_, g in TR.groupby('side')))
    for f in ['gap_rel', 'vxn_pct']:
        c1, c2 = TR[TR.half == 'A'][f].quantile([1 / 3, 2 / 3]).values
        print(f'  TRADE by {f} third: ' + ' | '.join(f'{nm}: mean {g.pts.mean():+.2f} win {np.mean(g.pts>0):.0%} terr {np.mean(g.tier.str[0]=="1"):.0%} (A {g[g.half=="A"].pts.mean():+.2f}/B {g[g.half=="B"].pts.mean():+.2f})' for nm, g in (('low', TR[TR[f] <= c1]), ('mid', TR[(TR[f] > c1) & (TR[f] <= c2)]), ('high', TR[TR[f] > c2]))))
    TR.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/tiers_trades.pkl'); D.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/tiers_days.pkl')
