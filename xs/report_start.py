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
    # ------------------------------------------------------------------ WHEN TO START AN EVAL (Flex 50K), variants I and P ----------------------------------------
    import yfinance as yf
    Y2 = yf.download(['QQQ', '^VXN'], start='2015-01-01', interval='1d', progress=False, auto_adjust=True)['Close'].ffill(); Y2.index = pd.to_datetime(Y2.index).normalize()
    def prior(s_, days): return s_.reindex(pd.DatetimeIndex(days) - pd.Timedelta(days=1), method='ffill').values
    vx = Y2['^VXN']; vz = (vx - vx.rolling(252, min_periods=120).mean()) / vx.rolling(252, min_periods=120).std(); vp = vx.rolling(252, min_periods=120).rank(pct=True); vmed = vx.rolling(252, min_periods=120).median()
    qq = Y2['QQQ']; qz = (qq - qq.rolling(20).mean()) / qq.rolling(20).std(); q5 = qq.pct_change(5)
    rngd = _Dn.rng.rolling(1).mean(); r20_ = _Dn.rng.rolling(20).mean(); rv_ = (_Dn.rng / _Dn.c.shift(1)) if 'c' in _Dn.columns else None
    yzp = np.nan
    F = pd.DataFrame(index=range(NS)); F['day'] = D_[:NS]
    F['vxn_z'] = prior(vz, D_)[:NS]; F['vxn_pct'] = prior(vp, D_)[:NS]; F['vxn_hi'] = (prior(vx, D_)[:NS] >= prior(vmed, D_)[:NS]).astype(float); F.loc[np.isnan(prior(vx, D_)[:NS]), 'vxn_hi'] = np.nan
    F['qqq_z20'] = prior(qz, D_)[:NS]; F['qqq_ret5'] = prior(q5, D_)[:NS]
    rg = _Dn.rng.reindex(pd.DatetimeIndex(D_[:NS])).values; F['rng20_pct'] = pd.Series(rg).rolling(20).mean().rolling(252, min_periods=120).rank(pct=True).shift(1).values
    for sp_nm in ('I', 'P'):
        pass
    VR = {nm: (lists_cap(sp, 'e'), lists_cap(sp, 'f')) for nm, sp in SPS.items() if nm[0] in 'IP'}
    def day_pnl(L): return np.array([sum(t[0][0] * t[1] for t in z) for z in L])
    OUT = {}
    for nm, (LE_, LF_) in VR.items():
        dp = pd.Series(day_pnl(LF_)); f = F.copy()
        f['mom5'] = dp.rolling(5).sum().shift(1).values; f['mom10'] = dp.rolling(10).sum().shift(1).values; f['mom20'] = dp.rolling(20).sum().shift(1).values
        f['dow'] = pd.DatetimeIndex(f.day).dayofweek; f['month'] = pd.DatetimeIndex(f.day).month; f['year'] = pd.DatetimeIndex(f.day).year
        rows = []
        for s_ in range(0, NS - 1, 2):
            r, k2 = EV(LE_, A, s_, NS, E2, True)
            if r == 'pass': fr, k3, pays = FU(LF_, A, k2 + 1, NS, F0, True)
            else: pays = []; k3 = k2
            rows.append(dict(s=s_, res=r, end=k2, npay=len(pays), net=sum(x for _, x in pays) - fee, paid=len(pays) > 0, passed=(r == 'pass'), dpass=(k2 - s_ + 1) if r == 'pass' else np.nan, dpay=(pays[0][0] - s_ + 1) if pays else np.nan))
        T = pd.DataFrame(rows).set_index('s'); T = T.join(f)
        # eval momentum: success rate of evals that FINISHED (pass or fail) in the 60 trading days before this start
        fin = T[T.res.isin(['pass', 'fail'])]; mom = []
        for s_ in T.index:
            w = fin[(fin.end < s_) & (fin.end >= s_ - 60)]; mom.append(w.passed.mean() if len(w) >= 5 else np.nan)
        T['eval_mom'] = mom; OUT[nm] = T
    import pickle; pickle.dump(OUT, open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/start_out.pkl', 'wb'))
    def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) >= 15 and len(b) >= 15 else np.nan
    for nm, T in OUT.items():
        T = T[T.index < NS - 120].copy(); print(f'\n##### {nm}: {len(T)} eval starts (every 2nd day, {len(T)} of them). outcome = net $/eval | pass % | paid-once % | median days to 1st payout')
        def line(lab, g): return f'{lab:34} n {len(g):4} | net ${g.net.mean():+6,.0f} | pass {g.passed.mean():4.0%} | paid {g.paid.mean():4.0%} | d1pay {g.dpay.median():4.0f}'
        print('-- by era'); 
        for lab, g in (('first 1500 days', T[T.index < 1500]), ('last 1500 days', T[T.index >= 1500]), ('2018-21', T[(T.year >= 2018) & (T.year <= 2021)]), ('2022-25', T[(T.year >= 2022) & (T.year <= 2025)])): print('  ' + line(lab, g))
        print('-- by feature at start (terciles for continuous; first/last 1500 days shown as net $)')
        for col in ('vxn_z', 'vxn_pct', 'qqq_z20', 'qqq_ret5', 'rng20_pct', 'mom5', 'mom10', 'mom20', 'eval_mom'):
            g = T.dropna(subset=[col]); 
            if len(g) < 100 or g[col].nunique() < 4: continue
            q = np.nanquantile(g[col], [1 / 3, 2 / 3]); lo, mid, hi = g[g[col] <= q[0]], g[(g[col] > q[0]) & (g[col] < q[1])], g[g[col] >= q[1]]
            h1 = lambda x: x[x.index < 1500].net.mean(); h2 = lambda x: x[x.index >= 1500].net.mean()
            print(f'  {col:10} LOW  {line("", lo)} | halves {h1(lo):+5,.0f}/{h2(lo):+5,.0f}'); print(f'  {"":10} MID  {line("", mid)} | halves {h1(mid):+5,.0f}/{h2(mid):+5,.0f}'); print(f'  {"":10} HIGH {line("", hi)} | halves {h1(hi):+5,.0f}/{h2(hi):+5,.0f}   hi-lo t {tt(hi.net, lo.net):+.2f}')
        g = T.dropna(subset=['vxn_hi'])
        for v, lab in ((1.0, 'VXN >= 1y median'), (0.0, 'VXN < 1y median')): x = g[g.vxn_hi == v]; print('  ' + line(lab, x) + f' | halves {x[x.index < 1500].net.mean():+5,.0f}/{x[x.index >= 1500].net.mean():+5,.0f}')
        print('-- day of week at start (0=Mon)'); [print('  ' + line(f'dow {d}', T[T.dow == d])) for d in range(5)]
        print('-- month at start'); [print('  ' + line(f'month {m:2}', T[T.month == m])) for m in range(1, 13)]
        print('-- by year'); [print('  ' + line(f'{y}', T[T.year == y])) for y in sorted(T.year.unique())]
        g = T.dropna(subset=['eval_mom']); print(f'-- EVAL MOMENTUM: corr(prior-60d eval success rate, this eval net) {np.corrcoef(g.eval_mom, g.net)[0,1]:+.2f} (n {len(g)}); first 1500 {np.corrcoef(g[g.index<1500].eval_mom, g[g.index<1500].net)[0,1]:+.2f} | last 1500 {np.corrcoef(g[g.index>=1500].eval_mom, g[g.index>=1500].net)[0,1]:+.2f}')
        print('   next-start outcome after a PASS vs after a FAIL (adjacent starts, 2 days apart, highly overlapping): pass%', f'{T.passed.shift(-1)[T.passed].mean():.0%} vs {T.passed.shift(-1)[~T.passed].mean():.0%}')
