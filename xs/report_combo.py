"""COMBO: STK-N4 (tier) x ECON tilt, eval and funded separately. MIX CHAIN (Nov 11 2026): top family mixes from report_mix_screen (tiers: bottom third x0.5, top third x1.5, thirds cut on era A) through the Lucid Flex 50K chain, eval and funded separately, all starts and 2022+ starts only. Scored legs V15a, RA; other legs untouched."""
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
    # ------------------------------------------------------------------ INTRADAY STAND-DOWN RULES on top of Z (skip small-gap days); Flex 50K chain -------------------------------
    import itertools
    c01 = lambda x, lo, hi: float(np.clip(x, lo, hi)); FPOL = lambda c, n: c01(c, .5, 1.5)
    starts = list(range(k18, k25_, 2)); yrs = np.array([D_[x].year for x in starts]); H2 = yrs >= 2022
    def intr(d, T):
        """features known by clock time T (minutes since midnight ET): VWAP crosses, first noise break, failed break"""
        g = Gd.get(d)
        if g is None: return None
        g = g[(g.index + 5) <= T]; c = g.c.values; vw = g.vw.values; sg_ = np.sign(c - vw); sg_ = sg_[(~np.isnan(sg_)) & (sg_ != 0)]
        ncross = int(np.sum(np.diff(sg_) != 0)) if len(sg_) > 1 else 0
        sig = SIGd.loc[d]; o = Od[d]; pc = PCd[d] if np.isfinite(PCd[d]) else o; st = []
        for t in g.index:
            if (t + 5) % 30 != 0 or t + 5 < 600 or not np.isfinite(sig.get(t, np.nan)): continue
            cc = g.c.loc[t]; ub, lb = max(o, pc) * (1 + sig[t]), min(o, pc) * (1 - sig[t]); st.append(1 if cc > ub else (-1 if cc < lb else 0))
        brk = [x for x in st if x != 0]; first = brk[0] if brk else 0
        failed = False
        if first != 0:
            idx = st.index(first); failed = idx + 1 < len(st) and st[idx + 1] != first
        return dict(ncross=ncross, nobreak=(len(brk) == 0), failed=failed)
    DAYS = sorted(d for d in Gd if pd.Timestamp('2014-02-01') <= d <= pd.Timestamp('2026-01-30'))
    IF = {T: {d: intr(d, T) for d in DAYS} for T in (630, 660, 690)}
    RULES_ = {'none (Z)': None,
              'X1 >=3 VWAP crosses by 11:00 -> x0': (660, lambda f: f['ncross'] >= 3, 0.0), 'X1h same -> x0.5': (660, lambda f: f['ncross'] >= 3, 0.5),
              'X2 no noise break by 11:00 -> x0': (660, lambda f: f['nobreak'], 0.0), 'X2h same -> x0.5': (660, lambda f: f['nobreak'], 0.5),
              'X2b no break by 10:30 -> x0.5': (630, lambda f: f['nobreak'], 0.5),
              'X3 crosses>=3 OR no break by 11:00 -> x0': (660, lambda f: f['ncross'] >= 3 or f['nobreak'], 0.0), 'X3h same -> x0.5': (660, lambda f: f['ncross'] >= 3 or f['nobreak'], 0.5),
              'X4 first break failed by 11:00 -> x0.5': (660, lambda f: f['failed'], 0.5), 'X4b first break failed by 11:30 -> x0': (690, lambda f: f['failed'], 0.0),
              'X5 >=5 VWAP crosses by 11:30 -> x0': (690, lambda f: f['ncross'] >= 5, 0.0)}
    def gate(spec, rule):
        if rule is None: return spec
        T, fn, mult = rule; out = []
        for (T_, u, yz), en in zip(spec, NAMES):
            if en in ('RA', 'M1', 'R1', 'RULES', 'V15a'):
                T_ = T_.copy(); d = pd.to_datetime(T_.day).values; fl = np.array([bool(IF[T].get(pd.Timestamp(x)) and fn(IF[T][pd.Timestamp(x)])) for x in d])
                T_['sz'] = T_.sz.values * np.where(fl & (T_.m.values >= T), mult, 1.0)
            out.append((T_, u, yz))
        return out
    def chain(spec):
        LE, LF = lists_cap(spec, 'e'), lists_cap(spec, 'f'); rows = []
        for s_ in starts:
            r, k2 = EV(LE, A, s_, NS, E2, True)
            if r != 'pass': rows.append(dict(out='evalfail' if r == 'fail' else 'open', t=k2 - s_ + 1, net=-fee, npay=0)); continue
            fr, k3, pays = FU(LF, A, k2 + 1, NS, FPOL, True)
            rows.append(dict(out='paid' if pays else ('fundedlost' if fr == 'lost' else 'open'), t=(pays[0][0] - s_ + 1) if pays else np.nan, net=sum(x for _, x in pays) - fee, npay=len(pays)))
        return pd.DataFrame(rows)
    def line(R):
        return (f"pass {1-np.mean(R.out=='evalfail')-np.mean(R.out=='open'):4.0%} evalfail {np.mean(R.out=='evalfail'):4.0%} paid {np.mean(R.out=='paid'):4.0%} fundedlost {np.mean(R.out=='fundedlost'):4.0%} | 1st payout median {R.t.median():3.0f}d | payouts/eval {R.npay.mean():.2f} net/eval ${R.net.mean():,.0f}")
    Z0 = gapmul(S6, 0.0, 1.0) + [(NZ, 0.0, None)]
    X1 = (660, lambda f: f['ncross'] >= 3, 0.0)
    def mod(spec, fn):
        out = []
        for (T_, u, yz), en in zip(spec, NAMES):
            T2 = T_.copy(); u2 = fn(en, T2, u); out.append((T2, u2 if u2 is not None else u, yz))
        return out
    def noON_minus(en, T_, u): return 0.0 if en == 'ON-' else None
    def noON_plus_tue(en, T_, u):
        if en == 'ON+': T_['sz'] = T_.sz.values * np.where(pd.to_datetime(T_.day).dt.dayofweek.values == 1, 0.0, 1.0)
        return None
    def noM1_late(en, T_, u):
        if en == 'M1': T_['sz'] = T_.sz.values * np.where(T_.m.values >= 660, 0.0, 1.0)
        return None
    def noRTH_after12(en, T_, u):
        if en not in ('ON+', 'ON-') and en != 'NZ': T_['sz'] = T_.sz.values * np.where(T_.m.values >= 720, 0.0, 1.0)
        return None
    def comb(*fs):
        def f(en, T_, u):
            r = None
            for g in fs:
                x = g(en, T_, u)
                if x is not None: r = x
            return r
        return f
    BASE = mod(gate(Z0, X1), noON_minus)           # stack: skip small-gap + 3-cross gate + no ON-
    def lists_cap2(spec, phase, capu):
        L = [[] for _ in range(NS)]
        for (T_, u, yzrule), en in zip(spec, NAMES):
            if u <= 0: continue
            for r_ in T_.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * r_.sz * (r_.ce if phase == 'e' else r_.cf)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m if en not in ('ON+', 'ON-') else -1, ((min(uu, capu),), r_.pts, r_.mae, (r_.m, END[en]))))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    def lev(spec, legs=None, mult=1.0, hi_gap=None, early=None):
        """multiply size on: given legs; top-third |gap| days (RTH legs); RTH entries before minute `early`"""
        out = []
        for (T_, u, yz), en in zip(spec, NAMES):
            T2 = T_.copy(); f = np.ones(len(T2))
            rth = en in ('RA', 'M1', 'R1', 'RULES', 'V15a')
            if legs is not None and en in legs: f = f * mult
            if legs is None and rth and hi_gap is None and early is None: f = f * mult
            if hi_gap is not None and rth:
                r = _gr.reindex(_pd.to_datetime(T2.day)).values; f = f * np.where(r >= .67, hi_gap, 1.0)
            if early is not None and rth: f = f * np.where(T2.m.values < early, mult, 1.0)
            T2['sz'] = T2.sz.values * f; out.append((T2, u, yz))
        return out
    LF_base = lists_cap2(BASE, 'f', 1.75)
    def chain2(spec_e, capu):
        LE = lists_cap2(spec_e, 'e', capu); rows = []
        for s_ in starts:
            r, k2 = EV(LE, A, s_, NS, E2, True)
            if r != 'pass': rows.append(dict(out='evalfail' if r == 'fail' else 'open', t=k2 - s_ + 1, net=-fee, npay=0)); continue
            fr, k3, pays = FU(LF_base, A, k2 + 1, NS, FPOL, True)
            rows.append(dict(out='paid' if pays else ('fundedlost' if fr == 'lost' else 'open'), t=(pays[0][0] - s_ + 1) if pays else np.nan, net=sum(x for _, x in pays) - fee, npay=len(pays)))
        return pd.DataFrame(rows)
    def line2(R):
        return (f"pass {1-np.mean(R.out=='evalfail')-np.mean(R.out=='open'):4.0%} evalfail {np.mean(R.out=='evalfail'):4.0%} paid {np.mean(R.out=='paid'):4.0%} fundedlost {np.mean(R.out=='fundedlost'):4.0%} | 1st payout median {R.t.median():3.0f}d <=45d {np.mean((R.out=='paid')&(R.t<=45)):4.0%} <=60d {np.mean((R.out=='paid')&(R.t<=60)):4.0%} | net/eval ${R.net.mean():,.0f}")
    def noON_plus(en, T_, u): return 0.0 if en == 'ON+' else None
    NOON = mod(BASE, noON_plus)
    def only(keep):
        def f(en, T_, u): return u if en in keep else 0.0
        return mod(NOON, f)
    def chain_s(spec):
        LE, LF = lists_cap2(spec, 'e', 1.75), lists_cap2(spec, 'f', 1.75); rows = []
        for s_ in starts:
            r, k2 = EV(LE, A, s_, NS, E2, True)
            if r != 'pass': rows.append(dict(out='evalfail' if r == 'fail' else 'open', dp=np.nan, t=k2 - s_ + 1, net=-fee, npay=0)); continue
            fr, k3, pays = FU(LF, A, k2 + 1, NS, FPOL, True)
            rows.append(dict(out='paid' if pays else ('fundedlost' if fr == 'lost' else 'open'), dp=k2 - s_ + 1, t=(pays[0][0] - s_ + 1) if pays else np.nan, net=sum(x for _, x in pays) - fee, npay=len(pays)))
        return pd.DataFrame(rows)
    SCp = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
    FE = {}
    for nm_ in ('V15a', 'RA'):
        a = pd.read_pickle(SCp + f'ict_{nm_}.pkl').set_index(['day', 'm']); b = pd.read_pickle(SCp + f'po3htf_{nm_}.pkl').set_index(['day', 'm']); c = pd.read_pickle(SCp + f'po3sess_{nm_}.pkl').set_index(['day', 'm'])
        F = a[['disp', 'mss', 'adx', 'rsi', 'smt', 'era']].join(b[['HTF', 'n_with']], how='left').join(c[['AL', 'al']], how='left'); F = F[~F.index.duplicated()]
        FE[nm_] = F
    ADXC = {k: FE[k][FE[k].era == 'A'].adx.quantile(1/3) for k in FE}; RSIC = {k: FE[k][FE[k].era == 'A'].rsi.quantile(1/3) for k in FE}
    def vmod(spec, rules):
        """rules: {leg: fn(F)->multiplier array}; applied on the leg's trade table by (day,m)."""
        out = []
        for (T_, u, yz), en in zip(spec, NAMES):
            if en in rules:
                T2 = T_.copy(); F = FE[en]; key = list(zip(pd.to_datetime(T2.day), T2.m.astype(int)))
                Fi = F.reindex(key); mult = np.asarray(rules[en](Fi), float); mult = np.where(np.isnan(mult), 1.0, mult)
                T2['sz'] = T2.sz.values * mult; out.append((T2, u, yz))
            else: out.append((T_, u, yz))
        return out
    sk_ds = lambda F: np.where((F.disp == -1) | (F.mss == -1), 0.0, 1.0)
    sk_adx = lambda F: np.where(F.adx <= ADXC['V15a'], 0.0, 1.0)
    for nm_ in ('V15a', 'RA'):
        N_ = pd.read_pickle(SCp + f'narr_{nm_}.pkl').set_index(['day', 'm']); N_ = N_[~N_.index.duplicated()]
        FE[nm_] = FE[nm_].join(N_[['score']], how='left')
    ALLRTH = {'RA', 'M1', 'R1', 'RULES', 'V15a'}; BASEP = only(ALLRTH)
    def rule(lo_m, hi_m, lo=2, hi=5): return lambda F: np.where(F.score <= lo, lo_m, np.where(F.score >= hi, hi_m, 1.0))
    ADXq = lambda F: np.where(F.adx <= ADXC['V15a'], 0.0, 1.0)
    for nm_ in ('V15a', 'RA'):
        S_ = pd.read_pickle(SCp + f'stack_{nm_}.pkl').set_index(['day', 'm']); S_ = S_[~S_.index.duplicated()]
        FE[nm_] = FE[nm_].join(S_[['tier', 'H', 'M', 'E', 'lH', 'lM', 'lE']], how='left'); FE[nm_]['me'] = FE[nm_].lM.astype(float) + FE[nm_].lE.astype(float)
    def both(f): return {'V15a': f, 'RA': f}
    def eval_stats(spec, pol=E2):
        LE = lists_cap2(spec, 'e', 1.75); rows = []
        for s_ in starts:
            r, k2 = EV(LE, A, s_, NS, pol, True); rows.append((r, k2 - s_ + 1))
        R = pd.DataFrame(rows, columns=['res', 'dur']); n = len(R); p = R[R.res == 'pass']
        out = dict(n=n, pass_=np.mean(R.res == 'pass'), fail=np.mean(R.res == 'fail'), open_=np.mean(R.res == 'open'))
        for T in (30, 60, 90): out[f'pass<={T}'] = np.mean((R.res == 'pass') & (R.dur <= T)); out[f'run>{T}'] = np.mean(R.dur > T)
        out['months'] = np.mean(np.ceil(R.dur / 21)); out['med'] = p.dur.median(); return out, R
    def fund_stats(spec):
        LF = lists_cap2(spec, 'f', 1.75); rows = []
        for s_ in starts:
            fr, k3, pays = FU(LF, A, s_, NS, FPOL, True)
            ds = [k - s_ + 1 for k, _ in pays]; amt = [x for _, x in pays]
            rows.append(dict(lost0=(fr == 'lost' and not pays), lost=(fr == 'lost'), paid=bool(pays), t1=ds[0] if ds else np.nan, n90=sum(d <= 90 for d in ds), n180=sum(d <= 180 for d in ds), d90=sum(a for d, a in zip(ds, amt) if d <= 90), d180=sum(a for d, a in zip(ds, amt) if d <= 180), tot=sum(amt), npay=len(pays)))
        R = pd.DataFrame(rows); out = dict(paid=R.paid.mean(), lost0=R.lost0.mean(), lost=R.lost.mean())
        for T in (30, 60, 90): out[f'1st<={T}'] = np.mean(R.t1 <= T)
        out.update(t1med=R.t1.median(), n90=R.n90.mean(), n180=R.n180.mean(), d90=R.d90.mean(), d180=R.d180.mean(), tot=R.tot.mean(), npay=R.npay.mean()); return out, R
    def pe(o): return (f"EVAL  pass {o['pass_']:4.0%} FAIL {o['fail']:4.0%} | pass<=30d {o['pass<=30']:4.0%} <=60d {o['pass<=60']:4.0%} <=90d {o['pass<=90']:4.0%} | still running >30d {o['run>30']:4.0%} >60d {o['run>60']:4.0%} >90d {o['run>90']:4.0%} | fee-months/eval {o['months']:.2f}")
    def pf(o): return (f"FUNDED paid {o['paid']:4.0%} lost-before-1st {o['lost0']:4.0%} lost-ever {o['lost']:4.0%} | 1st payout <=30d {o['1st<=30']:4.0%} <=60d {o['1st<=60']:4.0%} <=90d {o['1st<=90']:4.0%} (median {o['t1med']:.0f}d) | payouts in 90d {o['n90']:.2f} / 180d {o['n180']:.2f} | $ in 90d {o['d90']:,.0f} / 180d {o['d180']:,.0f} | $ total {o['tot']:,.0f}")
    N4R = both(lambda F: np.where(F.tier.isna(), 1.0, np.where(F.tier <= 1, .5, np.where(F.tier == 3, 1.5, 1.0))))
    import pickle
    MT = pickle.load(open(SCp + 'mix_tiers.pkl', 'rb')); R0 = MT['rows']; KEYS = []
    for key, v in MT['tops'].items():
        sg = np.where(v['top'], 1, np.where(v['bot'], -1, 0)); KEYS.append(key)
        for lg in ('V15a', 'RA'):
            m_ = (R0.leg == lg).values; s_ = pd.Series(sg[m_], index=pd.MultiIndex.from_arrays([pd.to_datetime(R0.day[m_]), R0.m[m_].astype(int)]))
            s_ = s_[~s_.index.duplicated()]; FE[lg]['mx_' + key] = s_.reindex(FE[lg].index)
    def mxrule(key, lo=.5, hi=1.5): return both(lambda F: np.where(F['mx_' + key].isna(), 1.0, np.where(F['mx_' + key] == -1, lo, np.where(F['mx_' + key] == 1, hi, 1.0))))
    VARS2 = {'base': {}, 'STK-N4 (reference)': N4R}
    ONLY = __import__('os').environ.get('MIXONLY')
    for k in KEYS:
        if ONLY is None or k in ONLY.split(','): VARS2['MIX ' + k] = mxrule(k)
    if ONLY: VARS2 = {k: v for k, v in VARS2.items() if k.startswith('MIX')}

    n4 = lambda F: np.where(F.tier.isna(), 1.0, np.where(F.tier <= 1, .5, np.where(F.tier == 3, 1.5, 1.0)))
    ecA = lambda F: np.where(F['mx_ECON'].isna(), 1.0, np.where(F['mx_ECON'] == -1, .5, np.where(F['mx_ECON'] == 1, 1.5, 1.0)))
    ecB = lambda F: np.where(F['mx_ECON'].isna(), 1.0, np.where(F['mx_ECON'] == -1, .75, np.where(F['mx_ECON'] == 1, 1.25, 1.0)))
    SPECS = {'N4': both(n4), 'ECON only': both(ecA), 'N4 x ECON(.5/1.5)': both(lambda F: n4(F) * ecA(F)), 'N4 x ECON(.75/1.25)': both(lambda F: n4(F) * ecB(F))}
    def l2(Re, Rf):
        e2 = Re[H2]; f2 = Rf[H2]
        return f'   2022+ only: eval pass {np.mean(e2.res=="pass"):.0%} fail {np.mean(e2.res=="fail"):.0%} pass<=30/60/90d {np.mean((e2.res=="pass")&(e2.dur<=30)):.0%}/{np.mean((e2.res=="pass")&(e2.dur<=60)):.0%}/{np.mean((e2.res=="pass")&(e2.dur<=90)):.0%} | funded lost-ever {f2.lost.mean():.0%} 1st payout<=30/60/90d {np.mean(f2.t1<=30):.0%}/{np.mean(f2.t1<=60):.0%}/{np.mean(f2.t1<=90):.0%} total ${f2.tot.mean():,.0f}'
    print('=== COMBO. Lucid Flex 50K. Each variant applied to BOTH phases (eval and funded lines are separate sims). Then eval-only K sweep on N4, and funded-only variants.', flush=True)
    for nm, rules in SPECS.items():
        V = vmod(BASEP, rules); oe, Re = eval_stats(V); of, Rf = fund_stats(V)
        print(f'{nm}\n   {pe(oe)}\n   {pf(of)}\n{l2(Re, Rf)}', flush=True)
    VN4 = vmod(BASEP, SPECS['N4'])
    for K in (1.25, 1.5):
        oe, Re = eval_stats(VN4, lambda cu, pr, K=K: K * E2(cu, pr)); print(f'EVALK N4 K={K}\n   {pe(oe)}', flush=True)
        for nm in ('N4 x ECON(.5/1.5)', 'N4 x ECON(.75/1.25)'):
            oe, Re = eval_stats(vmod(BASEP, SPECS[nm]), lambda cu, pr, K=K: K * E2(cu, pr)); print(f'EVALK {nm} K={K}\n   {pe(oe)}', flush=True)
