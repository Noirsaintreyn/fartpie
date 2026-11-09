"""Nov 9 2026: STATE SIZING + CONFIDENCE MICROS + TRAILING EXITS ON RULE A (Rey). DECLARED BEFORE RESULTS.
Base: the current 50K setup (V2 + skip slow flips). State sizing from xs/state_size.py: eval E2 (x clip(c^2, 0.3, 1), c = cushion / $2,000, chosen on 2018-21),
funded F0 (declared) or F1 (x clip(c, 0.4, 1), exploratory).
CONFIDENCE MICROS (Rule A): score = fast move into the flip (>= top-third cut, xs/flip_char.py) + flip-bar volume above its 20-bar EMA x 0.82 (top third,
xs/updown_vol.py); score 0 / 1 / 2 -> 5 / 10 / 15 micros-equivalent (x0.5 / x1 / x1.5 on the current HAR-scaled size; cap 1.75 NQ stays).
TRAILING instead of the fixed 75 target (Rule A; stop 50 to start, R = HAR forecast of the day's range, flat 15:55 instead of 12:00):
  T1 vol trail: once +0.15 R, stop trails the best price by 0.12 R | T3 NHP: once +0.15 R, stop to break-even and out when 4 5m bars pass without a new extreme |
  T4 chandelier: once +25 pts, stop trails the best price by 3 x the 5m ATR(14) | T5 range left (Rey's '60 range, down 20, up 10 -> 30 more'): target = the day's
  low so far + R (longs; mirror shorts) instead of 75, stop to break-even once +0.15 R.
The trailing exit used below is CHOSEN on 2018-21 points per trade, judged on 2022-25 and the fleets."""
_w = open('/home/claude/fartpie/xs/winners.py').read(); exec(_w[:_w.rindex("if __name__ == '__main__':")])
_s = open('/home/claude/fartpie/xs/state_size.py').read(); exec(_s[_s.index('def eval_p('):_s.rindex("if __name__ == '__main__':")])
_u = open('/home/claude/fartpie/xs/updown_vol.py').read(); exec(_u[_u.index("vol = (H['volume']"):_u.index('def path_v(')])
tr5 = np.maximum.reduce([h - l, np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])]); atr5 = pd.Series(tr5).rolling(14).mean().shift(1).values
def path_t(d, i, dr, kind, R):
    ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; S = e - dr * 50; best = e; worst = 0.0; mfe = 0.0; npb = 0
    lo0, hi0 = (l[ix[:j]].min(), h[ix[:j]].max()) if j > 0 else (e, e); T = (lo0 + R) if dr > 0 else (hi0 - R)
    if kind == 'T0': T = e + dr * 75
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= (720 if kind == 'T0' else 955): return (c[ix[p - 1]] - e) * dr - 1.0, worst - 1.0
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): return (S - e) * dr - 1.0, min(worst, (S - e) * dr) - 1.0
        worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr); ext = h[ii] if dr > 0 else l[ii]
        if kind in ('T0', 'T5') and (T - e) * dr > 0 and ((h[ii] >= T) if dr > 0 else (l[ii] <= T)): return (T - e) * dr - 1.0, worst - 1.0
        if (ext - best) * dr > 0: best = ext; npb = 0
        else: npb += 1
        mfe = (best - e) * dr
        if kind == 'T1' and mfe >= .15 * R: S = best - dr * .12 * R if (best - dr * .12 * R - S) * dr > 0 else S
        if kind in ('T3', 'T5') and mfe >= .15 * R and (S - e) * dr < 1: S = e + dr * 1.0
        if kind == 'T4' and mfe >= 25 and np.isfinite(atr5[ii]): S = best - dr * 3 * atr5[ii] if (best - dr * 3 * atr5[ii] - S) * dr > 0 else S
        if kind == 'T3' and mfe >= .15 * R and npb >= 4 and p + 1 < len(ix): return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
    return (c[ix[-1]] - e) * dr - 1.0, worst - 1.0
if __name__ == '__main__':
    V = run_sig2('s_vector'); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0)].copy()
    REF = V[V.day >= '2018-02-01'].join(feats(V[V.day >= '2018-02-01'])); q_ = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30; cut, top = q_.quantile(1 / 3), q_.quantile(2 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    base = []
    for en, (T_, u, yz) in zip(NAMES, variant(True, relm)):
        T_ = T_.copy()
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    iRA = NAMES.index('RA'); RT = base[iRA][0].copy(); RT = RT.join(feats(RT)); RT = RT.join(ent_feats(RT)[['FLIPVOL']])
    RT['score'] = (RT.mv30 >= top).astype(int) + (RT.FLIPVOL >= .82).astype(int); RT['per'] = np.where(pd.to_datetime(RT.day).dt.year >= 2022, 'b', 'a'); live = (RT.sz > 0) & (pd.to_datetime(RT.day) >= '2018-02-01')
    print('Rule A by confidence score: NQ pts per trade (n) 2018-21 / 2022-25', flush=True)
    for sc in (0, 1, 2):
        g_ = RT[live & (RT.score == sc)]; print(f'  score {sc}: {g_[g_.per == "a"].pts.mean():+6.2f} ({(g_.per == "a").sum()}) / {g_[g_.per == "b"].pts.mean():+6.2f} ({(g_.per == "b").sum()})', flush=True)
    print('\nRule A exits: NQ pts per trade 2018-21 / 2022-25 | total 2022-25 (score-weighted size not applied here)', flush=True); EX = {}
    for kind in ('T0', 'T1', 'T3', 'T4', 'T5'):
        out = []
        for r_ in RT.itertuples():
            d = pd.Timestamp(r_.day); ix = gi.get(d); w = np.where(m[ix] == r_.m)[0] if ix is not None else []
            R = FC.get(d, np.nan)
            if not len(w) or not np.isfinite(R) or r_.sz <= 0: out.append((r_.pts, r_.mae)); continue
            out.append(path_t(d, ix[w[0]], r_.dir, kind, R))
        EX[kind] = np.array(out); X = RT.assign(p2=EX[kind][:, 0])[live]
        print(f'  {kind}: {X[X.per == "a"].p2.mean():+6.2f} / {X[X.per == "b"].p2.mean():+6.2f} | {X[X.per == "b"].p2.sum():+6.0f}', flush=True)
    bt = max(('T1', 'T3', 'T4', 'T5'), key=lambda k_: RT.assign(p2=EX[k_][:, 0])[live & (RT.per == 'a')].p2.mean()); print(f'  chosen on 2018-21: {bt}', flush=True)
    def spec(tiers=False, trail=None):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base):
            T_ = T_.copy()
            if en == 'RA':
                if tiers: T_['sz'] = T_.sz * RT.score.map({0: .5, 1: 1.0, 2: 1.5}).values
                if trail: T_['pts'], T_['mae'] = EX[trail][:, 0], EX[trail][:, 1]; T_['m'] = T_['m']
            out.append((T_, u, yz))
        return out
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
    k22, k25 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))); SA, SB = list(range(0, k22, 2)), list(range(k22, k25, 2))
    def run(sp, ep, fp):
        LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
        hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
        def traj(k):
            if k not in TRJ:
                r, k2 = eval_p(LE_, A, k, NS, ep)
                if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fund_p(LF_, A, k2 + 1, NS, fp); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
            return TRJ[k]
        pe = {}
        for nm_, S_ in (('a', SA), ('b', SB)):
            tr_ = [traj(s) for s in S_]; pe[nm_] = (np.mean([t[0] == 'pass' for t in tr_]), np.mean([len(t[2]) > 0 for t in tr_]), np.mean([sum(x for _, x in t[2]) for t in tr_]) - fee)
        allv = []; f18 = None
        for k0 in [int(np.searchsorted(D_, p.start_time)) for p in pd.period_range('2022-01', '2023-12', freq='M')] + [0]:
            cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; NP = {}
            for k in range(k0, NS):
                mo = D_[k].to_period('M')
                while pq and pq[0][0] == k:
                    kk_, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
                act = [e for e in act if e >= k]
                if len(act) < 5 and cash >= fee and hot[k]:
                    r, end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                    for kk_, x in pays: heapq.heappush(pq, (kk_, x))
            months = pd.period_range(D_[k0].to_period('M'), D_[NS - 2].to_period('M'), freq='M'); N_ = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months})
            if k0 == 0: f18 = (N_, pd.Series({mo: NP.get(mo, 0) for mo in months}))
            else: allv.append(N_.values)
        allv = np.concatenate(allv); s = 0; run_ = mx = 0
        while s < NS - 1:
            r, end, pays = traj(s); run_ = run_ + 1 if not pays else 0; mx = max(mx, run_); s = end + 1
        N_, C_ = f18
        return (f'per eval 18-21 / 22-25: pass {pe["a"][0]:.0%} / {pe["b"][0]:.0%}, paid {pe["a"][1]:.0%} / {pe["b"][1]:.0%}, ${pe["a"][2]:,.0f} / ${pe["b"][2]:,.0f} | dry {mx} | '
                f'24 starts: ${allv.mean():,.0f} / ${np.median(allv):,.0f}, losing {np.mean(allv < 0):.0%} | from 2018: ${N_.mean():,.0f}/mo, losing {np.mean(N_ < 0):.0%}, worst ${N_.min():,.0f}, '
                f'{C_.mean():.2f} payouts/mo, >= 2 in {np.mean(C_ >= 2):.0%} of months')
    E0, E2, F0, F1 = EP['E0 x1'], EP['E2 harder de-risk'], FP['F0 x1'], FP['F1 x clip(c, .4, 1)']
    print('\n=== Lucid Flex 50K ===', flush=True)
    for lab, sp, ep, fp in (('now', spec(), E0, F0), ('E2 + F0 (state sizing, declared)', spec(), E2, F0), ('E2 + F1 (exploratory funded)', spec(), E2, F1),
                            ('E2 + F1 + Rule A confidence micros', spec(True), E2, F1), (f'E2 + F1 + Rule A trail {bt}', spec(False, bt), E2, F1),
                            (f'E2 + F1 + micros + trail {bt}', spec(True, bt), E2, F1), ('now + Rule A confidence micros', spec(True), E0, F0)):
        print(f'  {lab:<40} {run(sp, ep, fp)}', flush=True)
