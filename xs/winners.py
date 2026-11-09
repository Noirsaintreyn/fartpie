"""Nov 9 2026: CAPTURE THE WINNERS - are the exits cutting the big moves short, and do runners / exit-timeframe escalation / pyramiding fix it? (Rey). DECLARED BEFORE RESULTS.
Engines: Rule A (now: stop 50, target 75, flat 12:00) and V15a (now: stop 0.15 ADR, out on the opposite 15m flip, flat 15:55) - the two engines whose money is in big moves.
Both after the slow-flip skip (the current 50K setup). Real NQ 5m path, 1 pt cost per side per unit.
DIAGNOSTICS: average max favourable move (MFE) vs points kept; for trades out before 15:55, the move from the exit to 15:55 on the trade's side (> 0 = exits too early).
VARIANTS (all written before results):
  Rule A  RA-a no noon flat (75 / 50 until 15:55) | RA-b half off at +75, runner: stop to break-even, out on the opposite 15m flip or 15:55 |
          RA-c as b, runner out on the opposite 1h flip | RA-d no target: stop 50 -> break-even after +75, out on the opposite 15m flip or 15:55
  V15a    V-a after +0.25 ADR: stop to break-even and exit only on the opposite 30m flip | V-b same with the 1h flip |
          V-c PYRAMID: add a 2nd unit at +0.15 ADR, both stops to the first entry (break-even), same exits | V-d pyramid + 1h-flip escalation
30m / 1h vector: same settings on RTH 30m / 1h bars from 09:30, carried across days.
PASS: more points per signal in BOTH 2018-21 and 2022-25 -> Lucid Flex 50K on V2 + skip slow flips (per eval 2018-25; fleet from 2018; 24 starts 2022-23).
Pyramid adds go in as their own trades (same exit; their worst point is checked on its own, not combined with the first unit - a small approximation)."""
_fc = open('/home/claude/fartpie/xs/flip_char.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
REG = {15: {np.datetime64(t, 'ns'): r for t, r in zip(B.end.values, sv)}}
for k in (30, 60):
    g = pd.DataFrame({'day': dd, 'b': (m - 570) // k, 'o': o, 'h': h, 'l': l, 'c': c, 't': H.dt.values})
    Bk = g.groupby(['day', 'b']).agg(open=('o', 'first'), high=('h', 'max'), low=('l', 'min'), close=('c', 'last'), last=('t', 'last')).reset_index()
    Bk['end'] = Bk['last'] + np.timedelta64(5, 'm'); REG[k] = {np.datetime64(t, 'ns'): r for t, r in zip(Bk.end.values, NSL['vec'](Bk[['open', 'high', 'low', 'close']]))}
Hd5 = H.dt.values + np.timedelta64(5, 'm')
def path(d, i, dr, S0, T=None, part=0.0, until=955, tf=15, esc=None, be_at=None, add_at=None):
    """returns (pts per signal incl. adds, mae, mfe, exit price, exit minute, add (pts, mae) or None)"""
    ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; S = e - dr * S0; mfe = 0.0; worst = 0.0; tfx = tf; half = False; banked = 0.0; add = None; addw = 0.0
    def done(px, p_):
        run_ = (px - e) * dr - 1.0; tot = banked + (1 - part if half else 1.0) * run_ if half else run_
        ad = None if add is None else ((px - add) * dr - 1.0, addw - 1.0)
        return tot, worst - 1.0, mfe, px, m[ix[min(p_, len(ix) - 1)]], ad
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= until: return done(c[ix[p - 1]], p)
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): worst = min(worst, (S - e) * dr); return done(S, p)
        fav = ((h[ii] if dr > 0 else l[ii]) - e) * dr; worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
        if add is not None: addw = min(addw, ((l[ii] if dr > 0 else h[ii]) - add) * dr)
        mfe = max(mfe, fav)
        if T is not None and not half and fav >= T:
            if part == 0.0: return T - 1.0, worst - 1.0, mfe, e + dr * T, m[ii], None
            half = True; banked = part * (T - 1.0); S = e + dr * 1.0
        if be_at is not None and mfe >= be_at: S = e + dr * 1.0 if (S - e) * dr < 1.0 else S
        if esc is not None and mfe >= esc[0]: tfx = esc[1]; S = e + dr * 1.0 if (S - e) * dr < 1.0 else S
        if add_at is not None and add is None and mfe >= add_at: add = e + dr * add_at; addw = 0.0; S = e + dr * 1.0 if (S - e) * dr < 1.0 else S
        if tfx is not None and p + 1 < len(ix):
            r_ = REG[tfx].get(np.datetime64(Hd5[ii], 'ns'))
            if r_ is not None and r_ == -dr: return done(o[ix[p + 1]], p + 1)
    return done(c[ix[-1]], len(ix) - 1)
if __name__ == '__main__':
    V = run_sig2('s_vector'); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0)].copy()
    REF = V[V.day >= '2018-02-01'].join(feats(V[V.day >= '2018-02-01'])); cut = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30.quantile(1 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    base = []
    for en, (T_, u, yz) in zip(NAMES, variant(True, relm)):
        T_ = T_.copy()
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    BE = dict(zip(NAMES, base)); RES = {}
    VAR = {'RA': {'now': dict(S0=50, T=75, until=720, tf=None), 'RA-a no noon flat': dict(S0=50, T=75, until=955, tf=None),
                  'RA-b half at 75, runner 15m flip': dict(S0=50, T=75, part=.5, until=955, tf=15), 'RA-c half at 75, runner 1h flip': dict(S0=50, T=75, part=.5, until=955, tf=60),
                  'RA-d no target, BE after 75, 15m flip': dict(S0=50, until=955, tf=15, be_at=75)},
           'V15a': {'now': dict(tf=15), 'V-a escalate to 30m flip at +0.25 ADR': dict(tf=15, esc=(.25, 30)), 'V-b escalate to 1h flip at +0.25 ADR': dict(tf=15, esc=(.25, 60)),
                    'V-c pyramid at +0.15 ADR': dict(tf=15, add_at=.15), 'V-d pyramid + 1h escalation': dict(tf=15, add_at=.15, esc=(.25, 60))}}
    for en, vs in VAR.items():
        T_ = BE[en][0]; T_ = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')].copy(); T_['per'] = np.where(pd.to_datetime(T_.day).dt.year >= 2022, 'b', 'a')
        print(f'\n=== {en}: {len(T_)} trades | points per signal 2018-21 / 2022-25 | total 2022-25 | avg MFE | kept / MFE on winners | exit -> 15:55 on the trade side ===', flush=True)
        for vn, kw in vs.items():
            kw = dict(kw); adr_based = en == 'V15a'; out = []
            for r_ in T_.itertuples():
                d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
                if not len(w): out.append((np.nan,) * 7); continue
                a_ = DY.adr[d]; k2 = dict(kw)
                if adr_based:
                    k2['S0'] = .15 * a_
                    if 'esc' in k2: k2['esc'] = (k2['esc'][0] * a_, k2['esc'][1])
                    if 'add_at' in k2: k2['add_at'] = k2['add_at'] * a_
                pt, mae, mfe, xp, xm, ad = path(d, ix[w[0]], r_.dir, **k2); post = (c[ix[-1]] - xp) * r_.dir if xm < 950 else np.nan
                out.append((pt + (ad[0] if ad else 0.0), mae, mfe, post, ad[0] if ad else np.nan, ad[1] if ad else np.nan, pt))
            Q = pd.DataFrame(out, columns=['tot', 'mae2', 'mfe', 'post', 'addp', 'addw', 'p1'], index=T_.index); X = T_.join(Q); RES[(en, vn)] = X
            a, b = X[X.per == 'a'], X[X.per == 'b']; win = X[X.p1 > 0]
            print(f'  {vn:<40} {a.tot.mean():+6.2f} / {b.tot.mean():+6.2f} | {b.tot.sum():+6.0f} | MFE {X.mfe.mean():5.1f} | kept {win.p1.mean() / win.mfe.mean():4.0%} | post-exit {X.post.mean():+5.1f} (n{X.post.notna().sum()})', flush=True)
    PASS = []
    for en, vs in VAR.items():
        n0 = RES[(en, 'now')]
        for vn in vs:
            if vn == 'now': continue
            X = RES[(en, vn)]
            if all(X[X.per == p_].tot.mean() > n0[n0.per == p_].tot.mean() for p_ in 'ab'): PASS.append((en, vn))
    print(f'\npasses: {PASS if PASS else "none"}', flush=True)
    if PASS:
        LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
        def apply(chg):
            out = []
            for en, (T_, u, yz) in zip(NAMES, base):
                T_ = T_.copy(); extra = None
                if en in chg:
                    X = chg[en]; mp = {i_: (p_, w_) for i_, p_, w_ in zip(X.index, X.p1, X.mae2) if np.isfinite(p_)}
                    T_['pts'] = [mp.get(i_, (p0, w0))[0] for i_, p0, w0 in zip(T_.index, T_.pts, T_.mae)]; T_['mae'] = [mp.get(i_, (p0, w0))[1] for i_, p0, w0 in zip(T_.index, T_.pts, T_.mae)]
                    ad = X[X.addp.notna()]
                    if len(ad): extra = T_.loc[ad.index].assign(pts=ad.addp.values, mae=ad.addw.values, m=T_.loc[ad.index, 'm'].values + 1)
                out.append((T_, u, yz))
                if extra is not None: out.append((extra, u, yz))
            return out
        combos = [('V2 + skip slow flips (now)', base)] + [(f'{en} {vn}', apply({en: RES[(en, vn)]})) for en, vn in PASS]
        bra = [p_ for p_ in PASS if p_[0] == 'RA']; bv = [p_ for p_ in PASS if p_[0] == 'V15a']
        if bra and bv:
            ka = max(bra, key=lambda t: RES[t].tot.mean()); kv = max(bv, key=lambda t: RES[t].tot.mean()); combos.append((f'both: {ka[1]} + {kv[1]}', apply({'RA': RES[ka], 'V15a': RES[kv]})))
        print('\n=== Lucid Flex 50K ===', flush=True)
        for lab, sp in combos:
            LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
            def traj(k):
                if k not in TRJ:
                    r, k2 = eval_x(LE_, A, k, NS, 1000)
                    if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                    else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
                return TRJ[k]
            tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]; ppe = np.mean([len(t[2]) for t in tr_]); paid = np.mean([len(t[2]) > 0 for t in tr_])
            s = 0; chain = []
            while s < NS - 1:
                r, end, pays = traj(s); chain.append(pays); s = end + 1
            run = mx = 0
            for p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
            hot = (pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in LE_]).rolling(20).sum().shift(1) > 0).values
            cash = 40 * fee; act = []; pq = []; P = {}; Fe = {}; NP = {}; months = pd.period_range(D_[0].to_period('M'), D_[NS - 2].to_period('M'), freq='M')
            for k in range(0, NS):
                mo = D_[k].to_period('M')
                while pq and pq[0][0] == k:
                    kk_, x = heapq.heappop(pq); cash += x; P[mo] = P.get(mo, 0) + x; NP[mo] = NP.get(mo, 0) + 1
                act = [e for e in act if e >= k]
                if len(act) < 5 and cash >= fee and hot[k]:
                    r, end, pays = traj(k); cash -= fee; act.append(end); Fe[mo] = Fe.get(mo, 0) + fee
                    for kk_, x in pays: heapq.heappush(pq, (kk_, x))
            N_ = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C = pd.Series({mo: NP.get(mo, 0) for mo in months}); fl = multi_x({'x': sp}, '50K')
            print(f'  {lab[:58]:<58} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {paid:3.0%}, evals/payout {1 / ppe:4.2f}, dry {mx} | fleet 2018+: {C.mean():.2f} pay/mo, '
                  f'losing {np.mean(N_ < 0):3.0%}, ${N_.mean():,.0f}/mo | 24 starts 2022-23: avg ${fl["avg"]:,.0f} / med ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
