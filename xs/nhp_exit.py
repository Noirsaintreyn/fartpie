"""Nov 9 2026: NHP ('A HIGH IS FORMED') + VOLUME EXITS ON THE PROP ALGO (Rey; the screener exit from xs/high_formed.py moved to NQ). DECLARED BEFORE RESULTS.
NHP-k: once the trade is armed, exit at the next 5m open when k bars have closed since the trade's best bar without a new extreme (long: the running high is followed by
k lower highs = a confirmed pivot high; short: mirror). k = 2, 4, 6 (10 / 20 / 30 minutes).
+VOL: only when that best bar was a volume climax (its volume >= 1.5 x the average of the 20 bars before it) - the exhaustion read.
Armed once the open profit has reached +25 pts (Rule A) / +0.10 ADR (V15a). Everything else as now: Rule A stop 50 / target 75 / flat 12:00; V15a stop 0.15 ADR,
opposite 15m flip, flat 15:55. Trades = the current 50K setup (V2 + skip slow flips). 1 pt cost.
PASS: more points per trade in BOTH 2018-21 and 2022-25 -> Lucid Flex 50K (per eval 2018-25, fleet from 2018, 24 starts 2022-23)."""
_w = open('/home/claude/fartpie/xs/winners.py').read(); exec(_w[:_w.rindex("if __name__ == '__main__':")])
vol = (H['volume'] if 'volume' in H else H['v']).values.astype(float)
vavg = pd.Series(vol).groupby(dd).transform(lambda s_: s_.shift(1).rolling(20, min_periods=8).mean()).values
def path_nhp(d, i, dr, S0, T=None, until=955, tf=15, arm=None, k=None, needvol=False):
    ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; S = e - dr * S0; mfe = 0.0; worst = 0.0; pb = None; best = -np.inf
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= until: return (c[ix[p - 1]] - e) * dr - 1.0, worst - 1.0
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): return (S - e) * dr - 1.0, (S - e) * dr - 1.0
        fav = ((h[ii] if dr > 0 else l[ii]) - e) * dr; worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr); mfe = max(mfe, fav)
        if T is not None and fav >= T: return T - 1.0, worst - 1.0
        if fav > best: best, pb = fav, p
        if p + 1 < len(ix):
            if k is not None and mfe >= arm and p - pb >= k:
                ib = ix[pb]
                if not needvol or (np.isfinite(vavg[ib]) and vol[ib] >= 1.5 * vavg[ib]): return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
            if tf is not None:
                r_ = REG[tf].get(np.datetime64(Hd5[ii], 'ns'))
                if r_ is not None and r_ == -dr: return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
    return (c[ix[-1]] - e) * dr - 1.0, worst - 1.0
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
    VARS = [('now', None, False)] + [(f'NHP-{k}{" + VOL" if v else ""}', k, v) for v in (False, True) for k in (2, 4, 6)]
    for en in ('RA', 'V15a'):
        T_ = BE[en][0]; T_ = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')].copy(); T_['per'] = np.where(pd.to_datetime(T_.day).dt.year >= 2022, 'b', 'a')
        print(f'\n=== {en}: {len(T_)} trades | NQ pts per trade 2018-21 / 2022-25 | total 2022-25 | share of trades the NHP exit closed ===', flush=True)
        for vn, k, v in VARS:
            out = []
            for r_ in T_.itertuples():
                d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
                if not len(w): out.append((np.nan, np.nan)); continue
                a_ = DY.adr[d]
                kw = dict(S0=50, T=75, until=720, tf=None, arm=25) if en == 'RA' else dict(S0=.15 * a_, tf=15, arm=.10 * a_)
                out.append(path_nhp(d, ix[w[0]], r_.dir, k=k, needvol=v, **kw))
            Q = np.array(out); X = T_.assign(p1=Q[:, 0], mae2=Q[:, 1]); RES[(en, vn)] = X
            ch = np.nan if vn == 'now' else np.mean(np.abs(X.p1 - RES[(en, 'now')].p1) > 1e-9)
            a, b = X[X.per == 'a'], X[X.per == 'b']
            print(f'  {vn:<14} {a.p1.mean():+6.2f} / {b.p1.mean():+6.2f} | {b.p1.sum():+6.0f} | changed {ch:4.0%}' if vn != 'now' else f'  {vn:<14} {a.p1.mean():+6.2f} / {b.p1.mean():+6.2f} | {b.p1.sum():+6.0f}', flush=True)
    PASS = [(en, vn) for en in ('RA', 'V15a') for vn, _, _ in VARS[1:] if all(RES[(en, vn)][RES[(en, vn)].per == p_].p1.mean() > RES[(en, 'now')][RES[(en, 'now')].per == p_].p1.mean() for p_ in 'ab')]
    print(f'\npasses: {PASS if PASS else "none"}', flush=True)
    if PASS:
        LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
        def apply(chg):
            out = []
            for en, (T_, u, yz) in zip(NAMES, base):
                T_ = T_.copy()
                if en in chg:
                    X = chg[en]; mp = {i_: (p_, w_) for i_, p_, w_ in zip(X.index, X.p1, X.mae2) if np.isfinite(p_)}
                    T_['pts'] = [mp.get(i_, (p0, w0))[0] for i_, p0, w0 in zip(T_.index, T_.pts, T_.mae)]; T_['mae'] = [mp.get(i_, (p0, w0))[1] for i_, p0, w0 in zip(T_.index, T_.pts, T_.mae)]
                out.append((T_, u, yz))
            return out
        print('\n=== Lucid Flex 50K ===', flush=True)
        for lab, sp in [('V2 + skip slow flips (now)', base)] + [(f'{en} {vn}', apply({en: RES[(en, vn)]})) for en, vn in PASS[:4]]:
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
            print(f'  {lab[:40]:<40} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {paid:3.0%}, evals/payout {1 / ppe:4.2f}, dry {mx} | fleet 2018+: {C.mean():.2f} pay/mo, '
                  f'losing {np.mean(N_ < 0):3.0%}, ${N_.mean():,.0f}/mo | 24 starts 2022-23: avg ${fl["avg"]:,.0f} / med ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
