"""Nov 9 2026: UP VOLUME / DOWN VOLUME AND VOLUME-EMA BREAKS ON THE PROP ALGO (Rey). DECLARED BEFORE RESULTS.
NQ futures 5m RTH bars 2018-25 (bar volume only - a full-history stand-in for delta: volume on up-close bars vs down-close bars).
Trades: the current 50K setup (V2 + skip slow flips): Rule A, V15a, M1 noise momentum. Points per trade at 1 contract, 2018-21 / 2022-25, thirds cut on 2018-21.
AT ENTRY (known then):
  UD30 / UD60  up-volume / down-volume over the last 6 / 12 5m bars of the session, on the trade's side (long: up / down; short: down / up), as a log ratio
  FLIPVOL      volume of the 15m bar that triggered the entry vs its 20-bar EMA (15m bars): above = a volume EMA break on the signal bar
  VEMA         5-bar EMA of 5m volume above / below the 20-bar EMA at entry (volume expanding vs fading)
  RVOL         the last 30 minutes' volume vs the same 30 minutes' 20-day average
EXITS (Rule A and V15a, everything else as now):
  X1  out at the next open when a 5m bar closes AGAINST the trade on volume >= 2 x its 20-bar EMA (an adverse volume break)
  X2  out at the next open when, 30+ minutes in, the last 6 bars' up / down volume turns against the trade (log ratio < -0.4)
PASS: a filter third that loses in BOTH periods (skip it), or an exit with more points per trade in BOTH -> Lucid Flex 50K (per eval 2018-25, fleet from 2018, 24 starts 2022-23)."""
_w = open('/home/claude/fartpie/xs/winners.py').read(); exec(_w[:_w.rindex("if __name__ == '__main__':")])
_cb = open('/home/claude/fartpie/xs/cut_bottom.py').read(); exec(_cb[_cb.index('KD = {d: k'):_cb.index('DV = DY[')])
vol = (H['volume'] if 'volume' in H else H['v']).values.astype(float); up = (c > o).astype(float); dn = (c < o).astype(float)
ema = lambda s_, n: s_.ewm(span=n, adjust=False).mean()
VS = pd.Series(vol); v20 = VS.groupby(dd).transform(lambda s_: ema(s_, 20)).values; v5 = VS.groupby(dd).transform(lambda s_: ema(s_, 5)).values
v20p = VS.groupby(dd).transform(lambda s_: ema(s_, 20).shift(1)).values
TOD = pd.DataFrame({'d': dd, 'm': m, 'v': vol}); TOD['avg'] = TOD.groupby('m').v.transform(lambda s_: s_.shift(1).rolling(20, min_periods=10).mean()); vavg_tod = TOD.avg.values
B15v = pd.Series(np.nan, index=range(len(B)))
_g = pd.DataFrame({'t': H.dt.values - (((m - 570) % 15) * 60e9).astype('timedelta64[ns]'), 'v': vol}).groupby('t').v.sum()
B15v = pd.Series(B.index.map(lambda t: _g.get(t, np.nan)).values); B15e = B15v.groupby(Bday).transform(lambda s_: ema(s_, 20).shift(1)).values
def ent_feats(T_):
    out = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); ix = gi.get(d); dr = r_.dir
        w = np.where(m[ix] == r_.m)[0] if ix is not None else []
        if not len(w) or w[0] < 3 or not np.isfinite(dr): out.append([np.nan] * 5); continue
        j = w[0]; f = []
        for n in (6, 12):
            seg = ix[max(0, j - n):j]; u_, d_ = (vol[seg] * up[seg]).sum() + 1, (vol[seg] * dn[seg]).sum() + 1; f.append(np.log(u_ / d_) * dr)
        q = endmap.get(np.datetime64(d + pd.Timedelta(minutes=int(r_.m)), 'ns')); f.append(B15v[q] / B15e[q] if q is not None and np.isfinite(B15e[q]) and B15e[q] > 0 else np.nan)
        i1 = ix[j - 1]; f.append(1.0 if v5[i1] > v20[i1] else 0.0)
        seg = ix[max(0, j - 6):j]; f.append(vol[seg].sum() / np.nansum(vavg_tod[seg]) if np.isfinite(vavg_tod[seg]).all() else np.nan)
        out.append(f)
    return pd.DataFrame(out, columns=['UD30', 'UD60', 'FLIPVOL', 'VEMA', 'RVOL'], index=T_.index)
def path_v(d, i, dr, S0, T=None, until=955, tf=15, rule=None):
    ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; S = e - dr * S0; worst = 0.0
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= until: return (c[ix[p - 1]] - e) * dr - 1.0, worst - 1.0
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): return (S - e) * dr - 1.0, (S - e) * dr - 1.0
        fav = ((h[ii] if dr > 0 else l[ii]) - e) * dr; worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
        if T is not None and fav >= T: return T - 1.0, worst - 1.0
        if p + 1 < len(ix):
            if rule == 'X1' and (c[ii] - o[ii]) * dr < 0 and np.isfinite(v20p[ii]) and vol[ii] >= 2 * v20p[ii]: return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
            if rule == 'X2' and p - j >= 6:
                seg = ix[p - 5:p + 1]; r_ = np.log(((vol[seg] * up[seg]).sum() + 1) / ((vol[seg] * dn[seg]).sum() + 1)) * dr
                if r_ < -.4: return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
            if tf is not None:
                rg_ = REG[tf].get(np.datetime64(Hd5[ii], 'ns'))
                if rg_ is not None and rg_ == -dr: return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
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
        if en in ('RA', 'V15a', 'M1'): T_ = add_dir(en, T_)
        base.append((T_, u, yz))
    BE = dict(zip(NAMES, base)); SKIPS = []
    print('=== AT ENTRY: NQ pts per trade (n) by third, 2018-21 / 2022-25 (x = loses in both) ===', flush=True)
    for en in ('RA', 'V15a', 'M1'):
        T_ = BE[en][0]; T_ = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')].copy(); T_ = T_.join(ent_feats(T_)); T_['per'] = np.where(pd.to_datetime(T_.day).dt.year >= 2022, 'b', 'a')
        print(f'\n  {en}: all {T_[T_.per == "a"].pts.mean():+.2f} / {T_[T_.per == "b"].pts.mean():+.2f} (features for {T_.UD30.notna().mean():.0%})', flush=True)
        for f in ('UD30', 'UD60', 'FLIPVOL', 'VEMA', 'RVOL'):
            if f == 'VEMA': bk = [('fading', T_.VEMA == 0), ('expanding', T_.VEMA == 1)]
            else:
                a_, b_ = T_[T_.per == 'a'][f].quantile([1 / 3, 2 / 3]); bk = [(f'< {a_:.2f}', T_[f] < a_), (f'{a_:.2f}-{b_:.2f}', (T_[f] >= a_) & (T_[f] < b_)), (f'>= {b_:.2f}', T_[f] >= b_)]
            cells = []
            for bl, msk in bk:
                x, y = T_[msk & (T_.per == 'a')].pts, T_[msk & (T_.per == 'b')].pts; flag = len(x) >= 20 and len(y) >= 20 and x.mean() < 0 and y.mean() < 0
                cells.append(f'{bl}: {x.mean():+6.2f} ({len(x)}) / {y.mean():+6.2f} ({len(y)}){" x" if flag else ""}')
                if flag: SKIPS.append((en, f, bl, set(zip(pd.to_datetime(T_[msk].day), T_[msk].m)), x.mean() + y.mean()))
            print(f'    {f:<8} ' + ' | '.join(cells), flush=True)
    print('\n=== EXITS (Rule A, V15a): NQ pts per trade 2018-21 / 2022-25 | total 2022-25 ===', flush=True); EXR = {}
    for en in ('RA', 'V15a'):
        T_ = BE[en][0]; T_ = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')].copy(); T_['per'] = np.where(pd.to_datetime(T_.day).dt.year >= 2022, 'b', 'a')
        for rn in (None, 'X1', 'X2'):
            out = []
            for r_ in T_.itertuples():
                d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
                if not len(w): out.append((np.nan, np.nan)); continue
                a_ = DY.adr[d]; kw = dict(S0=50, T=75, until=720, tf=None) if en == 'RA' else dict(S0=.15 * a_, tf=15)
                out.append(path_v(d, ix[w[0]], r_.dir, rule=rn, **kw))
            Q = np.array(out); X = T_.assign(p1=Q[:, 0], mae2=Q[:, 1]); EXR[(en, rn)] = X; a, b = X[X.per == 'a'], X[X.per == 'b']
            print(f'  {en:<5} {rn or "now":<4} {a.p1.mean():+6.2f} / {b.p1.mean():+6.2f} | {b.p1.sum():+6.0f}', flush=True)
    PX = [(en, rn) for en in ('RA', 'V15a') for rn in ('X1', 'X2') if all(EXR[(en, rn)][EXR[(en, rn)].per == p_].p1.mean() > EXR[(en, None)][EXR[(en, None)].per == p_].p1.mean() for p_ in 'ab')]
    print(f'\nfilter passes: {[(s_[0], s_[1], s_[2]) for s_ in SKIPS] or "none"} | exit passes: {PX or "none"}', flush=True)
    if SKIPS or PX:
        LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
        def apply(skip=None, ex=None):
            out = []
            for en, (T_, u, yz) in zip(NAMES, base):
                T_ = T_.copy()
                if skip and en == skip[0]: T_.loc[np.array([(pd.Timestamp(d), mm_) in skip[3] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
                if ex and en == ex[0]:
                    X = EXR[ex]; mp = {i_: (p_, w_) for i_, p_, w_ in zip(X.index, X.p1, X.mae2) if np.isfinite(p_)}
                    T_['pts'] = [mp.get(i_, (p0, w0))[0] for i_, p0, w0 in zip(T_.index, T_.pts, T_.mae)]; T_['mae'] = [mp.get(i_, (p0, w0))[1] for i_, p0, w0 in zip(T_.index, T_.pts, T_.mae)]
                out.append((T_, u, yz))
            return out
        combos = [('V2 + skip slow flips (now)', base)] + [(f'skip {s_[0]} {s_[1]} {s_[2]}', apply(skip=s_)) for s_ in sorted(SKIPS, key=lambda t: t[4])[:3]] + [(f'{e_[0]} exit {e_[1]}', apply(ex=e_)) for e_ in PX[:2]]
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
            print(f'  {lab[:44]:<44} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {paid:3.0%}, evals/payout {1 / ppe:4.2f}, dry {mx} | fleet 2018+: {C.mean():.2f} pay/mo, '
                  f'losing {np.mean(N_ < 0):3.0%}, ${N_.mean():,.0f}/mo | 24 starts 2022-23: avg ${fl["avg"]:,.0f} / med ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
