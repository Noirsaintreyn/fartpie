"""Nov 9 2026: WHAT KIND OF FLIP PAYS? RECENCY, SPEED, SIZE, AND VOLATILITY AFTER THE FLIP (Rey). DECLARED BEFORE RESULTS.
15m vector flips on real NQ futures (RTH 15m bars carried across days, the live vector settings). Trades: every 15m flip (xs/dte_season2.py run_sig2: next 5m open,
exit on the opposite flip, stop 0.15 ADR, flat 15:55, <= 4 a day), V15a (gamma-gated subset, in the 50K setup) and Rule A (morning flips, 75 / 50).
Features of the flip bar (known at entry):
  RECENT  15m bars since the previous flip (short = the last regime barely lasted) | flips already today (0 / 1 / 2+)
  SPEED   bars from the old regime's extreme to the flip (few = a V-turn) | the 30-minute move into the flip close (ADR, on the trade's side)
  SIZE    how far the flip bar closed beyond the level it broke (ADR) | the flip bar's range / 15m ATR(20)
Each split into thirds with cuts from 2018-21; NQ points per trade by third, 2018-21 / 2022-25.
POST-FLIP VOLATILITY (known 15 min after entry): the first 15m bar after entry, range / ATR. By third, and as EXIT rules at that bar's close:
  E1 out if it is quiet (bottom third) | E2 out if it is wild (top third) AND closed against the trade | E3 out if quiet AND closed against.
PASS: a third that loses in BOTH periods -> skip it; an exit rule that beats the current exit per trade in BOTH periods. The best V15a / Rule A pass goes to the 50K Lucid Flex
test on the best setup (V2): per eval 2018-25 and the 5-account fleet (24 monthly starts 2022-23)."""
_ft = open('/home/claude/fartpie/xs/flex50.py').read(); exec(_ft[:_ft.rindex("if __name__ == '__main__':")])
src_ = open('/home/claude/fartpie/xs/signal_lab.py').read(); vsrc = src_[src_.index('def vec(df, **kw):'):src_.index('def donch(')]
vsrc = vsrc.replace('def vec(df, **kw):', 'def vec_lv(df, **kw):').replace('regs = np.zeros(n, int)', 'regs = np.zeros(n, int); lvs = np.full(n, np.nan)')
vsrc = vsrc.replace('        regs[i] = regime\n    return regs', '        regs[i] = regime; lvs[i] = level\n    return regs, lvs'); exec(vsrc, NSL)
rg, lv = NSL['vec_lv'](B[['open', 'high', 'low', 'close']]); sv = np.nan_to_num(B.s_vector.values.astype(float))
Bo, Bh, Bl, Bc = (B[k].values for k in ('open', 'high', 'low', 'close')); Bday = pd.DatetimeIndex(B.day.values)
tr15 = np.maximum.reduce([Bh - Bl, np.abs(Bh - np.r_[Bc[0], Bc[:-1]]), np.abs(Bl - np.r_[Bc[0], Bc[:-1]])]); atr15 = pd.Series(tr15).rolling(20).mean().values
endmap = {t: q for q, t in enumerate(B.end.values)}; bidx = {t: q for q, t in enumerate(B.index.values)}
FL = np.where((sv != np.r_[0, sv[:-1]]) & (sv != 0))[0]; prevf = {FL[k]: FL[k - 1] for k in range(1, len(FL))}
def feats(T_):
    out = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); q = endmap.get(np.datetime64(d + pd.Timedelta(minutes=int(r_.m)), 'ns')); dr = r_.dir
        if q is None or q not in prevf or q + 1 >= len(B): out.append([np.nan] * 8 + [-1]); continue
        p0 = prevf[q]; adr = DY.adr[d]; seg = np.arange(p0, q + 1)
        ext = seg[np.argmin(Bl[seg])] if dr > 0 else seg[np.argmax(Bh[seg])]
        ntd = int(((FL < q) & (FL >= q - 40) & (Bday[np.clip(FL, 0, len(B) - 1)] == Bday[q])).sum())
        rv1 = (Bh[q + 1] - Bl[q + 1]) / atr15[q] if Bday[q + 1] == Bday[q] else np.nan
        out.append([q - p0, ntd, q - ext, (Bc[q] - Bc[q - 2]) * dr / adr, (Bc[q] - lv[q - 1]) * dr / adr, (Bh[q] - Bl[q]) / atr15[q - 1], rv1,
                    (Bc[q + 1] - Bo[q + 1]) * dr if Bday[q + 1] == Bday[q] else np.nan, q])
    return pd.DataFrame(out, columns=['rec', 'ntd', 'spd', 'mv30', 'brk', 'bar', 'rv1', 'agst', 'q'], index=T_.index)
def sim_x(d, i, dr, q, rule, stop=None, until=955, flip=True, T=None):
    ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; S = e - dr * (.15 * DY.adr[d] if stop is None else stop); worst = 0.0
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= until: return (c[ix[p - 1]] - e) * dr - 1.0, worst - 1.0
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): return (S - e) * dr - 1.0, (S - e) * dr - 1.0
        if T is not None and ((h[ii] >= e + dr * T) if dr > 0 else (l[ii] <= e + dr * T)): return T - 1.0, worst - 1.0
        worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
        if (m[ii] + 5) % 15 == 0 and p + 1 < len(ix):
            bq = bidx.get(H.dt.values[ii] - np.timedelta64(10, 'm'))
            if flip and bq is not None and sv[bq] == -dr: return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
            if rule is not None and bq == q + 1 and rule(bq, dr, e): return (o[ix[p + 1]] - e) * dr - 1.0, worst - 1.0
    return (c[ix[-1]] - e) * dr - 1.0, worst - 1.0
if __name__ == '__main__':
    V = run_sig2('s_vector'); V = V[V.day >= '2018-02-01'].copy(); V['g'] = hig(V); RA_ = RAs[(RAs.sz > 0) & (RAs.day >= '2018-02-01')].copy()
    print(f'vector levels rebuilt: regime matches the live series on {np.mean(rg == sv):.1%} of 15m bars', flush=True)
    ENGS = {'15m flips, every': V, 'V15a (gamma-gated)': V[~V.g].copy(), 'Rule A': RA_}
    for nm, T_ in ENGS.items():
        F = feats(T_); ENGS[nm] = T_ = T_.join(F); T_['per'] = np.where(pd.to_datetime(T_.day).dt.year >= 2022, 'b', 'a'); ENGS[nm] = T_
        print(f'  {nm}: features found for {T_.rec.notna().mean():.0%} of {len(T_)} trades', flush=True)
    ref = ENGS['15m flips, every']; ref_a = ref[ref.per == 'a']
    CUT = {f: tuple(ref_a[f].quantile([1 / 3, 2 / 3]).values) for f in ('rec', 'spd', 'mv30', 'brk', 'bar', 'rv1')}
    LAB = {'rec': 'RECENT: bars since the last flip', 'ntd': 'RECENT: flips already today', 'spd': 'SPEED: bars from the old extreme', 'mv30': 'SPEED: 30-min move into the flip (ADR)',
           'brk': 'SIZE: close beyond the broken level (ADR)', 'bar': 'SIZE: flip bar range / ATR', 'rv1': 'POST-FLIP VOL: 1st bar range / ATR (known 15 min in)'}
    SKIP = []
    for nm, T_ in ENGS.items():
        print(f'\n=== {nm}: all {T_[T_.per == "a"].pts.mean():+.2f} ({(T_.per == "a").sum()}) / {T_[T_.per == "b"].pts.mean():+.2f} ({(T_.per == "b").sum()}) | NQ pts per trade (n), 2018-21 / 2022-25 ===', flush=True)
        for f, lab in LAB.items():
            if f == 'ntd': bk = [('0', T_.ntd == 0), ('1', T_.ntd == 1), ('2+', T_.ntd >= 2)]
            else:
                a, b = CUT[f]; bk = [(f'< {a:.2f}', T_[f] < a), (f'{a:.2f}-{b:.2f}', (T_[f] >= a) & (T_[f] < b)), (f'>= {b:.2f}', T_[f] >= b)]
            cells = []
            for bl, msk in bk:
                x, y = T_[msk & (T_.per == 'a')].pts, T_[msk & (T_.per == 'b')].pts; flag = ' x' if (x.mean() < 0 and y.mean() < 0 and len(x) >= 20 and len(y) >= 20) else ''
                cells.append(f'{bl}: {x.mean():+6.2f} ({len(x)}) / {y.mean():+6.2f} ({len(y)}){flag}')
                if flag and f != 'rv1': SKIP.append((nm, f, bl, msk, x.mean() + y.mean()))
            print(f'  {lab:<52} ' + ' | '.join(cells), flush=True)
    # exit rules on post-flip volatility
    lo_, hi_ = CUT['rv1']
    RULES = {'E1 out if quiet': lambda bq, dr, e: (Bh[bq] - Bl[bq]) / atr15[bq - 1] < lo_,
             'E2 out if wild AND against': lambda bq, dr, e: (Bh[bq] - Bl[bq]) / atr15[bq - 1] >= hi_ and (Bc[bq] - e) * dr < 0,
             'E3 out if quiet AND against': lambda bq, dr, e: (Bh[bq] - Bl[bq]) / atr15[bq - 1] < lo_ and (Bc[bq] - e) * dr < 0}
    print('\n=== POST-FLIP VOLATILITY AS AN EXIT (decided at the close of the first 15m bar after entry) | pts per trade 2018-21 / 2022-25 | total 2022-25 ===', flush=True)
    EX = {}
    for nm, T_ in ENGS.items():
        T_ = T_[T_.q >= 0]; ra = nm == 'Rule A'
        def run_rule(rule):
            res = []
            for r_ in T_.itertuples():
                d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
                if not len(w): res.append((np.nan, np.nan)); continue
                res.append(sim_x(d, ix[w[0]], r_.dir, int(r_.q), rule, stop=50 if ra else None, until=720 if ra else 955, flip=not ra, T=75 if ra else None))
            return np.array(res)
        b0 = run_rule(None); T0 = T_.assign(p0=b0[:, 0]); print(f'  {nm:<20} current exit (re-run) {T0[T0.per == "a"].p0.mean():+6.2f} / {T0[T0.per == "b"].p0.mean():+6.2f} | {T0[T0.per == "b"].p0.sum():+.0f}'
                                                                   f'   (logged: {T_[T_.per == "a"].pts.mean():+.2f} / {T_[T_.per == "b"].pts.mean():+.2f})', flush=True)
        for rn, rule in RULES.items():
            rr = run_rule(rule); T1 = T_.assign(p1=rr[:, 0], mae1=rr[:, 1]); a1, b1 = T1[T1.per == 'a'].p1.mean(), T1[T1.per == 'b'].p1.mean()
            ok_ = a1 > T0[T0.per == 'a'].p0.mean() and b1 > T0[T0.per == 'b'].p0.mean(); EX[(nm, rn)] = (T1, ok_, a1 + b1 - T0[T0.per == 'a'].p0.mean() - T0[T0.per == 'b'].p0.mean())
            print(f'    {rn:<28} {a1:+6.2f} / {b1:+6.2f} | {T1[T1.per == "b"].p1.sum():+.0f}{"   <- PASS" if ok_ else ""}', flush=True)
    # fleet
    cand = []
    for nm, f, bl, msk, sc in SKIP:
        if nm in ('V15a (gamma-gated)', 'Rule A'): cand.append(('skip', nm, f'skip {nm} when {LAB[f]} is {bl}', ENGS[nm][msk][['day', 'm']], sc))
    for (nm, rn), (T1, ok_, sc) in EX.items():
        if ok_ and nm in ('V15a (gamma-gated)', 'Rule A'): cand.append(('exit', nm, f'{nm}: {rn}', T1, sc))
    if not cand: print('\n  no V15a / Rule A pass - no fleet test', flush=True)
    else:
        print('\n=== Lucid Flex 50K, best setup V2 vs V2 + each pass ===', flush=True); LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']; base = variant(True, relm)
        def modify(kind, nm, X):
            key = 'V15a' if nm.startswith('V15a') else 'RA'; out = []
            for en, (T_, u, yz) in zip(NAMES, base):
                T_ = T_.copy()
                if en == key:
                    kk_ = set(zip(pd.to_datetime(X.day), X.m)); hit = np.array([(pd.Timestamp(d), mm_) in kk_ for d, mm_ in zip(T_.day, T_.m)])
                    if kind == 'skip': T_.loc[hit, 'sz'] = 0.0
                    else:
                        mp = {(pd.Timestamp(d), mm_): (p, mae) for d, mm_, p, mae in zip(X.day, X.m, X.p1, X.mae1) if np.isfinite(p)}
                        nv = [mp.get((pd.Timestamp(d), mm_), (p, mae)) for d, mm_, p, mae in zip(T_.day, T_.m, T_.pts, T_.mae)]; T_['pts'] = [x[0] for x in nv]; T_['mae'] = [x[1] for x in nv]
                out.append((T_, u, yz))
            return out
        for lab, sp in [('V2 as is', base)] + [(c_[2], modify(c_[0], c_[1], c_[3])) for c_ in sorted(cand, key=lambda t: -t[4])[:4]]:
            LE_, LF_ = lists(sp, 'e'), lists(sp, 'f'); TRJ = {}
            def traj(k):
                if k not in TRJ:
                    r, k2 = eval_x(LE_, A, k, NS, 1000)
                    if r != 'pass': TRJ[k] = (r, k2 if r == 'fail' else NS, [])
                    else: fr, k3, pays = fund_l(LF_, A, k2 + 1, NS, None); TRJ[k] = (r, NS if fr == 'open' else k3, pays)
                return TRJ[k]
            tr_ = [traj(s) for s in range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 2)]; ppe = np.mean([len(t[2]) for t in tr_]); s = 0; chain = []
            while s < NS - 1:
                r, end, pays = traj(s); chain.append(pays); s = end + 1
            run = mx = 0
            for p_ in chain: run = run + 1 if not p_ else 0; mx = max(mx, run)
            fl = multi_x({'x': sp}, '50K')
            print(f'  {lab[:70]:<70} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, evals/payout {1 / ppe if ppe else np.inf:4.1f}, longest dry run {mx:2d} | fleet avg ${fl["avg"]:,.0f}'
                  f' / median ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}, dry {fl["dry"]:.1f} mo', flush=True)
