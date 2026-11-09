"""Nov 9 2026: CUT THE BOTTOM - every trade of the 50K setup (V2 + skip slow flips) split by engine x regime; which slices lose, and does cutting them hold up out of
sample? DECLARED BEFORE RESULTS.
Engines: Rule A, M1 noise momentum, R1 80% rule, ON+ / ON- overnight, rules C + TPO, V15a. Points per trade at 1 contract (sized-out IV-jump trades left out).
Regime / context at entry (all known then): gamma sign, gamma third (QQQ g2p), DEX third, YZ20 third, HAR forecast vs its trailing-year median (third), prior-day NQ daily
vector (trade WITH / AGAINST it), opening gap vs the trade (with / against, and |gap| third), time of day (09:30-10:30 / 10:30-13:30 / 13:30+), weekday (Mon / Tue-Thu / Fri),
day type (FOMC / monthly OPEX Friday / other), side (long / short). Thirds cut on 2018-21.
CUT A (honest test): slices that lose in 2018-21 with n >= 30 there -> cut all their trades; judged on 2022-25 (never used to pick them) and the fleet started 2022-23.
CUT B (in-sample): slices that lose in BOTH 2018-21 and 2022-25, n >= 25 in each.
CUT C (priors, written before looking): trend engines (Rule A, M1, V15a) trading AGAINST the prior-day daily vector | trend engines in the top gamma third |
trend engines on quiet days (bottom HAR-relative third) | trend engines entering 11:30-13:30 (lunch).
FIX after the first run: the gap slices are left out for the overnight engines (the 09:30 gap contains the overnight trade's own move = lookahead).
Each cut -> Lucid Flex 50K: per eval 2018-25 (pass, paid, evals per payout), one at a time (longest run without a payout), 5-account fleet from 2018 (payouts a month,
losing months, net $ a month) and the 24-start fleet 2022-23."""
_fc = open('/home/claude/fartpie/xs/flip_char.py').read(); exec(_fc[:_fc.rindex("if __name__ == '__main__':")])
KD = {d: k for k, d in enumerate(days)}
def add_dir(nm, T_):
    T_ = T_.copy()
    if 'dir' in T_: return T_
    if nm in ('ON+', 'ON-'): T_['dir'] = 1.0; return T_
    if nm == 'RULES':
        if T_.index.isin(DS.index).all() and (pd.to_datetime(DS.loc[T_.index, 'day']).values == pd.to_datetime(T_.day).values).all(): T_['dir'] = DS.loc[T_.index, 'dir'].values.astype(float); return T_
        K_ = DS[['day', 'entry_m', 'ptsC', 'dir']].copy(); K_['day'] = pd.to_datetime(K_.day); K_['mm'] = K_.entry_m.clip(lower=0); K_['pk'] = K_.ptsC.round(3)
        g_ = K_.groupby(['day', 'mm', 'pk']).dir.agg(lambda s_: s_.iloc[0] if s_.nunique() == 1 else np.nan)
        T_['dir'] = [g_.get((pd.Timestamp(d), mm_, round(p_, 3)), np.nan) for d, mm_, p_ in zip(T_.day, T_.m, T_.pts)]; return T_
    out = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); ix = gi[d]; w = np.where(m[ix] == r_.m)[0]
        if not len(w) or w[0] == 0: out.append(np.nan); continue
        j = w[0]; o0 = o[ix[0]]
        if nm == 'M1': pc = c[gi[days[KD[d] - 1]][-1]]; cp = c[ix[j - 1]]; out.append(1 if cp > max(o0, pc) else (-1 if cp < min(o0, pc) else np.nan))
        else: out.append(float(np.sign(o[ix[j]] - o0)) if o[ix[j]] != o0 else np.nan)
    T_['dir'] = out; return T_
DV = DY[['o', 'h', 'l', 'c']].rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}).dropna(); DVr = pd.Series(NSL['vec'](DV), index=DV.index).shift(1)
GAP = (DY.o - DY.c.shift(1)) / DY.adr; HREL = FC / FC.rolling(252, min_periods=120).median()
def opex(d): return d.weekday() == 4 and 15 <= d.day <= 21
def context(T_):
    dd_ = pd.to_datetime(T_.day); E = pd.DataFrame(index=T_.index)
    E['gamma sign'] = np.where(dd_.map(DY.gex) > 0, 'pos', 'neg'); E['_g2p'] = dd_.map(DY.g2p).values; E['_dexp'] = [fget(d, 'dexp') for d in dd_]
    E['_yz'] = dd_.map(DY.yz20p).values; E['_har'] = dd_.map(HREL).values; dv = dd_.map(DVr).values
    E['daily vector'] = np.where(np.isnan(dv) | np.isnan(T_.dir.values.astype(float)), 'na', np.where(dv * T_.dir.values == 1, 'with', 'against'))
    gp = dd_.map(GAP).values; E['gap vs trade'] = np.where(np.isnan(T_.dir.values.astype(float)), 'na', np.where(gp * T_.dir.values > 0, 'with', 'against')); E['_gap'] = np.abs(gp)
    mm_ = T_.m.values; E['time'] = np.where(mm_ >= 1000, 'overnight', np.where(mm_ < 630, '09:30-10:30', np.where(mm_ < 810, '10:30-13:30', '13:30+')))
    wd = dd_.dt.weekday.values; E['weekday'] = np.where(wd == 0, 'Mon', np.where(wd == 4, 'Fri', 'Tue-Thu'))
    E['day type'] = ['FOMC' if d in FOMC else ('OPEX Fri' if opex(d) else 'other') for d in dd_]
    E['side'] = np.where(T_.dir.values > 0, 'long', np.where(T_.dir.values < 0, 'short', 'na'))
    return E
if __name__ == '__main__':
    V = run_sig2('s_vector'); V = V[(V.day >= '2018-02-01')].copy(); Vg = V[~hig(V)].copy(); RA_ = RAs[(RAs.sz > 0) & (RAs.day >= '2018-02-01')].copy()
    REF = V.join(feats(V)); cut = REF[pd.to_datetime(REF.day).dt.year < 2022].mv30.quantile(1 / 3); SL = {}
    for key, T_ in (('V15a', Vg), ('RA', RA_)):
        F = T_.join(feats(T_)); SL[key] = set(zip(pd.to_datetime(F[F.mv30 < cut].day), F[F.mv30 < cut].m))
    base0 = variant(True, relm); base = []
    for en, (T_, u, yz) in zip(NAMES, base0):
        T_ = add_dir(en, T_)
        if en in SL: T_.loc[np.array([(pd.Timestamp(d), mm_) in SL[en] for d, mm_ in zip(T_.day, T_.m)]), 'sz'] = 0.0
        base.append((T_, u, yz))
    ALL = []
    for en, (T_, u, yz) in zip(NAMES, base):
        X = T_[(T_.sz > 0) & (pd.to_datetime(T_.day) >= '2018-02-01')].copy(); X['eng'] = en; X['w'] = u * X.sz; X['key'] = [(en, pd.Timestamp(d), mm_, i_) for d, mm_, i_ in zip(X.day, X.m, X.index)]
        ALL.append(X.join(context(X)))
    Z = pd.concat(ALL); Z['per'] = np.where(pd.to_datetime(Z.day).dt.year >= 2022, 'b', 'a'); Za = Z[Z.per == 'a']
    for col, nm in (('_g2p', 'gamma third'), ('_dexp', 'DEX third'), ('_yz', 'YZ20 third'), ('_har', 'HAR vs normal'), ('_gap', '|gap| third')):
        q1, q2 = Za[col].quantile([1 / 3, 2 / 3]); Z[nm] = np.where(Z[col].isna(), 'na', np.where(Z[col] < q1, 'low', np.where(Z[col] < q2, 'mid', 'high')))
    DIMS = ['gamma sign', 'gamma third', 'DEX third', 'YZ20 third', 'HAR vs normal', 'daily vector', 'gap vs trade', '|gap| third', 'time', 'weekday', 'day type', 'side']
    print('=== engines (V2 + skip slow flips): trades a year, NQ pts per trade at 1 contract, weighted points (pts x size) a year | 2018-21 / 2022-25 ===', flush=True)
    for en in NAMES:
        a, b = Z[(Z.eng == en) & (Z.per == 'a')], Z[(Z.eng == en) & (Z.per == 'b')]; dirk = Z[Z.eng == en].dir.notna().mean()
        print(f'  {en:<6} {len(a) / 3.92:5.0f} / {len(b) / 4.0:5.0f} a yr | {a.pts.mean():+6.2f} / {b.pts.mean():+6.2f} pts | {(a.pts * a.w).sum() / 3.92:+6.0f} / {(b.pts * b.w).sum() / 4.0:+6.0f} wtd pts a yr'
              f'  (direction known {dirk:.0%})', flush=True)
    rows = []
    print('\n=== slices: pts per trade (n) 2018-21 / 2022-25; A = picked on 2018-21 (loses there, n >= 30); B = loses in both (n >= 25 each) ===', flush=True)
    for en in NAMES:
        E_ = Z[Z.eng == en]; lines = []
        for dmn in DIMS:
            for lv, g_ in E_.groupby(dmn):
                if lv == 'na' or (dmn == 'time' and lv == 'overnight') or (en in ('ON+', 'ON-') and dmn in ('gap vs trade', '|gap| third')): continue   # the RTH gap is the overnight move itself: lookahead for ON
                a, b = g_[g_.per == 'a'], g_[g_.per == 'b']
                if len(a) < 15 and len(b) < 15: continue
                fa = len(a) >= 30 and a.pts.mean() < 0; fb = len(a) >= 25 and len(b) >= 25 and a.pts.mean() < 0 and b.pts.mean() < 0
                rows.append(dict(eng=en, dim=dmn, lv=lv, na=len(a), nb=len(b), pa=a.pts.mean(), pb=b.pts.mean(), A=fa, B=fb, keys=set(g_.key)))
                if fa or fb: lines.append(f'    {dmn} = {lv:<12} {a.pts.mean():+7.2f} ({len(a):3d}) / {b.pts.mean():+7.2f} ({len(b):3d})  {"A" if fa else " "}{"B" if fb else " "}')
        print(f'  {en}: all {E_[E_.per == "a"].pts.mean():+.2f} / {E_[E_.per == "b"].pts.mean():+.2f}' + (' - no losing slices' if not lines else ''), flush=True)
        for ln in lines: print(ln, flush=True)
    R = pd.DataFrame(rows)
    TREND = {'RA', 'M1', 'V15a'}; tz = Z[Z.eng.isin(TREND)]
    CUTS = {'A picked on 2018-21': set().union(*R.loc[R.A, 'keys']) if R.A.any() else set(),
            'B loses in both periods': set().union(*R.loc[R.B, 'keys']) if R.B.any() else set(),
            'C priors': set(tz[(tz['daily vector'] == 'against') | (tz['gamma third'] == 'high') | (tz['HAR vs normal'] == 'low') | ((tz.m >= 690) & (tz.m < 810))].key)}
    print('\n=== the cuts: what they remove | removed trades pts per trade 2018-21 / 2022-25 | kept trades | weighted points a year removed ===', flush=True)
    for cn, ks in CUTS.items():
        rm = Z.key.isin(ks); a, b = Z[rm & (Z.per == 'a')], Z[rm & (Z.per == 'b')]; ka, kb = Z[~rm & (Z.per == 'a')], Z[~rm & (Z.per == 'b')]
        print(f'  {cn:<26} removes {rm.mean():.0%} of trades | removed {a.pts.mean():+6.2f} / {b.pts.mean():+6.2f} | kept {ka.pts.mean():+6.2f} / {kb.pts.mean():+6.2f} | '
              f'removed wtd pts {(a.pts * a.w).sum() / 3.92:+.0f} / {(b.pts * b.w).sum() / 4.0:+.0f} a yr', flush=True)
        print('     by engine (trades removed): ' + ', '.join(f'{en} {int((rm & (Z.eng == en)).sum())}' for en in NAMES), flush=True)
    LOCK['on'] = False; A = ACCX['50K']; fee = A['fee']
    def apply(ks):
        out = []
        for en, (T_, u, yz) in zip(NAMES, base):
            T_ = T_.copy(); hit = np.array([(en, pd.Timestamp(d), mm_, i_) in ks for d, mm_, i_ in zip(T_.day, T_.m, T_.index)])
            if hit.any(): T_.loc[hit, 'sz'] = 0.0
            out.append((T_, u, yz))
        return out
    print('\n=== Lucid Flex 50K ===', flush=True)
    for lab, sp in [('V2 + skip slow flips (now)', base)] + [(f'  + cut {cn}', apply(ks)) for cn, ks in CUTS.items()]:
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
        N = pd.Series({mo: P.get(mo, 0.0) - Fe.get(mo, 0.0) for mo in months}); C = pd.Series({mo: NP.get(mo, 0) for mo in months})
        fl = multi_x({'x': sp}, '50K')
        print(f'  {lab:<34} pass {np.mean([t[0] == "pass" for t in tr_]):3.0%}, paid {paid:3.0%}, evals/payout {1 / ppe:4.2f}, dry run {mx} | fleet 2018+: {C.mean():.2f} payouts/mo, '
              f'losing {np.mean(N < 0):3.0%}, ${N.mean():,.0f}/mo, worst ${N.min():,.0f} | 24 starts 2022-23: avg ${fl["avg"]:,.0f} / median ${fl["med"]:,.0f}, losing {fl["lose"]:.0%}', flush=True)
