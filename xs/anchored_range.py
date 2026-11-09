"""Nov 9 2026: ANCHORED RANGE + VECTOR FLIPS (Rey: if we know the day's range, the placed low / high tells the extent of the other side). DECLARED BEFORE RESULTS.
Expected RTH range R^: HAR forecast (prior close) | THOD 1h sigma at 09:30 x k (k = 2018-21 median of RTH range / sigma). Anchor at the flip: for a bull flip the day's low so far,
for a bear flip the day's high so far. Projected other extreme = anchor +/- R^. ROOM = (projected extreme - entry) / R^ on the trade's side.
A) FILTER: 15m vector flips (every flip; gamma-gated V15a) and Rule A, points per trade by room: < 0.2 / 0.2-0.5 / >= 0.5.
B) TAKE PROFIT at the projected extreme (100% / 80% / 60% of R^ from the anchor) on the 15m flips, keeping the stop (0.15 ADR) and the opposite-flip exit; flips whose
target is < 0.05 R^ away are skipped. Rule A: target = the nearer of +75 and the projected extreme.
Judged 2018-21 / 2022-25. A pass (better than now in both periods) goes to the 50K Lucid Flex fleet (24 monthly starts 2022-23) on the best 50K setup
(S6 + IV-jump skip + HAR relative sizing, xs/flex50.py V2)."""
_ft = open('/home/claude/fartpie/xs/flex50_thod.py').read(); exec(_ft[:_ft.rindex("if __name__ == '__main__':")])
rr = (DY.h - DY.l); kk = (rr / SG.h1_rth.reindex(DY.index))[(DY.index >= '2018-01-01') & (DY.index < '2022-01-01')].median()
RH = {'HAR': FC, 'THOD1h': SG.h1_rth.reindex(DY.index) * kk}
regend = dict(zip(B.end.values, B.reg.values)); Hd = H.dt.values
def sim(d, i, dr, S, T, flip_exit, until=955, ra=False):
    ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]
    for p in range(j, len(ix)):
        ii = ix[p]
        if m[ii] >= until: return (c[ix[p - 1]] - e) * dr - 1.0
        if (l[ii] <= S) if dr > 0 else (h[ii] >= S): return (S - e) * dr - 1.0
        if T is not None and ((h[ii] >= T) if dr > 0 else (l[ii] <= T)): return (T - e) * dr - 1.0
        if flip_exit:
            r_ = regend.get(np.datetime64(Hd[ii] + np.timedelta64(5, 'm')))
            if r_ is not None and r_ == -dr and p + 1 < len(ix): return (o[ix[p + 1]] - e) * dr - 1.0
    return (c[ix[-1]] - e) * dr - 1.0
def anchor(d, i, dr):
    ix = gi[d]; j = int(np.where(ix == i)[0][0])
    if j == 0: return np.nan
    return l[ix[:j]].min() if dr > 0 else h[ix[:j]].max()
if __name__ == '__main__':
    V = run_sig2('s_vector'); V = V[V.day >= '2018-02-01'].copy(); V['g'] = hig(V); RA_ = RAs[(RAs.sz > 0) & (RAs.day >= '2018-02-01')].copy()
    for T_ in (V, RA_):
        T_['i'] = [gi[pd.Timestamp(d)][np.where(m[gi[pd.Timestamp(d)]] == mm_)[0][0]] if (m[gi[pd.Timestamp(d)]] == mm_).any() else -1 for d, mm_ in zip(T_.day, T_.m)]
        T_['anc'] = [anchor(pd.Timestamp(d), i, dr) if i >= 0 else np.nan for d, i, dr in zip(T_.day, T_.i, T_.dir)]; T_['per'] = np.where(T_.day.dt.year >= 2022, 'b', 'a')
        for src, R in RH.items():
            T_[f'R_{src}'] = T_.day.map(R).values
            T_[f'room_{src}'] = ((T_.anc + T_.dir * T_[f'R_{src}']) - T_.i.map(lambda i: o[i] if i >= 0 else np.nan)) * T_.dir / T_[f'R_{src}']
    print('A) points per trade by ROOM left to the projected extreme | 2018-21 (n) / 2022-25 (n)', flush=True)
    for nm, T_ in (('15m flips, every', V), ('V15a (gamma-gated)', V[~V.g]), ('Rule A', RA_)):
        print(f'\n  {nm}: all {T_[T_.per == "a"].pts.mean():+.2f} / {T_[T_.per == "b"].pts.mean():+.2f}', flush=True)
        for src in RH:
            cells = []
            for lab, lo_, hi_ in (('<0.2', -9, .2), ('0.2-0.5', .2, .5), ('>=0.5', .5, 9)):
                msk = (T_[f'room_{src}'] >= lo_) & (T_[f'room_{src}'] < hi_)
                a, b = T_[msk & (T_.per == 'a')].pts, T_[msk & (T_.per == 'b')].pts; cells.append(f'{lab}: {a.mean():+6.2f} ({len(a)}) / {b.mean():+6.2f} ({len(b)})')
            print(f'    {src:<7} ' + '   '.join(cells), flush=True)
    print('\nB) take profit at the projected extreme | points per trade 2018-21 / 2022-25 (trades) | total points 2022-25', flush=True)
    RES = {}
    for nm, T_ in (('15m flips, every', V), ('V15a (gamma-gated)', V[~V.g])):
        a, b = T_[T_.per == 'a'], T_[T_.per == 'b']; print(f'\n  {nm} now: {a.pts.mean():+.2f} ({len(a)}) / {b.pts.mean():+.2f} ({len(b)}) | {b.pts.sum():+.0f}', flush=True)
        for src in RH:
            for f in (1.0, .8, .6):
                pts = []
                for r_ in T_.itertuples():
                    d = pd.Timestamp(r_.day); R = getattr(r_, f'R_{src}')
                    if r_.i < 0 or not np.isfinite(r_.anc) or not np.isfinite(R): pts.append(np.nan); continue
                    e = o[r_.i]; Tg = r_.anc + r_.dir * f * R
                    if (Tg - e) * r_.dir < .05 * R: pts.append(np.nan); continue
                    pts.append(sim(d, r_.i, r_.dir, e - r_.dir * .15 * DY.adr[d], Tg, True))
                T2 = T_.assign(pts2=pts); a, b = T2[T2.per == 'a'].pts2.dropna(), T2[T2.per == 'b'].pts2.dropna(); RES[(nm, src, f)] = T2
                print(f'    TP {int(f * 100):3d}% of {src:<7} {a.mean():+6.2f} ({len(a)}) / {b.mean():+6.2f} ({len(b)}) | {b.sum():+.0f}', flush=True)
    a, b = RA_[RA_.per == 'a'], RA_[RA_.per == 'b']; print(f'\n  Rule A now: {a.pts.mean():+.2f} / {b.pts.mean():+.2f}', flush=True)
    for src in RH:
        pts = []
        for r_ in RA_.itertuples():
            d = pd.Timestamp(r_.day); R = getattr(r_, f'R_{src}'); e = o[r_.i] if r_.i >= 0 else np.nan
            if r_.i < 0 or not np.isfinite(r_.anc) or not np.isfinite(R): pts.append(np.nan); continue
            Tg = r_.anc + r_.dir * R; Tg = e + r_.dir * min(75, (Tg - e) * r_.dir) if (Tg - e) * r_.dir > 0 else e + r_.dir * 75
            pts.append(sim(d, r_.i, r_.dir, e - r_.dir * 50, Tg, False, until=720))
        T2 = RA_.assign(pts2=pts); a, b = T2[T2.per == 'a'].pts2.dropna(), T2[T2.per == 'b'].pts2.dropna()
        print(f'    TP min(75, projected {src}) {a.mean():+6.2f} ({len(a)}) / {b.mean():+6.2f} ({len(b)})', flush=True)
    # fleet for any V15a TP variant better in both periods
    base_a, base_b = V[~V.g & (V.per == 'a')].pts.mean(), V[~V.g & (V.per == 'b')].pts.mean(); best = None
    for (nm, src, f), T2 in RES.items():
        if nm.startswith('V15a'):
            a, b = T2[T2.per == 'a'].pts2.mean(), T2[T2.per == 'b'].pts2.mean()
            if a > base_a and b > base_b and (best is None or a + b > best[0]): best = (a + b, src, f, T2)
    if best:
        _, src, f, T2 = best; Vn = T2.dropna(subset=['pts2']).assign(pts=lambda z: z.pts2)[['day', 'm', 'dir', 'pts', 'mae']]
        sp_base = variant(True, relm); sp_new = [(Vn, u, yz) if nm == 'V15a' else (T_, u, yz) for nm, (T_, u, yz) in zip(NAMES, sp_base)]
        for lab, sp in (('50K best setup (V2) as is', sp_base), (f'50K V2, V15a takes profit at {int(f * 100)}% of {src}', sp_new)):
            r = multi_x({'x': sp}, '50K'); print(f'  {lab:<44} avg ${r["avg"]:,.0f} / median ${r["med"]:,.0f}, >= $2.5k {r["ge25"]:.0%}, losing {r["lose"]:.0%}, dry {r["dry"]:.1f} mo', flush=True)
    else: print('\n  no V15a take-profit variant beat the current exit in both periods - no fleet test', flush=True)
