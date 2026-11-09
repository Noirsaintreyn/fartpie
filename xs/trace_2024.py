"""Nov 9 2026 (trace_2024: account bought 26 Mar 2024 - first payout, 2024 income and spend; setups: 50K = S6 as is, 150K = S6 + rules 5m pullback + noise zU skip;
one-account chain (a new eval the day after an account fails / blows / closes) and a 5-account fleet (one eval a day while under 5 and the trailing 20-day P&L is positive).
Previous header: (entry_z2, declared after entry_z.py and before this run: the fleet with the two pieces that held in BOTH periods - rules algo entered
on a 5m pullback limit (0.25 x average 5m range, 3 bars, unfilled = no trade), and noise momentum skipped when zU >= 1.04 ADR already used (the 2018-21 cut).
V15a z rule dropped (its skipped trades made as much as the kept ones in 2022-25). Previous header:) A SMARTER OVEREXTENSION Z AND 5M EXECUTION - only take the good entries. DECLARED BEFORE RESULTS.
Engines (real NQ futures, current trades): RA Rule A, M1 noise momentum, V15a vector all day, RULES C + TPO. Entry = the 5m bar open the engine trades at.
Z-scores at entry, signed by the trade side (high = stretched in the trade's direction), all from bars before the entry:
  zV  (entry - VWAP) / VWAP sd           zN  move from today's open / 14-day average move at this time of day (noise band)
  zU  range already used in the trade's direction: (entry - today's low) / ADR for longs, (today's high - entry) / ADR for shorts
  zE  (entry - 5m EMA20) / stdev of the last 20 5m closes
RULE (chosen on 2018-21 only): per engine, the z whose most-stretched fifth of trades did worst in 2018-21; if that fifth LOST money there, skip trades at or above
its 2018-21 80th percentile. Judged on 2022-25, then in the fleet (Lucid Flex x5, 24 monthly starts 2022-23, xs/fleet_compare2.py) on S6 (S3 + V15a).
5M EXECUTION (fixed in advance, RA and RULES, both periods): P pullback - limit at entry - 0.25 x average 5m range (last 12 bars), good for 3 bars, unfilled = no trade,
same target / stop distances from the fill; C confirm - the entry bar must close in the trade's direction, then enter at the next bar's open, else no trade."""
_fc2 = open('/home/claude/fartpie/xs/fleet_compare2.py').read(); exec(_fc2[:_fc2.rindex("if __name__ == '__main__':")])
RC2 = rules_take('C', GR['TPO'])
ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values; sd20 = pd.Series(c).rolling(20).std().values; rng12 = pd.Series(h - l).rolling(12).mean().values
tidx = {(d, mm_): i for d in days for i, mm_ in zip(gi[d], m[gi[d]])}
def zs(d, mm_, dr):
    d = pd.Timestamp(d); i = tidx.get((d, mm_))
    if i is None or i == gi[d][0]: return (np.nan,) * 4 + (None,)
    ix = gi[d]; p = int(np.where(ix == i)[0][0]); e = o[i]; adr = DY.adr[d]; o0 = o[ix[0]]; hi_, lo_ = h[ix[:p]].max(), l[ix[:p]].min()
    sg = SIGN.at[d, m[i - 1]] if (d in SIGN.index and m[i - 1] in SIGN.columns) else np.nan
    zV = (e - vw[i - 1]) / vsd[i - 1] * dr if vsd[i - 1] > 0 else np.nan; zN = (e / o0 - 1) / sg * dr if np.isfinite(sg) and sg > 0 else np.nan
    zU = ((e - lo_) if dr > 0 else (hi_ - e)) / adr; zE = (e - ema20[i - 1]) / sd20[i - 1] * dr if sd20[i - 1] > 0 else np.nan
    return zV, zN, zU, zE, i
m1d = m1_dir(); MD = {}
for r_ in m1d.itertuples():
    ix = gi[pd.Timestamp(r_.day)]; w = np.where(np.isclose(o[ix], r_.e))[0]
    if len(w): MD[(pd.Timestamp(r_.day), m[ix[w[0]]])] = int(r_.dir)
M1z = M1gg.copy(); M1z['dir'] = [MD.get((pd.Timestamp(d), mm_), np.nan) for d, mm_ in zip(M1z.day, M1z.m)]; M1z = M1z.dropna(subset=['dir'])
EZ = {'RA': RAs[RAs.sz > 0].copy(), 'M1': M1z, 'V15a': V15a.copy(), 'RULES': RC2.copy()}
for en, T_ in EZ.items():
    Z = [zs(d, mm_, dr) for d, mm_, dr in zip(T_.day, T_.m, T_.dir)]
    for k_, nm in enumerate(['zV', 'zN', 'zU', 'zE', 'i']): T_[nm] = [z_[k_] for z_ in Z]
    T_['per'] = np.where(T_.day.dt.year >= 2022, 'b', 'a'); EZ[en] = T_[T_.day >= '2018-02-01']
def pullback(T_, en):
    rows = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); i = r_.i
        if i is None or not np.isfinite(i): rows.append(np.nan); continue
        i = int(i)
        if not np.isfinite(rng12[i - 1]): rows.append(np.nan); continue
        ix = gi[d]; p = int(np.where(ix == i)[0][0]); dr = int(r_.dir); lim = o[i] - .25 * rng12[i - 1] * dr; fill = None
        for q in range(p, min(p + 3, len(ix))):
            if (l[ix[q]] <= lim) if dr > 0 else (h[ix[q]] >= lim): fill = q; break
        if fill is None: rows.append(np.nan); continue
        if en == 'RA': pts, mae = walk2(ix, fill, dr, lim, lim + 75 * dr, lim - 50 * dr, until_m=720); rows.append(pts - 1.0)
        else:
            adr = DY.adr[d]; T = .06 * adr + 1.0; S = .12 * adr; endp = np.where(m[ix] < 955)[0][-1]
            hi_, lo_ = h[ix[fill:endp + 1]], l[ix[fill:endp + 1]]
            adv = (lo_ <= lim - S) if dr > 0 else (hi_ >= lim + S); fav = (hi_ >= lim + T) if dr > 0 else (lo_ <= lim - T)
            ia = np.argmax(adv) if adv.any() else 10 ** 9; ib = np.argmax(fav) if fav.any() else 10 ** 9
            if ia == ib == 10 ** 9: rows.append((c[ix[endp]] - lim) * dr - 1.0)
            else: rows.append(-S - 1.0 if ia <= ib else T - 1.0)
    return np.array(rows, dtype=float)
def confirm(T_, en):
    rows = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); i = r_.i
        if i is None or not np.isfinite(i): rows.append(np.nan); continue
        i = int(i)
        ix = gi[d]; p = int(np.where(ix == i)[0][0]); dr = int(r_.dir)
        if (c[i] - o[i]) * dr <= 0 or p + 1 >= len(ix): rows.append(np.nan); continue
        j = p + 1; e = o[ix[j]]
        if en == 'RA': pts, mae = walk2(ix, j, dr, e, e + 75 * dr, e - 50 * dr, until_m=720); rows.append(pts - 1.0)
        else:
            adr = DY.adr[d]; T = .06 * adr + 1.0; S = .12 * adr; endp = np.where(m[ix] < 955)[0][-1]
            if j > endp: rows.append(np.nan); continue
            hi_, lo_ = h[ix[j:endp + 1]], l[ix[j:endp + 1]]
            adv = (lo_ <= e - S) if dr > 0 else (hi_ >= e + S); fav = (hi_ >= e + T) if dr > 0 else (lo_ <= e - T)
            ia = np.argmax(adv) if adv.any() else 10 ** 9; ib = np.argmax(fav) if fav.any() else 10 ** 9
            if ia == ib == 10 ** 9: rows.append((c[ix[endp]] - e) * dr - 1.0)
            else: rows.append(-S - 1.0 if ia <= ib else T - 1.0)
    return np.array(rows, dtype=float)
def rules_pb(T_):
    rows = []
    for r_ in T_.itertuples():
        d = pd.Timestamp(r_.day); i = r_.i
        if i is None or not np.isfinite(i): continue
        i = int(i)
        if not np.isfinite(rng12[i - 1]): continue
        ix = gi[d]; p = int(np.where(ix == i)[0][0]); dr = int(r_.dir); lim = o[i] - .25 * rng12[i - 1] * dr; fill = None
        for q in range(p, min(p + 3, len(ix))):
            if (l[ix[q]] <= lim) if dr > 0 else (h[ix[q]] >= lim): fill = q; break
        if fill is None: continue
        adr = DY.adr[d]; T = .06 * adr + 1.0; S = .12 * adr; endp = np.where(m[ix] < 955)[0][-1]
        if fill > endp: continue
        hi_, lo_ = h[ix[fill:endp + 1]], l[ix[fill:endp + 1]]; exc = (lo_ - lim) if dr > 0 else (lim - hi_)
        adv = (lo_ <= lim - S) if dr > 0 else (hi_ >= lim + S); fav = (hi_ >= lim + T) if dr > 0 else (lo_ <= lim - T)
        ia = int(np.argmax(adv)) if adv.any() else 10 ** 9; ib = int(np.argmax(fav)) if fav.any() else 10 ** 9
        if ia == ib == 10 ** 9: pts, mae = (c[ix[endp]] - lim) * dr - 1.0, exc.min() - 1.0
        elif ia <= ib: pts, mae = -S - 1.0, -S - 1.0
        else: pts, mae = T - 1.0, max(exc[:ib + 1].min(), -S) - 1.0
        rows.append((d, m[ix[fill]], pts, mae, r_.sz, 1.0, 1.0))
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf'])
if __name__ == '__main__':
    LOCK['on'] = False; RPB = rules_pb(EZ['RULES']); M1s = EZ['M1'].copy(); M1s['sz'], M1s['ce'], M1s['cf'] = 1.0, 1.0, 1.0; M1s.loc[M1s.zU >= 1.04, 'sz'] = 0.0
    base = [(RAs, 1.0, RULE['RA']), (M1gg, .3, RULE['M1']), (R1g, .5, RULE['R1']), (OBpos, .4, RULE['ON+']), (OBneg, .1, RULE['ON-'])]
    CF = {'50K': base + [(RC, 1.0, None), (V15a, .3, None)], '150K': [b_ if b_[0] is not M1gg else (M1s, .3, RULE['M1']) for b_ in base] + [(RPB, 1.0, None), (V15a, .3, None)]}
    k0 = int(np.searchsorted(D_, pd.Timestamp('2024-03-26'))); kY = int(np.searchsorted(D_, pd.Timestamp('2025-01-01'))) - 1; k12 = int(np.searchsorted(D_, pd.Timestamp('2025-03-26'))) - 1
    for zn, sp in CF.items():
        global LE, LF
        LE, LF = lists(sp, 'e'), lists(sp, 'f'); R = FIRM['Lucid Flex']; z = SZ[zn]; fee = R['fee'][z['i']]
        pts_ = pd.Series([sum(net * th[0] for th, net, mae, p in zz) for zz in LF], index=D_); yr_pts = pts_.iloc[k0:kY + 1].sum()
        print(f'\n=== Lucid Flex {zn} bought {D_[k0].date()} (fee ${fee}) | system points 26 Mar - 31 Dec 2024 at its size: {yr_pts:+.0f} ===', flush=True)
        def trace(s):
            r, k2 = eval_run(R, z, s, NS)
            if r != 'pass': return r, k2, None, [], (k2 if r == 'fail' else NS)
            fr, k3, pays = fund_run(R, z, k2 + 1, NS); return r, k2, fr, pays, (NS if fr == 'open' else k3)
        s = k0; n = 0; P = []; F = 0; first = None
        while s <= kY:
            r, k2, fr, pays, end = trace(s); n += 1; F += fee
            ev_txt = f'eval {"PASSED " + str(D_[k2].date()) if r == "pass" else ("failed " + str(D_[k2].date()) if r == "fail" else "still open")}'
            pay_txt = ', '.join(f'{D_[k].date()} ${x:,.0f}' for k, x in pays if k <= k12)
            end_txt = '' if r != 'pass' else (f' -> {"blown" if fr == "lost" else ("5 payouts, closed" if fr == "graduated" else "still open")} {D_[min(end, NS - 1)].date() if fr != "open" else ""}')
            print(f'  account {n}: bought {D_[s].date()} | {ev_txt}{end_txt} | payouts: {pay_txt or "none"}', flush=True)
            for k, x in pays:
                P.append((k, x))
                if first is None: first = (k, x)
            s = end + 1
        inY = sum(x for k, x in P if k <= kY); in12 = sum(x for k, x in P if k <= k12); fY = sum(fee for _ in range(n))
        print(f'  ONE ACCOUNT AT A TIME: first payout {D_[first[0]].date()} ${first[1]:,.0f} | 2024 (from 26 Mar): paid ${inY:,.0f}, spent ${F:,.0f} on {n} evals, net ${inY - F:,.0f} '
              f'| 12 months to 25 Mar 2025: paid ${in12:,.0f}' if first else '  no payout', flush=True)
        pnl = pd.Series([sum(net * 2 * max(1, round(10 * th[0] * z['scale'])) for th, net, mae, p in zz) for zz in LE]); hot = (pnl.rolling(20).sum().shift(1) > 0).values
        act = []; pq = []; paid = 0.0; spent = 0; nb = 0; firstf = None; npay = 0; TR2 = {}
        for k in range(k0, kY + 1):
            while pq and pq[0][0] == k:
                kk, x = heapq.heappop(pq); paid += x; npay += 1; firstf = firstf or (k, x)
            act = [e for e in act if e >= k]
            if len(act) < 5 and hot[k]:
                r, k2, fr, pays, end = trace(k); act.append(end); spent += fee; nb += 1
                for kk, x in pays: heapq.heappush(pq, (kk, x))
        print(f'  5-ACCOUNT FLEET: first payout {D_[firstf[0]].date()} ${firstf[1]:,.0f} | 26 Mar - 31 Dec 2024: {npay} payouts, paid ${paid:,.0f}, spent ${spent:,.0f} on {nb} evals, '
              f'net ${paid - spent:,.0f}' if firstf else f'  5-ACCOUNT FLEET: no payout in 2024, spent ${spent:,.0f}', flush=True)
