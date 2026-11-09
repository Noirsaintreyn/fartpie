"""Nov 9 2026: ICT ZONES THAT LINE UP WITH VOLUME PROFILE / TPO / OPTIONS STRUCTURE. DECLARED BEFORE RESULTS. Real NQ futures 5m RTH, 1 pt cost.
Zones built today from 09:30 on 5m bars (bullish shown; bearish mirrored):
  FVG  3-bar fair value gap: low[j] > high[j-2] -> zone [high[j-2], low[j]]
  OB   order block: bar j closes above the last 6 bars' high with range >= 1.5 x the 12-bar average (displacement); the last down-close bar in the 3 before it is the zone [low, high]
  RB   rejection block: swing low (lowest of 3 bars each side, confirmed 2 bars later) with lower wick >= 50% of its range -> zone [low, min(open, close)]
Entry: a later bar trades into the zone and closes at or above its low -> long at the next open; each zone used once, invalid after a close through it; one position per
type and exit, <= 3 a day, decisions 09:45-15:00, flat 15:55.  Exits: +10 net before -22 | +(0.06 ADR + 1) before -0.12 ADR | +(0.10 ADR + 1) before -0.15 ADR.
ALIGNMENT (zone within 0.03 ADR of a prior-day level): LVN (volume-profile low node: local minimum of the smoothed profile < 0.5 x median, inside the 10-90% volume range) |
SP (TPO single prints) | SHELF (edges of high-volume nodes where volume halves + VP VAL / POC / VAH + TPO VAL / POC / VAH) | WALL (QQQ call / put gamma walls mapped to NQ).
PASS: aligned zones beat non-aligned AND are positive per trade in BOTH 2018-21 and 2022-25 (>= 30 trades each) -> fleet test on S6 (sized so a win pays >= $150 at 50K)."""
_fc2 = open('/home/claude/fartpie/xs/fleet_compare2.py').read(); exec(_fc2[:_fc2.rindex("if __name__ == '__main__':")])
def vp_nodes(ix):
    bs = o[ix[0]] * .0005; b0 = np.floor(l[ix].min() / bs); nb = int(np.floor(h[ix].max() / bs) - b0) + 1; prof = np.zeros(nb)
    for a, b, w in zip(np.floor(l[ix] / bs) - b0, np.floor(h[ix] / bs) - b0, v[ix]): a, b = int(a), int(b); prof[a:b + 1] += w / (b - a + 1)
    sm = np.convolve(prof, np.ones(5) / 5, 'same'); med = np.median(sm[sm > 0]) if (sm > 0).any() else 0; cum = np.cumsum(prof) / max(prof.sum(), 1)
    lo_k, hi_k = int(np.searchsorted(cum, .1)), int(np.searchsorted(cum, .9)); pr = lambda k: (b0 + k + .5) * bs; lvn, shelf = [], []
    for k in range(max(3, lo_k), min(nb - 3, hi_k + 1)):
        w_ = sm[k - 3:k + 4]
        if sm[k] == w_.min() and sm[k] < .5 * med: lvn.append(pr(k))
        if sm[k] == w_.max() and sm[k] > 1.5 * med:
            for dk in (1, -1):
                q = k
                while 0 < q < nb - 1 and sm[q] >= .5 * sm[k]: q += dk
                shelf.append(pr(q))
    return np.array(lvn), np.array(shelf)
LEV = {}
for k, d in enumerate(days):
    if k == 0: continue
    pix = gi[days[k - 1]]; lvn, shelf = vp_nodes(pix); PS_ = prior_stats(pix); val, poc, vah = VA[days[k - 1]]
    wl = WL[WL.index < d]; walls = np.array([wl.call_wall_nq.iloc[-1], wl.put_wall_nq.iloc[-1]]) if len(wl) and (d - wl.index[-1]).days < 6 else np.array([])
    LEV[d] = {'LVN': lvn, 'SP': PS_['sp'], 'SHELF': np.r_[shelf, [val, poc, vah, PS_['tval'], PS_['tpoc'], PS_['tvah']]], 'WALL': walls}
def zones_and_trades():
    rows = []; EXITS = {'10pt': lambda adr: (11.0, 22.0), 'adr06': lambda adr: (.06 * adr + 1.0, .12 * adr), 'adr10': lambda adr: (.10 * adr + 1.0, .15 * adr)}
    for d in days:
        if not ok(d) or DY.roll[d] or d not in LEV: continue
        ix = gi[d]; endp = np.where(m[ix] < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; adr = DY.adr[d]; tol = .03 * adr; Z = []; busy = {}; rng = h[ix] - l[ix]
        for p in range(len(ix)):
            i = ix[p]
            if p >= 2:
                if l[i] > h[ix[p - 2]]: Z.append(['FVG', 1, h[ix[p - 2]], l[i], p])
                if h[i] < l[ix[p - 2]]: Z.append(['FVG', -1, h[i], l[ix[p - 2]], p])
            if p >= 6:
                avg = rng[max(0, p - 12):p].mean()
                if rng[p] >= 1.5 * avg:
                    if c[i] > h[ix[p - 6:p]].max():
                        for q in range(p - 1, max(-1, p - 4), -1):
                            if c[ix[q]] < o[ix[q]]: Z.append(['OB', 1, l[ix[q]], h[ix[q]], p]); break
                    if c[i] < l[ix[p - 6:p]].min():
                        for q in range(p - 1, max(-1, p - 4), -1):
                            if c[ix[q]] > o[ix[q]]: Z.append(['OB', -1, l[ix[q]], h[ix[q]], p]); break
            if p >= 5:
                s_ = p - 2; j_ = ix[s_]; r_ = h[j_] - l[j_]
                if r_ > 0:
                    if l[j_] < l[ix[s_ - 3:s_]].min() and l[j_] < l[ix[s_ + 1:p + 1]].min() and (min(o[j_], c[j_]) - l[j_]) / r_ >= .5: Z.append(['RB', 1, l[j_], min(o[j_], c[j_]), p])
                    if h[j_] > h[ix[s_ - 3:s_]].max() and h[j_] > h[ix[s_ + 1:p + 1]].max() and (h[j_] - max(o[j_], c[j_])) / r_ >= .5: Z.append(['RB', -1, max(o[j_], c[j_]), h[j_], p])
            t = m[i] + 5; keep = []
            for z in Z:
                ty, dr, lo_, hi_, fp = z
                if fp >= p: keep.append(z); continue
                if (dr > 0 and c[i] < lo_) or (dr < 0 and c[i] > hi_): continue
                hit = (dr > 0 and l[i] <= hi_ and c[i] >= lo_) or (dr < 0 and h[i] >= lo_ and c[i] <= hi_)
                if not hit or t < 585 or t > 900 or p + 1 > endp: keep.append(z); continue
                al = {cat: bool(len(L_) and ((L_ >= lo_ - tol) & (L_ <= hi_ + tol)).any()) for cat, L_ in LEV[d].items()}
                j = p + 1; e = o[ix[j]]
                for xn, fx in EXITS.items():
                    cnt, bz = busy.get((ty, xn), (0, -1))
                    if cnt >= 3 or j <= bz: continue
                    T, S = fx(adr); hi2, lo2, op2, cl2 = h[ix[j:endp + 1]], l[ix[j:endp + 1]], o[ix[j:endp + 1]], c[ix[j:endp + 1]]
                    adv = (lo2 <= e - S) if dr > 0 else (hi2 >= e + S); fav = (hi2 >= e + T) if dr > 0 else (lo2 <= e - T); exc = (lo2 - e) if dr > 0 else (e - hi2)
                    ia = int(np.argmax(adv)) if adv.any() else 10 ** 9; ib = int(np.argmax(fav)) if fav.any() else 10 ** 9
                    if ia == ib == 10 ** 9: pts, mae, xe = (cl2[-1] - e) * dr - 1.0, exc.min() - 1.0, len(hi2) - 1
                    elif ia < ib or (ia == ib and (cl2[ia] >= op2[ia]) == (dr > 0)): pts, mae, xe = -S - 1.0, -S - 1.0, ia
                    else: pts, mae, xe = T - 1.0, max(exc[:ib + 1].min(), -S) - 1.0, ib
                    rows.append(dict(day=d, ty=ty, xn=xn, dir=dr, m=m[ix[j]], pts=pts, mae=mae, T=T, gneg=DY.gex.get(d, np.nan) < 0, **al)); busy[(ty, xn)] = (cnt + 1, j + xe)
            Z = keep
    return pd.DataFrame(rows)
if __name__ == '__main__':
    LOCK['on'] = False; TR = zones_and_trades(); TR = TR[TR.day >= '2018-02-01']; TR['per'] = np.where(TR.day.dt.year >= 2022, 'b', 'a'); TR['ANY'] = TR[['LVN', 'SP', 'SHELF', 'WALL']].any(axis=1)
    nd = {p_: sum(1 for d in days if ok(d) and ((d.year >= 2022) == (p_ == 'b')) and d >= pd.Timestamp('2018-02-01')) for p_ in 'ab'}
    print(f'ICT trades: {len(TR):,}; aligned share: ' + ', '.join(f'{k_} {TR[k_].mean():.0%}' for k_ in ('LVN', 'SP', 'SHELF', 'WALL', 'ANY')), flush=True)
    print('points per trade (hit rate) [trades/mo] | 2018-21 / 2022-25', flush=True)
    PASS = []
    for (ty, xn), g in TR.groupby(['ty', 'xn']):
        print(f'\n  {ty} {xn}', flush=True)
        def cell(msk):
            a, b = g[msk & (g.per == 'a')], g[msk & (g.per == 'b')]
            return (a.pts.mean(), b.pts.mean(), len(a), len(b)), f'{a.pts.mean():+6.2f} ({np.mean(a.pts > 0):.0%}) [{len(a) / nd["a"] * 21:4.1f}] / {b.pts.mean():+6.2f} ({np.mean(b.pts > 0):.0%}) [{len(b) / nd["b"] * 21:4.1f}]'
        base, txt = cell(np.ones(len(g), bool)); print(f'    {"all":<26} {txt}', flush=True)
        _, txt = cell(~g.ANY.values); print(f'    {"not aligned":<26} {txt}', flush=True); na_ = cell(~g.ANY.values)[0]
        for cat in ('LVN', 'SP', 'SHELF', 'WALL', 'ANY'):
            st_, txt = cell(g[cat].values); print(f'    {"aligned " + cat:<26} {txt}', flush=True)
            if st_[0] > 0 and st_[1] > 0 and st_[0] > na_[0] and st_[1] > na_[1] and st_[2] >= 30 and st_[3] >= 30: PASS.append((ty, xn, cat))
        for gl, gm in (('aligned ANY, neg gamma', g.ANY.values & g.gneg.values), ('aligned ANY, pos gamma', g.ANY.values & ~g.gneg.values)):
            _, txt = cell(gm); print(f'    {gl:<26} {txt}', flush=True)
    print(f'\nPASS: {PASS or "none"}', flush=True)
    if PASS:
        S6 = SET['S6 S3 + V15a']; VAR = {'S6 as is': S6}
        for ty, xn, cat in PASS[:4]:
            T_ = TR[(TR.ty == ty) & (TR.xn == xn) & TR[cat]].copy(); T_['sz'] = np.minimum(np.ceil(150 / (2 * (T_['T'] - 1.0))), 17) / 10; T_['ce'] = T_['cf'] = 1.0
            VAR[f'S6 + {ty} {xn} aligned {cat}'] = S6 + [(T_[['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf']], 1.0, None)]
        print('\nfleet: Lucid Flex x5, 24 monthly starts 2022-23, budget 40 fees', flush=True)
        for zn in ('150K', '50K'):
            for nm, sp in VAR.items():
                o_ = multi(sp, sp, zn, budgets=(40,))[40]
                print(f'  {zn:<5} {nm:<34} avg ${o_["avg"]:6,.0f} / median ${o_["med"]:6,.0f} | >= $2.5k {o_["ge25"]:4.0%}, >= $5k {o_["ge5"]:4.0%}, losing {o_["lose"]:4.0%}, dry {o_["dry"]:.1f} mo', flush=True)
