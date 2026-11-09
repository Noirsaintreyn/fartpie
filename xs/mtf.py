"""Nov 9 2026: HIGHER-TIMEFRAME VECTORS AND 1H-CONTEXT SCALPS. DECLARED BEFORE RESULTS. Real NQ futures 5m, RTH bars (09:30-16:00) carried across days, 1 pt cost.
A) VECTOR ENGINES at 30m / 45m / 1h (bars start 09:30 each day): flip entries at bar ends 09:45-15:30 (next 5m open), exit on the opposite flip at a bar end, stop 0.15 ADR,
   flat 15:55, one position, <= 4 a day; each with and without the top-third-gamma gate (V15a-style). Also 15m flips (V15a) only when the 1h vector agrees.
B) 1H-CONTEXT SCALPS: long setups only while the last completed 1h vector bar is BULL, shorts only while BEAR; decided at a 5m close 09:45-15:00, entry next open,
   one position per setup, <= 3 a day:  VW  VWAP rejection (bar trades through VWAP and closes back on the trade side)   FVG  ICT fair-value-gap retrace (price trades
   into the latest unfilled 3-bar gap from the last 12 bars in the trade direction and closes beyond the gap's far edge)   EMA  pullback to the 5m EMA20 and close back beyond it.
   Exits: +10 net (+11 gross) before -22, or +(0.06 ADR + 1) before -0.12 ADR; flat 15:55. Each setup also run IGNORING the 1h vector (both sides) and AGAINST it.
Pass for B: with-context beats ignoring-context per trade in BOTH 2018-21 and 2022-25 AND is positive in both. Passing engines then go into the fleet (Lucid Flex x5,
24 monthly starts 2022-23, xs/fleet_compare2.py) on top of S6 at a size chosen BEFORE: A engines 0.3 (3 micros); B scalps sized so a win pays $150+ at 50K."""
_fc2 = open('/home/claude/fartpie/xs/fleet_compare2.py').read(); exec(_fc2[:_fc2.rindex("if __name__ == '__main__':")])
Hd = H.dt.values; ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
def tf_bars(k):
    g = pd.DataFrame({'day': dd, 'b': (m - 570) // k, 'o': o, 'h': h, 'l': l, 'c': c, 't': Hd})
    Bk = g.groupby(['day', 'b']).agg(open=('o', 'first'), high=('h', 'max'), low=('l', 'min'), close=('c', 'last'), start=('t', 'first'), last=('t', 'last')).reset_index()
    Bk['end'] = Bk['last'] + np.timedelta64(5, 'm'); Bk['reg'] = np.asarray(vec(Bk[['open', 'high', 'low', 'close']])); return Bk
TFB = {k: tf_bars(k) for k in (30, 45, 60)}
def run_tf(Bk, gate):
    regend = dict(zip(Bk.end.values, Bk.reg.values)); s = np.nan_to_num(Bk.reg.values.astype(float)); ends = Bk.end.values; rows = []; busy = {}
    for q in range(1, len(Bk)):
        if s[q] == s[q - 1] or s[q] == 0: continue
        t_end = pd.Timestamp(ends[q]); mm_ = t_end.hour * 60 + t_end.minute
        if not (585 <= mm_ <= 930): continue
        i = tpos5.get(np.datetime64(ends[q], 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i])
        if not ok(d) or (gate and DY.g2p.get(d, np.nan) >= 2 / 3): continue
        cnt, bz = busy.get(d, (0, -1))
        if cnt >= 4 or i <= bz: continue
        dr = int(s[q]); e = o[i]; S = e - dr * .15 * DY.adr[d]; ix = gi[d]; j = int(np.where(ix == i)[0][0]); res = None; worst = 0.0
        for p in range(j, len(ix)):
            ii = ix[p]
            if m[ii] >= 955: res = ((c[ix[p - 1]] - e) * dr, ii); break
            if (l[ii] <= S) if dr > 0 else (h[ii] >= S): res = ((S - e) * dr, ii); worst = (S - e) * dr; break
            worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
            r_end = regend.get(np.datetime64(Hd[ii] + np.timedelta64(5, 'm')))
            if r_end is not None and r_end == -dr and p + 1 < len(ix): res = ((o[ix[p + 1]] - e) * dr, ix[p + 1]); break
        if res is None: res = ((c[ix[-1]] - e) * dr, ix[-1])
        rows.append((d, m[i], dr, res[0] - 1.0, worst - 1.0)); busy[d] = (cnt + 1, res[1])
    return pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts', 'mae'])
B60 = TFB[60]; e60 = B60.end.values.astype('datetime64[ns]'); r60 = B60.reg.values
def reg1h_at(i):
    k = np.searchsorted(e60, Hd[i] + np.timedelta64(5, 'm'), 'right') - 1; return r60[k] if k >= 0 else 0
REG1H = np.array([reg1h_at(i) for i in range(len(Hd))])
def setups_at(ix, p):
    i = ix[p]; out = []
    if l[i] <= vw[i] < c[i]: out.append(('VW', 1))
    if h[i] >= vw[i] > c[i]: out.append(('VW', -1))
    if l[i] <= ema20[i] < c[i]: out.append(('EMA', 1))
    if h[i] >= ema20[i] > c[i]: out.append(('EMA', -1))
    for dr in (1, -1):
        for j in range(p - 1, max(1, p - 12), -1):
            a_, b_ = ix[j - 2], ix[j]
            gap = (h[a_], l[b_]) if dr > 0 else (h[b_], l[a_])
            if gap[1] <= gap[0]: continue
            filled = (l[ix[j + 1:p]] <= gap[0]).any() if dr > 0 else (h[ix[j + 1:p]] >= gap[1]).any()
            if filled: break
            if (dr > 0 and l[i] <= gap[1] and c[i] > gap[0]) or (dr < 0 and h[i] >= gap[0] and c[i] < gap[1]): out.append(('FVG', dr))
            break
    return out
def scalps():
    rows = []
    for d in days:
        if not ok(d) or DY.roll[d]: continue
        ix = gi[d]; endp = np.where(m[ix] < 955)[0]
        if not len(endp): continue
        endp = endp[-1]; adr = DY.adr[d]; busy = {}
        for p in range(2, len(ix)):
            t = m[ix[p]] + 5
            if t < 585 or t > 900 or p + 1 > endp: continue
            for st, dr in setups_at(ix, p):
                for tg, (T, S) in (('10pt', (11.0, 22.0)), ('adr', (.06 * adr + 1.0, .12 * adr))):
                    key = (st, tg, 'any'); ctx = REG1H[ix[p]] * dr
                    for mode in ('any', 'with' if ctx > 0 else ('against' if ctx < 0 else None)):
                        if mode is None: continue
                        key = (st, tg, mode); cnt, bz = busy.get(key, (0, -1))
                        if cnt >= 3 or p + 1 <= bz: continue
                        j = p + 1; e = o[ix[j]]; hi_, lo_, op_, cl_ = h[ix[j:endp + 1]], l[ix[j:endp + 1]], o[ix[j:endp + 1]], c[ix[j:endp + 1]]
                        adv = (lo_ <= e - S) if dr > 0 else (hi_ >= e + S); fav = (hi_ >= e + T) if dr > 0 else (lo_ <= e - T)
                        ia = int(np.argmax(adv)) if adv.any() else 10 ** 9; ib = int(np.argmax(fav)) if fav.any() else 10 ** 9; exc = (lo_ - e) if dr > 0 else (e - hi_)
                        if ia == ib == 10 ** 9: pts, mae, xe = (cl_[-1] - e) * dr - 1.0, exc.min() - 1.0, len(hi_) - 1
                        elif (ia < ib) or (ia == ib and (cl_[ia] >= op_[ia]) == (dr > 0)): pts, mae, xe = -S - 1.0, -S - 1.0, ia
                        else: pts, mae, xe = T - 1.0, max(exc[:ib + 1].min(), -S) - 1.0, ib
                        rows.append((d, st, tg, mode, dr, m[ix[j]], pts, mae, T)); busy[key] = (cnt + 1, j + xe)
    return pd.DataFrame(rows, columns=['day', 'st', 'tg', 'mode', 'dir', 'm', 'pts', 'mae', 'T'])
if __name__ == '__main__':
    LOCK['on'] = False; per = lambda T_: np.where(T_.day.dt.year >= 2022, 'b', 'a')
    nd = {p_: sum(1 for d in days if ok(d) and ((d.year >= 2022) == (p_ == 'b')) and d >= pd.Timestamp('2018-02-01')) for p_ in 'ab'}
    print('A) vector engines: points per trade (trades/month) | 2018-21 / 2022-25 | total points 2022-25', flush=True)
    AE = {}
    for k in (30, 45, 60):
        for gate in (False, True):
            T_ = run_tf(TFB[k], gate); T_ = T_[T_.day >= '2018-02-01']; T_['per'] = per(T_); AE[(k, gate)] = T_
            a, b = T_[T_.per == 'a'].pts, T_[T_.per == 'b'].pts
            print(f'  {k}m {"gamma-gated" if gate else "every flip ":<11} {a.mean():+6.2f} ({len(a) / nd["a"] * 21:4.1f}/mo) / {b.mean():+6.2f} ({len(b) / nd["b"] * 21:4.1f}/mo) | {b.sum():+7.0f}', flush=True)
    Va = V15a[V15a.day >= '2018-02-01'].copy(); Va['per'] = per(Va); Va['i'] = [tidx.get((pd.Timestamp(d), mm_)) for d, mm_ in zip(Va.day, Va.m)] if 'tidx' in globals() else None
    agree = np.array([REG1H[gi[pd.Timestamp(d)][np.where(m[gi[pd.Timestamp(d)]] == mm_)[0][0]] - 1] * dr > 0 if (m[gi[pd.Timestamp(d)]] == mm_).any() else False for d, mm_, dr in zip(Va.day, Va.m, Va.dir)])
    for lab, msk in (('15m V15a, all', np.ones(len(Va), bool)), ('15m V15a, 1h agrees', agree), ('15m V15a, 1h disagrees', ~agree)):
        a, b = Va[msk & (Va.per == 'a')].pts, Va[msk & (Va.per == 'b')].pts
        print(f'  {lab:<26} {a.mean():+6.2f} ({len(a) / nd["a"] * 21:4.1f}/mo) / {b.mean():+6.2f} ({len(b) / nd["b"] * 21:4.1f}/mo) | {b.sum():+7.0f}', flush=True)
    print('\nB) 1h-context scalps: hit rate, points per trade, trades/month | 2018-21 / 2022-25   (with 1h vector vs ignoring it vs against it)', flush=True)
    SC = scalps(); SC = SC[SC.day >= '2018-02-01']; SC['per'] = per(SC); PASS = []
    for (st, tg), g in SC.groupby(['st', 'tg']):
        cells = {}
        for mode in ('with', 'any', 'against'):
            a, b = g[(g['mode'] == mode) & (g.per == 'a')], g[(g['mode'] == mode) & (g.per == 'b')]; cells[mode] = (a.pts.mean(), b.pts.mean())
            print(f'  {st:<4} {tg:<5} {mode:<8} hit {np.mean(a.pts > 0):.0%} / {np.mean(b.pts > 0):.0%} | {a.pts.mean():+6.2f} / {b.pts.mean():+6.2f} | {len(a) / nd["a"] * 21:5.1f} / {len(b) / nd["b"] * 21:5.1f} per mo', flush=True)
        w, an = cells['with'], cells['any']
        if w[0] > 0 and w[1] > 0 and w[0] > an[0] and w[1] > an[1]: PASS.append((st, tg))
    print(f'  PASS (with-context > ignoring, and positive, in both periods): {PASS or "none"}', flush=True)
    AP = [(k, gt) for (k, gt), T_ in AE.items() if T_[T_.per == 'a'].pts.mean() > 0 and T_[T_.per == 'b'].pts.mean() > 0]
    print(f'  A engines positive in both periods: {AP}', flush=True)
    S6 = SET['S6 S3 + V15a']; VAR = {'S6 as is': S6}
    for k, gt in AP:
        if gt: VAR[f'S6 + {k}m vector (gated)'] = S6 + [(AE[(k, gt)], .3, None)]
    for st, tg in PASS:
        T_ = SC[(SC.st == st) & (SC.tg == tg) & (SC['mode'] == 'with')].copy(); T_['sz'] = np.ceil(150 / (2 * (T_['T'] - 1.0))) / 10; T_['ce'] = T_['cf'] = 1.0
        VAR[f'S6 + {st} {tg} scalps (1h context)'] = S6 + [(T_[['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf']], 1.0, None)]
    print('\nC) fleet: Lucid Flex x5, 24 monthly starts 2022-23, budget 40 fees', flush=True)
    for zn in ('150K', '50K'):
        for nm, sp in VAR.items():
            o_ = multi(sp, sp, zn, budgets=(40,))[40]
            print(f'  {zn:<5} {nm:<34} avg ${o_["avg"]:6,.0f} / median ${o_["med"]:6,.0f} | >= $2.5k {o_["ge25"]:4.0%}, >= $5k {o_["ge5"]:4.0%}, losing {o_["lose"]:4.0%}, dry {o_["dry"]:.1f} mo', flush=True)
