"""Nov 9 2026: VECTOR FLIP -> VWAP (Rey's idea). DECLARED BEFORE RESULTS. Real NQ futures 5m, RTH vector bars carried across days, 1 pt cost.
Rule: when the vector flips (bar close 09:45-15:30) and VWAP is on the trade's side of the entry (long: VWAP above the entry; short: below), enter at the next 5m open;
STOP at the vector line (the ratchet level the flip sets), TARGET = VWAP (touch of the live session VWAP), out on an opposite flip or at 15:55. One position per
timeframe, <= 4 a day. Stop variants: the line fixed at entry | the line trailing as the vector ratchets. Timeframes: 15m, 30m, 45m, 1h (bars start 09:30).
Reported: trades / month, VWAP reached, points per trade, average distance to VWAP and to the stop (ADR), by gamma sign at the prior close (dealers long gamma pull price
back -> VWAP should work better in positive gamma), and by distance to VWAP (< 0.05 / 0.05-0.15 / > 0.15 ADR); 2018-21 and 2022-25.
Comparison: the same flips WITHOUT the VWAP condition, plain vector exits (opposite flip, stop 0.15 ADR).
Pass: positive per trade in BOTH periods with >= 30 trades each -> fleet test on S6 at 3 micros (Lucid Flex x5, 24 monthly starts 2022-23)."""
_m = open('/home/claude/fartpie/xs/mtf.py').read(); exec(_m[:_m.rindex("if __name__ == '__main__':")])
src = open('/home/claude/fartpie/xs/signal_lab.py').read(); vsrc = src[src.index('def vec(df, **kw):'):src.index('def donch(')]
vsrc = vsrc.replace('def vec(df, **kw):', 'def vec_lv(df, **kw):').replace('regs = np.zeros(n, int)', 'regs = np.zeros(n, int); lvs = np.full(n, np.nan)')
vsrc = vsrc.replace('        regs[i] = regime\n    return regs', '        regs[i] = regime; lvs[i] = level\n    return regs, lvs')
exec(vsrc, NSL); vec_lv = NSL['vec_lv']
TFL = {}
for k in (15, 30, 45, 60):
    Bk = tf_bars(k) if k != 15 else None
    if Bk is None:
        g = pd.DataFrame({'day': dd, 'b': (m - 570) // 15, 'o': o, 'h': h, 'l': l, 'c': c, 't': Hd})
        Bk = g.groupby(['day', 'b']).agg(open=('o', 'first'), high=('h', 'max'), low=('l', 'min'), close=('c', 'last'), start=('t', 'first'), last=('t', 'last')).reset_index()
        Bk['end'] = Bk['last'] + np.timedelta64(5, 'm')
    rg, lv = vec_lv(Bk[['open', 'high', 'low', 'close']]); Bk['reg'], Bk['lvl'] = rg, lv; TFL[k] = Bk
def run_vw(Bk, need_vwap=True, trail=False, plain=False):
    endmap = {t: q for q, t in enumerate(Bk.end.values)}; s = Bk.reg.values; lvv = Bk.lvl.values; ends = Bk.end.values; rows = []; busy = {}
    for q in range(1, len(Bk)):
        if s[q] == s[q - 1] or s[q] == 0: continue
        t_end = pd.Timestamp(ends[q]); mm_ = t_end.hour * 60 + t_end.minute
        if not (585 <= mm_ <= 930): continue
        i = tpos5.get(np.datetime64(ends[q], 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i])
        if not ok(d): continue
        cnt, bz = busy.get(d, (0, -1))
        if cnt >= 4 or i <= bz: continue
        dr = int(s[q]); e = o[i]; adr = DY.adr[d]; vwe = vw[i - 1]
        if need_vwap and (vwe - e) * dr <= 0: continue
        S = e - dr * .15 * adr if plain else lvv[q]
        if not np.isfinite(S) or (e - S) * dr <= 0: continue
        ix = gi[d]; j = int(np.where(ix == i)[0][0]); res = None; worst = 0.0; hitv = False
        for p in range(j, len(ix)):
            ii = ix[p]
            if m[ii] >= 955: res = ((c[ix[p - 1]] - e) * dr, ii); break
            if (l[ii] <= S) if dr > 0 else (h[ii] >= S): res = ((S - e) * dr, ii); worst = (S - e) * dr; break
            worst = min(worst, ((l[ii] if dr > 0 else h[ii]) - e) * dr)
            if not plain:
                tv = vw[ii - 1] if p > 0 else vwe
                if (h[ii] >= tv) if dr > 0 else (l[ii] <= tv):
                    if (tv - e) * dr > 0: res = ((tv - e) * dr, ii); hitv = True; break
            qe = endmap.get(np.datetime64(Hd[ii] + np.timedelta64(5, 'm')))
            if qe is not None:
                if s[qe] == -dr and p + 1 < len(ix): res = ((o[ix[p + 1]] - e) * dr, ix[p + 1]); break
                if trail and np.isfinite(lvv[qe]) and (lvv[qe] - S) * dr > 0: S = lvv[qe]
        if res is None: res = ((c[ix[-1]] - e) * dr, ix[-1])
        rows.append((d, m[i], dr, res[0] - 1.0, worst - 1.0, hitv, (vwe - e) * dr / adr, (e - (lvv[q] if not plain else S)) * dr / adr, DY.gex.get(d, np.nan)))
        busy[d] = (cnt + 1, res[1])
    T_ = pd.DataFrame(rows, columns=['day', 'm', 'dir', 'pts', 'mae', 'hitv', 'dv', 'ds', 'gex']); T_ = T_[T_.day >= '2018-02-01']; T_['per'] = np.where(T_.day.dt.year >= 2022, 'b', 'a'); return T_
if __name__ == '__main__':
    LOCK['on'] = False; nd = {p_: sum(1 for d in days if ok(d) and ((d.year >= 2022) == (p_ == 'b')) and d >= pd.Timestamp('2018-02-01')) for p_ in 'ab'}
    def fmt(T_):
        a, b = T_[T_.per == 'a'], T_[T_.per == 'b']
        return (f'{a.pts.mean():+6.2f} [{len(a) / nd["a"] * 21:4.1f}/mo, VWAP hit {a.hitv.mean():3.0%}] / {b.pts.mean():+6.2f} [{len(b) / nd["b"] * 21:4.1f}/mo, VWAP hit {b.hitv.mean():3.0%}]',
                a.pts.mean(), b.pts.mean(), len(a), len(b))
    print('points per trade [trades/mo, VWAP reached] | 2018-21 / 2022-25', flush=True); PASS = {}
    for k, Bk in TFL.items():
        print(f'\n=== {k}m vector ===', flush=True)
        for lab, kw in (('flip toward VWAP, stop = line', dict()), ('flip toward VWAP, line trails', dict(trail=True)), ('every flip, stop = line, target VWAP if ahead', dict(need_vwap=False)),
                        ('plain flips (0.15 ADR stop, flip exit)', dict(need_vwap=False, plain=True))):
            T_ = run_vw(Bk, **kw); txt, a, b, na, nb = fmt(T_)
            print(f'  {lab:<46} {txt} | to VWAP {T_.dv.mean():.2f} ADR, to stop {T_.ds.mean():.2f} ADR', flush=True)
            if lab.startswith('flip toward') and a > 0 and b > 0 and na >= 30 and nb >= 30: PASS[(k, lab)] = T_
            if lab == 'flip toward VWAP, stop = line':
                for gl, gm in (('  positive gamma', T_.gex > 0), ('  negative gamma', T_.gex < 0)):
                    print(f'    {gl:<44} {fmt(T_[gm])[0]}', flush=True)
                for bl, bm in (('  VWAP < 0.05 ADR away', T_.dv < .05), ('  VWAP 0.05-0.15 ADR', (T_.dv >= .05) & (T_.dv < .15)), ('  VWAP > 0.15 ADR', T_.dv >= .15)):
                    print(f'    {bl:<44} {fmt(T_[bm])[0]}', flush=True)
    print(f'\nPASS: {list(PASS) or "none"}', flush=True)
    if PASS:
        S6 = SET['S6 S3 + V15a']; VAR = {'S6 as is': S6}
        for (k, lab), T_ in list(PASS.items())[:4]: VAR[f'S6 + {k}m {lab[:22]}'] = S6 + [(T_[['day', 'm', 'pts', 'mae']], .3, None)]
        print('\nfleet: Lucid Flex x5, 24 monthly starts 2022-23, budget 40 fees', flush=True)
        for zn in ('150K', '50K'):
            for nm, sp in VAR.items():
                o_ = multi(sp, sp, zn, budgets=(40,))[40]
                print(f'  {zn:<5} {nm:<44} avg ${o_["avg"]:6,.0f} / median ${o_["med"]:6,.0f} | >= $2.5k {o_["ge25"]:4.0%}, >= $5k {o_["ge5"]:4.0%}, losing {o_["lose"]:4.0%}, dry {o_["dry"]:.1f} mo', flush=True)
