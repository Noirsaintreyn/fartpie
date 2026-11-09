"""Nov 9 2026 (mtf_why: why did the 30m vector - best per trade - make the fleet worse? Overlap and correlation with the engines already in S6, effect on the worst days,
and per-account eval / funded stats with vs without it; same for 45m. Previous header:) HIGHER-TIMEFRAME VECTORS AND 1H-CONTEXT SCALPS. DECLARED BEFORE RESULTS. Real NQ futures 5m, RTH bars (09:30-16:00) carried across days, 1 pt cost.
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
    LOCK['on'] = False; A30 = run_tf(TFB[30], True); A45 = run_tf(TFB[45], True)
    S6 = SET['S6 S3 + V15a']; names = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a']
    def daily_usd(spec):
        L_ = lists(spec, 'f'); return pd.Series([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in zz) for zz in L_], index=D_)
    ED = pd.DataFrame({nm: daily_usd([s_]) for nm, s_ in zip(names, S6)}); B6 = ED.sum(axis=1)
    for lab, A in (('30m', A30), ('45m', A45)):
        X_ = daily_usd([(A, .3, None)]); act = X_ != 0; w = D_ >= pd.Timestamp('2022-01-01')
        print(f'\n=== {lab} vector (gated, 3 micros) ===', flush=True)
        print(f'  trades on {act.mean():.0%} of days; correlation of its daily $ with: ' + ', '.join(f'{n_} {np.corrcoef(X_[act & (ED[n_] != 0)], ED[n_][act & (ED[n_] != 0)])[0, 1]:+.2f}' for n_ in ('RA', 'M1', 'V15a', 'RULES')) +
              f' | with the S6 book {np.corrcoef(X_[act], B6[act])[0, 1]:+.2f}', flush=True)
        v15 = V15a.set_index('day').dir.groupby(level=0).first(); a_dir = A.groupby('day').dir.first(); both = a_dir.index.intersection(v15.index)
        print(f'  days it trades that V15a also trades: {len(both) / len(a_dir):.0%}; same direction on those days: {np.mean(a_dir[both].values == v15[both].values):.0%}', flush=True)
        for per, msk in (('all', np.ones(len(D_), bool)), ('2022+', w)):
            b0, b1 = B6[msk], (B6 + X_)[msk]; q0 = b0.quantile(.05)
            print(f'  {per}: S6 avg day ${b0.mean():,.0f} -> ${b1.mean():,.0f}; worst 5% of S6 days ${b0[b0 <= q0].mean():,.0f} -> ${b1[b0 <= q0].mean():,.0f}; '
                  f'its own worst 5% days ${X_[msk][X_[msk] <= X_[msk][X_[msk] != 0].quantile(.05)].mean():,.0f}; days S6 >= $150 {np.mean(b0 >= 150):.0%} -> {np.mean(b1 >= 150):.0%}; '
                  f'biggest day share check: max day / month total', flush=True)
        R = FIRM['Lucid Flex']
        for zn in ('50K', '150K'):
            z = SZ[zn]; out = {}
            for tag, sp in (('S6', S6), ('S6+' + lab, S6 + [(A, .3, None)])):
                globals()['LE'], globals()['LF'] = lists(sp, 'e'), lists(sp, 'f'); ev_ = [eval_run(R, z, s_, NS) for s_ in range(k22, k25, 5)]
                fu_ = [fund_run(R, z, s_, NS) for s_ in range(k22, k25, 5)]; pays = [x for f_ in fu_ for _, x in f_[2]]
                life = sum(((NS if f_[0] == 'open' else f_[1]) - s_ + 1) / 21 for f_, s_ in zip(fu_, range(k22, k25, 5)))
                out[tag] = (np.mean([r_ == 'pass' for r_, k_ in ev_]), np.median([k_ - s_ for (r_, k_), s_ in zip(ev_, range(k22, k25, 5)) if r_ == 'pass']),
                            np.mean([f_[0] == 'lost' for f_ in fu_]), len(pays) / life, np.mean(pays) if pays else 0)
            print(f'  {zn}: eval pass {out["S6"][0]:.0%} -> {out["S6+" + lab][0]:.0%}, days to pass {out["S6"][1]:.0f} -> {out["S6+" + lab][1]:.0f} | funded blown {out["S6"][2]:.0%} -> {out["S6+" + lab][2]:.0%}, '
                  f'payouts/mo {out["S6"][3]:.2f} -> {out["S6+" + lab][3]:.2f}, avg payout ${out["S6"][4]:,.0f} -> ${out["S6+" + lab][4]:,.0f}', flush=True)
