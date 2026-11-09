"""Nov 8 2026: THE DAY-TYPE ENGINE INSIDE LUCID - can it reach 2 payouts a month per account? DECLARED BEFORE RESULTS.
From xs/dte_engines.py on real NQ futures 5m bars (Feb 2018 - Jan 2026, the GEX years): noise momentum only paid on NEGATIVE-gamma days (+6.87 vs +0.34 a trade),
the 80% rule only on POSITIVE-gamma days (+2.10 vs 0.00), the VWAP fade lost everywhere (dropped), Rule A paid in both.
Streams (NQ points per contract after 1 pt, with max adverse excursion, in entry-time order within the day):
  RA   Rule A trades re-filled on NQ futures (the saved entry time and direction, 75 / 50, flat 12:00)
  M1g  noise momentum on negative-gamma days only (hard stop 0.15 ADR)        R1g  80% rule on positive-gamma days only
Versions (micros at 50K; x2.25 at 150K): V0 RA 10 | V1 RA 10 + M1g 2 | V2 RA 10 + M1g 3 | V3 RA 10 + M1g 5 | V4 RA 10 + M1g 3 + R1g 5 | V5 RA 5 + M1g 5 + R1g 5 | V6 M1g 5 alone
  '+LOCK' = stop trading for the day once the day is up a qualifying amount ($150 / $250): bank the qualifying day.
Lucid engine as xs/nq_account_sizes.py (EOD trailing max loss, +$100 lock, 5 winning days, 50% up to the cap, 90% split, 5 payouts, day stop). One account bought every
5th session Feb 2018 - Jun 2025, run to its end or Jan 2026. Points = $ / 20."""
_e = open('/home/claude/fartpie/xs/dte_engines.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
def walk2(ix, j, d_, e, T, S, until_m=955, maxb=None):
    worst = 0.0
    for q, i in enumerate(ix[j:]):
        if m[i] >= until_m or (maxb and q >= maxb): return ((c[ix[j + q - 1]] - e) * d_ if q > 0 else 0.0), worst
        adv = ((l[i] if d_ > 0 else h[i]) - e) * d_
        if ((l[i] <= S) if d_ > 0 else (h[i] >= S)): return (S - e) * d_, min(worst, (S - e) * d_)
        worst = min(worst, adv)
        if ((h[i] >= T) if d_ > 0 else (l[i] <= T)): return (T - e) * d_, worst
    return (c[ix[-1]] - e) * d_, worst
ts_ = H.dt.values; tpos = {t: i for i, t in enumerate(ts_)}
def ra_trades():
    R = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); rows = []
    for r_ in R.itertuples():
        i = tpos.get(np.datetime64(pd.Timestamp(r_.entry), 'ns'))
        if i is None: continue
        d = dd[i]; ix = gi[pd.Timestamp(d)]; j = int(np.where(ix == i)[0][0]); e = o[i]; dr = int(r_.dir)
        pts, mae = walk2(ix, j, dr, e, e + 75 * dr, e - 50 * dr, until_m=720); rows.append((pd.Timestamp(d), m[i], pts - 1.0, mae - 1.0))
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae'])
def m1_trades():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or d not in SIG.index: continue
        ix = gi[d]; o0 = o[ix[0]]; pc = c[gi[days[k - 1]][-1]]; S = .15 * DY.adr[d]; pos = 0; e = None; pend = None; worst = 0.0; me = 0
        for p, i in enumerate(ix):
            if pend is not None: pos, e, worst, me = pend, o[i], 0.0, m[i]; pend = None
            if pos != 0:
                if (l[i] <= e - S) if pos > 0 else (h[i] >= e + S): rows.append((d, me, -S - 1.0, -S - 1.0)); pos = 0; continue
                worst = min(worst, ((l[i] if pos > 0 else h[i]) - e) * pos)
            if p == len(ix) - 1: break
            if (m[i] + 5) % 30 != 0 or m[i] + 5 < 600 or m[i] + 5 > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); nxt = o[ix[p + 1]]
            if pos == 1 and c[i] < max(ub, vw[i]): rows.append((d, me, nxt - e - 1.0, worst - 1.0)); pos = 0
            elif pos == -1 and c[i] > min(lb, vw[i]): rows.append((d, me, e - nxt - 1.0, worst - 1.0)); pos = 0
            if pos == 0 and c[i] > ub: pend = 1
            elif pos == 0 and c[i] < lb: pend = -1
        if pos != 0: rows.append((d, me, (c[ix[-1]] - e) * pos - 1.0, worst - 1.0))
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae'])
def r1_trades():
    rows = []
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d]: continue
        val, poc, vah = VA[days[k - 1]]; ix = gi[d]; o0 = o[ix[0]]; side = 1 if o0 > vah else (-1 if o0 < val else 0)
        if side == 0: continue
        run = 0
        for p, i in enumerate(ix[:-1]):
            if m[i] >= 780: break
            run = run + 1 if val <= c[i] <= vah else 0
            if run >= 6:
                d_ = -side; e = o[ix[p + 1]]; T = val if d_ < 0 else vah; S = (vah + .08 * DY.adr[d]) if d_ < 0 else (val - .08 * DY.adr[d])
                if (T - e) * d_ <= 0 or (e - S) * d_ <= 0: break
                pts, mae = walk2(ix, p + 1, d_, e, T, S); rows.append((d, m[ix[p + 1]], pts - 1.0, mae - 1.0)); break
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae'])
_as = open('/home/claude/fartpie/xs/nq_account_sizes.py').read(); exec(_as[_as.index('ACC = {'):_as.index('def day(')]); exec(_as[_as.index('def ev('):_as.index('def slot2(')])
LOCK = {'on': False}
def day(D, k, st, A):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in D[k]:
        if cum <= -1000 * A['scale'] or (LOCK['on'] and cum >= A['qday']): break
        mic = max(1, int(round(10 * th[0] * A['scale']))); mm = 2.0 * mic
        if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
        cum += net * mm
    return cum, False
if __name__ == '__main__':
    gneg = (DY.gex < 0); RAt, M1t, R1t = ra_trades(), m1_trades(), r1_trades()
    M1g = M1t[M1t.day.map(gneg) == True]; R1g = R1t[R1t.day.map(gneg) == False]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    for nm, T in (('RA on NQ futures', RAt), ('M1g noise, neg gamma', M1g), ('R1g 80% rule, pos gamma', R1g)):
        T = T[T.day.isin(D_)]; print(f'{nm}: {len(T)} trades, win {np.mean(T.pts > 0):.0%}, avg {T.pts.mean():+.2f} pts, median MAE {T.mae.median():+.1f}', flush=True)
    def lists(spec):
        L = [[] for _ in range(NS)]
        for T, u in spec:
            for r_ in T.itertuples():
                k = kpos.get(r_.day)
                if k is not None and u > 0: L[k].append((r_.m, ((u,), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    V = {'V0 RA 10': [(RAt, 1.0)], 'V1 RA 10 + M1g 2': [(RAt, 1.0), (M1g, .2)], 'V2 RA 10 + M1g 3': [(RAt, 1.0), (M1g, .3)], 'V3 RA 10 + M1g 5': [(RAt, 1.0), (M1g, .5)],
         'V4 RA 10 + M1g 3 + R1g 5': [(RAt, 1.0), (M1g, .3), (R1g, .5)], 'V5 RA 5 + M1g 5 + R1g 5': [(RAt, .5), (M1g, .5), (R1g, .5)], 'V6 M1g 5 alone': [(M1g, .5)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    print(f'\n  {"version":<30} {"acct":<5} pass  paid  payouts x size  blown | per acct-month | 1st payout  every   payouts/mo funded', flush=True)
    for nm, spec in V.items():
        DE = DF = lists(spec)
        for lock in (False, True):
            LOCK['on'] = lock
            for a in ('50K', '150K'):
                A = ACC[a]; R = []
                for s in starts:
                    res, k2 = ev(A, s, NS)
                    if res != 'pass': R.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                    fr, k3, pays = fu(A, k2 + 1, NS, 5); R.append((res, k2, pays, fr, NS if fr == 'open' else k3))
                net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(R, starts)])
                fund = [r for r in R if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
                fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(R, starts) if r[2]]
                flife = sum((r[4] - r[1]) / 21 for r in fund); pm = len(pays) / flife if flife > 0 else 0
                print(f'  {nm + (" +LOCK" if lock else ""):<30} {a:<5} {np.mean([r[0] == "pass" for r in R]):4.0%} {np.mean([len(r[2]) > 0 for r in R]):5.0%}  {np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts '
                      f'{np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | {net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {pm:12.2f}', flush=True)
