"""Nov 8 2026: DAY-TYPE ENGINE + OVERNIGHT DRIFT WITH A STOP inside Lucid. DECLARED BEFORE RESULTS. The stop is always on (k x the 20-day average overnight range,
checked every 5m bar). Candidates from xs/dte_overnight_stops.py (each positive in every period it covers):
  OA 18:00 - 09:25, stop 0.50, negative-gamma nights (+3.45 a night, worst -203)      OB 18:00 - 04:00, stop 0.35, every night (+2.05, worst -150)
  OC 01:00 - 05:00, stop 0.50, negative-gamma nights (+2.62, worst -171)
Each at 2 and 3 micros at 50K (x2.25 at 150K) on top of B0 = V4 (RA 10, M1g 3, R1g 5). One account bought every 5th session Feb 2018 - Jun 2025 (as xs/dte_prop.py)."""
_p = open('/home/claude/fartpie/xs/dte_prop.py').read(); exec(_p[:_p.rindex("if __name__ == '__main__':")])
_s = open('/home/claude/fartpie/xs/dte_overnight_stops.py').read(); exec(_s[_s.index("on = X[~X.rth"):_s.index("if __name__ == '__main__':")])
def on_list(a, b, k, flt):
    rows = []
    for d in R.index:
        if d not in gi or not symok.get(d, False): continue
        if flt == 'neg' and not (bool(gknown.get(d, False)) and bool(gneg.get(d, False))): continue
        r_ = night(d, a, b, k)
        if r_ is not None: rows.append((d, -1, r_[0], min(r_[1], 0) - 1.0))
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae'])
if __name__ == '__main__':
    gx = (DY.gex < 0); RAt, M1t, R1t = ra_trades(), m1_trades(), r1_trades(); M1g = M1t[M1t.day.map(gx) == True]; R1g = R1t[R1t.day.map(gx) == False]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OA, OB, OC = on_list(0, 930, .5, 'neg'), on_list(0, 600, .35, 'all'), on_list(420, 660, .5, 'neg')
    for nm, t in (('OA', OA), ('OB', OB), ('OC', OC)):
        t = t[t.day.isin(D_)]; print(f'{nm}: {len(t)} nights 2018+, win {np.mean(t.pts > 0):.0%}, avg {t.pts.mean():+.2f}, worst {t.pts.min():+.0f}', flush=True)
    def lists(spec):
        L = [[] for _ in range(NS)]
        for T, u in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is not None and u > 0: L[k].append((r_.m, ((u,), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAt, 1.0), (M1g, .3), (R1g, .5)]
    V = {'B0 day-type engine (V4)': base}
    for nm, t in (('OA', OA), ('OB', OB), ('OC', OC)):
        for mic in (2, 3): V[f'B0 + {nm} {mic} micros'] = base + [(t, mic / 10)]
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'\n  {"version":<26} {"acct":<5} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, spec in V.items():
        DE = DF = lists(spec); usd = np.array([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in z) for z in DE])
        for a in ('50K', '150K'):
            A = ACC[a]; Rr = []
            for s in starts:
                res, k2 = ev(A, s, NS)
                if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                fr, k3, pays = fu(A, k2 + 1, NS, 5); Rr.append((res, k2, pays, fr, NS if fr == 'open' else k3))
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, starts)])
            fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(Rr, starts) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
            print(f'  {nm:<26} {a:<5} {np.mean(usd * A["scale"] >= A["qday"]) * 21:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                  f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                  f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
