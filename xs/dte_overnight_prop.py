"""Nov 8 2026: DAY-TYPE ENGINE + THE OVERNIGHT DRIFT inside Lucid. DECLARED BEFORE RESULTS. Session day = 18:00 - 17:00 ET; the overnight long (buy 18:00, sell by 09:25)
is the session's first trade, then the RTH engines (xs/dte_prop.py). Overnight versions: no stop, or a hard stop of 0.25 ADR; every night, or only nights when QQQ net GEX at
the prior close is negative (+7.35 a night vs +2.35). Sizes in micros at 50K (x2.25 at 150K).
  B0 V4 (RA 10, M1g 3, R1g 5)        N1 B0 + overnight every night, 2 micros, no stop          N2 B0 + overnight every night, 3 micros, stop 0.25 ADR
  N3 B0 + negative-gamma nights, 3 micros, stop 0.25 ADR       N4 B0 + negative-gamma nights, 5 micros, stop 0.25 ADR       N5 B0 + negative-gamma nights, 3 micros, no stop"""
_p = open('/home/claude/fartpie/xs/dte_prop.py').read(); exec(_p[:_p.rindex("if __name__ == '__main__':")])
_o = open('/home/claude/fartpie/xs/dte_overnight.py').read(); exec(_o[_o.index("on = X[(X.m"):_o.index("DTE = None")])
S25 = .25 * N.adr; N['pts25'] = np.where(N.mae <= -S25, -S25 - 1.0, N.pts); N['mae25'] = np.maximum(N.mae, -S25) - 1.0; N['mae0'] = N.mae - 1.0
if __name__ == '__main__':
    gneg = (DY.gex < 0); RAt, M1t, R1t = ra_trades(), m1_trades(), r1_trades(); M1g = M1t[M1t.day.map(gneg) == True]; R1g = R1t[R1t.day.map(gneg) == False]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    ng_night = pd.Series(N.gex_prev.values < 0, index=N.index)
    ONall = pd.DataFrame({'day': N.index, 'm': -1, 'pts': N.pts, 'mae': N.mae0}); ONs = pd.DataFrame({'day': N.index, 'm': -1, 'pts': N.pts25, 'mae': N.mae25})
    ONng_s = ONs[ng_night.values]; ONng = ONall[ng_night.values]
    def lists(spec):
        L = [[] for _ in range(NS)]
        for T, u in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is not None and u > 0: L[k].append((r_.m, ((u,), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAt, 1.0), (M1g, .3), (R1g, .5)]
    V = {'B0 day-type engine (V4)': base, 'N1 + overnight all nights 2 mic, no stop': base + [(ONall, .2)], 'N2 + overnight all nights 3 mic, stop .25 ADR': base + [(ONs, .3)],
         'N3 + overnight neg-gamma 3 mic, stop .25 ADR': base + [(ONng_s, .3)], 'N4 + overnight neg-gamma 5 mic, stop .25 ADR': base + [(ONng_s, .5)], 'N5 + overnight neg-gamma 3 mic, no stop': base + [(ONng, .3)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'  {"version":<46} {"acct":<5} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, spec in V.items():
        DE = DF = lists(spec)
        usd = np.array([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in z) for z in DE])
        for a in ('50K', '150K'):
            A = ACC[a]; R = []
            for s in starts:
                res, k2 = ev(A, s, NS)
                if res != 'pass': R.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                fr, k3, pays = fu(A, k2 + 1, NS, 5); R.append((res, k2, pays, fr, NS if fr == 'open' else k3))
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(R, starts)])
            fund = [r for r in R if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(R, starts) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
            q = np.mean(usd * A['scale'] >= A['qday']) * 21
            print(f'  {nm:<46} {a:<5} {q:7.1f} {np.mean([r[0] == "pass" for r in R]):4.0%} {np.mean([len(r[2]) > 0 for r in R]):5.0%}  {np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts '
                  f'{np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | {net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
