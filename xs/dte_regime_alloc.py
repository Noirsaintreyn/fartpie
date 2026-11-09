"""Nov 8 2026: ONE IDEOLOGY - dealers set the day type. Short gamma: follow the move intraday (Rule A + noise momentum with low DEX). Long gamma: intraday has no edge
on NQ (every mean-reversion fade tested lost, xs/dte_meanrev.py), so trade the 80% rule only and carry the OVERNIGHT drift, which is stronger after long-gamma closes
(+4.34 vs +1.98 a night). DECLARED BEFORE RESULTS. Base U0 = G1 from xs/dte_greeks_prop.py (RA production, M1 negative gamma + DEX not high 3 micros, R1 positive gamma 5,
overnight 18:00 - 04:00 stop 0.35 x overnight range every night 2). Variants size the overnight leg by the prior close's gamma:
  U1 overnight 3 micros after positive-gamma closes, 1 after negative      U2 overnight only after positive-gamma closes, 3 micros      U3 U1 with 4 / 1"""
_g = open('/home/claude/fartpie/xs/dte_greeks_prop.py').read(); exec(_g[:_g.rindex("if __name__ == '__main__':")])
if __name__ == '__main__':
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    print(f'overnight after positive-gamma closes {OBpos[OBpos.day.isin(D_)].pts.mean():+.2f} (n{OBpos.day.isin(D_).sum()}), after negative {OBneg[OBneg.day.isin(D_)].pts.mean():+.2f} (n{OBneg.day.isin(D_).sum()})', flush=True)
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T, u in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if uu > 0: L[k].append((r_.m, ((uu,), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    core = [(RAs, 1.0), (M1gg, .3), (R1g, .5)]
    V = {'U0 G1 (overnight 2 every night)': core + [(OB, .2)], 'U1 overnight 3 pos-gamma / 1 neg': core + [(OBpos, .3), (OBneg, .1)],
         'U2 overnight pos-gamma only, 3': core + [(OBpos, .3)], 'U3 overnight 4 pos-gamma / 1 neg': core + [(OBpos, .4), (OBneg, .1)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'\n  {"version":<34} {"acct":<5} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, spec in V.items():
        DE = lists(spec, 'e'); DF = lists(spec, 'f'); usd = np.array([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in z) for z in DF])
        for a in ('50K', '150K'):
            A = ACC[a]; Rr = []
            for s in starts:
                res, k2 = ev(A, s, NS)
                if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                fr, k3, pays = fu(A, k2 + 1, NS, 5); Rr.append((res, k2, pays, fr, NS if fr == 'open' else k3))
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, starts)])
            fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(Rr, starts) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
            print(f'  {nm:<34} {a:<5} {np.mean(usd * A["scale"] >= A["qday"]) * 21:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                  f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                  f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
