"""Nov 8 2026: YANG-ZHANG VOLATILITY as a regime input on every engine of the dealer-regime model. DECLARED BEFORE RESULTS.
Yang-Zhang volatility from the RTH daily bars of real NQ futures (overnight gap + open-to-close + Rogers-Satchell), 1-day, 5-day and 20-day, each as a percentile of its
trailing year, known at the prior close (and for the overnight leg, at the close it starts after). Engines (2018 - Jan 2026, as xs/dte_regime_alloc.py):
  M1 noise momentum on negative-gamma days with DEX not high | ON overnight 18:00 - 04:00 stop 0.35 (split by the prior close's gamma) | R1 80% rule (positive gamma)
  | RA Rule A (75 / 50 noon, before the ladder)
Per engine and YZ window: average per trade by third (low / mid / high), 2018-21 and 2022-25; '*' = best third beats worst in both periods.
Then in Lucid: the best variant so far (U3: overnight 4 micros after positive-gamma closes / 1 after negative) with YZ sizing on each engine where YZ held in both periods."""
_r = open('/home/claude/fartpie/xs/dte_regime_alloc.py').read(); exec(_r[:_r.rindex("if __name__ == '__main__':")])
lr2 = lambda x: np.log(x) ** 2
on_ = np.log(DY.o / DY.c.shift(1)); oc_ = np.log(DY.c / DY.o); hh_, ll2 = np.log(DY.h / DY.o), np.log(DY.l / DY.o); rs_ = hh_ * (hh_ - oc_) + ll2 * (ll2 - oc_)
def yzn(n):
    k = .34 / (1.34 + (n + 1) / max(n - 1, 1)) if n > 1 else .34
    if n == 1: return np.sqrt(on_ ** 2 + k * oc_ ** 2 + (1 - k) * rs_)
    return np.sqrt(on_.rolling(n).var() + k * oc_.rolling(n).var() + (1 - k) * rs_.rolling(n).mean())
for n in (1, 5, 20): DY[f'yz{n}p'] = yzn(n).rolling(252, min_periods=120).rank(pct=True).shift(1)
def t3(x): return np.where(np.isnan(x), None, np.where(x <= 1 / 3, 'low', np.where(x >= 2 / 3, 'high', 'mid')))
if __name__ == '__main__':
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    ENG = {'M1 noise (neg gamma, DEX ok)': M1gg, 'ON after positive gamma': OBpos, 'ON after negative gamma': OBneg, 'R1 80% rule (pos gamma)': R1g, 'RA Rule A': RAs}
    HELD = {}
    for en, T in ENG.items():
        T = T[T.day.isin(D_)]; yr = T.day.dt.year.values; print(f'\n=== {en}: {len(T)} trades, avg {T.pts.mean():+.2f}', flush=True)
        for n in (1, 5, 20):
            lab = t3(T.day.map(DY[f'yz{n}p']).values.astype(float)); out = []
            for v in ('low', 'mid', 'high'):
                msk = lab == v; a = T.pts.values[msk & (yr < 2022)]; b = T.pts.values[msk & (yr >= 2022)]
                out.append((v, msk.sum(), T.pts.values[msk].mean(), a.mean() if len(a) >= 15 else np.nan, b.mean() if len(b) >= 15 else np.nan))
            srt = sorted(out, key=lambda z: -z[2]); bst, wst = srt[0], srt[-1]; cons = all(np.isfinite([bst[3], bst[4], wst[3], wst[4]])) and bst[3] > wst[3] and bst[4] > wst[4]
            if cons: HELD[(en, n)] = (bst[0], wst[0])
            print(f'  YZ {n:>2}d {"*" if cons else " "} ' + ' | '.join(f'{v} n{k} {mu:+.2f} ({a:+.1f}/{b:+.1f})' for v, k, mu, a, b in out), flush=True)
    print('\nheld in both periods: ' + ', '.join(f'{en} YZ{n}d best {b} worst {w}' for (en, n), (b, w) in HELD.items()), flush=True)
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T, u, yzrule in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]
                    uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    def rule(en):
        for n in (5, 20, 1):
            if (en, n) in HELD: b, w = HELD[(en, n)]; return (n, b, w)
        return None
    base = [(RAs, 1.0, None), (M1gg, .3, None), (R1g, .5, None), (OBpos, .4, None), (OBneg, .1, None)]
    yzv = [(RAs, 1.0, rule('RA Rule A')), (M1gg, .3, rule('M1 noise (neg gamma, DEX ok)')), (R1g, .5, rule('R1 80% rule (pos gamma)')),
           (OBpos, .4, rule('ON after positive gamma')), (OBneg, .1, rule('ON after negative gamma'))]
    print('YZ sizing rules used (1.5x in the best third, 0.5x in the worst, by the first window that held: 5d, 20d, 1d): ' + ', '.join(f'{nm}: {r}' for nm, r in
          zip(('RA', 'M1', 'R1', 'ON+', 'ON-'), [x[2] for x in yzv])), flush=True)
    V = {'U3 dealer-regime model': base, 'U3 + YZ sizing per engine': yzv}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'\n  {"version":<28} {"acct":<5} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
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
            print(f'  {nm:<28} {a:<5} {np.mean(usd * A["scale"] >= A["qday"]) * 21:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                  f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                  f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
