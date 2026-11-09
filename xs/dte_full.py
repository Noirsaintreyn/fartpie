"""Nov 8 2026: THE FULL STACK on real NQ futures inside Lucid - production sizing rules + day-type engine + stopped overnight drift. DECLARED BEFORE RESULTS.
Production rules re-built on futures data (all known at entry):
  LADDER   Rule A size 10 micros; 5 if prior-day Yang-Zhang vol (1-day YZ) is in the top third of its trailing year OR the entry is stretched past session VWAP in the
           trade's direction (top third of 2018-21 Rule A entries, in ADR units); 3 if both
  SKIP     no Rule A entry inside the prior-day value area between POC and the edge it is heading to (long POC-VAH / short VAL-POC)
  COLD     x0.3 in eval and x0.7 funded while the last 20 Rule A trades are net negative
Streams: RA (Rule A, 75 / 50, flat 12:00, with the three rules), M1g noise momentum on negative-gamma days (stop 0.15 ADR), R1g 80% rule on positive-gamma days,
ON overnight long with a stop always on (OA 18:00 - 09:25 stop 0.50 x overnight range, negative-gamma nights | OB 18:00 - 04:00 stop 0.35, every night).
  F0 RA production alone | F1 F0 + M1g 3 + R1g 5 | F2 F1 + OA 2 | F3 F1 + OB 2 | F4 F1 + OA 3 | F5 F2 with every stream halved on high-Yang-Zhang days
Micros at 50K (x2.25 at 150K). One account bought every 5th session Feb 2018 - Jun 2025; Lucid engine as xs/dte_prop.py. Points = $ / 20."""
_p = open('/home/claude/fartpie/xs/dte_prop.py').read(); exec(_p[:_p.rindex("if __name__ == '__main__':")])
NSO = {}; _s = open('/home/claude/fartpie/xs/dte_overnight_stops.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")], NSO)
def on_list(a, b, k, flt):
    rows = []
    for d in NSO['R'].index:
        if d not in NSO['gi'] or not NSO['symok'].get(d, False): continue
        if flt == 'neg' and not (bool(NSO['gknown'].get(d, False)) and bool(NSO['gneg'].get(d, False))): continue
        r_ = NSO['night'](d, a, b, k)
        if r_ is not None: rows.append((d, -1, r_[0], min(r_[1], 0) - 1.0))
    return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae'])
# Yang-Zhang 1-day (prior day) percentile
DY['pc'] = DY.c.shift(1); lo_ = np.log(DY.o / DY.pc); lc = np.log(DY.c / DY.o); lh, ll_ = np.log(DY.h / DY.o), np.log(DY.l / DY.o); rs = lh * (lh - lc) + ll_ * (ll_ - lc)
DY['yz'] = np.sqrt(lo_ ** 2 + .34 * lc ** 2 + .66 * rs); DY['yzhi'] = (DY.yz.rolling(252, min_periods=120).rank(pct=True) >= 2 / 3).shift(1).fillna(False).astype(bool)
def ra_full():
    R0 = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); rows = []
    for r_ in R0.itertuples():
        i = tpos.get(np.datetime64(pd.Timestamp(r_.entry), 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i]); ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; dr = int(r_.dir); k = di_[d]
        val, poc, vah = VA[days[k - 1]] if k > 0 else (np.nan,) * 3
        skip = (dr > 0 and poc <= e <= vah) or (dr < 0 and val <= e <= poc)
        vwx = (e - vw[i - 1]) * dr / DY.adr[d] if i > 0 and dd[i - 1] == dd[i] and np.isfinite(DY.adr[d]) else 0.0
        pts, mae = walk2(ix, j, dr, e, e + 75 * dr, e - 50 * dr, until_m=720)
        rows.append((d, m[i], pts - 1.0, mae - 1.0, skip, vwx, bool(DY.yzhi.get(d, False))))
    T = pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae', 'skip', 'vwx', 'yzhi']); cut = T.vwx[T.day < '2022-01-01'].quantile(2 / 3)
    T['sz'] = np.select([~T.yzhi & (T.vwx < cut), T.yzhi & (T.vwx >= cut)], [1.0, .3], .5); T.loc[T.skip, 'sz'] = 0.0
    hist = []; ce, cf = [], []
    for p in T.pts.values:
        cold = len(hist) >= 20 and sum(hist[-20:]) < 0; ce.append(.3 if cold else 1.0); cf.append(.7 if cold else 1.0); hist.append(p)
    T['ce'], T['cf'] = ce, cf; return T, cut
if __name__ == '__main__':
    gx = (DY.gex < 0); RAf, vcut = ra_full(); M1t, R1t = m1_trades(), r1_trades(); M1g = M1t[M1t.day.map(gx) == True]; R1g = R1t[R1t.day.map(gx) == False]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OA, OB = on_list(0, 930, .5, 'neg'), on_list(0, 600, .35, 'all')
    k_ = RAf.sz > 0; print(f'Rule A on futures: {len(RAf)} trades, skipped {RAf.skip.mean():.0%}, ladder 10/5/3 micros {np.mean(RAf.sz[k_] == 1):.0%} / {np.mean(RAf.sz[k_] == .5):.0%} / {np.mean(RAf.sz[k_] == .3):.0%} '
          f'(VWAP cut {vcut:.2f} ADR) | kept avg {RAf.pts[k_].mean():+.2f}, skipped avg {RAf.pts[RAf.skip].mean():+.2f}', flush=True)
    yzh = DY.yzhi
    def lists(spec, phase, yzcut=False):
        L = [[] for _ in range(NS)]
        for T, u in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0) * (.5 if yzcut and bool(yzh.get(pd.Timestamp(r_.day), False)) else 1.0)
                if uu > 0: L[k].append((r_.m, ((uu,), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAf, 1.0), (M1g, .3), (R1g, .5)]
    V = {'F0 RA production alone': ([(RAf, 1.0)], False), 'F1 + M1g 3 + R1g 5': (base, False), 'F2 F1 + OA 2 (neg-gamma nights)': (base + [(OA, .2)], False),
         'F3 F1 + OB 2 (every night)': (base + [(OB, .2)], False), 'F4 F1 + OA 3': (base + [(OA, .3)], False), 'F5 F2, halve all on high-YZ days': (base + [(OA, .2)], True)}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'\n  {"version":<34} {"acct":<5} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, (spec, yc) in V.items():
        DE = lists(spec, 'e', yc); DF = lists(spec, 'f', yc); usd = np.array([sum(net * 2 * max(1, round(10 * th[0])) for th, net, mae, p in z) for z in DF])
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
