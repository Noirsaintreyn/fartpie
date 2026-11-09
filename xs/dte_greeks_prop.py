"""Nov 8 2026: GREEK GATES inside Lucid on the full stack (real NQ futures). DECLARED BEFORE RESULTS - gates are the strongest results of xs/dte_greeks.py (found on the
same 2014-2025 history, so in-sample; the M1 / ON gates held in all three periods 2014-17 / 2018-21 / 2022-25, the RA gate in both of its periods):
  M1 noise   trade only when net GEX is negative AND dealer DEX is not in its top third (low-DEX +6.11 / high-DEX +0.22 a trade)
  ON night   skip nights when ATM IV is in its bottom third or GEX is in its top third (low IV +0.01, high GEX +0.09 a night)
  RA         skip Rule A trades that go the same way as dealer DEX (agrees +2.45 vs against +15.43)
Base = F3 from xs/dte_full.py: RA production (ladder, POC-to-VAH skip, cold cuts) + M1 on negative-gamma days 3 micros + R1 80% rule on positive-gamma days 5 micros
+ overnight 18:00 - 04:00 stop 0.35 x overnight range every night 2 micros (micros at 50K, x2.25 at 150K).
  G0 F3 | G1 F3 + M1 gate | G2 G1 + ON gate | G3 G2 + RA gate | G4 G3 without the POC-to-VAH skip (it lost on futures: skipped trades averaged +15)"""
_f = open('/home/claude/fartpie/xs/dte_full.py').read(); exec(_f[:_f.rindex("if __name__ == '__main__':")])
_g = open('/home/claude/fartpie/xs/dte_greeks.py').read(); exec(_g[_g.index("U = '/mnt"):_g.index("def m1_dir")])
def ra_full2(use_skip=True):
    R0 = pd.read_csv('/home/claude/fartpie/live_data/nq_rule_a_trades.csv'); rows = []
    for r_ in R0.itertuples():
        i = tpos.get(np.datetime64(pd.Timestamp(r_.entry), 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i]); ix = gi[d]; j = int(np.where(ix == i)[0][0]); e = o[i]; dr = int(r_.dir); k = di_[d]
        val, poc, vah = VA[days[k - 1]] if k > 0 else (np.nan,) * 3
        skip = use_skip and ((dr > 0 and poc <= e <= vah) or (dr < 0 and val <= e <= poc))
        vwx = (e - vw[i - 1]) * dr / DY.adr[d] if i > 0 and dd[i - 1] == dd[i] and np.isfinite(DY.adr[d]) else 0.0
        pts, mae = walk2(ix, j, dr, e, e + 75 * dr, e - 50 * dr, until_m=720); fz = DAYF.get(d)
        dex = fz['dex'] if fz is not None else np.nan
        rows.append((d, m[i], pts - 1.0, mae - 1.0, skip, vwx, bool(DY.yzhi.get(d, False)), dr, dex))
    T = pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae', 'skip', 'vwx', 'yzhi', 'dir', 'dex']); cut = T.vwx[T.day < '2022-01-01'].quantile(2 / 3)
    T['sz'] = np.select([~T.yzhi & (T.vwx < cut), T.yzhi & (T.vwx >= cut)], [1.0, .3], .5); T.loc[T.skip, 'sz'] = 0.0
    hist = []; ce, cf = [], []
    for p in T.pts.values:
        cold = len(hist) >= 20 and sum(hist[-20:]) < 0; ce.append(.3 if cold else 1.0); cf.append(.7 if cold else 1.0); hist.append(p)
    T['ce'], T['cf'] = ce, cf; T['dexag'] = np.sign(T.dex) * T.dir > 0; return T
fget = lambda d, col: (DAYF.get(d)[col] if DAYF.get(d) is not None else np.nan)
if __name__ == '__main__':
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == True == False] if False else R1t[R1t.day.map(gx) == False]
    M1g = M1t[M1t.day.map(gx) == True]; M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); OBg = OB[[not (fget(d, 'ivp') <= 1 / 3 or fget(d, 'gex2p') >= 2 / 3) for d in OB.day]]
    RAs, RAn = ra_full2(True), ra_full2(False); RAsg = RAs.copy(); RAsg.loc[RAsg.dexag, 'sz'] = 0.0; RAng = RAn.copy(); RAng.loc[RAng.dexag, 'sz'] = 0.0
    for nm, t in (('M1 neg gamma', M1g), ('M1 neg gamma + DEX not high', M1gg), ('ON every night', OB), ('ON gated', OBg)):
        t = t[t.day.isin(D_)]; print(f'{nm}: {len(t)} trades 2018+, avg {t.pts.mean():+.2f}', flush=True)
    for nm, t in (('RA (POC skip)', RAs), ('RA + DEX gate', RAsg), ('RA no POC skip + DEX gate', RAng)):
        k_ = t.sz > 0; print(f'{nm}: {k_.sum()} trades taken, avg {t.pts[k_].mean():+.2f}', flush=True)
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T, u in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if uu > 0: L[k].append((r_.m, ((uu,), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    V = {'G0 F3 (no Greek gates)': [(RAs, 1.0), (M1g, .3), (R1g, .5), (OB, .2)], 'G1 + M1 DEX gate': [(RAs, 1.0), (M1gg, .3), (R1g, .5), (OB, .2)],
         'G2 + ON IV/GEX gate': [(RAs, 1.0), (M1gg, .3), (R1g, .5), (OBg, .2)], 'G3 + RA DEX gate': [(RAsg, 1.0), (M1gg, .3), (R1g, .5), (OBg, .2)],
         'G4 G3 without POC skip': [(RAng, 1.0), (M1gg, .3), (R1g, .5), (OBg, .2)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); LOCK['on'] = False
    print(f'\n  {"version":<26} {"acct":<5} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
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
            print(f'  {nm:<26} {a:<5} {np.mean(usd * A["scale"] >= A["qday"]) * 21:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                  f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                  f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
