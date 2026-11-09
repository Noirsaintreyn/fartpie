"""Nov 8 2026: A QUALIFIER ENGINE IN LUCID. DECLARED BEFORE RESULTS.
xs/dte_qual.py: a fixed +7.5 target turns every edge into a thin one (+0.4 to +1.6 pts a trade); the best is the FIRST noise-momentum signal of the day, which hit +7.5
before a 40-pt stop 88% of the time in 2022-25 (79% the other way). Its only job here: make qualifying days. Q = that trade at 10 micros (1 NQ; on the 150K x2.25 as always),
target +7.5 net, stop S, out at 15:55; added to the current best (U3 + YZ). The noise engine's own trade (M1, 3 micros, natural exit) stays.
Variants: Q stop 40 every noise day | Q stop 40 off in top-third gamma | Q stop 25 off in top-third gamma | Q stop 40 off in top-third gamma + V15a (vector all day,
off in top-third gamma, 3 micros). Lucid 50K / 150K, accounts started Feb 2018 - Jun 2025 and those started 2022+; qualifying days a month on the funded lists."""
_q = open('/home/claude/fartpie/xs/dte_qual.py').read(); exec(_q[:_q.rindex("if __name__ == '__main__':")])
def touch2(j, jend, dr, S):
    pts, jx = touch(j, jend, dr, S); e = Xo[j]; seg = Xl[j:jx + 1] if dr > 0 else Xh[j:jx + 1]
    worst = max(min(((seg - e) * dr).min(), 0.0), -S) if pts > -S - 1 else -S
    return pts, worst - 1.0
if __name__ == '__main__':
    E = entries(); hm = {}
    for d in days:
        for i in gi[d]: hm[h2x[i]] = m[i]
    def qlist(S, skip_hg):
        rows = []
        for d, j, dr in E['NOISE']:
            if skip_hg and DY.g2p.get(d, np.nan) >= 2 / 3: continue
            p, mae = touch2(j, end_x[d], dr, S); rows.append((d, hm[j], p, mae))
        return pd.DataFrame(rows, columns=['day', 'm', 'pts', 'mae'])
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    V = run_sig2('s_vector'); V15a = V[~hig(V)]
    RULE = {'RA': (5, 'low', 'mid'), 'M1': (5, 'high', 'low'), 'R1': (1, 'low', 'mid'), 'ON+': (5, 'high', 'mid'), 'ON-': (1, 'high', 'low')}
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T, u, yzrule in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                if yzrule is not None:
                    n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    base = [(RAs, 1.0, RULE['RA']), (M1gg, .3, RULE['M1']), (R1g, .5, RULE['R1']), (OBpos, .4, RULE['ON+']), (OBneg, .1, RULE['ON-'])]
    Q40, Q40g, Q25g = qlist(40, False), qlist(40, True), qlist(25, True)
    for nm_, T in (('Q stop 40 all', Q40), ('Q stop 40 gamma ok', Q40g), ('Q stop 25 gamma ok', Q25g)):
        t = T[T.day >= '2018-02-01']; print(f'{nm_}: {len(t)} trades since Feb 2018 ({len(t) / NS * 21:.1f} a month), hit {np.mean(t.pts > 7):.0%}, avg {t.pts.mean():+.2f}, '
                                            f'2022+ avg {t[t.day >= "2022-01-01"].pts.mean():+.2f}', flush=True)
    VAR = {'U3 + YZ (current best)': base, '+ Q stop 40, every noise day': base + [(Q40, 1.0, None)], '+ Q stop 40, gamma not top': base + [(Q40g, 1.0, None)],
           '+ Q stop 25, gamma not top': base + [(Q25g, 1.0, None)], '+ Q40 gamma not top + V15a': base + [(Q40g, 1.0, None), (V15a, .3, None)]}
    starts = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); s22 = [s for s in starts if D_[s] >= pd.Timestamp('2022-01-01')]; LOCK['on'] = False
    print(f'\n  {"version":<30} {"acct":<5} {"starts":<6} qual/mo pass  paid  payouts x size  blown | per acct-month | 1st payout  every  payouts/mo funded', flush=True)
    for nm, spec in VAR.items():
        DE = lists(spec, 'e'); DF = lists(spec, 'f')
        for a in ('50K', '150K'):
            A = ACC[a]; usd = np.array([sum(net * 2 * max(1, round(10 * th[0] * A['scale'])) for th, net, mae, p in z) for z in DF]); qm = np.mean(usd >= A['qday']) * 21
            for slab, ST in (('all', starts), ('2022+', s22)):
                Rr = []
                for s in ST:
                    res, k2 = ev(A, s, NS)
                    if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else NS)); continue
                    fr, k3, pays = fu(A, k2 + 1, NS, 5); Rr.append((res, k2, pays, fr, NS if fr == 'open' else k3))
                net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, ST)])
                fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
                fp = [(D_[min(r[2][0][0], NS - 1)] - D_[s]).days / 30.44 for r, s in zip(Rr, ST) if r[2]]; flife = sum((r[4] - r[1]) / 21 for r in fund)
                print(f'  {nm:<30} {a:<5} {slab:<6} {qm:7.1f} {np.mean([r[0] == "pass" for r in Rr]):4.0%} {np.mean([len(r[2]) > 0 for r in Rr]):5.0%}  '
                      f'{np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts {np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | '
                      f'{net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d {len(pays) / flife if flife else 0:10.2f}', flush=True)
