"""Nov 8 2026: ACCOUNT-STATE SIZING - size each day by how much room the account has above its drawdown floor, separately in eval and funded. OUT OF SAMPLE:
the policy is chosen on accounts started Feb 2018 - Dec 2021 and judged on accounts started Jan 2022 - Jun 2025. DECLARED BEFORE RESULTS.
Trades: the dealer-regime model + YZ sizing (xs/dte_yz.py 'U3 + YZ sizing per engine'). Cushion = (balance - floor) / max loss at the start of the day.
  EVAL:   size x e_lo when cushion < 0.5, x e_hi otherwise          e_lo, e_hi in {0.5, 1, 1.5}
  FUNDED: size x f_lo when cushion < 0.3, x f_mid at 0.3 - 0.6, x f_hi above 0.6      f_lo in {0.25, 0.5, 1}, f_mid in {0.5, 1, 1.5}, f_hi in {1, 1.5}
162 policies; pick the best per account-month on the train starts (each size, 50K and 150K), then report it next to 'all 1x' on the test starts."""
_y = open('/home/claude/fartpie/xs/dte_yz.py').read(); exec(_y[:_y.rindex("if __name__ == '__main__':")])
POL = {'e': (1.0, 1.0), 'f': (1.0, 1.0, 1.0)}; PH = {'DE': None}
def day(D, k, st, A):
    c0 = (st['bal'] - st['floor']) / A['mll']
    if D is PH['DE']: e_lo, e_hi = POL['e']; s = e_lo if c0 < .5 else e_hi
    else: f_lo, f_mid, f_hi = POL['f']; s = f_lo if c0 < .3 else (f_mid if c0 < .6 else f_hi)
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in D[k]:
        if cum <= -1000 * A['scale']: break
        mic = max(1, int(round(10 * th[0] * s * A['scale']))) if th[0] * s > 0 else 0
        if mic == 0: continue
        mm = 2.0 * mic
        if bal + cum + mae * mm <= st['floor']: return st['floor'] - bal, True
        cum += net * mm
    return cum, False
if __name__ == '__main__':
    gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
    M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
    D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
    OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
    OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
    RULE = {'RA': (5, 'low', 'mid'), 'M1': (5, 'high', 'low'), 'R1': (1, 'low', 'mid'), 'ON+': (5, 'high', 'mid'), 'ON-': (1, 'high', 'low')}
    def lists(spec, phase):
        L = [[] for _ in range(NS)]
        for T, u, yzrule in spec:
            for r_ in T.itertuples():
                k = kpos.get(pd.Timestamp(r_.day))
                if k is None: continue
                uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
                n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
                if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
        return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
    spec = [(RAs, 1.0, RULE['RA']), (M1gg, .3, RULE['M1']), (R1g, .5, RULE['R1']), (OBpos, .4, RULE['ON+']), (OBneg, .1, RULE['ON-'])]
    DE = lists(spec, 'e'); DF = lists(spec, 'f'); PH['DE'] = DE
    s_tr = list(range(0, int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), 5)); s_te = list(range(int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))), int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    def run(A, starts, end_cap=None):
        Rr = []
        for s in starts:
            end = NS if end_cap is None else min(NS, end_cap)
            res, k2 = ev(A, s, end)
            if res != 'pass': Rr.append((res, k2, [], 'none', k2 if res == 'fail' else end)); continue
            fr, k3, pays = fu(A, k2 + 1, end, 5); Rr.append((res, k2, pays, fr, end if fr == 'open' else k3))
        net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in Rr]); life = np.array([(r[4] - s + 1) / 21 for r, s in zip(Rr, starts)])
        fund = [r for r in Rr if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
        flife = sum((r[4] - r[1]) / 21 for r in fund)
        return dict(pam=net.sum() / life.sum(), paid=np.mean([len(r[2]) > 0 for r in Rr]), npay=np.mean([len(r[2]) for r in fund]) if fund else 0, size=np.mean(pays) / 20 if pays else 0,
                    blown=np.mean([r[3] == 'lost' for r in fund]) if fund else 0, gap=np.median(gaps) if gaps else np.nan, pfm=len(pays) / flife if flife else 0, pas=np.mean([r[0] == 'pass' for r in Rr]))
    trn_end = int(np.searchsorted(D_, pd.Timestamp('2022-07-01')))     # train accounts may run 6 months into 2022 at most
    GRID = [((a, b), (c, d, e)) for a in (.5, 1., 1.5) for b in (.5, 1., 1.5) for c in (.25, .5, 1.) for d in (.5, 1., 1.5) for e in (1., 1.5)]
    for acct in ('50K', '150K'):
        A = ACC[acct]; res = []
        for g in GRID:
            POL['e'], POL['f'] = g; res.append((run(A, s_tr, trn_end)['pam'], g))
        res.sort(key=lambda z: -z[0]); best = res[0][1]
        print(f'\n{acct}: best policy on 2018-21 starts: eval {best[0]} funded {best[1]} (train {res[0][0]:+.1f} per account-month; 1x policy '
              f'{[r for r in res if r[1] == ((1., 1.), (1., 1., 1.))][0][0]:+.1f}); next best: ' + '; '.join(f'{r[1]} {r[0]:+.1f}' for r in res[1:4]), flush=True)
        for lab, g in (('all 1x (no policy)', ((1., 1.), (1., 1., 1.))), ('chosen policy', best)):
            POL['e'], POL['f'] = g; r_ = run(A, s_te)
            print(f'  TEST 2022-25 starts, {lab:<20} pass {r_["pas"]:.0%} payout rate {r_["paid"]:.0%}, {r_["npay"]:.1f} payouts x {r_["size"]:.0f} pts, blown {r_["blown"]:.0%}, '
                  f'per account-month {r_["pam"]:+.1f}, payouts every {r_["gap"]:.0f} d, {r_["pfm"]:.2f} per funded month', flush=True)
