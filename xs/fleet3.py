"""Nov 9 2026 (fleet3: how big a starting budget survives a bad start - 5 / 10 / 20 / 40 eval fees, single setup; same keep-alive staggered policy). (fleet2): run 1 showed copy-traded accounts started the same day share one fate, and the payout-only rule dies after a bad start.
POLICY (declared before results): keep the fleet alive and staggered - every trading day, if active accounts < cap and cash >= the fee, buy ONE eval; cash starts at
5 fees out of pocket; every payout goes to cash. Eval = single setup (per-phase eval tuning failed out of sample); funded = profit x1.25, rules x1.5 (held out of sample),
vs the single setup in both phases. Caps 5 (Lucid) and 20 (what-if). From Jan 2022 and from Jul 2023.
Original header:
EVAL ALGO vs FUNDED ALGO SOLVED SEPARATELY, THEN A SELF-GROWING FLEET (every payout buys a new eval). DECLARED BEFORE RESULTS.
Trades: profit algo (U3 + YZ, xs/dte_yz.py) + rules algo C + TPO (xs/rules_algo3.py), with a risk rule from xs/rules_algo4.py: no rules-algo entries after 13:30 on FOMC days.
Copy-traded: every account takes the same trades; accounts differ only by start day (state).
1) Per-phase sizing, chosen on accounts / funded stints started Jan 2022 - Jun 2023, judged on those started Jul 2023 - Jun 2025 (the rules model never saw 2022+):
   EVAL   profit algo x {0.75, 1, 1.25, 1.5}, rules algo x {0, 1}          objective: pass rate (ties: fewer days to pass)
   FUNDED profit algo x {0.5, 0.75, 1, 1.25}, rules algo x {0.5, 1, 1.5}   objective: $ paid per funded month (every start day simulated as a fresh funded account)
2) Fleet: start with N evals bought out of pocket (N = 1 or 5); afterwards each payout buys one new eval the next day while active accounts (eval + funded) < cap
   (5 = Lucid, 20 = a what-if cap). An account ends when it fails, blows or takes its 5th payout. 50K and 150K, current single setup vs per-phase setup.
   Fleet from Jan 2022 (sizing partly chosen on it) and from Jul 2023 (test window only). Report: payouts a month, $ a month after fees, months with >= 2 payouts, accounts."""
import heapq
_r4 = open('/home/claude/fartpie/xs/rules_algo4.py').read(); exec(_r4[:_r4.rindex("if __name__ == '__main__':")])
DS = pd.read_parquet('/tmp/claude-0/rules_ds2.parquet').merge(pd.read_parquet('/tmp/claude-0/rules_ex.parquet'), on=['day', 'i', 'dir'], how='left'); DS['yr'] = DS.day.dt.year
tr = DS.yr <= 2019; va = DS.yr.between(2020, 2021); cols = [x if x != 'tadr' else 'tadrC' for x in FEAT] + GR['TPO']
mdl = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=.05, min_samples_leaf=300, l2_regularization=1.0, random_state=0).fit(DS.loc[tr, cols], DS.loc[tr, 'yC'])
DS['_p'] = mdl.predict_proba(DS[cols])[:, 1]
def daily(msk, cut):
    z = DS[msk & (DS._p >= cut)]; return z.sort_values(['day', 'tmin', '_p'], ascending=[True, True, False]).drop_duplicates('day')
res = [(cut, daily(va, cut).usdC.mean(), len(daily(va, cut))) for cut in np.round(np.arange(.40, .97, .01), 2)]; res = [r for r in res if r[2] >= 40]
good = [r for r in res if r[1] >= 10]; CUT = min(good)[0] if good else max(res, key=lambda r: r[1])[0]
RC = daily(DS.yr >= 2014, CUT); RC = RC[~(RC.day.isin(FOMC) & (RC.entry_m >= 810))]
RC = RC.assign(m=RC.entry_m, pts=RC.ptsC, mae=RC.maeC, sz=RC.micC / 10, ce=1.0, cf=1.0)[['day', 'm', 'pts', 'mae', 'sz', 'ce', 'cf']]
gx = (DY.gex < 0); M1t, R1t = m1_trades(), r1_trades(); R1g = R1t[R1t.day.map(gx) == False]; M1g = M1t[M1t.day.map(gx) == True]
M1gg = M1g[[not (fget(d, 'dexp') >= 2 / 3) for d in M1g.day]]
D_ = pd.DatetimeIndex([d for d in days if d >= pd.Timestamp('2018-02-01') and ok(d)]); NS = len(D_); kpos = {d: i for i, d in enumerate(D_)}
OB = on_list(0, 600, .35, 'all'); RAs = ra_full2(True)
OBpos = OB[[bool(DY.gex.get(d, np.nan) > 0) for d in OB.day]]; OBneg = OB[[bool(DY.gex.get(d, np.nan) < 0) for d in OB.day]]
RULE = {'RA': (5, 'low', 'mid'), 'M1': (5, 'high', 'low'), 'R1': (1, 'low', 'mid'), 'ON+': (5, 'high', 'mid'), 'ON-': (1, 'high', 'low')}
def lists(spec, phase):
    L = [[] for _ in range(NS)]
    for T_, u, yzrule in spec:
        if u <= 0: continue
        for r_ in T_.itertuples():
            k = kpos.get(pd.Timestamp(r_.day))
            if k is None: continue
            uu = u * (r_.sz * (r_.ce if phase == 'e' else r_.cf) if hasattr(r_, 'sz') else 1.0)
            if yzrule is not None:
                n, best, worst = yzrule; v = DY[f'yz{n}p'].get(pd.Timestamp(r_.day), np.nan); b3 = t3(np.array([v]))[0]; uu *= 1.5 if b3 == best else (.5 if b3 == worst else 1.0)
            if uu > 0: L[k].append((r_.m, ((min(uu, 1.75),), r_.pts, r_.mae, None)))
    return [[x for _, x in sorted(z, key=lambda t: t[0])] for z in L]
def spec(pm, rm): return [(RAs, 1.0 * pm, RULE['RA']), (M1gg, .3 * pm, RULE['M1']), (R1g, .5 * pm, RULE['R1']), (OBpos, .4 * pm, RULE['ON+']), (OBneg, .1 * pm, RULE['ON-']), (RC, rm, None)]
k22, k23, k25 = (int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2023-07-01', '2025-07-01'))
SEL, TST = list(range(k22, k23)), list(range(k23, k25))
def eval_stats(A, starts):
    out = [ev(A, s, NS) for s in starts]; p = [r for r in out if r[0] == 'pass']
    return np.mean([r[0] == 'pass' for r in out]), (np.median([k - s for (r, k), s in zip(out, starts) if r == 'pass']) if p else np.nan)
def fund_stats(A, starts):
    paid, months, n, lost = 0.0, 0.0, 0, 0
    for s in starts:
        fr, k3, pays = fu(A, s, NS, 5); end = NS if fr == 'open' else k3; paid += sum(x for _, x in pays); months += (end - s + 1) / 21; n += len(pays); lost += fr == 'lost'
    return paid / months, n / months, (paid / n if n else 0), lost / len(starts)
if __name__ == '__main__':
    LOCK['on'] = False
    for a_ in ('50K', '150K'):
        A = ACC[a_]; globals()['DE'] = lists(spec(1.0, 1.0), 'e'); globals()['DF'] = lists(spec(1.0, 1.0), 'f'); TRJ = {}
        def traj(s):
            if s not in TRJ:
                r, k2 = ev(A, s, NS)
                if r != 'pass': TRJ[s] = (k2 if r == 'fail' else NS, [])
                else: fr, k3, pays = fu(A, k2 + 1, NS, 5); TRJ[s] = (NS if fr == 'open' else k3, pays)
            return TRJ[s]
        for k0, start_lab in ((k22, 'Jan 2022'), (k23, 'Jul 2023')):
            for cap in (5, 20):
                for nb in (5, 10, 20, 40):
                    cash = nb * A['fee']; act = []; pq = []; P = {}; Cn = {}; bought = 0; low = cash; first = None
                    for k in range(k0, NS):
                        while pq and pq[0][0] == k:
                            kk, x = heapq.heappop(pq); cash += x; mo = D_[k].to_period('M'); P[mo] = P.get(mo, 0) + x; Cn[mo] = Cn.get(mo, 0) + 1; first = first or D_[k]
                        act = [e for e in act if e >= k]
                        if len(act) < cap and cash >= A['fee']:
                            end, pays = traj(k); cash -= A['fee']; bought += 1; act.append(end)
                            for kk, x in pays: heapq.heappush(pq, (kk, x))
                        low = min(low, cash)
                    months = pd.period_range(D_[k0].to_period('M'), D_[NS - 1].to_period('M'), freq='M')
                    Ps = pd.Series({mo: P.get(mo, 0.0) for mo in months}); Cs = pd.Series({mo: Cn.get(mo, 0) for mo in months}); fees = bought * A['fee']
                    print(f'  {a_:<5} from {start_lab} cap {cap:>2} budget {nb:>2} fees (${nb * A["fee"]:>6,}): payouts {Cs.sum():3d} = {Cs.mean():4.1f}/mo, months >= 2 payouts {np.mean(Cs >= 2):4.0%}, '
                          f'first payout {first.date() if first is not None else "never"}, {bought:3d} evals, net ${Ps.sum() - fees:9,.0f} = ${(Ps.sum() - fees) / len(months):7,.0f}/mo, '
                          f'cash low ${low:7,.0f}', flush=True)
