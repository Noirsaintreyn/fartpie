"""Nov 9 2026: EVAL ALGO vs FUNDED ALGO SOLVED SEPARATELY, THEN A SELF-GROWING FLEET (every payout buys a new eval). DECLARED BEFORE RESULTS.
Trades: profit algo (U3 + YZ, xs/dte_yz.py) + rules algo C + TPO (xs/rules_algo3.py), with a risk rule from xs/rules_algo4.py: no rules-algo entries after 13:30 on FOMC days.
Copy-traded: every account takes the same trades; accounts differ only by start day (state).
1) Per-phase sizing, chosen on accounts / funded stints started Jan 2022 - Jun 2023, judged on those started Jul 2023 - Jun 2025 (the rules model never saw 2022+):
   EVAL   profit algo x {0.75, 1, 1.25, 1.5}, rules algo x {0, 1}          objective: pass rate (ties: fewer days to pass)
   FUNDED profit algo x {0.5, 0.75, 1, 1.25}, rules algo x {0.5, 1, 1.5}   objective: $ paid per funded month (every start day simulated as a fresh funded account)
2) Fleet: start with N evals bought out of pocket (N = 1 or 5); afterwards each payout buys one new eval the next day while active accounts (eval + funded) < cap
   (5 = Lucid, 20 = a what-if cap). An account ends when it fails, blows or takes its 5th payout. 50K and 150K, current single setup vs per-phase setup.
   Fleet from Jan 2022 (sizing partly chosen on it) and from Jul 2023 (test window only). Report: payouts a month, $ a month after fees, months with >= 2 payouts, accounts."""
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
    print(f'rules algo C + TPO: cut {CUT:.2f}, {len(RC)} trades 2014 - Jan 2026 (FOMC-afternoon entries removed)', flush=True)
    LOCK['on'] = False; CH = {}
    for a_ in ('50K', '150K'):
        A = ACC[a_]; print(f'\n=== {a_} ===', flush=True)
        er = []
        for pm in (.75, 1.0, 1.25, 1.5):
            for rm in (0.0, 1.0):
                DE = lists(spec(pm, rm), 'e'); globals()['DE'] = DE; p1, d1 = eval_stats(A, SEL[::2]); er.append((p1, -d1 if np.isfinite(d1) else -999, pm, rm))
        er.sort(reverse=True); ebest = er[0][2:]; print('  EVAL grid on Jan22-Jun23 starts (pass rate, median days to pass): ' + '; '.join(f'p x{pm} r x{rm}: {p1:.0%} {-d1:.0f}d' for p1, d1, pm, rm in er[:5]), flush=True)
        fr_ = []
        for pm in (.5, .75, 1.0, 1.25):
            for rm in (.5, 1.0, 1.5):
                DF = lists(spec(pm, rm), 'f'); globals()['DF'] = DF; usd_m, pay_m, avg, lost = fund_stats(A, SEL[::2]); fr_.append((usd_m, pay_m, avg, lost, pm, rm))
        fr_.sort(reverse=True); fbest = fr_[0][4:]; print('  FUNDED grid on Jan22-Jun23 stints ($ per funded month, payouts/mo, avg payout, blown): '
                                                         + '; '.join(f'p x{pm} r x{rm}: ${u:.0f} {pmo:.2f} ${av:.0f} {lo:.0%}' for u, pmo, av, lo, pm, rm in fr_[:5]), flush=True)
        CH[a_] = (ebest, fbest)
        for lab, (ep, fp) in (('single setup (profit x1, rules x1)', ((1.0, 1.0), (1.0, 1.0))), (f'per-phase: eval p x{ebest[0]} r x{ebest[1]}, funded p x{fbest[0]} r x{fbest[1]}', (ebest, fbest))):
            globals()['DE'] = lists(spec(*ep), 'e'); globals()['DF'] = lists(spec(*fp), 'f'); p1, d1 = eval_stats(A, TST); usd_m, pay_m, avg, lost = fund_stats(A, TST)
            print(f'  TEST Jul23-Jun25 | {lab:<58} | eval pass {p1:.0%} in {d1:.0f} trading days | funded: ${usd_m:5.0f} a month, {pay_m:.2f} payouts/mo x ${avg:4.0f}, blown {lost:.0%}', flush=True)
    # ---- fleet ----
    import heapq
    print('\n=== FLEET: each payout buys a new eval (copy-traded, same trades on every account) ===', flush=True)
    for a_ in ('50K', '150K'):
        A = ACC[a_]
        for lab, (ep, fp) in (('single', ((1.0, 1.0), (1.0, 1.0))), ('per-phase', CH[a_])):
            globals()['DE'] = lists(spec(*ep), 'e'); globals()['DF'] = lists(spec(*fp), 'f'); TRJ = {}
            def traj(s):
                if s not in TRJ:
                    r, k2 = ev(A, s, NS)
                    if r != 'pass': TRJ[s] = (k2 if r == 'fail' else NS, [])
                    else: fr, k3, pays = fu(A, k2 + 1, NS, 5); TRJ[s] = (NS if fr == 'open' else k3, pays)
                return TRJ[s]
            for k0, start_lab in ((k22, 'Jan 2022'), (k23, 'Jul 2023')):
                for N0, cap in ((1, 5), (5, 5), (5, 20)):
                    act = []; ev_q = []; pays_by_m = {}; cnt_by_m = {}; fees = N0 * A['fee']; bought = N0
                    for _ in range(N0):
                        end, pays = traj(k0); heapq.heappush(act, end)
                        for k, x in pays: heapq.heappush(ev_q, (k, x))
                    maxact = N0
                    while ev_q:
                        k, x = heapq.heappop(ev_q); mo = D_[k].to_period('M'); pays_by_m[mo] = pays_by_m.get(mo, 0) + x; cnt_by_m[mo] = cnt_by_m.get(mo, 0) + 1
                        while act and act[0] < k: heapq.heappop(act)
                        if len(act) < cap and k + 1 < NS:
                            end, pays2 = traj(k + 1); heapq.heappush(act, end); fees += A['fee']; bought += 1; maxact = max(maxact, len(act))
                            for kk, xx in pays2: heapq.heappush(ev_q, (kk, xx))
                    months = pd.period_range(D_[k0].to_period('M'), D_[NS - 1].to_period('M'), freq='M'); P = pd.Series({mo: pays_by_m.get(mo, 0.0) for mo in months})
                    C_ = pd.Series({mo: cnt_by_m.get(mo, 0) for mo in months})
                    print(f'  {a_:<5} {lab:<9} from {start_lab} start {N0} cap {cap:>2}: payouts ${P.sum():8,.0f} over {len(months)} mo = ${P.mean():6,.0f}/mo paid, fees ${fees:6,.0f} '
                          f'({bought} evals), net ${P.sum() - fees:8,.0f} = ${(P.sum() - fees) / len(months):6,.0f}/mo; payouts/mo {C_.mean():.1f} (last 12 mo {C_.iloc[-12:].mean():.1f}), months with >= 2 payouts {np.mean(C_ >= 2):.0%}, '
                          f'best / worst 6 months ${P.rolling(6).sum().max():,.0f} / ${P.rolling(6).sum().min():,.0f}; out of pocket ${N0 * A["fee"]:,.0f}, max active {maxact}', flush=True)
