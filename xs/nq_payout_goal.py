"""Nov 8 2026: THE PAYOUT GOAL - 20 NQ points a day of payouts, paid weekly or every two weeks. DECLARED BEFORE RESULTS.
System: S1 (Rule A, 75 / 50, flat 12:00, Yang-Zhang + VWAP ladder, POC-to-VAH skip, funded x0.7 cold streaks), Lucid rules as xs/nq_account_sizes.py, buys on GREEN days.
Part 1, the three numbers for the current setup (2022 - Oct 2026): points per trade (1 contract) and trades a month; evals bought before the first payout (a slot that
  rebuys after every fail, started every 5th session Jan 2022 - Jun 2025: median / mean / 80th percentile, days to the first payout, fees spent); per funded account:
  payouts, trading days between payouts, share that reach 5 payouts / blow up; and for 5 accounts: payout points a day (payouts net of fees / trading days), share of
  weeks, two-week windows and months with a payout.
Part 2, the levers (each changes one thing, then combined):
  E1.5 / E2   eval size x1.5 / x2 with no cold-streak cut in eval (evals are cheap: pass faster, fail more, rebuy)
  N10 / N20   10 / 20 accounts (only possible across several firms; modelled as if each firm had Lucid's rules)
  LIVE        keep a funded account trading after 5 payouts (if the live account keeps the same rules) instead of ending it
  combos      E2 + LIVE, E2 + N10, E2 + N10 + LIVE, E2 + N20 + LIVE
Points = $ / 20."""
_b = open('xs/nq_buy_policy.py').read(); exec(_b[:_b.rindex("if __name__ == '__main__':")])
ndays = NS - s0
def make_get2(A, maxpay):
    C = {}
    def get(k):
        if k not in C:
            res, k2 = ev(A, k, NS)
            if res == 'pass': fr, k3, pays = fu(A, k2 + 1, NS, maxpay); last = NS if fr == 'open' else k3
            else: fr, k3, pays, last = res, k2, [], (NS if res == 'open' else k2)
            C[k] = (res, NS if res == 'open' else k2, last, pays, fr)
        return C[k]
    return get
def run_n(get, A, s, e, cap):
    accts = []; cash = np.zeros(NS + 1); pay = np.zeros(NS + 1); nb = 0
    for k in range(s, e):
        if not GREEN[k]: continue
        active = [a for a in accts if a >= k]
        while len(active) < cap:
            res, k2, last, pays, fr = get(k); accts.append(last); active.append(last); nb += 1; cash[k] -= A['fee']
            for kp, amt in pays:
                if kp < e: cash[kp] += amt; pay[kp] += amt
    return cash, pay, nb
def first_pay(get, A, s):
    k = s; n = 0
    while k < NS:
        while k < NS and not GREEN[k]: k += 1
        if k >= NS: return None
        res, k2, last, pays, fr = get(k); n += 1
        if res == 'open': return None
        if pays: return n, (D_[pays[0][0]] - D_[s]).days, n * A['fee']
        k = last + 1
    return None
def cadence(pay, s, e):
    p = pd.Series(pay[s:e], index=D_[s:e]); wk = p.groupby(p.index.to_period('W')).sum(); mo = p.groupby(p.index.to_period('M')).sum()
    bi = p.groupby(np.arange(len(p)) // 10).sum()
    return (wk > 0).mean(), (bi > 0).mean(), (mo > 0).mean()
def report(lab, me_m, mf_m, maxpay, cap, evcut=True):
    global DE, DF
    DE = to_D2(ev_res, (eqm(.3) if evcut else 1.0) * me_m, 'YZV'); DF = to_D2(full, eqm(.7) * mf_m, 'YZV')
    for a in ('50K', '150K'):
        A = ACC[a]; get = make_get2(A, maxpay)
        fp = [first_pay(get, A, s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)]; fp = np.array([x for x in fp if x], dtype=float)
        o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]; fund = [x for x in o if x[0] == 'pass']
        npay = np.mean([len(x[3]) for x in fund]); gaps = [b[0] - a_[0] for x in fund for a_, b in zip(x[3][:-1], x[3][1:])]
        lost = np.mean([x[4] == 'lost' for x in fund]); grad = np.mean([len(x[3]) >= 5 for x in fund]); avgp = np.mean([amt for x in fund for _, amt in x[3]]) / 20
        cash, pay, nb = run_n(get, A, s0, NS, cap); wk, bi, mo = cadence(pay, s0, NS)
        W = []
        for w in WS:
            e = min(w + 504, NS); c2, p2, _ = run_n(get, A, w, e, cap); W.append(c2[w:e].sum() / 20 / (e - w))
        print(f'  {lab:<26} {a:<5} pass {np.mean([x[0] == "pass" for x in o]):3.0%} | evals to 1st payout median {np.median(fp[:, 0]):.0f} mean {fp[:, 0].mean():.1f} p80 {np.percentile(fp[:, 0], 80):.0f}, '
              f'{np.median(fp[:, 1]):.0f} days, fees ${np.median(fp[:, 2]):,.0f} | funded: {npay:.1f} payouts of {avgp:.0f} pts, {np.median(gaps) if gaps else np.nan:.0f} days apart, '
              f'blown {lost:.0%} | {cap} accts: {cash[s0:NS].sum() / 20 / ndays:+5.1f} pts/day net ({pay[s0:NS].sum() / 20 / ndays:.1f} gross), 24-mo median {np.median(W):+.1f}/day | '
              f'paid weeks {wk:.0%}, 2-wk {bi:.0%}, months {mo:.0%} | evals {nb}', flush=True)
if __name__ == '__main__':
    n1 = nets; kept = (skip2.loc[n1.index] > 0).values; yy = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year.values; m22 = yy >= 2022
    mon = (D_[NS - 1] - D_[s0]).days / 30.44
    print(f'PART 1. Points per trade, 1 contract (S1 trades): 2018-21 {n1.values[kept & ~m22].mean():+.1f}, 2022-26 {n1.values[kept & m22].mean():+.1f} '
          f'({(kept & m22).sum() / mon:.1f} trades a month, win {np.mean(n1.values[kept & m22] > 0):.0%})', flush=True)
    one_ = pd.Series(1.0, index=XA.index)
    print('\n  version                    size  ...', flush=True)
    report('current (S1)', skip2, skip2, 5, 5)
    print('\nPART 2. Levers', flush=True)
    report('E1.5 eval x1.5', skip2 * 1.5, skip2, 5, 5, evcut=False); report('E2 eval x2', skip2 * 2, skip2, 5, 5, evcut=False)
    report('N10 10 accounts', skip2, skip2, 5, 10); report('N20 20 accounts', skip2, skip2, 5, 20); report('LIVE keep after 5 payouts', skip2, skip2, 99, 5)
    report('E2 + LIVE', skip2 * 2, skip2, 99, 5, evcut=False); report('E2 + N10', skip2 * 2, skip2, 5, 10, evcut=False)
    report('E2 + N10 + LIVE', skip2 * 2, skip2, 99, 10, evcut=False); report('E2 + N20 + LIVE', skip2 * 2, skip2, 99, 20, evcut=False)
