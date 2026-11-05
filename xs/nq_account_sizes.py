"""Nov 5 2026: BIGGER LUCID ACCOUNTS and the REAL 5-PAYOUT LIMIT. DECLARED BEFORE RESULTS.
LucidFlex rules (proptradingvibes.com, Sept 2026): EOD trailing drawdown that locks at start + $100; payout = 90% of min(50% of profit, cap), min $500, needs 5 qualifying days
per cycle and positive profit since the last payout; a payout moves the floor to the lock; up to 5 payouts per funded account (then a live review - modelled as the account ending
and the slot buying a new eval); max 5 funded accounts per household.
               fee      target  drawdown  qualifying day  cap     size vs 50K (drawdown ratio)
  25K          $56*     1,250   1,000     $100            1,000   0.5x
  50K          $88      3,000   2,000     $150            2,000   1x    (10 / 5 / 3 MNQ ladder)
  100K         $165*    6,000   3,000     $200            2,500   1.5x
  150K         $244     9,000   4,500     $250            3,000   2.25x
  (* 25K / 100K prices not published on the page: 25K from the earlier Lucid test, 100K interpolated.)
System: final NQ system (Rule A, 75 / 50, Yang-Zhang + VWAP ladder in whole micros x size, eval gap-gated x0.3 cold streaks, funded x0.7), day stop -$1,000 x size.
Book: 5 accounts, a slot buys a new eval only on GREEN days (account-start signal), copied trades. One run Jan 2022 -> Oct 2026 and 24-month runs from 6 start dates.
Also 50K with UNLIMITED payouts = what the earlier tests assumed."""
_fs = open('xs/nq_full_system.py').read(); exec(_fs[:_fs.rindex("if __name__ == '__main__':")])
ACC = {'25K': dict(fee=56, target=1250, mll=1000, qday=100, cap=1000, scale=.5), '50K': dict(fee=88, target=3000, mll=2000, qday=150, cap=2000, scale=1.0),
       '100K': dict(fee=165, target=6000, mll=3000, qday=200, cap=2500, scale=1.5), '150K': dict(fee=244, target=9000, mll=4500, qday=250, cap=3000, scale=2.25)}
def day(D, k, st, A):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in D[k]:
        if cum <= -1000 * A['scale']: break
        mic = max(1, int(round(10 * th[0] * A['scale']))); m = 2.0 * mic
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        cum += net * m
    return cum, False
def ev(A, s, end):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); best = 0.0
    for k in range(s, end):
        pnl, br = day(DE, k, st, A)
        if br: return 'fail', k
        st['bal'] += pnl; best = max(best, pnl)
        if st['bal'] >= A['target'] and best <= .5 * st['bal']: return 'pass', k
        st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
    return 'open', end
def fu(A, s, end, maxpay):
    st = dict(bal=0.0, floor=-A['mll'], peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
    for k in range(s, end):
        pnl, br = day(DF, k, st, A)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= A['qday']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - A['mll']), 100.0)
        if wins >= 5 and b > 0 and (not pays or since > 0):
            amt = min(.5 * b, A['cap'])
            if amt >= 500:
                pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = 100.0; fixed = True; wins = 0; since = 0.0
                if len(pays) >= maxpay: return 'graduated', k, pays
    return 'open', end, pays
def slot2(A, s, end, gate, maxpay):
    E = []; k = s
    while k < end:
        while k < end and not gate[k]: k += 1
        if k >= end: break
        E.append(('buy', k, -A['fee'])); res, k2 = ev(A, k, end)
        if res == 'fail': E.append(('fail', k2, 0.0)); k = k2 + 1; continue
        if res == 'open': break
        E.append(('pass', k2, 0.0)); fr, k3, pays = fu(A, k2 + 1, end, maxpay); E += [('pay', kp, a) for kp, a in pays]
        if fr in ('lost', 'graduated'): E.append((fr, k3, 0.0)); k = k3 + 1; continue
        break
    return E
def book(A, s0, end, maxpay, n=5):
    E = pd.DataFrame([(t, k, a, i) for i in range(n) for t, k, a in slot2(A, s0, end, GREEN, maxpay)], columns=['t', 'k', 'amt', 'slot'])
    cash = np.zeros(NS + 1); np.add.at(cash, E.k.clip(upper=NS).values, E.amt.values); cs = np.cumsum(cash[s0:end]); months = (end - s0) / 21
    mo = pd.Series(cash[s0:end], index=D_[s0:end]).groupby(D_[s0:end].to_period('M')).sum(); pm = (mo > 0).values
    gap, run = 0, 0
    for v in pm: run = 0 if v else run + 1; gap = max(gap, run)
    done = E.t.isin(['pass', 'fail']).sum(); pays = E[E.t == 'pay']
    return E, dict(net=cs[-1], mo=cs[-1] / months, fees=-E[E.t == 'buy'].amt.sum(), evals=(E.t == 'buy').sum(), passed=(E.t == 'pass').sum() / max(done, 1), npay=len(pays),
                   avg=pays.amt.mean() if len(pays) else 0, worst=cs.min(), mpos=pm.mean(), gap=gap, grad=(E.t == 'graduated').sum(), lost=(E.t == 'lost').sum())
if __name__ == '__main__':
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
    RUNS = [('50K, unlimited payouts (old assumption)', '50K', 99)] + [(f'{a}, 5-payout limit', a, 5) for a in ('25K', '50K', '100K', '150K')]
    print(f'5 accounts, buy on GREEN days, copied trades. Jan 2022 -> {D_[NS - 1]:%b %d %Y}', flush=True)
    for nm, a, mp in RUNS:
        A = ACC[a]; E, r = book(A, s0, NS, mp)
        yr = []
        for y in (2022, 2023, 2024, 2025, 2026):
            lo, hi = int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01'))); g = E[(E.k >= lo) & (E.k < min(hi, NS))]; yr.append(f'{y} {g.amt.sum():+,.0f}')
        W = [book(A, s, min(s + 504, NS), mp)[1] for s in starts]
        print(f'\n  {nm}\n    full run: NET {r["net"]:+,.0f} (${r["mo"]:+,.0f}/month) | evals {r["evals"]} pass {r["passed"]:.0%} | payouts {r["npay"]} avg ${r["avg"]:,.0f} | fees ${r["fees"]:,.0f} | accounts graduated {r["grad"]} blown {r["lost"]}'
              f' | most cash out {-r["worst"]:,.0f} | months with a payout {r["mpos"]:.0%}, longest dry spell {r["gap"]} months', flush=True)
        print(f'    by year: ' + ' | '.join(yr), flush=True)
        print(f'    24-month runs: ' + ' '.join(f'{w["net"]:+,.0f}' for w in W) + f' | median {np.median([w["net"] for w in W]):+,.0f} (${np.median([w["mo"] for w in W]):,.0f}/month) | worst {min(w["net"] for w in W):+,.0f}'
              f' | longest dry spell median {np.median([w["gap"] for w in W]):.0f} months', flush=True)
