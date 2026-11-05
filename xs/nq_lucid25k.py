"""Nov 4 2026: Lucid Flex 25K at half size vs 50K at full size, Rule A (skip thin 60-day path / hard break / monthly LVN; 75 / 50; flat 12:00). DECLARED BEFORE RESULTS.
Rules (Lucid help-page summaries, Oct 2026):
  50K  eval $88 (code VAULT), +$3,000, $2,000 EOD trailing locking at +$100, best day <= 50%, no DLL; funded 5 days >= $150 + cycle profit > 0, 50% of profit up to $2,000 (min $500), 90%, max loss -> +$100 after a payout
  25K  eval ~$56 (list $93, same ~40% code assumed), +$1,250, $1,000 EOD trailing locking at +$100, best day <= 50%, >= 2 trading days, no DLL;
       funded 5 days >= $100 + cycle profit > 0, 50% of profit up to $1,000 (min $500), 90%, max loss -> +$100 after a payout
Sizing (NQ fractions, by HAR high + VWAP stretched): 50K full 1 / 0.5 / 0.3 | 25K half 0.5 / 0.3 / 0.2 (5 / 3 / 2 MNQ) | 25K third 0.3 / 0.2 / 0.1 (3 / 2 / 1 MNQ).
Own day stop: no new trade once the day is down half the max loss ($1,000 on 50K, $500 on 25K). Drawdown enforced in real time on the 5m path.
Output: per eval bought (starts every session 2022-25), 12-month runs (start each month 2022-01 .. 2025-10), one run Jan 2022 -> Oct 2026, one run Jan 2 -> Oct 8 2026."""
_ss = open('xs/nq_structure_size.py').read(); exec(_ss[:_ss.rindex("if __name__ == '__main__':")])
L50 = dict(fee=88, target=3000, mll=2000, lock=100, cons=.5, min_days=0, pstop=1000, win_thr=150, cap=2000, minpay=500)
L25 = dict(fee=56, target=1250, mll=1000, lock=100, cons=.5, min_days=2, pstop=500, win_thr=100, cap=1000, minpay=500)
def ladder(a, b, c): return lambda th: (a, b, c)[min(th[1] + th[2], 2)]
SET = {'Lucid 50K, full size (10/5/3 MNQ)': (L50, ladder(1.0, .5, .3)), 'Lucid 25K, half size (5/3/2 MNQ)': (L25, ladder(.5, .3, .2)),
       'Lucid 25K, third size (3/2/1 MNQ)': (L25, ladder(.3, .2, .1))}
DAY = SYSX['cutA']
def dayp(k, st, size, f):
    cum = 0.0; bal = st['bal']
    for th, net, mae, p in DAY[k]:
        if cum <= -f['pstop']: break
        m = 20.0 * size(th)
        if bal + cum + mae * m <= st['floor']: return st['floor'] - bal, True
        cum += net * m
    return cum, False
def evalp(f, s, end, size):
    st = dict(bal=0.0, floor=-float(f['mll']), peak=0.0); best = 0.0; td = 0
    for k in range(s, end):
        pnl, br = dayp(k, st, size, f); td += bool(DAY[k])
        if br: return 'fail', k
        st['bal'] += pnl; best = max(best, pnl)
        if st['bal'] >= f['target'] and td >= f['min_days'] and best <= f['cons'] * st['bal']: return 'pass', k
        st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - f['mll']), f['lock'])
    return 'open', end
def fundp(f, s, end, size):
    st = dict(bal=0.0, floor=-float(f['mll']), peak=0.0); fixed = False; pays = []; wins = 0; since = 0.0
    for k in range(s, end):
        pnl, br = dayp(k, st, size, f)
        if br: return 'lost', k, pays
        st['bal'] += pnl; since += pnl; wins += pnl >= f['win_thr']; b = st['bal']
        if not fixed: st['peak'] = max(st['peak'], b); st['floor'] = min(max(st['floor'], st['peak'] - f['mll']), f['lock'])
        if wins >= 5 and since > 0:
            amt = min(.5 * b, f['cap'])
            if amt >= f['minpay']: pays.append((k, .9 * amt)); st['bal'] -= amt; st['floor'] = float(f['lock']); fixed = True; wins = 0; since = 0.0
    return 'open', end, pays
def career(f, size, s, end):
    k = s; fees = got = 0.0; ev = blown = npay = 0
    while k < end:
        ev += 1; fees += f['fee']; res, k2 = evalp(f, k, end, size)
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        fr, k3, pays = fundp(f, k2 + 1, end, size); got += sum(a for _, a in pays); npay += len(pays)
        if fr == 'lost': blown += 1; k = k3 + 1; continue
        break
    return dict(fees=fees, got=got, net=got - fees, evals=ev, blown=blown, payouts=npay)
def job6(nm):
    f, size = SET[nm]; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []
    for s in range(last):
        res, k = evalp(f, s, NS, size); r = dict(res=res, npay=0, got=0.0, fres=None)
        if res == 'pass': fr, k2, pays = fundp(f, k + 1, NS, size); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']; p = (A.npay > 0).mean()
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([career(f, size, s, min(s + 252, NS)) for s in starts])
    return dict(nm=nm, passed=(A.res == 'pass').mean(), blown0=((Fd.fres == 'lost') & (Fd.npay == 0)).mean(), per_pay=1 / p if p else np.inf, net_eval=(A.got).mean() - f['fee'],
                k_fees=K.fees.mean(), k_got=K.got.mean(), k_med=K.net.median(), k_p10=K.net.quantile(.1), k_p90=K.net.quantile(.9), k_pos=(K.net > 0).mean(), k_paid=(K.payouts > 0).mean(),
                full=career(f, size, 0, NS), y26=career(f, size, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), NS))
if __name__ == '__main__':
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job6, list(SET))
    for r in RES:
        fu, y = r['full'], r['y26']
        print(f'\n{r["nm"]}', flush=True)
        print(f'  per eval: passed {r["passed"]:.0%} | funded blown before 1st payout {r["blown0"]:.0%} | evals per 1st payout {r["per_pay"]:.1f} | net per eval bought {r["net_eval"]:+,.0f}', flush=True)
        print(f'  12-month runs: avg fees {r["k_fees"]:,.0f} | avg paid {r["k_got"]:,.0f} | net median {r["k_med"]:+,.0f} (10th {r["k_p10"]:+,.0f} / 90th {r["k_p90"]:+,.0f}) | paid {r["k_paid"]:.0%} | net positive {r["k_pos"]:.0%}', flush=True)
        print(f'  Jan 2022 -> Oct 2026: evals {fu["evals"]}, blown {fu["blown"]}, payouts {fu["payouts"]} | fees {fu["fees"]:,.0f} | paid {fu["got"]:,.0f} | net {fu["net"]:+,.0f}', flush=True)
        print(f'  2026 (Jan 2 -> Oct 8): evals {y["evals"]}, blown {y["blown"]}, payouts {y["payouts"]} | fees {y["fees"]:,.0f} | paid {y["got"]:,.0f} | net {y["net"]:+,.0f}', flush=True)
