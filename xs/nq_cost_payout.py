"""Nov 4 2026: COST vs PAYOUT for the three best set-ups, one trader running ONE account at a time (buy a new eval after every fail / blown account).
Set-ups (Rule A = skip thin 60-day path / hard break / monthly LVN; size 1 / 0.5 / 0.3 NQ by HAR high + VWAP stretched; stop 50; flat 12:00):
  M1 Rule A, target 50 in the eval / 75 funded | M2 Rule A, 75 / 75 | M3 Rule A, 50 / 50 | M0 reference: every flip, flat 1 NQ, 75 / 50
Firms: Topstep ($49 per 21 sessions + $149 activation), Lucid Flex ($146 one-time, no activation). Payouts = what reaches you (after split and fees).
Windows: (a) one continuous run Jan 2022 -> Oct 2026 | (b) 12-month runs starting each month 2022-01 .. 2025-10 | (c) 2026: runs starting each month Jan - Jul 2026, to Oct 8.
NOTE 2026 is not clean out-of-sample: the Rule A labels and the 50 eval target were chosen after looking at 2018-2026 (their cut-offs were set on 2018-21 only)."""
_e2 = open('xs/nq_exits2.py').read(); exec(_e2[:_e2.rindex("if __name__ == '__main__':")])
SETUPS = {'M1 Rule A, 50 eval / 75 funded': ('T50/S50', 'T75/S50 (current)', CUT), 'M2 Rule A, 75 / 75': ('T75/S50 (current)', 'T75/S50 (current)', CUT),
          'M3 Rule A, 50 / 50': ('T50/S50', 'T50/S50', CUT), 'M0 every flip, flat 1 NQ': ('all', 'all', lambda th: 1.0)}
def run_career(fn, es_n, fs_n, sz, s, end):
    global DAY
    f = F[fn]; k = s; fees = got = 0.0; ev = fu = blown = npay = 0; first = None; open_bal = 0.0
    while k < end:
        DAY = SYSX[es_n]; ev += 1; res, k2, fe = eval_phase(f, k, end, sz); fees += fe
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        fees += f['act']; fu += 1; DAY = SYSX[fs_n]; fr, k3, pays = funded(f, k2 + 1, end, sz)
        for kp, a in pays:
            got += a; npay += 1
            if first is None: first = kp
        if fr == 'lost': blown += 1; k = k3 + 1; continue
        break
    return dict(fees=fees, got=got, net=got - fees, evals=ev, funded=fu, blown=blown, payouts=npay, first=None if first is None else D_[first])
def job5(a):
    nm, fn = a; es_n, fs_n, sz = SETUPS[nm]; out = {'setup': nm, 'firm': fn}
    full = run_career(fn, es_n, fs_n, sz, 0, NS); out.update({f'full_{k}': v for k, v in full.items()})
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([run_career(fn, es_n, fs_n, sz, s, min(s + 252, NS)) for s in starts])
    out.update(k_fees=K.fees.mean(), k_got=K.got.mean(), k_net_med=K.net.median(), k_p10=K.net.quantile(.1), k_p90=K.net.quantile(.9), k_paid=(K.payouts > 0).mean(), k_win=(K.net > 0).mean())
    S26 = [int(np.searchsorted(D_, pd.Timestamp(f'2026-{m:02d}-01'))) for m in range(1, 8)]; Y = [run_career(fn, es_n, fs_n, sz, s, NS) for s in S26]
    out['y26'] = Y; return out
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in SETUPS for fn in ('TOPSTEP', 'LUCID FLEX')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job5, jobs)
    print('(a) ONE CONTINUOUS RUN Jan 2022 -> Oct 8 2026 (one account at a time)', flush=True)
    for r in RES:
        print(f'  {r["setup"]:<32} {r["firm"]:<11} | evals bought {r["full_evals"]:>3}, funded {r["full_funded"]:>2}, blown {r["full_blown"]:>2}, payouts {r["full_payouts"]:>3} | fees {r["full_fees"]:>7,.0f} | paid to you {r["full_got"]:>8,.0f} | net {r["full_net"]:+8,.0f} | first payout {r["full_first"]}', flush=True)
    print('\n(b) 12-MONTH RUNS starting each month 2022-01 .. 2025-10 (46 runs)', flush=True)
    for r in RES:
        print(f'  {r["setup"]:<32} {r["firm"]:<11} | avg fees {r["k_fees"]:>6,.0f} | avg paid {r["k_got"]:>7,.0f} | net median {r["k_net_med"]:+7,.0f} (10th {r["k_p10"]:+7,.0f} / 90th {r["k_p90"]:+7,.0f}) | runs paid {r["k_paid"]:4.0%} | runs net positive {r["k_win"]:4.0%}', flush=True)
    print('\n(c) 2026 (Jan 2 -> Oct 8), runs starting the first session of each month', flush=True)
    for r in RES:
        y = r['y26']; j = y[0]
        print(f'  {r["setup"]:<32} {r["firm"]:<11} | from Jan: evals {j["evals"]}, funded {j["funded"]}, blown {j["blown"]}, payouts {j["payouts"]} | fees {j["fees"]:,.0f} | paid {j["got"]:,.0f} | net {j["net"]:+,.0f} | '
              + 'net by start month Jan-Jul: ' + ' '.join(f'{x["net"]:+,.0f}' for x in y), flush=True)
