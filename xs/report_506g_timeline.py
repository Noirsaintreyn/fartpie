"""506g: PCV, 14 perp names, cushion 6x5. Eval -> funded timeline: days to pass, days to first payout (rule: flat and profit >= $400 above 10000, withdraw profit-300, floor locks at 10000), $/day, and the 2026 YTD path.
Funded phase restarts at 10000 with floor 9700 (same rules, same sizing), fail = floor or daily 300."""
exec(open('xs/report_506_perps.py').read().split("if __name__")[0])
PERP = {'NVDA', 'MU', 'AMD', 'GOOGL', 'TSLA', 'COIN', 'META', 'AVGO', 'AAPL', 'HOOD', 'ORCL', 'PLTR', 'AMZN', 'MSFT'}
T = SETS['PCV']; T = T[T.tk.isin(PERP)].sort_values('entry').reset_index(drop=True); END = pd.Timestamp('2026-10-20')
def life(start, horizon=400, c=6, K=5):
    X = T[(T.entry >= start) & (T.entry < min(start + pd.Timedelta(days=horizon), END))]
    ev = sorted([(r.entry, 0, i) for i, r in enumerate(X.itertuples())] + [(r.exit, 1, i) for i, r in enumerate(X.itertuples())])
    rets = X.ret.values; hold = ((X.exit - X.entry).dt.total_seconds() / 86400).values; sz = {}; op = set(); bal = 10000.0; floor = 9700.0; cur = None; ds = bal
    pn = lambda i: sz[i] * (rets[i] - 2 * COM - SWAP * hold[i]); out = dict(pass_day=None, pay_day=None, fail_day=None, bal=None, status='open'); phase = 'eval'; path = []
    for ts, k, i in ev:
        d = ts.normalize()
        if d != cur: cur = d; ds = bal + sum(min(0, pn(j)) for j in op)
        if k == 0:
            if len(op) >= K: continue
            sz[i] = min(30000 / K, c * max(0, bal - floor)); op.add(i)
        else:
            if i not in sz or i not in op: continue
            op.discard(i); bal += pn(i); path.append((ts, bal, phase))
            if phase == 'eval' and bal >= 10900:
                phase = 'funded'; out['pass_day'] = (ts - start).days; bal = 10000.0; ds = bal; floor = 9700.0; op = set(); sz = {}
                continue
            if phase == 'funded' and out['pay_day'] is None and not op and bal >= 10400:
                out['pay_day'] = (ts - start).days; out['payout'] = bal - 300 - 10000; bal = 10300.0; floor = 10000.0; return {**out, 'status': 'paid', 'bal': bal, 'path': path}
        eq = bal + sum(min(0, pn(j)) for j in op)
        if eq < floor or eq < ds - 300: out['fail_day'] = (ts - start).days; return {**out, 'status': f'failed in {phase}', 'bal': eq, 'path': path}
    return {**out, 'status': f'open in {phase}', 'bal': bal, 'path': path}
days = pd.date_range('2019-02-01', '2025-06-30', freq='B')[::3]
L = [(d, life(d)) for d in days]
def summ(sel, tag):
    R = [r for d, r in sel]; n = len(R); P = np.array([r['pass_day'] or np.nan for r in R], float); Y = np.array([r['pay_day'] or np.nan for r in R], float)
    f = lambda a, c: round(100 * float(np.mean(a <= c)))
    pay = [r['payout'] for r in R if r['pay_day']]
    print(f"{tag}: n={n} pass by 30/60/90/180/365d {f(P,30)}/{f(P,60)}/{f(P,90)}/{f(P,180)}/{f(P,365)}%  | passers median {np.nanmedian(P):.0f}d (25-75%: {np.nanpercentile(P,25):.0f}-{np.nanpercentile(P,75):.0f}) | first payout by 90/180/365d {f(Y,90)}/{f(Y,180)}/{f(Y,365)}%  median {np.nanmedian(Y):.0f}d from purchase (25-75%: {np.nanpercentile(Y,25):.0f}-{np.nanpercentile(Y,75):.0f}) | failed anywhere {round(100*np.mean([r['fail_day'] is not None for r in R]))}% | payout amount median ${np.median(pay):.0f}")
summ(L, 'all 2019-25'); summ([x for x in L if x[0].year <= 2021], 'train 19-21'); summ([x for x in L if x[0].year >= 2022], 'test 22-25')
pa = [(r['pass_day']) for d, r in L if r['pass_day']]; print('eval $/calendar day for passers (900/days): median', round(900 / np.median(pa), 1), ' | % of account per calendar day', round(9 / np.median(pa), 3))
# 2026
print('\n2026 accounts bought on the 1st trading day of each month (data through 2026-10-20):')
for m in range(1, 10):
    d = pd.Timestamp(f'2026-{m:02d}-01'); d = d + pd.offsets.BDay(0); r = life(d)
    print(f"  bought {d:%b %d}: {r['status']:16} pass day {r['pass_day']} payout day {r['pay_day']} fail day {r['fail_day']} bal ${r['bal']:.0f}")
X = T[(T.entry >= '2026-01-01') & (T.entry <= END)]; print('\n2026 PCV 14-name flips:', len(X), 'avg', round(X.ret.mean() * 1e4, 1), 'bp, win', round((X.ret > 0).mean() * 100), '%')
M = X.assign(mon=X.exit.dt.strftime('%m')).groupby('mon').ret.agg(n='size', bp=lambda x: x.mean() * 1e4, tot=lambda x: x.sum() * 100).round(1); print(M.to_string())
