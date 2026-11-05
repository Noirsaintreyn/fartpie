"""Nov 5 2026: WHEN to start prop accounts, HOW MANY (up to 5), and HOW MUCH to keep in reserve vs pocket. DECLARED BEFORE RESULTS.
System = final pairing (eval: Rule A, gap-gated, Yang-Zhang + VWAP ladder, x0.3 cold streaks; funded: x0.7), Lucid 50K ($88) unless stated.
(1) START TIMING: an eval started on every session 2022-2025; outcome (passed, ended in a payout, net $) grouped by the start day's weekday, first week of the month / quarter / year,
    VIX and VXN level (terciles), QQQ 20d z, Yang-Zhang vol; each split shown for 2022-23 and 2024-25 separately (a timing rule must hold in both).
(2) MULTIPLE ACCOUNTS: 1 / 3 / 5 slots, each slot rebuys after every fail or blown account; all started the same day vs staggered (1 month apart / 2 weeks apart);
    3 Lucid + 2 Topstep staggered. Same trades in every account (copier), so slots differ only by account state. Runs start Jan 2022 / Jul 2022 / Jan 2023 / Jul 2023 / Jan 2024 /
    Jul 2024, 24 months each (the Jul 2024 run ends Oct 2026). Report net, worst cash drawdown (cash you must have up front), % of months with new cash out.
(3) RESERVE RULE: fees one slot burns in its worst 12 months (95th percentile over all 12-month windows) -> reserve per slot; pocket everything above it."""
_rt = open('xs/nq_reach_tp.py').read(); exec(_rt[:_rt.rindex("if __name__ == '__main__':")])
DE, DF = DS['e_EQ20 x0.3'], DS['f_T75']
def flows(f, s, end):
    """cash flows of one slot from session s: (session index, $) - fees negative, payouts positive"""
    out = []; k = s
    while k < end:
        res, k2, fe = evS(f, k, end, DE); out.append((k, -fe))
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        if f['act']: out.append((k2, -f['act']))
        fr, k3, pays = fuS(f, k2 + 1, end, DF, DF); out += [(kp, a) for kp, a in pays]
        if fr == 'lost': k = k3 + 1; continue
        break
    return out
def attempt(f, s):
    res, k, fees = evS(f, s, NS, DE); r = dict(start=D_[s], res=res, fees=fees + (f['act'] if res == 'pass' else 0), npay=0, got=0.0)
    if res == 'pass': fr, k2, pays = fuS(f, k + 1, NS, DF, DF); r.update(npay=len(pays), got=sum(a for _, a in pays))
    return r
def job_t(chunk): return [attempt(LUC, s) for s in chunk]
def job_p(arg):
    cfg, s0 = arg; end = min(s0 + 504, NS); cash = np.zeros(NS + 1)
    for firm, off in cfg:
        f = LUC if firm == 'L' else TOP; st = s0 + off
        for k, a in flows(f, st, end): cash[k] += a
    cs = np.cumsum(cash[s0:end]); mo = pd.Series(cash[s0:end], index=D_[s0:end]).groupby(D_[s0:end].to_period('M')).sum()
    return dict(net=cs[-1], worst=cs.min(), months_pos=(mo > 0).mean(), fees=-cash[s0:end][cash[s0:end] < 0].sum(), paid=cash[s0:end][cash[s0:end] > 0].sum())
def job_r(s): f = LUC; end = min(s + 252, NS); fl = flows(f, s, end); return -sum(a for k, a in fl if a < 0)
if __name__ == '__main__':
    last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); chunks = np.array_split(np.arange(last), 8)
    with mp.get_context('fork').Pool(2) as pool:
        A = pd.DataFrame([r for c in pool.map(job_t, chunks) for r in c])
        A['net'] = A.got - A.fees; A['paid'] = A.npay > 0; A['h'] = np.where(A.start < '2024-01-01', '22-23', '24-25')
        VIX = yf.download('^VIX', start='2021-01-01', interval='1d', progress=False, auto_adjust=True)['Close'].squeeze(); VIX.index = VIX.index.normalize()
        vix = pd.Series(prior(VIX, A.start), index=A.index); vxn = pd.Series(prior(Y2['^VXN'], A.start), index=A.index); qz = pd.Series(prior(XQ.qz, A.start), index=A.index)
        yzp = pd.Series(prior(yz.dropna().rolling(252, min_periods=120).rank(pct=True), A.start), index=A.index)
        pos_in_month = A.start.groupby(A.start.dt.to_period('M')).rank(method='first'); pos_in_q = A.start.groupby(A.start.dt.to_period('Q')).rank(method='first'); pos_in_y = A.start.groupby(A.start.dt.year).rank(method='first')
        T3 = lambda s: pd.qcut(s, 3, labels=['low', 'mid', 'high']).astype(str)
        G = {'weekday': A.start.dt.day_name().str[:3], 'month start (first 5 sessions)': np.where(pos_in_month <= 5, 'first week', 'rest'), 'quarter start': np.where(pos_in_q <= 5, 'first week', 'rest'),
             'year start': np.where(pos_in_y <= 10, 'first 2 weeks', 'rest'), 'VIX tercile': T3(vix), 'VXN tercile': T3(vxn), 'QQQ z20': np.where(qz < -1, 'z < -1', np.where(qz > 1, 'z > 1', '-1..1')),
             'Yang-Zhang vol tercile': T3(yzp)}
        print(f'(1) START TIMING: {len(A)} evals started 2022-2025 | all: passed {np.mean(A.res == "pass"):.0%}, ended in a payout {A.paid.mean():.0%}, net per eval {A.net.mean():+,.0f}', flush=True)
        for nm, lab in G.items():
            print(f'  -- {nm}', flush=True)
            for v_, g in A.groupby(np.asarray(lab)):
                cells = ' | '.join(f'{h} payout {g[g.h == h].paid.mean():4.0%} net {g[g.h == h].net.mean():+6,.0f} (n{(g.h == h).sum()})' for h in ('22-23', '24-25'))
                print(f'    {str(v_):<14} passed {np.mean(g.res == "pass"):4.0%} payout {g.paid.mean():4.0%} net {g.net.mean():+6,.0f} | {cells}', flush=True)
        starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
        CFG = {'1 Lucid': [('L', 0)], '3 Lucid same day': [('L', 0)] * 3, '3 Lucid, 1 month apart': [('L', 0), ('L', 21), ('L', 42)], '5 Lucid same day': [('L', 0)] * 5,
               '5 Lucid, 1 month apart': [('L', 21 * i) for i in range(5)], '5 Lucid, 2 weeks apart': [('L', 10 * i) for i in range(5)],
               '3 Lucid + 2 Topstep, 1 month apart': [('L', 0), ('T', 21), ('L', 42), ('T', 63), ('L', 84)]}
        P = {nm: pool.map(job_p, [(cfg, s) for s in starts]) for nm, cfg in CFG.items()}
        print('\n(2) MULTIPLE ACCOUNTS: 24-month runs from Jan 22 / Jul 22 / Jan 23 / Jul 23 / Jan 24 / Jul 24 (copied trades)', flush=True)
        for nm, rr in P.items():
            nets = [r['net'] for r in rr]; worst = [r['worst'] for r in rr]
            print(f'  {nm:<34} net by start: ' + ' '.join(f'{x:+7,.0f}' for x in nets) + f' | median {np.median(nets):+7,.0f} | worst cash drawdown median {np.median(worst):+6,.0f} (worst {min(worst):+6,.0f}) | months with cash in {np.mean([r["months_pos"] for r in rr]):.0%} | fees median {np.median([r["fees"] for r in rr]):,.0f}', flush=True)
        F12 = np.array(pool.map(job_r, [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]))
    print(f'\n(3) FEES ONE SLOT SPENDS IN 12 MONTHS (Lucid, 46 windows): median {np.median(F12):,.0f}, 90th {np.quantile(F12, .9):,.0f}, 95th {np.quantile(F12, .95):,.0f}, max {F12.max():,.0f}', flush=True)
