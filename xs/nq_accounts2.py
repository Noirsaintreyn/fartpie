"""Nov 5 2026: follow-up to nq_accounts. DECLARED BEFORE RESULTS.
(a) WAIT-FOR-VOL REBUY: a slot only buys an eval on a session whose prior-close Yang-Zhang 1-year percentile >= 1/3 (YZ33) or >= 1/2 (YZ50), or prior VXN >= its
    1-year median (VXNMED); otherwise it waits. Funded accounts are never paused. 1 Lucid and 5 Lucid 1 month apart, same 6 start dates, 24 months.
(b) BOOTSTRAP: start 1 Lucid; open the next slot (up to 5) the session after each payout, vs 5 from day one.
(c) RESERVE / POCKET RULE on the 5-slot run: keep reserve = slots x 6 eval fees ($528 per Lucid slot = most one slot burned in any 12 months 2022-25);
    every payout above the reserve is pocketed. Report what you put in, what you pocket, and whether the reserve ever ran dry."""
_ac = open('xs/nq_accounts.py').read(); exec(_ac[:_ac.rindex("if __name__ == '__main__':")])
yzr = yz.dropna().rolling(252, min_periods=120).rank(pct=True); yzp_s = prior(yzr, D_)
vxn_s = prior(Y2['^VXN'], D_); vxm_s = prior(Y2['^VXN'].rolling(252, min_periods=120).median(), D_)
GATE = {'any day': np.ones(NS + 1, bool), 'YZ33': np.r_[np.nan_to_num(yzp_s, nan=1) >= 1 / 3, True], 'YZ50': np.r_[np.nan_to_num(yzp_s, nan=1) >= .5, True],
        'VXNMED': np.r_[~(np.nan_to_num(vxn_s, nan=99) < np.nan_to_num(vxm_s, nan=0)), True]}
def flows_g(f, s, end, gate):
    out = []; k = s
    while k < end:
        while k < end and not gate[k]: k += 1
        if k >= end: break
        res, k2, fe = evS(f, k, end, DE); out.append((k, -fe))
        if res == 'fail': k = k2 + 1; continue
        if res == 'open': break
        if f['act']: out.append((k2, -f['act']))
        fr, k3, pays = fuS(f, k2 + 1, end, DF, DF); out += [(kp, a) for kp, a in pays]
        if fr == 'lost': k = k3 + 1; continue
        break
    return out
def summ(cash, s0, end):
    cs = np.cumsum(cash[s0:end]); mo = pd.Series(cash[s0:end], index=D_[s0:end]).groupby(D_[s0:end].to_period('M')).sum()
    return dict(net=cs[-1], worst=cs.min(), months_pos=(mo > 0).mean(), fees=-cash[s0:end][cash[s0:end] < 0].sum())
def job_g(arg):
    gname, offs, s0 = arg; end = min(s0 + 504, NS); cash = np.zeros(NS + 1)
    for off in offs:
        for k, a in flows_g(LUC, s0 + off, end, GATE[gname]): cash[k] += a
    return summ(cash, s0, end)
def job_b(s0):
    end = min(s0 + 504, NS); cash = np.zeros(NS + 1); fl = flows_g(LUC, s0, end, GATE['any day']); slots = 1; used = set()
    while slots < 5:
        pays = sorted(k for k, a in fl if a > 0 and k not in used)
        if len(pays) < 1: break
        used.add(pays[0]); fl += flows_g(LUC, pays[0] + 1, end, GATE['any day']); slots += 1
    for k, a in fl: cash[k] += a
    r = summ(cash, s0, end); r['slots_end'] = slots; return r
def job_c(s0):
    end = min(s0 + 504, NS); cash = np.zeros(NS + 1)
    for i in range(5):
        for k, a in flows_g(LUC, s0 + 21 * i, end, GATE['any day']): cash[k] += a
    res = 5 * 528.0; bank = res; put_in = res; pocket = 0.0; dry = 0
    for k in range(s0, end):
        bank += cash[k]
        if bank < 0: put_in += -bank; bank = 0.0; dry += 1
        if cash[k] > 0 and bank > res: pocket += bank - res; bank = res
    return dict(put_in=put_in, pocket=pocket, dry=dry, left=bank)
if __name__ == '__main__':
    starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
    print(f'gate open share of sessions 2022-26: ' + ', '.join(f'{g} {GATE[g][:NS].mean():.0%}' for g in GATE), flush=True)
    with mp.get_context('fork').Pool(2) as pool:
        print('(a) WAIT-FOR-VOL REBUY (24-month runs from Jan22 / Jul22 / Jan23 / Jul23 / Jan24 / Jul24)', flush=True)
        for nm, offs in (('1 Lucid', [0]), ('5 Lucid, 1 month apart', [21 * i for i in range(5)])):
            for g in GATE:
                rr = pool.map(job_g, [(g, offs, s) for s in starts]); nets = [r['net'] for r in rr]
                print(f'  {nm:<24} {g:<8} net ' + ' '.join(f'{x:+7,.0f}' for x in nets) + f' | median {np.median(nets):+7,.0f} min {min(nets):+7,.0f} | worst cash dd {min(r["worst"] for r in rr):+6,.0f} | fees median {np.median([r["fees"] for r in rr]):,.0f}', flush=True)
        rr = pool.map(job_b, starts); nets = [r['net'] for r in rr]
        print(f'\n(b) BOOTSTRAP 1 -> 5 Lucid (add a slot after each payout): net ' + ' '.join(f'{x:+7,.0f}' for x in nets) + f' | median {np.median(nets):+7,.0f} | worst cash dd {min(r["worst"] for r in rr):+6,.0f} | slots at end ' + ' '.join(str(r['slots_end']) for r in rr), flush=True)
        rr = pool.map(job_c, starts)
        print('\n(c) RESERVE $2,640 (5 slots x $528), pocket the rest, 5 Lucid 1 month apart', flush=True)
        for s, r in zip(('Jan22', 'Jul22', 'Jan23', 'Jul23', 'Jan24', 'Jul24'), rr):
            print(f'  start {s}: put in {r["put_in"]:,.0f} | pocketed {r["pocket"]:+,.0f} | reserve left {r["left"]:,.0f} | times reserve ran dry {r["dry"]}', flush=True)
