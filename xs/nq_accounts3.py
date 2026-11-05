"""Nov 5 2026: (1) a START SIGNAL for when to open accounts, (2) DIVERSIFYING 5 accounts across system variants instead of copying one. DECLARED BEFORE RESULTS.
Lucid 50K ($88), same engine as nq_accounts (real-time drawdown, rebuy after every fail / blown account, 24-month runs from Jan22 / Jul22 / Jan23 / Jul23 / Jan24 / Jul24).
(1) Signals known at the prior close: VOL-HI Yang-Zhang 1-year percentile >= 2/3 | VXN >= its 1-year median | HOT the system's last 20 trades (1 lot) net > 0 | COLD the opposite |
    DIP QQQ 20-day z < -1.  Per-eval outcome when the signal is on vs off, split 2022-23 / 2024-25. Portfolio rule: 2 accounts always + 3 that only buy an eval while the signal is on,
    vs 5 always.
(2) Five accounts on five variants that each held up on their own: BASE | VWAP exit (5m close back through VWAP) | ADX-TF (skip ADX >= 25 and falling; post-hoc) |
    NEWS-UP + SHADOW | TRAIL (best - 35 after +40). Daily P&L correlation between variants; 5 copies of BASE vs one account per variant, same day and 1 month apart."""
def _strip(path, first_exec=True):
    s = open(path).read(); s = s[:s.rindex("if __name__ == '__main__':")]
    if first_exec: s = '\n'.join(l for l in s.split('\n') if not l.startswith(('_rt = open(', '_cg = open(', '_ac = open(')))
    return s
exec(_strip('xs/nq_trail_exits.py', first_exec=False))
exec(_strip('xs/nq_context_guards.py')); exec(_strip('xs/nq_adx_posthoc.py')); exec(_strip('xs/nq_accounts.py'))
def flows_v(f, s, end, ek, fk, gate=None):
    out = []; k = s; de, df = DS[ek], DS[fk]
    while k < end:
        if gate is not None:
            while k < end and not gate[k]: k += 1
            if k >= end: break
        res, k2, fe = evS(f, k, end, de); out.append((k, -fe))
        if res == 'fail': out.append((k2, 0.0, 'fail')); k = k2 + 1; continue
        if res == 'open': break
        if f['act']: out.append((k2, -f['act']))
        fr, k3, pays = fuS(f, k2 + 1, end, df, df); out += [(kp, a, 'pay') for kp, a in pays]
        if fr == 'lost': out.append((k3, 0.0, 'lost')); k = k3 + 1; continue
        break
    return [(x[0], x[1], x[2] if len(x) > 2 else 'fee') for x in out]
def summ(cash, s0, end):
    cs = np.cumsum(cash[s0:end]); return dict(net=cs[-1], worst=cs.min(), fees=-cash[s0:end][cash[s0:end] < 0].sum())
yzr = yz.dropna().rolling(252, min_periods=120).rank(pct=True); yzp_s = np.nan_to_num(prior(yzr, D_), nan=.5)
vxn_s = prior(Y2['^VXN'], D_); vxm_s = prior(Y2['^VXN'].rolling(252, min_periods=120).median(), D_); qz_s = prior(XQ.qz, D_)
tday = pd.DatetimeIndex(XA.loc[nets.index, 'day']).normalize(); tn = pd.Series(nets.values, index=tday).sort_index()
last20 = np.array([tn[tn.index < d].iloc[-20:].sum() if (tn.index < d).sum() >= 20 else 0.0 for d in D_])
SIG = {'VOL-HI': yzp_s >= 2 / 3, 'VXN>=median': np.nan_to_num(vxn_s, nan=0) >= np.nan_to_num(vxm_s, nan=0), 'HOT last20 > 0': last20 > 0, 'COLD last20 < 0': last20 < 0,
       'DIP z20 < -1': np.nan_to_num(qz_s, nan=0) < -1}
SIG = {k: np.r_[v, True] for k, v in SIG.items()}
VARS = {'BASE': ('e_BASE', 'f_BASE'), 'VWAP exit': ('e_VWAP', 'f_VWAP'), 'ADX-TF': ('e_ADX-TF', 'f_ADX-TF'), 'NEWSUP+SHADOW': ('e_NEWSUP+SHADOW', 'f_NEWSUP+SHADOW'), 'TRAIL': ('e_TRAIL', 'f_TRAIL')}
def job_a(s): r = attempt(LUC, s); return r
def job_sig(arg):
    sg, s0 = arg; end = min(s0 + 504, NS); cash = np.zeros(NS + 1)
    for i, off in enumerate((0, 21, 42, 63, 84)):
        for k, a, _ in flows_v(LUC, s0 + off, end, 'e_BASE', 'f_BASE', None if (sg is None or i < 2) else SIG[sg]): cash[k] += a
    return summ(cash, s0, end)
def job_div(arg):
    cfg, s0 = arg; end = min(s0 + 504, NS); cash = np.zeros(NS + 1); lost = []
    for vn, off in cfg:
        for k, a, kind in flows_v(LUC, s0 + off, end, *VARS[vn]):
            cash[k] += a
            if kind in ('lost', 'fail'): lost.append(k)
    r = summ(cash, s0, end); c = pd.Series(lost).value_counts(); r['same_day'] = int((c >= 3).sum()) if len(c) else 0; return r
if __name__ == '__main__':
    starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
    DE, DF = DS['e_BASE'], DS['f_BASE']; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01')))
    with mp.get_context('fork').Pool(2) as pool:
        A = pd.DataFrame(pool.map(job_a, range(last))); A['net'] = A.got - A.fees; A['paid'] = A.npay > 0; A['h'] = np.where(A.start < '2024-01-01', '22-23', '24-25')
        print(f'(1) START SIGNAL - {len(A)} evals started 2022-2025 (all: payout {A.paid.mean():.0%}, net {A.net.mean():+,.0f})', flush=True)
        for sg, on in SIG.items():
            o = pd.Series(on[:last], index=A.index)
            for lab, mk in (('on ', o), ('off', ~o)):
                g = A[mk]; cells = ' | '.join(f'{h} payout {g[g.h == h].paid.mean():4.0%} net {g[g.h == h].net.mean():+6,.0f} (n{(g.h == h).sum()})' for h in ('22-23', '24-25'))
                print(f'  {sg:<16} {lab} {mk.mean():4.0%} of days | payout {g.paid.mean():4.0%} net {g.net.mean():+6,.0f} | {cells}', flush=True)
        print('\n  portfolio: 2 accounts always + 3 that only buy while the signal is on (24-month runs; net by start | median | min | worst cash dd | fees)', flush=True)
        for sg in [None] + list(SIG):
            rr = pool.map(job_sig, [(sg, s) for s in starts]); nets_ = [r['net'] for r in rr]
            print(f'  {"5 always (no signal)" if sg is None else sg:<22} ' + ' '.join(f'{x:+7,.0f}' for x in nets_) + f' | {np.median(nets_):+7,.0f} | {min(nets_):+7,.0f} | {min(r["worst"] for r in rr):+6,.0f} | {np.median([r["fees"] for r in rr]):,.0f}', flush=True)
        print('\n(2) DIVERSIFY ACROSS VARIANTS', flush=True)
        PL = {}
        for vn, (ek, fk) in VARS.items(): PL[vn] = pd.Series([sum(t[0][0] * t[1] for t in DS[fk][k]) for k in range(NS)], index=D_[:NS])
        PL = pd.DataFrame(PL); act = PL[(PL != 0).any(axis=1)]
        print(f'  daily P&L correlation on trading days (funded sizing): mean off-diagonal {act.corr().values[np.triu_indices(5, 1)].mean():.2f}, min {act.corr().values[np.triu_indices(5, 1)].min():.2f}', flush=True)
        print('  ' + act.corr().round(2).to_string().replace('\n', '\n  '), flush=True)
        for vn in VARS: rr = pool.map(job_div, [([(vn, 0)], s) for s in starts]); print(f'  1 account {vn:<15} net ' + ' '.join(f'{r["net"]:+7,.0f}' for r in rr) + f' | median {np.median([r["net"] for r in rr]):+7,.0f}', flush=True)
        CFG = {'5 x BASE, same day': [('BASE', 0)] * 5, '5 x BASE, 1 month apart': [('BASE', 21 * i) for i in range(5)],
               '5 variants, same day': [(v, 0) for v in VARS], '5 variants, 1 month apart': [(v, 21 * i) for i, v in enumerate(VARS)]}
        for nm, cfg in CFG.items():
            rr = pool.map(job_div, [(cfg, s) for s in starts]); nets_ = [r['net'] for r in rr]
            print(f'  {nm:<26} net ' + ' '.join(f'{x:+7,.0f}' for x in nets_) + f' | median {np.median(nets_):+7,.0f} min {min(nets_):+7,.0f} | worst cash dd {min(r["worst"] for r in rr):+6,.0f} | days 3+ accounts died together {np.median([r["same_day"] for r in rr]):.0f}', flush=True)
