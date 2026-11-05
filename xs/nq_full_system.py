"""Nov 5 2026: THE FULL PROP SYSTEM, all accounts together, with the account-start signal. DECLARED BEFORE RESULTS.
Trading: final NQ system (Rule A, 75 / 50, Yang-Zhang + VWAP ladder in micros, eval gap-gated with x0.3 cold streaks, funded x0.7), Lucid 50K ($88 one-time, no monthly fee),
real-time trailing drawdown, payouts 90% of min(50% of balance, $2,000), same trades copied into every account. Max 5 accounts at once.
GREEN = any of VOL-HI / VXN >= median / COLD / DIP at the prior close (tools/account_signals.py).
  A  5 accounts, rebuy the day after any fail / blown account
  B  5 accounts, every purchase (incl. the first) waits for a GREEN day
  C  2 accounts always + 3 that only buy on GREEN days
One continuous run Jan 3 2022 -> Oct 2026 (calendar years from it), plus 24-month runs from Jan22 / Jul22 / Jan23 / Jul23 / Jan24 / Jul24 for the spread."""
_cg = open('xs/nq_context_guards.py').read(); exec(_cg[:_cg.rindex("if __name__ == '__main__':")])
yzr = yz.dropna().rolling(252, min_periods=120).rank(pct=True); yzp_s = np.nan_to_num(prior(yzr, D_), nan=.5)
vxn_s = prior(Y2['^VXN'], D_); vxm_s = prior(Y2['^VXN'].rolling(252, min_periods=120).median(), D_); qz_s = prior(XQ.qz, D_)
tday = pd.DatetimeIndex(XA.loc[nets.index, 'day']).normalize(); tn = pd.Series(nets.values, index=tday).sort_index()
last20 = np.array([tn[tn.index < d].iloc[-20:].sum() if (tn.index < d).sum() >= 20 else 0.0 for d in D_])
GREEN = np.r_[(yzp_s >= 2 / 3) | (np.nan_to_num(vxn_s, nan=0) >= np.nan_to_num(vxm_s, nan=0)) | (last20 < 0) | (np.nan_to_num(qz_s, nan=0) < -1), True]
ALWAYS = np.ones(NS + 1, bool); DE, DF = DS['e_BASE'], DS['f_BASE']; f = LUC
def slot(s, end, gate):
    ev = []; k = s
    while k < end:
        while k < end and not gate[k]: k += 1
        if k >= end: break
        res, k2, fe = evS(f, k, end, DE); ev.append(('buy', k, -fe))
        if res == 'fail': ev.append(('fail', k2, 0.0)); k = k2 + 1; continue
        if res == 'open': ev.append(('in eval at end', k2, 0.0)); break
        ev.append(('pass', k2, 0.0))
        fr, k3, pays = fuS(f, k2 + 1, end, DF, DF); ev += [('pay', kp, a) for kp, a in pays]
        if fr == 'lost': ev.append(('lost', k3, 0.0)); k = k3 + 1; continue
        ev.append(('funded at end', end, 0.0)); break
    return ev
CFG = {'A 5 accounts, rebuy at once': [ALWAYS] * 5, 'B 5 accounts, buy only on GREEN': [GREEN] * 5, 'C 2 always + 3 on GREEN': [ALWAYS] * 2 + [GREEN] * 3}
def run(gates, s0, end):
    E = []
    for i, g in enumerate(gates): E += [(t, k, a, i) for t, k, a in slot(s0, end, g)]
    return pd.DataFrame(E, columns=['t', 'k', 'amt', 'slot'])
def report(E, s0, end):
    cash = np.zeros(NS + 1); np.add.at(cash, E.k.clip(upper=NS).values, E.amt.values); cs = np.cumsum(cash[s0:end]); months = (end - s0) / 21
    buys, pays = E[E.t == 'buy'], E[E.t == 'pay']; done = E.t.isin(['pass', 'fail']).sum()
    mo = pd.Series(cash[s0:end], index=D_[s0:end]).groupby(D_[s0:end].to_period('M')).sum()
    return dict(evals=len(buys), passed=(E.t == 'pass').sum() / max(done, 1), blown=(E.t == 'lost').sum(), npay=len(pays), avg_pay=pays.amt.mean() if len(pays) else 0, paid=pays.amt.sum(),
                fees=-buys.amt.sum(), fees_mo=-buys.amt.sum() / months, net=cs[-1], net_mo=cs[-1] / months, worst=cs.min(), mpos=(mo > 0).mean(), months=months)
if __name__ == '__main__':
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); print(f'GREEN on {GREEN[s0:NS].mean():.0%} of sessions 2022 - Oct 2026 ({GREEN[int(np.searchsorted(D_, pd.Timestamp("2026-01-01"))):NS].mean():.0%} in 2026); sessions {NS - s0}', flush=True)
    for nm, gates in CFG.items():
        E = run(gates, s0, NS); r = report(E, s0, NS)
        print(f'\n{nm}  (Jan 2022 -> {D_[NS - 1]:%b %d %Y}, {r["months"]:.0f} months)', flush=True)
        print(f'  evals bought {r["evals"]} | eval pass rate {r["passed"]:.0%} | funded accounts blown {r["blown"]} | payouts {r["npay"]} avg ${r["avg_pay"]:,.0f} total ${r["paid"]:,.0f}', flush=True)
        print(f'  fees ${r["fees"]:,.0f} (${r["fees_mo"]:,.0f}/month) | NET ${r["net"]:+,.0f} (${r["net_mo"]:+,.0f}/month) | most cash out of pocket ${-r["worst"]:,.0f} | months with cash in {r["mpos"]:.0%}', flush=True)
        yr = []
        for y in (2022, 2023, 2024, 2025, 2026):
            a, b = int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01'))); g = E[(E.k >= a) & (E.k < min(b, NS))]
            yr.append(f'{y}{" YTD" if y == 2026 else ""} {g.amt.sum():+,.0f} (evals {(g.t == "buy").sum()}, payouts {(g.t == "pay").sum()})')
        print('  by year: ' + ' | '.join(yr), flush=True)
        st = E.groupby('slot').t.apply(lambda s: s.iloc[-1]); print(f'  state at the end: ' + ', '.join(f'{k} {v}' for k, v in st.value_counts().items()), flush=True)
        W = [report(run(gates, s, min(s + 504, NS)), s, min(s + 504, NS)) for s in [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]]
        print(f'  24-month runs: net ' + ' '.join(f'{w["net"]:+,.0f}' for w in W) + f' | median {np.median([w["net"] for w in W]):+,.0f} | pass {np.mean([w["passed"] for w in W]):.0%} | evals {np.median([w["evals"] for w in W]):.0f} | fees/month ${np.median([w["fees_mo"] for w in W]):,.0f}', flush=True)
