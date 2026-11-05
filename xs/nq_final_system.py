"""Nov 4 2026, POST-HOC: the best combination of everything that held up, with pass and payout rates. Lucid 50K ($88) and Topstep, real-time drawdown, one account at a time.
All variants: Rule A trades (15m QQQ vector flip 09:30-11:30, skip thin 60-day path / hard break / monthly LVN), stop 50 / target 75, flat 12:00, <= 2 a day,
size 1 / 0.5 / 0.3 NQ (10 / 5 / 3 MNQ) by Yang-Zhang 20d high + VWAP stretched, no new trade after -$1,000 on the day.
  S1 eval: skip flips against the 09:30 gap | funded: every Rule A trade
  S2 eval: gap-gated | funded: half size against the gap
  S3 eval: gap-gated | funded: half size against vanna flow (vanna data through Dec 2025)
  S4 eval: gap-gated | funded: size by the count of gap / vanna-flow against (0 x1, 1 x0.5, 2 x0.3)
  S5 eval: gap-gated + half size against the 1h vector | funded: half size against the 1h vector (built for 2026-type years)
  S0 reference: no gates (Rule A + Yang-Zhang / VWAP sizing)
Rates: eval pass | funded accounts paid at least once | evals bought that end in a payout | payouts per funded account | average payout to you | net per eval bought |
       12-month runs (start each month 2022-01 .. 2025-10): median net, 10th pct, % positive | Jan 2022 -> Oct 2026 net | 2026 net."""
_ex = open('xs/nq_extract.py').read(); exec(_ex[:_ex.rindex("if __name__ == '__main__':")])
cntGV = AG['GAP'].astype(int) + AG['VFLOW'].astype(int)
DS['xGV'] = build(mult=cntGV.map({0: 1.0, 1: .5, 2: .3}))
DS['GAPg_xH1'] = build(~AG['GAP'], mult=pd.Series(np.where(AG['H1'], .5, 1.0), index=XA.index))
VAR = {'S0 no gates': ('base', 'base', 'base'), 'S1 gap-gated eval / funded all': ('GAPg', 'base', 'base'), 'S2 gap eval / funded x0.5 vs gap': ('GAPg', 'xGAP', 'xGAP'),
       'S3 gap eval / funded x0.5 vs vanna': ('GAPg', 'xVF', 'xVF'), 'S4 gap eval / funded gap+vanna count': ('GAPg', 'xGV', 'xGV'), 'S5 gap+1h eval / funded x0.5 vs 1h': ('GAPg_xH1', 'xH1', 'xH1')}
def job10(a):
    nm, fn = a; v = VAR[nm]; f = TOP if fn == 'TOPSTEP' else LUC; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []; pays_all = []
    De, Dpre, Dpost = (DS[x] for x in v)
    for s in range(last):
        res, k, fees = evS(f, s, NS, De); r = dict(res=res, fees=fees + (f['act'] if res == 'pass' else 0), npay=0, got=0.0, fres=None)
        if res == 'pass':
            fr, k2, pays = fuS(f, k + 1, NS, Dpre, Dpost); r.update(fres=fr, npay=len(pays), got=sum(a_ for _, a_ in pays)); pays_all += [a_ for _, a_ in pays]
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']; p = (A.npay > 0).mean()
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    K = pd.DataFrame([carS(f, s, min(s + 252, NS), v) for s in starts])
    return dict(nm=nm, fn=fn, passed=(A.res == 'pass').mean(), paid_f=(Fd.npay > 0).mean(), paid_e=p, npay_f=Fd.npay.mean(), avg_pay=np.mean(pays_all) if pays_all else 0.0,
                net_eval=(A.got - A.fees).mean(), k_med=K.net.median(), k_p10=K.net.quantile(.1), k_pos=(K.net > 0).mean(), k_paid=(K.payouts > 0).mean(),
                full=carS(f, 0, NS, v)['net'], y26=carS(f, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), NS, v)['net'])
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in VAR for fn in ('LUCID 50K', 'TOPSTEP')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job10, jobs)
    print('\nFINAL CANDIDATES', flush=True)
    for r in RES:
        print(f'  {r["nm"]:<38} {r["fn"]:<9} | eval pass {r["passed"]:4.0%} | funded paid {r["paid_f"]:4.0%} | evals -> payout {r["paid_e"]:4.0%} | payouts/funded {r["npay_f"]:3.1f} | avg payout ${r["avg_pay"]:,.0f} | net/eval {r["net_eval"]:+6,.0f} '
              f'| 12m median {r["k_med"]:+6,.0f} (10th {r["k_p10"]:+5,.0f}) positive {r["k_pos"]:4.0%} paid {r["k_paid"]:4.0%} | 2022-Oct26 {r["full"]:+7,.0f} | 2026 {r["y26"]:+5,.0f}', flush=True)
