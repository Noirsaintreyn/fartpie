"""Nov 24 2026 report 448. Rey: 2x on negative gamma AND half on positive, in the eval (funded flat or half-on-positive). Same sim/pass rule as report 447 (Lucid Flex 50K, base NQ flip, 2022-26 starts). DECLARED: pass = beats flat AND beats P.5 (half on positive only) on net/eval in BOTH halves 2022-23 and 2024-25 and on 12m paid-once in both."""
_s = open('xs/report_447_nq_neggamma_size.py').read(); exec(_s[:_s.index("print('\\nLUCID FLEX")])
SZ.update({'N2P.5': lambda th: 2.0 if th[1] else .5, 'N3P.5': lambda th: 3.0 if th[1] else .5}); SZ['P.5'] = lambda th: 1.0 if th[1] else .5
print('\neval/funded | pass | blown before 1st payout | paid | net/eval all | 22-23 | 24-25 | 12m paid all | 22-23 | 24-25 | 12m median | 12m p10', flush=True)
for en in ('flat', 'P.5', 'N2P.5', 'N3P.5'):
    for fn in ('flat', 'P.5'):
        es, fs = SZ[en], SZ[fn]; A = []
        for s in range(last):
            res, k, fees = eval_phase(f, s, NS, es); r = dict(res=res, fees=fees, npay=0, got=0.0, fres=None)
            if res == 'pass': r['fees'] += f['act']; fr, k2, pays = funded(f, k + 1, NS, fs); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays))
            A.append(r)
        A = pd.DataFrame(A); A['net'] = A.got - A.fees; Fd = A[A.res == 'pass']; K = pd.DataFrame([career(f, s, min(s + 252, NS), es, fs) for s in starts])
        print(f"{en:6s}/{fn:5s} | {(A.res=='pass').mean()*100:3.0f}% | {((Fd.fres=='lost')&(Fd.npay==0)).mean()*100:3.0f}% | {(Fd.npay>0).mean()*100:3.0f}% | {A.net.mean():+6.0f} | {A.net[half==0].mean():+6.0f} | {A.net[half==1].mean():+6.0f} | {K.paid.mean()*100:3.0f}% | {K.paid[sh==0].mean()*100:3.0f}% | {K.paid[sh==1].mean()*100:3.0f}% | {K.net.median():+6.0f} | {K.net.quantile(.1):+6.0f}", flush=True)
