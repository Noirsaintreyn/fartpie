"""Nov 24 2026 report 447. Rey: leverage / increase size on NQ negative-gamma days. DECLARED BEFORE RESULTS.
Base trade = nq_firms.py (QQQ 15m vector flip 09:30-11:30, both ways, <= 2/day, TP 75 / SL 50 NQ pts, flat 12:00, 1 pt cost). NEG = QQQ prior-close net_gex < 0 (column C.neg, real OI-based file).
Mechanism: dealer short gamma amplifies moves -> flips follow through more; report 441 on QQQ-proxy: long NEG-POS +19.5bp t+3.5 (train and test same sign), shorts weaker (+8.6bp t+1.3).
PART 1 per-trade (all years 2018-26): pts/trade NEG vs POS, by side, month-clustered t NEG-POS, by era 18-21 | 22-23 | 24-26. PASS: NEG-POS > 0 with t >= 2 pooled and same sign in 18-21 and 22+.
PART 2 Lucid Flex 50K sims (nq_firms rules, real-time drawdown, 2022-26 starts; eval and funded sizing set separately): flat 1 NQ | N1.5 | N2 | N1.5/P0.5 (1.5 on NEG, 0.5 on POS) | P0.5.  Net/eval, 12m paid, by start half 2022-23 | 2024-25.
PASS for sizing: beats flat on net/eval AND 12m paid-once in BOTH halves, eval AND funded."""
_s = open('xs/nq_firms.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
import numpy as np, pandas as pd
neg = C['neg'].astype(float)
rows = [dict(day=d, idx=i, net=n, side=C.dir.loc[i], neg=bool(neg.loc[i] == 1)) for d, i, n, mae, p in TR]; T = pd.DataFrame(rows); T['mon'] = pd.to_datetime(T.day).dt.to_period('M'); T['era'] = pd.cut(pd.to_datetime(T.day).dt.year, [2017, 2021, 2023, 2100], labels=['18-21', '22-23', '24-26'])
def tst(A, B):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); g = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.net[x.s == 1].sum(), s0=x.net[x.s == -1].sum()))); m1, m0 = A.net.mean(), B.net.mean()
    se = np.sqrt((((g.s1 - g.n1 * m1) / len(A) - (g.s0 - g.n0 * m0) / len(B)) ** 2).sum()); return m1 - m0, (m1 - m0) / se
print(f'NEG share {T.neg.mean():.2f}; trades {len(T)}')
for lab, D in (('ALL', T), ('LONG', T[T.side == 1]), ('SHORT', T[T.side == -1])):
    A, B = D[D.neg], D[~D.neg]; d, t = tst(A, B)
    print(f"{lab:5s} NEG n{len(A):4d} avg {A.net.mean():+6.1f} win {(A.net>0).mean()*100:3.0f}% | POS n{len(B):4d} avg {B.net.mean():+6.1f} win {(B.net>0).mean()*100:3.0f}% | NEG-POS {d:+5.1f} pts t {t:+.2f} | " + ' '.join(f"{e}: {A[A.era==e].net.mean():+.1f}/{B[B.era==e].net.mean():+.1f}(n{(A.era==e).sum()}/{(B.era==e).sum()})" for e in ('18-21', '22-23', '24-26')), flush=True)
DAY2 = [[] for _ in D_]
for day, idx, net_, mae, p in TR:
    if day in dpos: DAY2[dpos[day]].append(((bool(thin.loc[idx]), bool(neg.loc[idx] == 1)), net_, mae, p))
DAY[:] = DAY2
SZ = {'flat': lambda th: 1.0, 'N1.5': lambda th: 1.5 if th[1] else 1.0, 'N2': lambda th: 2.0 if th[1] else 1.0, 'N1.5P.5': lambda th: 1.5 if th[1] else .5, 'P.5': lambda th: 1.0 if th[1] else .5}
f = F['LUCID FLEX']; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); half = np.array([0 if D_[s] < pd.Timestamp('2024-01-01') else 1 for s in range(last)])
starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]; sh = np.array([0 if D_[s] < pd.Timestamp('2024-01-01') else 1 for s in starts])
print('\nLUCID FLEX 50K | eval size / funded size | eval passed | funded blown before 1st payout | paid | net/eval all | 22-23 | 24-25 | 12m paid all | 22-23 | 24-25 | 12m median net', flush=True)
for en in ('flat', 'N1.5', 'N2', 'N1.5P.5', 'P.5'):
    for fn in ('flat', 'N1.5', 'N2', 'N1.5P.5'):
        es, fs = SZ[en], SZ[fn]; A = []
        for s in range(last):
            res, k, fees = eval_phase(f, s, NS, es); r = dict(res=res, fees=fees, npay=0, got=0.0, fres=None)
            if res == 'pass': r['fees'] += f['act']; fr, k2, pays = funded(f, k + 1, NS, fs); r.update(fres=fr, npay=len(pays), got=sum(a for _, a in pays))
            A.append(r)
        A = pd.DataFrame(A); A['net'] = A.got - A.fees; Fd = A[A.res == 'pass']; K = pd.DataFrame([career(f, s, min(s + 252, NS), es, fs) for s in starts])
        print(f"{en:8s}/{fn:8s} | {(A.res=='pass').mean()*100:3.0f}% | {((Fd.fres=='lost')&(Fd.npay==0)).mean()*100:3.0f}% | {(Fd.npay>0).mean()*100:3.0f}% | {A.net.mean():+6.0f} | {A.net[half==0].mean():+6.0f} | {A.net[half==1].mean():+6.0f} | {K.paid.mean()*100:3.0f}% | {K.paid[sh==0].mean()*100:3.0f}% | {K.paid[sh==1].mean()*100:3.0f}% | {K.net.median():+6.0f}", flush=True)
