"""Nov 4 2026: (1) Rey's website volatility model as the take-profit: anchored at the DAY's extreme so far (long target = RTH low so far + m x sigma, short = RTH high so far - m x sigma),
sigma = the site's Garman-Klass daily vol (mean GK variance of the prior 60 RTH sessions) x price, m in 0.75 / 1 / 1.5; < 25 pts ahead -> fixed 75; capped at 250.
(2) different exits for the eval and the funded account (eval wants frequent smaller wins, funded wants survival). DECLARED BEFORE RESULTS.
Rule-A trades and sizing (eval and funded), stop 50, flat 12:00, 1 pt cost; Topstep + Lucid Flex sims (real-time drawdown).
  A T75 eval / T75 funded (current) | B T50 / T50 | C T50 eval / RUN-BE funded (half at 75, half at the anchored P90-from-open level, breakeven after the first) |
  D T50 eval / T75 funded | E AD1 / AD1 (day-extreme anchored, 1 sigma) | F AD0.75 / AD0.75 | G AD1.5 / AD1.5 |
  H T50 eval / RUN-AD funded (half at 75, half at the day-extreme 1-sigma level, breakeven after the first)"""
_ex = open('xs/nq_exits.py').read(); exec(_ex[:_ex.rindex("if __name__ == '__main__':")])
gk = 0.5 * np.log(RTD.h / RTD.l) ** 2 - (2 * np.log(2) - 1) * np.log(RTD.c / RTD.o) ** 2
sig = np.sqrt(gk.clip(lower=1e-9).rolling(60, min_periods=20).mean().shift(1))
ext = []
for j, d_, day, k_ in zip(XA.j.values, XA.dir.values, XA.day.values, XA.k.values):
    i0 = j
    while i0 > 0 and d5[i0 - 1] == d5[j] and m5[i0 - 1] >= 570: i0 -= 1
    lo_, hi_ = (l5[i0:j].min(), h5[i0:j].max()) if j > i0 else (o5[j], o5[j]); s = sig.get(pd.Timestamp(day), np.nan) * o5[j]
    ext.append((lo_, hi_, s))
ext = np.array(ext)
def adt(m):
    lv = np.where(XA.dir.values > 0, ext[:, 0] + m * ext[:, 2], ext[:, 1] - m * ext[:, 2]); dist = (lv - o5[XA.j.values]) * XA.dir.values * XA.k.values
    return np.where(np.isfinite(dist) & (dist >= 25), np.minimum(dist, 250), 75.0)
for m in (.75, 1.0, 1.5): XA[f'AD{m}'] = adt(m)
print('sigma (NQ pts) median ' + f'{np.nanmedian(ext[:, 2] * XA.k.values):.0f}' + ' | day-extreme targets, median pts: ' + ', '.join(f'AD{m} {XA[f"AD{m}"].median():.0f} (75 fallback {np.mean(XA[f"AD{m}"] == 75):.0%})' for m in (.75, 1.0, 1.5)), flush=True)
EX2 = {'AD0.75': lambda r: ([r['AD0.75']], [1.0], 50, False, None), 'AD1': lambda r: ([r['AD1.0']], [1.0], 50, False, None), 'AD1.5': lambda r: ([r['AD1.5']], [1.0], 50, False, None),
       'RUN-AD half 75 / half AD1 + BE': lambda r: ([75, max(r['AD1.0'], 100.0)], [.5, .5], 50, True, None)}
for nm, spec in EX2.items():
    T_ = runx(XA, spec); R_ = pd.DataFrame([(d, i, n) for d, i, n, _, _ in T_], columns=['day', 'idx', 'net']); per = np.where(R_.day < P0, '18-21', np.where(R_.day < '2024-01-01', '22-23', '24-26'))
    print(f'  {nm:<32} trades {len(R_)} win {R_.net.gt(0).mean():.0%} | ' + ' / '.join(f'{p} {R_.net[per == p].mean():+5.1f}' for p in ('18-21', '22-23', '24-26')), flush=True)
    D = [[] for _ in D_]
    for day, idx, net, mae, p in T_:
        if day in dpos: D[dpos[day]].append((tuple(bool(X.at[idx, c]) for c in FL), net, mae, p))
    SYSX[nm] = D
PAIRS = {'A T75 / T75 (current)': ('T75/S50 (current)', 'T75/S50 (current)'), 'B T50 / T50': ('T50/S50', 'T50/S50'),
         'C T50 eval / RUN-BE funded': ('T50/S50', 'RUN-BE half 75 / half P90 + BE'), 'D T50 eval / T75 funded': ('T50/S50', 'T75/S50 (current)'),
         'E AD1 / AD1': ('AD1', 'AD1'), 'F AD0.75 / AD0.75': ('AD0.75', 'AD0.75'), 'G AD1.5 / AD1.5': ('AD1.5', 'AD1.5'), 'H T50 eval / RUN-AD funded': ('T50/S50', 'RUN-AD half 75 / half AD1 + BE')}
def job4(a):
    global DAY
    nm, fn = a; es_n, fs_n = PAIRS[nm]; f = F[fn]; last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))); A = []
    def ev(s, end): global DAY; DAY = SYSX[es_n]; return eval_phase(f, s, end, CUT)
    def fu(s, end): global DAY; DAY = SYSX[fs_n]; return funded(f, s, end, CUT)
    for s in range(last):
        res, k, fees = ev(s, NS); r = dict(res=res, fees=fees, npay=0, got=0.0, fres=None)
        if res == 'pass': r['fees'] += f['act']; fr, k2, pays = fu(k + 1, NS); r.update(fres=fr, npay=len(pays), got=sum(a_ for _, a_ in pays))
        A.append(r)
    A = pd.DataFrame(A); Fd = A[A.res == 'pass']; K = []
    starts = [int(np.searchsorted(D_, pd.Timestamp(f'{y}-{m:02d}-01'))) for y in range(2022, 2026) for m in range(1, 13) if (y, m) <= (2025, 10)]
    for s in starts:
        k = s; end = min(s + 252, NS); fees = got = 0.0; paid = False
        while k < end:
            res, k2, fe = ev(k, end); fees += fe
            if res == 'fail': k = k2 + 1; continue
            if res == 'open': break
            fees += f['act']; fr, k3, pays = fu(k2 + 1, end); got += sum(a_ for _, a_ in pays); paid = paid or bool(pays)
            if fr == 'lost': k = k3 + 1; continue
            break
        K.append((got - fees, paid))
    K = pd.DataFrame(K, columns=['net', 'paid']); p = (A.npay > 0).mean()
    return dict(var=nm, firm=fn, passed=(A.res == 'pass').mean(), blown0=((Fd.fres == 'lost') & (Fd.npay == 0)).mean(), per_pay=1 / p if p else np.inf,
                net_eval=(A.got - A.fees).mean(), k_paid=K.paid.mean(), k_med=K.net.median())
if __name__ == '__main__':
    jobs = [(nm, fn) for nm in PAIRS for fn in ('TOPSTEP', 'LUCID FLEX')]
    with mp.get_context('fork').Pool(2) as pool: RES = pool.map(job4, jobs)
    print('\nFIRM SIMS (eval exit / funded exit)', flush=True)
    for r in RES:
        print(f'  {r["var"]:<30} {r["firm"]:<11} | eval passed {r["passed"]:4.0%} | funded blown before 1st payout {r["blown0"]:4.0%} | evals per 1st payout {r["per_pay"]:5.1f} | net/eval {r["net_eval"]:+6,.0f} | 12m paid {r["k_paid"]:4.0%} median {r["k_med"]:+6,.0f}', flush=True)
