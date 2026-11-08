"""Nov 8 2026: TIGHTER STOPS + MORE CONTRACTS on the own-account trend-hold (A6x3). DECLARED BEFORE RESULTS.
Every RTH flip held to the 15m vector exit (flat 15:55, <= 4 a day, no overlap), A6x3 contracts (0 / 1 / 2 / 3 by the six checks), re-simulated per stop, 1 pt cost.
Part 1 per flip: stop 20 / 25 / 30 / 35 / 50 for checks 3 and 4+: average, win, R = avg / stop, points at equal risk (avg x 50 / stop), 2018-21 and 2022-26.
Part 2 book (2022 - Oct 2026 and 2018-21):
  B0 A6x3 stop 50 (Nov 8 best)       B1 stop 30 on 4+ checks, contracts x1.67        B2 stop 30 on every trade, x1.67
  B3 stop 35 every trade, x1.43      B4 stop 25 on 5-6 checks x2, rest stop 50      B5 stop 30 every trade, same contracts"""
_q = open('xs/nq_quality.py').read(); exec(_q[:_q.rindex("if __name__ == '__main__':")])
A6 = np.asarray(a6); base = np.select([A6 <= 2, A6 == 3, A6 == 4], [0.0, 1.0, 2.0], 3.0)
CACHE = {}
def run1(S):
    if S not in CACHE:
        o = [sim(int(r.j), int(r.dir), None, S, 955, True) for r in X.itertuples()]
        CACHE[S] = (np.array([x[0] - 1.0 if x else np.nan for x in o]), np.array([x[1] if x else int(j) for x, j in zip(o, X.j.values)]))
    return CACHE[S]
def book(stopv, sz, a0, b0):
    AD = ALLD[(ALLD >= a0) & (ALLD < b0)]; pts = np.full(len(X), np.nan); qx = np.zeros(len(X), int)
    for S in np.unique(stopv):
        m = stopv == S; p_, q_ = run1(float(S)); pts[m] = p_[m]; qx[m] = q_[m]
    rows = []; msk = ((X.day >= a0) & (X.day < b0)).values
    for day, g in X[msk].assign(sz=sz[msk], pts=pts[msk], qx=qx[msk]).groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy or r.sz <= 0 or not np.isfinite(r.pts): continue
            rows.append((day, r.pts * r.sz, r.sz)); busy = r.qx; n += 1
    T = pd.DataFrame(rows, columns=['day', 'pts', 'sz']); Pd = T.groupby('day').pts.sum().reindex(AD).fillna(0.0); M = Pd.groupby(Pd.index.to_period('M')).sum(); cum = Pd.cumsum()
    return Pd, M, cum, T
if __name__ == '__main__':
    p50, _ = run1(50.0); print(f'check: stop-50 re-sim vs saved th, mean diff {np.nanmean(p50 - X.th.values):+.2f} pts', flush=True)
    print('PART 1 per flip (n, avg, win, R, pts at equal risk)', flush=True)
    yy = pd.DatetimeIndex(X.day).year.values
    for S in (20, 25, 30, 35, 50):
        p_, _ = run1(float(S)); out = []
        for lab, mq in (('3 checks', A6 == 3), ('4+ checks', A6 >= 4)):
            for plab, mp in (('18-21', yy < 2022), ('22-26', yy >= 2022)):
                g = p_[mq & mp]; g = g[np.isfinite(g)]; out.append(f'{lab} {plab}: n{len(g):3d} {g.mean():+6.1f} win {np.mean(g > 0):3.0%} R {g.mean() / S:+.2f} eq {g.mean() * 50 / S:+6.1f}')
        print(f'  stop {S}: ' + ' | '.join(out), flush=True)
    V = {'B0 A6x3 stop 50': (np.full(len(X), 50.0), base), 'B1 stop 30 on 4+, x1.67': (np.where(A6 >= 4, 30.0, 50.0), base * np.where(A6 >= 4, 50 / 30, 1.0)),
         'B2 stop 30 all, x1.67': (np.full(len(X), 30.0), base * 50 / 30), 'B3 stop 35 all, x1.43': (np.full(len(X), 35.0), base * 50 / 35),
         'B4 stop 25 on 5-6 x2, rest 50': (np.where(A6 >= 5, 25.0, 50.0), base * np.where(A6 >= 5, 2.0, 1.0)), 'B5 stop 30 all, same contracts': (np.full(len(X), 30.0), base)}
    for a0, b0, lab in (('2018-01-01', '2022-01-01', '2018-21'), ('2022-01-01', '2027-01-01', '2022-26')):
        print(f'\n{lab}, own account, NQ points   pts/day  pts/year  avg size  Sharpe  worst mo  max DD  months>=50 | by year', flush=True)
        for nm, (sv, sz) in V.items():
            Pd, M, cum, T = book(sv, sz, a0, b0)
            print(f'  {nm:<30} {Pd.mean():+7.2f} {Pd.mean() * 252:+9.0f} {T.sz.mean():9.2f} {Pd.mean() / Pd.std() * np.sqrt(252):7.2f} {M.min():+9.0f} {(cum - cum.cummax()).min():+7.0f} {(M >= 50).mean():10.0%} | '
                  + ' '.join(f'{y} {v:+.0f}' for y, v in Pd.groupby(Pd.index.year).sum().items()), flush=True)
