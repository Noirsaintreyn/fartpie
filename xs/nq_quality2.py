"""Nov 8 2026: QUALITY SCORE round 2. DECLARED BEFORE RESULTS. Round 1 showed Q 0-2 is about zero per trade in BOTH 2018-21 and 2022-26 (prop: -3 / -11 / -5 in 2018-21;
trend-hold: +3.5 / +0.5 / +0.1), and Q 3-4 carries the edge, so round 2 sizes on 'Q 3+' (3 of the 4 pillars).
OWN ACCOUNT trend-hold (as xs/nq_quality.py): QTOP 0/0/0/1/2 | QTOP3 0/0/0/2/3 | QTOP3-R QTOP3 x0.5 when prior-day Yang-Zhang vol is high
PROP (as xs/nq_quality_prop.py, S1 base = current + POC-to-VAH skip, ladder kept for risk):
  P1 Q3+ only (skip 0-2)                P2 Q3+ only, 1.5x at Q4
  P3 keep S1 trades; FUNDED x1.5 at Q3-4 (never skip, lever the good ones once funded)        P4 P3 + eval x1.25 at Q3-4
  P5 keep S1; funded x0.5 at Q 0-1, x1.5 at Q3-4"""
import sys
if sys.argv[1] == 'th':
    _q = open('xs/nq_quality.py').read(); exec(_q[:_q.rindex("if __name__ == '__main__':")])
    t3 = np.select([Qv <= 2, Qv == 3], [0.0, 2.0], 3.0)
    SZ = {'A6x3 (Nov 8 best)': SZ['A6x3 (Nov 8 best)'], 'QTOP 0/0/0/1/2': np.select([Qv <= 2, Qv == 3], [0.0, 1.0], 2.0), 'QTOP3 0/0/0/2/3': t3, 'QTOP3-R x0.5 high YZ': t3 * np.where(yzh, .5, 1.0)}
    for nm0, a0, b0 in PERS:
        AD = ALLD[(ALLD >= a0) & (ALLD < b0)]
        print(f'\n{nm0}, own account, NQ points   pts/day  pts/year  avg size  per contract  Sharpe  worst mo  max DD  months>=50 | by year', flush=True)
        for nm, sz in SZ.items():
            rows = []; m = (X.day >= a0) & (X.day < b0)
            for day, g in X[m].assign(sz=sz[m.values]).groupby('day', sort=True):
                busy = -1; n = 0
                for r in g.itertuples():
                    if n >= 4 or r.j <= busy or r.sz <= 0: continue
                    rows.append((day, r.th * r.sz, r.sz)); busy = r.q; n += 1
            T = pd.DataFrame(rows, columns=['day', 'pts', 'sz']); Pd = T.groupby('day').pts.sum().reindex(AD).fillna(0.0); M = Pd.groupby(Pd.index.to_period('M')).sum(); cum = Pd.cumsum()
            print(f'  {nm:<24} {Pd.mean():+7.2f} {Pd.mean() * 252:+9.0f} {T.sz.mean():9.2f} {Pd.mean() / T.sz.mean():+12.2f} {Pd.mean() / Pd.std() * np.sqrt(252):7.2f} {M.min():+9.0f} {(cum - cum.cummax()).min():+7.0f} '
                  f'{(M >= 50).mean():10.0%} | ' + ' '.join(f'{y} {v:+.0f}' for y, v in Pd.groupby(Pd.index.year).sum().items()), flush=True)
else:
    _q = open('xs/nq_quality_prop.py').read(); exec(_q[:_q.rindex("if __name__ == '__main__':")])
    hi = (Q >= 3).astype(float); lo = (Q <= 1).astype(float)
    MP = {'S1 current + skip POC-to-VAH': (skip2, skip2), 'P1 Q3+ only': (hi * skip2, hi * skip2), 'P2 Q3+ only, 1.5x at Q4': (hi * skip2 * np.where(Q == 4, 1.5, 1.0), hi * skip2 * np.where(Q == 4, 1.5, 1.0)),
          'P3 S1, funded x1.5 at Q3-4': (skip2, skip2 * (1 + .5 * hi)), 'P4 P3 + eval x1.25 at Q3-4': (skip2 * (1 + .25 * hi), skip2 * (1 + .5 * hi)),
          'P5 S1, funded x0.5 at Q0-1, x1.5 at Q3-4': (skip2, skip2 * (1 + .5 * hi - .5 * lo))}
    for nm, (me_m, mf_m) in MP.items():
        DE = to_D2(ev_res, eqm(.3) * me_m, 'YZV'); DF = to_D2(full, eqm(.7) * mf_m, 'YZV'); print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
