"""Nov 8 2026: the COMPOSITE on the own-account trend-hold (every RTH flip, vector exit, flat 15:55, <= 4 a day, no overlap). DECLARED BEFORE RESULTS.
S = the same 8 conditions as xs/nq_composite.py (+ Rule A as a 9th here, since every flip is eligible). Contracts = max(0, S - 3) ('the more, the most'), capped at 5.
Compared with A6x3 (0 / 1 / 2 / 3 by six checks); each at stop 50 and at stop 30 with x1.67 contracts. 2018-21 and 2022-26, NQ points."""
_t = open('xs/nq_tight_stop_th.py').read(); exec(_t[:_t.rindex("if __name__ == '__main__':")])
dqc = H.groupby('day').close.last(); a50 = (dqc / dqc.rolling(50).mean() - 1).shift(1); fade = (np.sign(a50.reindex(pd.DatetimeIndex(X.day).normalize()).values) * X.dir.values) < 0
SCt = ((X.gex < 0).astype(int) + (X.rv5 >= cut['rv5']).astype(int) + (X.rvol >= cut['rvol']).astype(int) + (X.comp >= cut['comp']).astype(int) + locok.astype(int) + early.astype(int)
       + X.rng.isin(['discount half', 'beyond trade-side extreme']).astype(int) + fade.astype(int) + (X.ruleA == 1).astype(int)).values
comp_sz = np.clip(SCt - 3, 0, 5).astype(float)
if __name__ == '__main__':
    yy = pd.DatetimeIndex(X.day).year.values; p50, _ = run1(50.0)
    print('PER FLIP by composite S (held to vector exit, stop 50): 2018-21 | 2022-26', flush=True)
    for lab, m in (('S 0-3', SCt <= 3), ('S 4', SCt == 4), ('S 5', SCt == 5), ('S 6', SCt == 6), ('S 7+', SCt >= 7)):
        print(f'  {lab:<6} {np.nanmean(p50[m & (yy < 2022)]):+6.1f} (n{(m & (yy < 2022)).sum():3d}) | {np.nanmean(p50[m & (yy >= 2022)]):+6.1f} (n{(m & (yy >= 2022)).sum():3d})', flush=True)
    V = {'A6x3 stop 50': (np.full(len(X), 50.0), base), 'A6x3 stop 30 x1.67': (np.full(len(X), 30.0), base * 50 / 30),
         'COMPOSITE S-3 stop 50': (np.full(len(X), 50.0), comp_sz), 'COMPOSITE S-3 stop 30 x1.67': (np.full(len(X), 30.0), comp_sz * 50 / 30)}
    for a0, b0, lab in (('2018-01-01', '2022-01-01', '2018-21'), ('2022-01-01', '2027-01-01', '2022-26')):
        print(f'\n{lab}, own account   pts/day  pts/year  avg size  per contract  Sharpe  worst mo  max DD | by year', flush=True)
        for nm, (sv, sz) in V.items():
            Pd, M, cum, T = book(sv, sz, a0, b0)
            print(f'  {nm:<28} {Pd.mean():+7.2f} {Pd.mean() * 252:+9.0f} {T.sz.mean():9.2f} {Pd.mean() / T.sz.mean():+12.2f} {Pd.mean() / Pd.std() * np.sqrt(252):7.2f} {M.min():+9.0f} {(cum - cum.cummax()).min():+7.0f} | '
                  + ' '.join(f'{y} {v:+.0f}' for y, v in Pd.groupby(Pd.index.year).sum().items()), flush=True)
