"""Nov 8 2026: TREND-HOLD BOT with the A+ count + the GEX regime, sized up on great trades. OUT OF SAMPLE: cut points from 2018-2021 flips, results 2022 - Oct 2026.
DECLARED BEFORE RESULTS (the GEX effect itself was found today on 2018-2026, so 2022+ is not fully clean for GEX). NQ points per contract after costs; every RTH flip, held to the
15m vector exit (stop 50, flat 15:55, <= 4 a day, no overlap).
Conditions (6): breakout-bar RVOL top third, 5-day realized vol top third, VWAP distance not top third, signal by 10:15, 15m ATR expansion top third, QQQ net GEX negative (prior close).
  FLAT     every flip, 1 contract                          GEXONLY   skip flips when GEX is positive
  A5       5 A+ conditions: skip 0-2, 1 at 3, 2 at 4-5      A6        6 conditions: skip 0-2, 1 at 3, 2 at 4-6
  A6x3     6 conditions: skip 0-2, 1 at 3, 2 at 4, 3 at 5-6 (more leverage the more it aligns)
  Rule A versions: RA flat, RA + GEX skip, RA + A6 sizing
Report 2022 - Oct 2026: points a day, points a year, average size, points a day per contract of average size, Sharpe, worst month, max drawdown, months >= 50, by year."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
X = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet').sort_values('j').reset_index(drop=True)
q = []
for r in X.itertuples():
    res = sim(int(r.j), int(r.dir), None, 50, 955, True); q.append(res[1] if res else int(r.j))
X['q'] = q; P = X[X.day < '2022-01-01']
cut = {c: P[c].quantile(2 / 3) for c in ('rvol', 'rv5', 'vwd', 'comp')}
a5 = (X.rvol >= cut['rvol']).astype(int) + (X.rv5 >= cut['rv5']).astype(int) + (X.vwd <= cut['vwd']).astype(int) + (X.me <= 615).astype(int) + (X.comp >= cut['comp']).astype(int)
gneg = (X.gex < 0).astype(int); a6 = a5 + gneg
SZ = {'FLAT every flip': np.ones(len(X)), 'GEXONLY skip positive GEX': gneg.astype(float).values,
      'A5 skip 0-2 / 1 / 2': np.select([a5 <= 2, a5 == 3], [0.0, 1.0], 2.0), 'A6 skip 0-2 / 1 / 2 (with GEX)': np.select([a6 <= 2, a6 == 3], [0.0, 1.0], 2.0),
      'A6x3 skip 0-2 / 1 / 2 / 3': np.select([a6 <= 2, a6 == 3, a6 == 4], [0.0, 1.0, 2.0], 3.0),
      'RA Rule A entries flat': (X.ruleA == 1).astype(float).values, 'RA + GEX skip': ((X.ruleA == 1) & (gneg == 1)).astype(float).values,
      'RA + A6 sizing': np.where(X.ruleA == 1, np.select([a6 <= 2, a6 == 3], [0.0, 1.0], 2.0), 0.0)}
ALLD = pd.DatetimeIndex(sorted(H.day.unique())); ALLD = ALLD[ALLD >= '2022-01-01']
print(f'cut points from 2018-21: ' + ', '.join(f'{k} {v:.2f}' for k, v in cut.items()), flush=True)
print(f'\n2022 - Oct 2026, NQ points        pts/day  pts/year  avg size  per contract  Sharpe  worst mo  max DD  months>=50 | by year (points)', flush=True)
for nm, sz in SZ.items():
    rows = []
    for day, g in X[X.day >= '2022-01-01'].assign(sz=sz[X.day >= '2022-01-01']).groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy or r.sz <= 0: continue
            rows.append((day, r.th * r.sz, r.sz)); busy = r.q; n += 1
    T = pd.DataFrame(rows, columns=['day', 'pts', 'sz']); Pd = T.groupby('day').pts.sum().reindex(ALLD).fillna(0.0); M = Pd.groupby(Pd.index.to_period('M')).sum(); cum = Pd.cumsum()
    yr = Pd.groupby(Pd.index.year).sum()
    print(f'  {nm:<32} {Pd.mean():+7.2f} {Pd.mean() * 252:+9.0f} {T.sz.mean():9.2f} {Pd.mean() / T.sz.mean():+12.2f} {Pd.mean() / Pd.std() * np.sqrt(252):7.2f} {M.min():+9.0f} {(cum - cum.cummax()).min():+7.0f} {(M >= 50).mean():10.0%} | '
          + ' '.join(f'{y} {v:+.0f}' for y, v in yr.items()), flush=True)
