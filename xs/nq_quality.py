"""Nov 8 2026: ONE IDEA FOR TRADE QUALITY instead of a pile of patches. DECLARED BEFORE RESULTS.
The thesis: a vector flip pays when (1) the market has fuel, (2) the break has real participation, (3) it starts from a good place, (4) it comes early. Leverage only when the
four line up; let volatility set the risk per contract (already in the Yang-Zhang ladder).
  ENERGY         QQQ net GEX negative at the prior close (dealers chase moves) OR 5-day realized vol in its top third
  PARTICIPATION  breakout-bar RVOL in its top third OR 15m ATR expansion in its top third
  LOCATION       not in the dead zone: NOT inside prior-day value on the trade side of POC, NOT in the premium half of the prior-day range (still below the prior-day extreme),
                 NOT 0-1 VWAP standard deviations past VWAP (mirrored for shorts)
  TIMING         signal by 10:15
Quality Q = number of pillars (0-4). Cut points from 2018-2021 flips only; 2022 - Oct 2026 is out of sample for every cut (GEX itself was found on the full sample).
This file: EVERY RTH FLIP held to the 15m vector exit (stop 50, flat 15:55, <= 4 a day, no overlap) on an account with no prop rules, NQ points:
  FLAT 1 contract every flip | A6x3 (Nov 8 best: 0 / 1 / 2 / 3 by the six A+ / GEX conditions) | QSKIP 1 contract at Q >= 3 | QSIZE 0 / 0 / 1 / 2 / 3 contracts at Q 0-4 |
  QSIZE-R  QSIZE x 0.5 when prior-day Yang-Zhang vol is in its top third (risk per contract) | QBIG 0 / 0 / 1 / 2 / 4"""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
X = pd.read_parquet('/tmp/claude-0/ml/nq_location_flips.parquet').sort_values('j').reset_index(drop=True)
q = []
for r in X.itertuples():
    res = sim(int(r.j), int(r.dir), None, 50, 955, True); q.append(res[1] if res else int(r.j))
X['q'] = q; P = X[X.day < '2022-01-01']; cut = {c: P[c].quantile(2 / 3) for c in ('rvol', 'rv5', 'vwd', 'comp', 'yz1')}
energy = (X.gex < 0) | (X.rv5 >= cut['rv5']); part = (X.rvol >= cut['rvol']) | (X.comp >= cut['comp'])
dead = (X['value'] == 'value, trade side of POC') | (X.rng == 'premium half') | ((X.vz >= 0) & (X.vz < 1)); locok = X['value'].notna() & ~dead; early = X.me <= 615
X['Q'] = energy.astype(int) + part.astype(int) + locok.astype(int) + early.astype(int); yzh = (X.yz1 >= cut['yz1']).values
a5 = (X.rvol >= cut['rvol']).astype(int) + (X.rv5 >= cut['rv5']).astype(int) + (X.vwd <= cut['vwd']).astype(int) + early.astype(int) + (X.comp >= cut['comp']).astype(int); a6 = a5 + (X.gex < 0).astype(int)
Qv = X.Q.values; qs = np.select([Qv <= 1, Qv == 2, Qv == 3], [0.0, 1.0, 2.0], 3.0)
SZ = {'FLAT every flip': np.ones(len(X)), 'A6x3 (Nov 8 best)': np.select([a6 <= 2, a6 == 3, a6 == 4], [0.0, 1.0, 2.0], 3.0), 'QSKIP 1 at Q>=3': (Qv >= 3).astype(float),
      'QSIZE 0/0/1/2/3': qs, 'QSIZE-R x0.5 high YZ': qs * np.where(yzh, .5, 1.0), 'QBIG 0/0/1/2/4': np.select([Qv <= 1, Qv == 2, Qv == 3], [0.0, 1.0, 2.0], 4.0)}
ALLD = pd.DatetimeIndex(sorted(H.day.unique())); ALLD = ALLD[ALLD >= '2018-01-01']
PERS = [('2018-21', '2018-01-01', '2022-01-01'), ('2022-26', '2022-01-01', '2027-01-01')]
if __name__ == '__main__':
    print('cut points (2018-21): ' + ', '.join(f'{k} {v:.2f}' for k, v in cut.items()), flush=True)
    print('pillars on: ' + ', '.join(f'{n} {v.mean():.0%}' for n, v in (('energy', energy), ('participation', part), ('location', locok), ('timing', early))), flush=True)
    print('\nPER FLIP by quality, NQ points held to the vector exit (TH) and at 75/50 (F75): 2018-21 | 2022-26 | by year 2022 2023 2024 2025 2026', flush=True)
    yr = pd.DatetimeIndex(X.day).year
    for v in range(5):
        g = X[X.Q == v]; a = g[g.day < '2022-01-01']; b = g[g.day >= '2022-01-01']
        print(f'  Q{v}: n {len(g):4d} ({len(g) / len(X):3.0%}) | TH {a.th.mean():+6.1f} | {b.th.mean():+6.1f} | F75 {a.f75.mean():+6.1f} | {b.f75.mean():+6.1f} | TH by year ' +
              ' '.join(f'{g[yr[X.Q == v] == y].th.mean():+6.1f}' if (yr[X.Q == v] == y).sum() >= 5 else '     .' for y in range(2022, 2027)), flush=True)
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
    X[['day', 'j', 'dir', 'Q']].to_parquet('/tmp/claude-0/ml/nq_quality_flips.parquet')
