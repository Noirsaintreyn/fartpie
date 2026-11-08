"""Nov 8 2026: THE QUALITY SCORE inside the prop system (Lucid). DECLARED BEFORE RESULTS. Pillars and cut points exactly as xs/nq_quality.py (2018-21 cuts):
ENERGY (GEX < 0 or 5-day vol top third), PARTICIPATION (RVOL or ATR expansion top third), LOCATION (not in the dead zone), TIMING (by 10:15). Q = 0-4.
Applied to the current prop trades (Rule A, 75 / 50, flat 12:00, Yang-Zhang + VWAP ladder for risk, eval x0.3 / funded x0.7 cold streaks), 5 accounts bought together on GREEN days:
  S1    current + skip the POC-to-VAH zone (Nov 8 recommendation)
  QS    skip Q 0-1
  QS15  skip Q 0-1, 1.5x at Q 4
  QL    0 / 0 / 0.5 / 1 / 1.5 at Q 0-4
  QF    eval: skip Q 0-1 at base size; funded: 0 / 0 / 1 / 1 / 1.75 (lever only once funded, where the drawdown is locked; 1.75 keeps 150K under 40 micros)
  QF+   QF with the dead zone skipped outright in both
Single evals every 5th session 2022-2025; full run Jan 2022 - Oct 2026; 34 rolling 24-month windows (median, worst). Points = $ / 20 after fees."""
_b = open('xs/nq_buy_policy.py').read(); exec(_b[:_b.rindex("if __name__ == '__main__':")])
XQ = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features_opt.parquet'); XQ['t'] = Hq.dt.values[XQ.j.values]; P = XQ[XQ.day < '2022-01-01']
cut = {c: P[c].quantile(2 / 3) for c in ('rvol', 'rv5', 'comp')}
KQ = {(np.datetime64(t, 'ns'), int(d)): i for i, (t, d) in enumerate(zip(XQ.t.values, XQ.dir.values))}
ii = [KQ.get((np.datetime64(t5[int(j)], 'ns'), int(d))) for j, d in zip(XA.j.values, XA.dir.values)]
col = lambda c: pd.Series([XQ[c].values[i] if i is not None else np.nan for i in ii], index=XA.index)
gx, r5, rv, cp = col('gex'), col('rv5'), col('rvol'), col('comp'); me_ = pd.Series(m5[XA.j.values.astype(int)] - 5, index=XA.index)
energy = (gx < 0) | (r5 >= cut['rv5']); part = (rv >= cut['rvol']) | (cp >= cut['comp'])
dead = (F['value'] == 'value, trade side of POC') | (F.rng == 'premium half') | ((F.vz >= 0) & (F.vz < 1)); locok = F['value'].notna() & ~dead; early = me_ <= 615
Q = energy.astype(int) + part.astype(int) + locok.astype(int) + early.astype(int)
mapq = lambda vals: pd.Series(np.select([Q <= 1, Q == 2, Q == 3], vals[:3], vals[3]), index=XA.index)
MP = {'S1 current + skip POC-to-VAH': (skip2, skip2), 'QS skip Q 0-1': (mapq([0, 1, 1, 1]), mapq([0, 1, 1, 1])), 'QS15 skip 0-1, 1.5x at Q4': (mapq([0, 1, 1, 1.5]), mapq([0, 1, 1, 1.5])),
      'QL 0/0/0.5/1/1.5': (mapq([0, .5, 1, 1.5]), mapq([0, .5, 1, 1.5])), 'QF eval skip 0-1, funded 1.75x at Q4': (mapq([0, 1, 1, 1]), mapq([0, 1, 1, 1.75])),
      'QF+ QF + dead zone skipped': (mapq([0, 1, 1, 1]) * (~dead).astype(float), mapq([0, 1, 1, 1.75]) * (~dead).astype(float))}
if __name__ == '__main__':
    n1 = nets; yy = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year; Qn = Q.loc[n1.index]
    print(f'prop trades {len(n1)}, matched {np.mean([i is not None for i in ii]):.0%}; ' + ', '.join(f'{n} {v.loc[n1.index].mean():.0%}' for n, v in (('energy', energy), ('participation', part), ('location', locok), ('timing', early))), flush=True)
    print('\n1 CONTRACT per prop trade by quality: n | 2018-21 avg / full-stop share | 2022-26 avg / stops | by year 2022 2023 2024 2025 2026', flush=True)
    for v in range(5):
        m = (Qn == v).values; a = m & (yy < 2022); b = m & (yy >= 2022); st = lambda mm: (n1.values[mm] <= -50.5).mean()
        print(f'  Q{v}: n {m.sum():4d} | {n1.values[a].mean():+6.1f} / {st(a):3.0%} | {n1.values[b].mean():+6.1f} / {st(b):3.0%} | ' +
              ' '.join(f'{n1.values[m & (yy == y)].mean():+6.1f}' if (m & (yy == y)).sum() >= 5 else '     .' for y in range(2022, 2027)), flush=True)
    for nm, (me_m, mf_m) in MP.items():
        DE = to_D2(ev_res, eqm(.3) * me_m, 'YZV'); DF = to_D2(full, eqm(.7) * mf_m, 'YZV'); print(f'\n  {nm}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; get = make_get(A); o = [get(s) for s in range(s0, int(np.searchsorted(D_, pd.Timestamp('2026-01-01'))), 5)]
            pe = np.mean([x[0] == 'pass' for x in o]); pp = np.mean([len(x[3]) > 0 for x in o]); r, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: single eval pass {pe:.0%}, evals paid {pp:.0%}', flush=True); show('ALL5-GREEN', a, r, py, med, worst, mm)
