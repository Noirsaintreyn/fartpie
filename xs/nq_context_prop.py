"""Nov 8 2026: RISK BY CONTEXT inside Lucid - payout rate and average payout. DECLARED BEFORE RESULTS.
Context score C (0-3) on each Rule A prop trade, from the context map (xs/nq_context_map.py): +1 QQQ net GEX negative | +1 entry NOT at a 20-day LVN |
+1 entry NOT below the prior-day VAL (longs bought below VAL and shorts sold below VAL both averaged about zero or worse).
Sizes are multiples of today's ladder size (Yang-Zhang + VWAP: 10 / 5 / 3 micros at 50K, x2.25 at 150K), cold-streak cuts kept:
  V0 S1 current + POC-to-VAH skip          V1 S1, size 1x at C3, 0.75x at C2, 0.5x at C0-1 (risk down when context disagrees)
  V2 S1, skip C0-1                          V3 S1, 1.25x at C3, 1x at C2, 0.5x at C0-1          V4 V1 without the POC-to-VAH skip
  V5 V1 + noise-area momentum at 2 micros (4 at 150K), 40-point stop (the payout-frequency add-on)
One account bought every 5th session Jan 2022 - Jun 2025 and 5 accounts on GREEN days. Points = $ / 20."""
_e = open('xs/nq_account_ev.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
CM = pd.read_parquet('/tmp/claude-0/ml/nq_context_map.parquet'); CM['t'] = Hq.dt.values[CM.j.values]
KC = {(np.datetime64(t, 'ns'), int(d)): i for i, (t, d) in enumerate(zip(CM.t.values, CM.dir.values))}
ic = [KC.get((np.datetime64(t5[int(j)], 'ns'), int(d))) for j, d in zip(XA.j.values, XA.dir.values)]
cc = lambda c: pd.Series([CM[c].values[i] if i is not None else None for i in ic], index=XA.index)
GAMc, NDc, VAc = cc('GAM'), cc('ND'), cc('VA')
C = (GAMc == 'neg gamma').astype(int) + ((NDc != 'LVN') & NDc.notna()).astype(int) + ((VAc != 'below VAL') & VAc.notna()).astype(int)
m3 = lambda a, b, c_: pd.Series(np.select([C >= 3, C == 2], [a, b], c_), index=XA.index)
NT = pd.read_parquet('/tmp/claude-0/ml/noise_stop_40.parquet'); dposm = {pd.Timestamp(d): i for i, d in enumerate(D_)}
def nlists(u):
    L = [[] for _ in D_]
    for r in NT.itertuples():
        k = dposm.get(pd.Timestamp(r.day))
        if k is not None: L[k].append(((u,), r.pts, r.mae, None))
    return L
merge = lambda A_, B_: [a + b for a, b in zip(A_, B_)]
V = {'V0 S1 (current + POC-VAH skip)': skip2, 'V1 context risk-down 1 / 0.75 / 0.5': skip2 * m3(1, .75, .5), 'V2 skip context 0-1': skip2 * m3(1, 1, 0),
     'V3 context tilt 1.25 / 1 / 0.5': skip2 * m3(1.25, 1, .5), 'V4 V1 without POC-VAH skip': m3(1, .75, .5), 'V5 V1 + noise 2 micros': skip2 * m3(1, .75, .5)}
if __name__ == '__main__':
    n1 = nets; yy = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year.values; cv = C.loc[n1.index].values; kp = (skip2.loc[n1.index] > 0).values
    print(f'prop trades matched {np.mean([i is not None for i in ic]):.0%}; per trade (1 contract, S1 trades) by context score: 2018-21 | 2022-26 | share', flush=True)
    for v in (3, 2, 1, 0):
        m = kp & (cv == v); print(f'  C{v}: {n1.values[m & (yy < 2022)].mean():+6.1f} (n{(m & (yy < 2022)).sum():3d}) | {n1.values[m & (yy >= 2022)].mean():+6.1f} (n{(m & (yy >= 2022)).sum():3d}) | {m.sum() / kp.sum():.0%}', flush=True)
    starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    print(f'\n  {"version":<36} {"acct":<5} pass  PAYOUT RATE  payouts/funded  AVG PAYOUT  blown | net/eval  per acct-month | 1st payout  every | 5 accounts: pts/yr  months paid', flush=True)
    for nm, sz in V.items():
        e_, f_ = to_D2(ev_res, eqm(.3) * sz, 'YZV'), to_D2(full, eqm(.7) * sz, 'YZV')
        for a in ('50K', '150K'):
            A = ACC[a]
            if nm.startswith('V5'): DE, DF = merge(e_, nlists(.2)), merge(f_, nlists(.2))
            else: DE, DF = e_, f_
            R = [get_one(A, k, 5) for k in starts]; net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)])
            fund = [r for r in R if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[k]).days / 30.44 for r, k in zip(R, starts) if r[2]]
            get = make_get(A); r_, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'  {nm:<36} {a:<5} {np.mean([r[0] == "pass" for r in R]):4.0%} {np.mean([len(r[2]) > 0 for r in R]):11.0%} {np.mean([len(r[2]) for r in fund]):15.1f} {np.mean(pays) / 20:7.0f} pts '
                  f'{np.mean([r[3] == "lost" for r in fund]):5.0%} | {net.mean():+8.0f} {net.sum() / life.sum():+15.1f} | {np.median(fp):6.1f} mo {np.median(gaps):4.0f} d | {r_["net"] / yrs:+15,.0f} {r_["mpay"]:5d}/{r_["nmo"]}', flush=True)
