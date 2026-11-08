"""Nov 8 2026: PICK THE DAYS, NOT THE TRADES - noise-area momentum (40-pt stop) by day context, and sized up only on good-context days inside Lucid. DECLARED BEFORE RESULTS.
Day context at the prior close: QQQ net GEX sign, 5-day realized vol percentile (thirds, trailing year). Part 1: noise per day by context (avg, share of days >= +7.5 pts at
1 contract, 2018-21 / 2022-26). Part 2 (Lucid, as xs/nq_noise_prop2.py): S1 + noise at 2 micros every day (baseline) vs S1 + noise at 5 / 8 micros on negative-gamma days only
(0 otherwise), and on negative-gamma + high-vol days only."""
_np = open('xs/nq_noise_prop.py').read(); exec(_np[_np.index('_e0 = open'):_np.rindex("if __name__ == '__main__':")])
NT = pd.read_parquet('/tmp/claude-0/ml/noise_stop_40.parquet'); NT['day'] = pd.DatetimeIndex(NT.day).normalize()
O3 = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); gx = O3.net_gex.copy(); gx.index = gx.index.normalize()
dq = Hq.groupby('day').close.last(); rv = np.log(dq).diff().rolling(5).std(); rvp = rv.rolling(252, min_periods=120).rank(pct=True)
ctx = pd.DataFrame({'gneg': gx.reindex(dq.index, method='ffill').shift(1) < 0, 'rvp': rvp.shift(1)})
NDp = NT.groupby('day').pts.sum(); ALLd = pd.DatetimeIndex(dq.index[dq.index >= '2018-02-01']); NDp = NDp.reindex(ALLd).fillna(0.0); cx = ctx.reindex(ALLd)
def nlists_m(u_by_day):
    L = [[] for _ in D_]
    for r in NT.itertuples():
        k = dposm.get(pd.Timestamp(r.day)); u = u_by_day.get(pd.Timestamp(r.day), 0.0)
        if k is not None and u > 0: L[k].append(((u,), r.pts, r.mae, None))
    return L
if __name__ == '__main__':
    print('PART 1 noise (stop 40) per day by context, 1 contract: avg pts/day, share of days >= +7.5 | 2018-21 avg / 2022-26 avg', flush=True)
    vb = np.select([cx.rvp >= 2 / 3, cx.rvp <= 1 / 3], ['high vol', 'low vol'], 'mid vol')
    for gl, gm in (('neg gamma', cx.gneg.values), ('pos gamma', ~cx.gneg.values)):
        for vl in ('high vol', 'mid vol', 'low vol'):
            m = gm & (vb == vl); P = NDp[m]
            print(f'  {gl} + {vl:<8}: days {m.sum():4d} avg {P.mean():+6.2f} | days >= +7.5: {np.mean(P >= 7.5):3.0%} | {P[P.index < "2022-01-01"].mean():+6.2f} / {P[P.index >= "2022-01-01"].mean():+6.2f}', flush=True)
    s1e, s1f = to_D2(ev_res, eqm(.3) * skip2, 'YZV'), to_D2(full, eqm(.7) * skip2, 'YZV')
    good1 = {d: 1.0 for d, v in cx.gneg.items() if v}; good2 = {d: 1.0 for d, (g_, r_) in cx.iterrows() if g_ and r_ >= 2 / 3}; every = {d: 1.0 for d in ALLd}
    V = {'S1 + noise 2 micros every day': (every, .2), 'S1 + noise 5 micros, neg gamma days only': (good1, .5), 'S1 + noise 8 micros, neg gamma days only': (good1, .8),
         'S1 + noise 5 micros, neg gamma + high vol only': (good2, .5), 'S1 + noise 8 micros, neg gamma + high vol only': (good2, .8)}
    starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    print(f'\n  {"version":<46} {"acct":<5} pass  PAYOUT RATE  payouts x size  blown | net/eval  per acct-month | 1st payout  every', flush=True)
    for nm, (days_on, u) in V.items():
        nl = nlists_m({d: u for d in days_on}); DE, DF = merge(s1e, nl), merge(s1f, nl)
        for a in ('50K', '150K'):
            A = ACC[a]; R = [get_one(A, k, 5) for k in starts]; net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)])
            fund = [r for r in R if r[0] == 'pass']; pays = [x for r in fund for _, x in r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[k]).days / 30.44 for r, k in zip(R, starts) if r[2]]
            print(f'  {nm:<46} {a:<5} {np.mean([r[0] == "pass" for r in R]):4.0%} {np.mean([len(r[2]) > 0 for r in R]):11.0%}  {np.mean([len(r[2]) for r in fund]):4.1f} x {np.mean(pays) / 20 if pays else 0:3.0f} pts '
                  f'{np.mean([r[3] == "lost" for r in fund]):5.0%} | {net.mean():+8.0f} {net.sum() / life.sum():+15.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:4.0f} d', flush=True)
