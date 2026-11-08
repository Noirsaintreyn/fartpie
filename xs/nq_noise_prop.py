"""Nov 8 2026: NOISE-AREA MOMENTUM (Zarattini, Aziz & Barbon 2024) INSIDE LUCID, alone and next to the Rule A system. DECLARED BEFORE RESULTS.
Trades from xs/nq_public_strats.py (QQQ 5m -> NQ points, fills at the next 5m open, 1 pt cost, max adverse excursion kept). Engine as xs/nq_account_sizes.py.
  N5 / N10 / N15   noise only, 5 / 10 / 15 micros at 50K (x2.25 at 150K), no cold-streak cut
  S1               current Rule A system (75 / 50, POC-to-VAH skip, ladder)
  S1 + N5 / N10    both in the same account (Rule A in the morning, noise from 10:00)
One account bought every 5th session Jan 2022 - Jun 2025 (pass, paid, payouts x size, blown, net per eval, per account-month, months to the first payout, trading days
between payouts) and 5 accounts on GREEN days (payout points a day net of fees, share of weeks / two-week windows / months with a payout). Points = $ / 20."""
_e0 = open('xs/nq_account_ev.py').read(); exec(_e0[:_e0.rindex("if __name__ == '__main__':")])
NT = pd.read_parquet(glob.glob('/tmp/claude-0/ml/pub_noise_*.parquet')[0]); dposm = {pd.Timestamp(d): i for i, d in enumerate(D_)}; import glob
def nlists(u):
    L = [[] for _ in D_]
    for r in NT.itertuples():
        k = dposm.get(pd.Timestamp(r.day))
        if k is not None: L[k].append(((u,), r.pts, r.mae, None))
    return L
def merge(A_, B_): return [a + b for a, b in zip(A_, B_)]
if __name__ == '__main__':
    print(f'noise trades: {len(NT)}, worst trade {NT.pts.min():+.0f} pts, worst adverse excursion {NT.mae.min():+.0f}, 5th pct trade {NT.pts.quantile(.05):+.0f}', flush=True)
    s1e, s1f = to_D2(ev_res, eqm(.3) * skip2, 'YZV'), to_D2(full, eqm(.7) * skip2, 'YZV')
    V = {'N5 noise, 5 micros': (nlists(.5), nlists(.5)), 'N10 noise, 10 micros': (nlists(1.), nlists(1.)), 'N15 noise, 15 micros': (nlists(1.5), nlists(1.5)),
         'S1 Rule A system': (s1e, s1f), 'S1 + N5': (merge(s1e, nlists(.5)), merge(s1f, nlists(.5))), 'S1 + N10': (merge(s1e, nlists(1.)), merge(s1f, nlists(1.)))}
    starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    print(f'  {"version":<22} {"acct":<5} pass  paid  payouts x size  blown | net/eval  median | per acct-month | 1st payout  days apart || 5 accounts: pts/day  weeks  2-wk  months', flush=True)
    for nm, (de, df) in V.items():
        DE, DF = de, df
        for a in ('50K', '150K'):
            A = ACC[a]; R = [get_one(A, k, 5) for k in starts]
            net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)]); fund = [r for r in R if r[0] == 'pass']
            fp = [(D_[min(r[2][0][0], NS - 1)] - D_[k]).days / 30.44 for r, k in zip(R, starts) if r[2]]; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            gg = make_get(A); acc_ = []; cash = np.zeros(NS + 1); pay = np.zeros(NS + 1)
            for k in range(s0, NS):
                if not GREEN[k]: continue
                act = [x for x in acc_ if x >= k]
                while len(act) < 5:
                    res, k2, last, pays = gg(k); acc_.append(last); act.append(last); cash[k] -= A['fee']
                    for kp, amt in pays: cash[min(kp, NS)] += amt; pay[min(kp, NS)] += amt
            p = pd.Series(pay[s0:NS], index=D_[s0:NS]); wk = (p.groupby(p.index.to_period('W')).sum() > 0).mean(); bi = (p.groupby(np.arange(len(p)) // 10).sum() > 0).mean(); mo = (p.groupby(p.index.to_period('M')).sum() > 0).mean()
            print(f'  {nm:<22} {a:<5} {np.mean([r[0] == "pass" for r in R]):4.0%} {np.mean([len(r[2]) > 0 for r in R]):5.0%}  {np.mean([len(r[2]) for r in fund]) if fund else 0:4.1f} x {np.mean([x for r in fund for _, x in r[2]]) / 20 if any(r[2] for r in fund) else 0:3.0f} pts '
                  f'{np.mean([r[3] == "lost" for r in fund]) if fund else 0:5.0%} | {net.mean():+8.0f} {np.median(net):+7.0f} | {net.sum() / life.sum():+14.1f} | {np.median(fp) if fp else np.nan:6.1f} mo {np.median(gaps) if gaps else np.nan:7.0f}'
                  f' || {cash[s0:NS].sum() / 20 / (NS - s0):+16.1f} {wk:6.0%} {bi:5.0%} {mo:6.0%}', flush=True)
