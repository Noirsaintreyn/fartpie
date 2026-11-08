"""Nov 8 2026: NOISE-AREA MOMENTUM WITH A HARD STOP inside Lucid (round 2). DECLARED BEFORE RESULTS. Round 1 (xs/nq_noise_prop.py): without a hard stop the
noise strategy blew 89-100% of funded accounts (worst trade -326 pts). Trades from xs/nq_noise_stop.py, stop 30 and 40, sizes 3 / 5 / 8 micros at 50K (x2.25 at 150K),
alone and next to the Rule A system (S1). Same report as round 1."""
_np = open('xs/nq_noise_prop.py').read(); exec(_np[_np.index('_e0 = open'):_np.rindex("if __name__ == '__main__':")])
if __name__ == '__main__':
    s1e, s1f = to_D2(ev_res, eqm(.3) * skip2, 'YZV'), to_D2(full, eqm(.7) * skip2, 'YZV')
    V = {}
    for S in (30, 40):
        NT = pd.read_parquet(f'/tmp/claude-0/ml/noise_stop_{S}.parquet')
        for mic in (3, 5, 8): V[f'N{mic} stop {S}'] = (nlists(mic / 10), nlists(mic / 10))
        for mic in (3, 5): V[f'S1 + N{mic} stop {S}'] = (merge(s1e, nlists(mic / 10)), merge(s1f, nlists(mic / 10)))
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
