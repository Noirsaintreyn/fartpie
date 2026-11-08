"""Nov 8 2026: SKIP THE 'EVERYTHING ALIGNED' RULE A TRADES inside Lucid. DECLARED BEFORE RESULTS. From xs/nq_align_ema.py: Rule A morning flips where the QQQ 5m, 1h AND
daily vectors all agree averaged -8.9 at 75 / 50 (2018-21 -12.4, 22-23 -9.3, 24-25 -23.0) vs +13.8 otherwise; an EMA 9/21 cross in the same direction within 2 bars
(EMA flip confluence) +3.8 vs +13.0. Versions on the S1 system (current + POC-to-VAH skip): S1 | S1 + skip ALL3 | S1 + skip ALL3 or EMA-flip confluence.
One account bought every 5th session Jan 2022 - Jun 2025 (as xs/nq_account_ev.py) and 5 accounts on GREEN days; 1-contract points per trade."""
_e = open('xs/nq_account_ev.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
XL2 = pd.read_parquet('/tmp/claude-0/ml/nq_align_ema_flips.parquet'); KA = {(np.datetime64(t, 'ns'), int(d)): i for i, (t, d) in enumerate(zip(pd.to_datetime(XL2.t).values, XL2.dir.values))}
ia = [KA.get((np.datetime64(t5[int(j)], 'ns'), int(d))) for j, d in zip(XA.j.values, XA.dir.values)]
colA = lambda c: pd.Series([XL2[c].values[i] if i is not None else np.nan for i in ia], index=XA.index)
all3 = colA('ALL3') > 0; efl = colA('EFLIP') > 0
V = {'S1': skip2, 'S1 + skip ALL3 aligned': skip2 * (~all3).astype(float), 'S1 + skip ALL3 or EMA-flip confluence': skip2 * (~(all3 | efl)).astype(float)}
if __name__ == '__main__':
    n1 = nets; yy = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year.values
    print(f'prop trades matched {np.mean([i is not None for i in ia]):.0%}; ALL3 aligned {all3.loc[n1.index].mean():.0%}, EMA-flip confluence {efl.loc[n1.index].mean():.0%}', flush=True)
    starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5))
    for nm, sz in V.items():
        k_ = (sz.loc[n1.index] > 0).values; sk = ~k_ & (skip2.loc[n1.index] > 0).values
        print(f'\n  {nm}: kept trades 2018-21 {n1.values[k_ & (yy < 2022)].mean():+.1f} / 2022-26 {n1.values[k_ & (yy >= 2022)].mean():+.1f} per trade; '
              f'skipped {sk.sum()} trades: {n1.values[sk & (yy < 2022)].mean() if (sk & (yy < 2022)).any() else np.nan:+.1f} / {n1.values[sk & (yy >= 2022)].mean() if (sk & (yy >= 2022)).any() else np.nan:+.1f}', flush=True)
        DE = to_D2(ev_res, eqm(.3) * sz, 'YZV'); DF = to_D2(full, eqm(.7) * sz, 'YZV')
        for a in ('50K', '150K'):
            A = ACC[a]; R = [get_one(A, k, 5) for k in starts]; net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)])
            fund = [r for r in R if r[0] == 'pass']; gaps = [b[0] - a_[0] for r in fund for a_, b in zip(r[2][:-1], r[2][1:])]
            get = make_get(A); r_, py, med, worst, mm = line(get, A, GREEN, 'ALL5')
            print(f'    {a}: one account: pass {np.mean([r[0] == "pass" for r in R]):.0%}, paid {np.mean([len(r[2]) > 0 for r in R]):.0%}, {np.mean([len(r[2]) for r in fund]):.1f} payouts, blown {np.mean([r[3] == "lost" for r in fund]):.0%}, '
                  f'net/eval {net.mean():+.0f}, per account-month {net.sum() / life.sum():+.1f}, payouts {np.median(gaps):.0f} days apart', flush=True); show('5 accts ALL5', a, r_, py, med, worst, mm)
