"""Nov 8 2026: ONE COMPOSITE SIGNAL FROM EVERYTHING WITH EDGE, size proportional to it ('the more, the most leverage; the less, the least'). DECLARED BEFORE RESULTS.
Base filter: Rule A (60-day volume-profile path / hard break / monthly LVN). Composite S = count of 8 conditions known at entry (cut points from 2018-21):
  1 QQQ net GEX negative       2 5-day realized vol top third      3 breakout RVOL top third     4 15m ATR expansion top third
  5 volume-profile location OK (not inside value POC-to-edge, not premium half, not 0-1 VWAP sd)   6 signal by 10:15
  7 good range location (discount half, or beyond the prior-day extreme with the trade)           8 against the 50-day trend (fade the daily stretch)
Per trade (1 contract) by S, 2018-21 and 2022-26. Sizing on the prop trades:
  BASE  C1 (current + POC-to-VAH skip)
  P1 CAPPED     0.33x at S <= 3, 0.5x at 4, 0.75x at 5, 1x at 6+ (best trades at today's size, the rest smaller; respects Lucid's drawdown)
  P2 TILT       0.5x / 0.75x / 1x / 1.25x / 1.5x at S <= 3 / 4 / 5 / 6 / 7+
  P3 LINEAR     size = S / 5, clipped 0.2 - 1.75
Report: no prop rules (1 contract x size, points per trade and a day, 2022-26) and one Lucid 50K account (as xs/nq_account_ev.py)."""
_e = open('xs/nq_account_ev.py').read(); exec(_e[:_e.rindex("if __name__ == '__main__':")])
c = {'gex': (gx < 0), 'vol': (r5 >= cut['rv5']), 'rvol': (rv >= cut['rvol']), 'atr': (cp >= cut['comp']), 'vp': locok, 'early': early,
     'range': F.rng.isin(['discount half', 'beyond trade-side extreme']), 'fade50': ~w50}
SC = sum(v.astype(int) for v in c.values())
tier = lambda vals: pd.Series(np.select([SC <= 3, SC == 4, SC == 5, SC == 6], vals[:4], vals[4]), index=XA.index)
SZ = {'BASE C1': skip2, 'P1 CAPPED 0.33-1x': skip2 * tier([.33, .5, .75, 1, 1]), 'P2 TILT 0.5-1.5x': skip2 * tier([.5, .75, 1, 1.25, 1.5]), 'P3 LINEAR S/5': skip2 * (SC / 5).clip(.2, 1.75)}
if __name__ == '__main__':
    n1 = nets; yy = pd.DatetimeIndex(XA.loc[n1.index, 'day']).year.values; kept = (skip2.loc[n1.index] > 0).values; sc = SC.loc[n1.index].values
    print('conditions on (prop trades): ' + ', '.join(f'{k} {v.loc[n1.index].mean():.0%}' for k, v in c.items()), flush=True)
    print('\nPER TRADE by composite S (1 contract): 2018-21 avg (n) | 2022-26 avg (n) | by year 2022-2026', flush=True)
    for lab, m in (('S 0-3', sc <= 3), ('S 4', sc == 4), ('S 5', sc == 5), ('S 6', sc == 6), ('S 7-8', sc >= 7)):
        a = m & kept & (yy < 2022); b = m & kept & (yy >= 2022)
        print(f'  {lab:<6} {n1.values[a].mean():+6.1f} (n{a.sum():3d}) | {n1.values[b].mean():+6.1f} (n{b.sum():3d}) | ' + ' '.join(f'{n1.values[b & (yy == y)].mean():+6.1f}' if (b & (yy == y)).sum() >= 4 else '     .' for y in range(2022, 2027)), flush=True)
    A = ACC['50K']; starts = list(range(s0, int(np.searchsorted(D_, pd.Timestamp('2025-07-01'))), 5)); AD = pd.DatetimeIndex(D_[s0:NS])
    print(f'\n  {"version":<22} no prop rules 2022-26: per trade  per day  avg size | ONE 50K: pass  paid  payouts x size  blown | net/eval  median | per acct-month', flush=True)
    for nm, sz in SZ.items():
        idx = [i for i in full if pd.Timestamp(XA.at[i, 'day']) >= pd.Timestamp('2022-01-01') and sz.loc[i] > 0]
        pts = pd.Series([(full[i][0] - 1.0) * sz.loc[i] for i in idx], index=pd.DatetimeIndex(XA.loc[idx, 'day']).normalize()); pday = pts.groupby(level=0).sum().reindex(AD).fillna(0.0)
        DE = to_D2(ev_res, eqm(.3) * sz, 'YZV'); DF = to_D2(full, eqm(.7) * sz, 'YZV'); R = [get_one(A, k, 5) for k in starts]
        net = np.array([sum(x for _, x in r[2]) / 20 - A['fee'] / 20 for r in R]); life = np.array([(r[4] - k + 1) / 21 for r, k in zip(R, starts)]); fund = [r for r in R if r[0] == 'pass']
        print(f'  {nm:<22} {pts.mean():+30.1f} {pday.mean():+8.2f} {np.mean([sz.loc[i] for i in idx]):9.2f} | {np.mean([r[0] == "pass" for r in R]):10.0%} {np.mean([len(r[2]) > 0 for r in R]):5.0%}  '
              f'{np.mean([len(r[2]) for r in fund]):4.1f} x {np.mean([x for r in fund for _, x in r[2]]) / 20:3.0f} pts {np.mean([r[3] == "lost" for r in fund]):5.0%} | {net.mean():+8.0f} {np.median(net):+7.0f} | {net.sum() / life.sum():+14.1f}', flush=True)
