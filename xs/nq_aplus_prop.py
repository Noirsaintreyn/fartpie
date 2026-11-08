"""Nov 8 2026: the A+ count inside the 75 / 50 PROP system - OUT OF SAMPLE for the prop window: the cut points come from 2018-2021 flips only, the prop test runs 2022 - Oct 2026.
DECLARED BEFORE RESULTS. A+ conditions: breakout-bar RVOL in its 2018-21 top third, 5-day realized vol in its top third, VWAP distance not in its top third, signal by 10:15,
15m ATR expansion in its top third. Applied to the current prop system's Rule A trades (75 / 50, flat 12:00, eval gap-gated, x0.3 / x0.7 cold streaks, 10 / 5 / 3 micro ladder):
  P0 as is | SKIP01 skip trades with 0-1 conditions | SKIP02 skip 0-2 | SKIP02+UP skip 0-2, 1.5x at 4-5 | TILT 0.5x at 0-2, 1x at 3, 1.5x at 4-5
Lucid 50K and 150K, 5-payout limit, 5 accounts on GREEN days; single evals every 5th session 2022-2025. Points = $ / 20. Also 1-contract points a day (no prop rules)."""
_as = open('xs/nq_account_sizes.py').read(); exec(_as[:_as.rindex("if __name__ == '__main__':")])
XF = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features.parquet')
Hq = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); Hq = Hq[(Hq.m >= 570) & (Hq.m < 960)].reset_index(drop=True)
XF['t'] = Hq.dt.values[XF.j.values]; P = XF[XF.day < '2022-01-01']
cut = {c: P[c].quantile(2 / 3) for c in ('rvol', 'rv5', 'vwd', 'comp')}
XF['ap'] = (XF.rvol >= cut['rvol']).astype(int) + (XF.rv5 >= cut['rv5']).astype(int) + (XF.vwd <= cut['vwd']).astype(int) + (XF.me <= 615).astype(int) + (XF.comp >= cut['comp']).astype(int)
APK = dict(zip(zip(XF.t.values.astype('datetime64[ns]'), XF.dir.values), XF.ap.values))
AP = pd.Series([APK.get((np.datetime64(t5[int(j)], 'ns'), int(d)), np.nan) for j, d in zip(XA.j.values, XA.dir.values)], index=XA.index)
print(f'cut points from 2018-21: ' + ', '.join(f'{k} {v:.2f}' for k, v in cut.items()) + f' | Rule A trades matched {AP.notna().mean():.0%}', flush=True)
FMS = {'P0 as is': lambda a: 1.0, 'SKIP01 skip 0-1 conditions': lambda a: 0.0 if a <= 1 else 1.0, 'SKIP02 skip 0-2': lambda a: 0.0 if a <= 2 else 1.0,
       'SKIP02+UP skip 0-2, 1.5x at 4-5': lambda a: 0.0 if a <= 2 else (1.5 if a >= 4 else 1.0), 'TILT 0.5x / 1x / 1.5x': lambda a: .5 if a <= 2 else (1.5 if a >= 4 else 1.0)}
nets1 = pd.Series({i: v[0] - 1.0 for i, v in full.items()}); dd = pd.DatetimeIndex(XA.loc[nets1.index, 'day']).normalize()
ALLD = pd.DatetimeIndex(D_[:NS])
def single(A, s_list):
    o = []
    for s in s_list:
        res, k = ev(A, s, NS); paid = 0.0; npay = 0
        if res == 'pass': fr, k2, pays = fu(A, k + 1, NS, 5); paid = sum(a for _, a in pays); npay = len(pays)
        o.append((res == 'pass', npay > 0, paid - A['fee']))
    o = np.array(o, dtype=float); return o[:, 0].mean(), o[:, 1].mean(), o[:, 2].mean()
if __name__ == '__main__':
    s0 = int(np.searchsorted(D_, pd.Timestamp('2022-01-01'))); last = int(np.searchsorted(D_, pd.Timestamp('2026-01-01')))
    starts = [int(np.searchsorted(D_, pd.Timestamp(x))) for x in ('2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01', '2024-01-01', '2024-07-01')]
    for nm, fn in FMS.items():
        fm = pd.Series([fn(a) if np.isfinite(a) else 1.0 for a in AP.values], index=XA.index)
        DE = to_D(ev_res, eqm(.3) * fm); DF = to_D(full, eqm(.7) * fm)
        p = (nets1 * fm.loc[nets1.index]); pday = pd.Series(p.values, index=dd).groupby(level=0).sum().reindex(ALLD).fillna(0.0)
        print(f'\n  {nm}: 1 contract, no prop rules, 2022 - Oct 2026: {pday.mean():+.2f} pts a day, trades kept {(fm.loc[nets1.index] > 0).mean():.0%}', flush=True)
        for a in ('50K', '150K'):
            A = ACC[a]; E, r = book(A, s0, NS, 5); W = [book(A, s, min(s + 504, NS), 5)[1] for s in starts]; pe, pp, ne = single(A, list(range(s0, last, 5)))
            yr = []
            for y in (2022, 2023, 2024, 2025, 2026):
                lo, hi = int(np.searchsorted(D_, pd.Timestamp(f'{y}-01-01'))), int(np.searchsorted(D_, pd.Timestamp(f'{y + 1}-01-01'))); g = E[(E.k >= lo) & (E.k < min(hi, NS))]
                yr.append(f'{y} {g.amt.sum() / 20:+,.0f}')
            print(f'    {a:<5} eval pass {pe:.0%}, paid {pp:.0%}, net/eval {ne / 20:+,.0f} pts | 5 accounts: {r["net"] / 20:+,.0f} pts ({r["mo"] / 20:+.0f}/month), evals {r["evals"]}, payouts {r["npay"]}, '
                  f'dry {r["gap"]} mo | 24-month median {np.median([w["net"] for w in W]) / 20:+,.0f}, worst {min(w["net"] for w in W) / 20:+,.0f} | ' + ' '.join(yr), flush=True)
