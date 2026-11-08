"""Nov 8 2026: NOISE-AREA MOMENTUM WITH A HARD STOP (for prop drawdown rules). DECLARED BEFORE RESULTS. Same rules as xs/nq_public_strats.py NOISE, plus a hard stop of
S NQ points checked on every 5m bar (filled at the stop price), S = 30 / 40 / 50 / 75; none = the paper. 1 contract, 1 pt cost. Per trade / per day / by year / correlations;
trade lists saved for the Lucid run (xs/nq_noise_prop2.py)."""
_p = open('xs/nq_public_strats.py').read(); exec(_p[:_p.index('STR = {')])
def noise2(S):
    rows = []
    for di, d in enumerate(days):
        if di == 0 or d not in SIG.index or SIG.loc[d].isna().all(): continue
        ix = grp[d]; o0 = o5[ix[0]]; pc = c5[grp[days[di - 1]][-1]]; kk = k5[ix[0]]; pos = 0; e = None; ie = None; pend = None
        for p, i in enumerate(ix):
            if pend is not None: pos, e, ie = pend[0], o5[i], i; pend = None
            if pos != 0 and S is not None:
                stp = e - pos * S / kk
                if (l5[i] <= stp) if pos > 0 else (h5[i] >= stp): rows.append((d, -S - 1.0, -S - 1.0)); pos = 0; continue
            if p == len(ix) - 1: break
            if (m5[i] + 5) % 30 != 0 or m5[i] + 5 < 600 or m5[i] + 5 > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); c = c5[i]; nxt = o5[ix[p + 1]]
            if pos == 1 and c < max(ub, vwap[i]): rows.append((d, (nxt - e) * kk - 1.0, (l5[ie:i + 2].min() - e) * kk - 1.0)); pos = 0
            elif pos == -1 and c > min(lb, vwap[i]): rows.append((d, (e - nxt) * kk - 1.0, (e - h5[ie:i + 2].max()) * kk - 1.0)); pos = 0
            if pos == 0 and c > ub: pend = (1,)
            elif pos == 0 and c < lb: pend = (-1,)
        if pos != 0:
            x = c5[ix[-1]]; mae = ((l5[ie:ix[-1] + 1].min() - e) if pos > 0 else (e - h5[ie:ix[-1] + 1].max())) * kk - 1.0
            rows.append((d, (x - e) * pos * kk - 1.0, max(mae, -S - 1.0) if S else mae))
    return pd.DataFrame(rows, columns=['day', 'pts', 'mae'])
if __name__ == '__main__':
    print(f'{"version":<16} trades  win  pts/trade  pts/day  Sharpe  worst mo  max DD  worst trade  p5 trade  corr RA  corr C30 | 2018-21 / 2022-26 pts/day | by year', flush=True)
    for S in (None, 30, 40, 50, 75):
        T = noise2(S); T.to_parquet(f'/tmp/claude-0/ml/noise_stop_{S}.parquet')
        P = T.groupby(pd.DatetimeIndex(T.day)).pts.sum().reindex(AD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum(); cum = P.cumsum()
        print(f'{"stop " + str(S):<16} {len(T):6d} {np.mean(T.pts > 0):4.0%} {T.pts.mean():+9.2f} {P.mean():+8.2f} {P.mean() / P.std() * np.sqrt(252):7.2f} {M.min():+9.0f} {(cum - cum.cummax()).min():+7.0f} '
              f'{T.pts.min():+11.0f} {T.pts.quantile(.05):+9.0f} {np.corrcoef(P, RA)[0, 1]:+8.2f} {np.corrcoef(P, C30)[0, 1]:+9.2f} | {P[P.index < "2022-01-01"].mean():+5.2f} / {P[P.index >= "2022-01-01"].mean():+5.2f} | '
              + ' '.join(f'{y} {v:+.1f}' for y, v in P.groupby(P.index.year).mean().items()), flush=True)
