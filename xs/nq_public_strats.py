"""Nov 8 2026: THE PUBLISHED INTRADAY STRATEGIES MANY TRADERS RUN, on our NQ data. DECLARED BEFORE RESULTS. QQQ 5m RTH bars 2018 - Oct 2026 converted to NQ points
(QQQ move x NQ / QQQ at the prior close), 1 contract, 1 pt round-trip cost per trade, fills at the next 5m open (more conservative than the papers' same-minute fills).
  ORB   Zarattini & Aziz (2023) opening range breakout: direction of the first N-minute bar(s) (open vs close), enter at the next open, stop at the range low (long) / high
        (short), target 10x the risk, else out at the close. N = 5 (the paper), 15, 30.
  NOISE Zarattini, Aziz & Barbon (2024) 'Beat the Market' noise-area momentum: bounds = max/min(today's open, yesterday's close) x (1 +/- average 14-day absolute move
        from the open at that time of day); checked every HH:00 / HH:30 from 10:00; above the upper bound = long, below the lower = short; trailing stop = the bound or
        VWAP (whichever is tighter) at the same checks; flat at the close.
  IMOM  Gao, Han, Li & Zhou (2018) intraday momentum: return from yesterday's close to 10:00 sets the direction of the last half hour (15:30 - 16:00). Also IMOM-STRONG:
        only when that return is beyond its trailing 1-year median size.
Report: trades, win, points per trade, trades a day, points a day, Sharpe, worst month, max drawdown, by year (points a day), and the correlation of daily results with
our two streams (RA = Rule A prop trades, C30 = composite trend-hold)."""
_f = open('xs/nq_floor.py').read(); exec(_f[:_f.rindex("if __name__ == '__main__':")])
vol = H.volume.values.astype(float); tpv = (h5 + l5 + c5) / 3 * vol
grp = H.groupby('day').indices; days = sorted(grp.keys()); vwap = np.empty(len(H))
for d in days:
    ix = grp[d]; vwap[ix] = np.cumsum(tpv[ix]) / np.maximum(np.cumsum(vol[ix]), 1)
def run_exit(i_in, d, e, stop, tgt, ix_day):
    """walk 5m bars from i_in; stop first, then target; else the day's last close. returns (exit price, exit idx, mae price move)"""
    worst = 0.0
    for q in ix_day[ix_day >= i_in]:
        adv = (l5[q] - e) * d if d > 0 else (h5[q] - e) * d; worst = min(worst, adv)
        if stop is not None and ((l5[q] <= stop) if d > 0 else (h5[q] >= stop)): return stop, q, min(worst, (stop - e) * d)
        if tgt is not None and ((h5[q] >= tgt) if d > 0 else (l5[q] <= tgt)): return tgt, q, worst
    q = ix_day[-1]; return c5[q], q, worst
def orb(nb):
    rows = []
    for d in days:
        ix = grp[d]
        if len(ix) < 70 or m5[ix[0]] != 570: continue
        o0, cN = o5[ix[0]], c5[ix[nb - 1]]; dr = int(np.sign(cN - o0))
        if dr == 0: continue
        e = o5[ix[nb]]; stop = (l5[ix[:nb]].min() if dr > 0 else h5[ix[:nb]].max()); R = (e - stop) * dr
        if R <= 0: continue
        px, q, mae = run_exit(ix[nb], dr, e, stop, e + 10 * R * dr, ix); kk = k5[ix[0]]
        rows.append((d, (px - e) * dr * kk - 1.0, mae * kk - 1.0, R * kk))
    return pd.DataFrame(rows, columns=['day', 'pts', 'mae', 'risk'])
POS = {}
for d in days:
    for p, i in enumerate(grp[d]): POS[i] = p
mv = pd.DataFrame({'day': d5, 'p': [POS[i] for i in range(len(H))], 'mv': np.abs(c5 / np.array([o5[grp[d][0]] for d in d5]) - 1)})
MV = mv.pivot_table(index='day', columns='p', values='mv'); SIG = MV.rolling(14, min_periods=14).mean().shift(1)
def noise():
    rows = []
    for di, d in enumerate(days):
        if di == 0 or d not in SIG.index or SIG.loc[d].isna().all(): continue
        ix = grp[d]; o0 = o5[ix[0]]; pc = c5[grp[days[di - 1]][-1]]; kk = k5[ix[0]]; pos = 0; e = None; ie = None
        for p, i in enumerate(ix[:-1]):
            if (m5[i] + 5) % 30 != 0 or m5[i] + 5 < 600 or m5[i] + 5 > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); c = c5[i]; nxt = o5[ix[p + 1]]
            if pos == 1 and c < max(ub, vwap[i]): rows.append((d, (nxt - e) * kk - 1.0, (l5[ie:i + 2].min() - e) * kk - 1.0)); pos = 0
            elif pos == -1 and c > min(lb, vwap[i]): rows.append((d, (e - nxt) * kk - 1.0, (e - h5[ie:i + 2].max()) * kk - 1.0)); pos = 0
            if pos == 0 and c > ub: pos, e, ie = 1, nxt, ix[p + 1]
            elif pos == 0 and c < lb: pos, e, ie = -1, nxt, ix[p + 1]
        if pos != 0:
            x = c5[ix[-1]]; rows.append((d, (x - e) * pos * kk - 1.0, ((l5[ie:ix[-1] + 1].min() - e) if pos > 0 else (e - h5[ie:ix[-1] + 1].max())) * kk - 1.0))
    return pd.DataFrame(rows, columns=['day', 'pts', 'mae'])
def imom(strong):
    rows = []; r1s = []
    for di, d in enumerate(days):
        if di == 0: continue
        ix = grp[d]; i10 = [i for i in ix if m5[i] == 595]; i330 = [i for i in ix if m5[i] == 930]
        if not i10 or not i330: continue
        pc = c5[grp[days[di - 1]][-1]]; r1 = c5[i10[0]] / pc - 1; r1s.append(abs(r1))
        if strong and (len(r1s) < 120 or abs(r1) < np.median(r1s[-253:-1])): continue
        dr = int(np.sign(r1)); e = o5[i330[0]]; kk = k5[ix[0]]; x = c5[ix[-1]]
        mae = ((l5[i330[0]:ix[-1] + 1].min() - e) if dr > 0 else (e - h5[i330[0]:ix[-1] + 1].max())) * kk - 1.0
        rows.append((d, (x - e) * dr * kk - 1.0, mae))
    return pd.DataFrame(rows, columns=['day', 'pts', 'mae'])
STR = {'ORB 5-min (paper)': orb(1), 'ORB 15-min': orb(3), 'ORB 30-min': orb(6), 'NOISE area momentum': noise(), 'IMOM last half hour': imom(False), 'IMOM-STRONG': imom(True)}
if __name__ == '__main__':
    for nm, T in STR.items(): T.to_parquet(f'/tmp/claude-0/ml/pub_{nm.split()[0].lower()}_{len(T)}.parquet')
    print(f'{"strategy":<22} trades  win  pts/trade  trades/day  pts/day  Sharpe  worst mo  max DD  corr RA  corr C30 | by year (points a day)', flush=True)
    for nm, T in STR.items():
        P = T.groupby(pd.DatetimeIndex(T.day)).pts.sum().reindex(AD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum(); cum = P.cumsum()
        extra = f' | risk median {T.risk.median():.0f} pts, R per trade {np.mean(T.pts / T.risk):+.3f}' if 'risk' in T else ''
        print(f'{nm:<22} {len(T):6d} {np.mean(T.pts > 0):4.0%} {T.pts.mean():+9.2f} {len(T) / len(AD):10.2f} {P.mean():+8.2f} {P.mean() / P.std() * np.sqrt(252):7.2f} {M.min():+9.0f} {(cum - cum.cummax()).min():+7.0f} '
              f'{np.corrcoef(P, RA)[0, 1]:+8.2f} {np.corrcoef(P, C30)[0, 1]:+9.2f} | ' + ' '.join(f'{y} {v:+.1f}' for y, v in P.groupby(P.index.year).mean().items()) + extra, flush=True)
