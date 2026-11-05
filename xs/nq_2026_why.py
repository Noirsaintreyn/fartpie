"""Nov 4 2026: why the NQ prop system paid nothing in 2026. Diagnostics, year by year 2022 - 2026 (2026 = Jan 2 -> Oct 8):
  MARKET  QQQ return, RTH day range and 09:30-12:00 range in NQ pts, HAR percentile, % negative-gamma days, flips per day
  TRADES  (Rule A and every flip, 75/50, 1 lot) win, avg pts, % target / stop / noon exit, avg best open profit (MFE) of losers, longs vs shorts, trades per month
  EVAL    the 2026 Lucid 50K path (Rule A sizing): each eval's start, end, result, best balance."""
_lz = open('xs/nq_lucid25k.py').read(); exec(_lz[:_lz.rindex("if __name__ == '__main__':")])
def sim1(j, d, kk, T=75, S=50):
    e = o5[j]; mfe = 0.0
    for q in range(j, len(H5)):
        if d5[q] != d5[j] or m5[q] >= 720: return (c5[q - 1] - e) * d * kk - 1, 'noon', mfe
        lo_ = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi_ = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
        if lo_ <= -S: return -S - 1, 'stop', max(mfe, min(hi_, T))
        if hi_ >= T: return T - 1, 'target', T
        mfe = max(mfe, hi_)
    return None
def trades(Cc):
    out = []
    for day, g in Cc.sort_values('j').groupby('day'):
        busy = -1; n = 0
        for j, d, kk in zip(g.j.values, g.dir.values, g.k.values):
            if n >= 2 or j <= busy: continue
            r = sim1(j, d, kk)
            if r is None: continue
            out.append((day, d, *r)); n += 1
            q = j
            while q < len(H5) and d5[q] == d5[j] and m5[q] < 720: q += 1
            busy = j if r[1] != 'noon' else q          # approximate: the next flip after a non-noon exit is allowed
    return pd.DataFrame(out, columns=['day', 'dir', 'net', 'how', 'mfe'])
YR = lambda s: np.where(s.dt.year >= 2026, '2026', s.dt.year.astype(str))
RT = pd.DataFrame({'o': o5, 'h': h5, 'l': l5, 'c': c5, 'd': d5, 'm': m5})[rthm]
day_ = RT.groupby('d').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last')); am = RT[RT.m < 720].groupby('d').agg(h=('h', 'max'), l=('l', 'min'))
day_.index = pd.DatetimeIndex(day_.index); am.index = pd.DatetimeIndex(am.index); lvl = NQs.reindex(day_.index - pd.Timedelta(days=1), method='ffill').values / day_.o.values
day_['rng'] = (day_.h - day_.l) * lvl; day_['amr'] = (am.h - am.l).reindex(day_.index).values * lvl; day_['trend'] = (day_.c - day_.o).abs() / (day_.h - day_.l)
day_ = day_[day_.index >= P0]; y_ = np.where(day_.index.year >= 2026, '2026', day_.index.year.astype(str))
print('MARKET (2022 - Oct 8 2026)', flush=True)
for y in sorted(set(y_)):
    d = day_[y_ == y]; qr = d.c.iloc[-1] / d.o.iloc[0] - 1
    hp = np.nanmean(prior(HP, d.index)); neg = np.mean([NEG.get(x, False) for x in d.index]) if 'NEG' in globals() else np.nan
    nfl = (X.day.dt.year.astype(str).replace({'2026': '2026'}) == y).sum() / len(d)
    print(f'  {y}: QQQ {qr:+.0%} | day range median {d.rng.median():.0f} pts | 09:30-12:00 range median {d.amr.median():.0f} pts | trend days (|close-open| > 60% of range) {np.mean(d.trend > .6):.0%} | HAR pct avg {hp:.2f} | AM flips/day {nfl:.2f}', flush=True)
print('\nTRADES (75/50, 1 lot)', flush=True)
for nm, Cc in (('Rule A', X[~cutA]), ('every flip', X)):
    T_ = trades(Cc); T_['y'] = YR(T_.day)
    for y, g in T_.groupby('y'):
        if y < '2022': continue
        nmo = 9.25 if y == '2026' else 12
        print(f'  {nm:<10} {y}: n {len(g):>3} ({len(g) / nmo:.1f}/mo) win {g.net.gt(0).mean():.0%} avg {g.net.mean():+5.1f} | target {np.mean(g.how == "target"):.0%} stop {np.mean(g.how == "stop"):.0%} noon {np.mean(g.how == "noon"):.0%} '
              f'| losers\' best open profit {g[g.net <= 0].mfe.mean():4.0f} pts | longs {g[g.dir > 0].net.mean():+5.1f} (n{(g.dir > 0).sum()}) shorts {g[g.dir < 0].net.mean():+5.1f} (n{(g.dir < 0).sum()})', flush=True)
print('\nEVAL PATH 2026, Lucid 50K, Rule A sizing', flush=True)
f, size = SET['Lucid 50K, full size (10/5/3 MNQ)']; k = int(np.searchsorted(D_, pd.Timestamp('2026-01-01')))
while k < NS:
    st = dict(bal=0.0, floor=-2000.0, peak=0.0); best = 0.0; td = 0; s0 = k; res = 'open'; hi = 0.0
    for kk in range(s0, NS):
        pnl, br = dayp(kk, st, size, f); td += bool(DAY[kk])
        if br: res = 'FAILED'; k = kk + 1; break
        st['bal'] += pnl; best = max(best, pnl); hi = max(hi, st['bal'])
        if st['bal'] >= 3000 and best <= .5 * st['bal']: res = 'PASSED'; k = kk + 1; break
        st['peak'] = max(st['peak'], st['bal']); st['floor'] = min(max(st['floor'], st['peak'] - 2000), 100)
    else: k = NS
    end = D_[min(kk, NS - 1)]
    print(f'  eval from {D_[s0]:%b %d} to {end:%b %d}: {res:<7} | trading days {td} | best balance {hi:+,.0f} | final {st["bal"]:+,.0f} | best day {best:+,.0f}', flush=True)
    if res == 'PASSED': break
