"""Nov 8 2026: THE 7.5-POINT DAY. Lucid 50K counts a day as qualifying at +$150 = +7.5 NQ points at 10 micros; 2 payouts a month needs ~12 of them a month (57% of days).
The current dealer-regime model makes 6.6. DECLARED BEFORE RESULTS.
Question: is there an entry where price reaches +7.5 net (+8.5 gross, 1 pt cost) before the stop often enough AND without losing money?
Yardstick: with no edge (random walk), P(+T before -S) = S / (S + T). So 12 days a month is free with a ~10-pt stop - at zero edge minus cost. What matters is the hit rate
ABOVE that line. Real NQ futures 5m, one trade a day, first touch, out at 15:55 if neither. A bar that touches both: bullish bar = open-low-high-close, bearish = open-high-low-close.
Baseline = the SAME entries taken the OTHER way (same time of day, same bar sizes), so 5m resolution bias cancels.
Entries:  LONG at 09:35 | ON long at the 18:00 open the evening before (the session's start), held into RTH | VEC first vector 15m flip 09:45 - 15:30
          | NOISE first noise-momentum signal | each also split by the prior close's QQQ gamma: top third vs the rest.
Stops 4 / 6 / 8.5 / 12 / 17 / 25 / 40 points. Reported 2018-21 and 2022-25: hit rate signal side vs other side, points per trade, qualifying days a month."""
_s2 = open('/home/claude/fartpie/xs/dte_season2.py').read(); exec(_s2[:_s2.rindex("if __name__ == '__main__':")])
_gr = open('/home/claude/fartpie/xs/dte_greeks.py').read(); exec(_gr[_gr.index('def m1_dir'):_gr.index('def r1_dir')])
Xo, Xh, Xl, Xc, Xm, Xs = (X[k].values for k in ('open', 'high', 'low', 'close', 'm', 'sym')); Xdt = X.dt.values; h2x = np.where(X.rth.values)[0]
TG = 8.5; STOPS = [4, 6, 8.5, 12, 17, 25, 40]
end_x = {}
for d in days:
    ix = gi[d]; w = ix[m[ix] < 955]
    if len(w): end_x[d] = h2x[w[-1]]
def touch(j, jend, dr, S):
    e = Xo[j]; hi, lo = Xh[j:jend + 1], Xl[j:jend + 1]
    adv = (lo <= e - S) if dr > 0 else (hi >= e + S); fav = (hi >= e + TG) if dr > 0 else (lo <= e - TG)
    ia = np.argmax(adv) if adv.any() else 10 ** 9; ib = np.argmax(fav) if fav.any() else 10 ** 9
    if ia == 10 ** 9 and ib == 10 ** 9: return (Xc[jend] - e) * dr - 1.0, j + len(hi) - 1
    if ia == ib: stop_first = (Xc[j + ia] >= Xo[j + ia]) == (dr > 0)       # same bar: bullish bar = open-low-high-close, bearish = open-high-low-close
    else: stop_first = ia < ib
    return (-S - 1.0, j + ia) if stop_first else (TG - 1.0, j + ib)
def entries():
    E = {'LONG': [], 'ON': [], 'VEC': [], 'NOISE': []}
    for k, d in enumerate(days):
        if k == 0 or not ok(d) or DY.roll[d] or d not in end_x: continue
        ix = gi[d]; a = np.where(m[ix] == 575)[0]
        if len(a): E['LONG'].append((d, h2x[ix[a[0]]], 1))
        j0 = int(np.searchsorted(Xdt, np.datetime64(days[k - 1] + pd.Timedelta(hours=18), 'ns')))
        if j0 < end_x[d] and Xs[j0] == Xs[end_x[d]] and (pd.Timestamp(Xdt[j0]) - days[k - 1]) < pd.Timedelta(days=4): E['ON'].append((d, j0, 1))
    s = np.nan_to_num(B.s_vector.values.astype(float)); ends = B.end.values; mm_ = (pd.DatetimeIndex(ends).hour * 60 + pd.DatetimeIndex(ends).minute).values; seen = {}
    for q in range(1, len(B)):
        if s[q] == s[q - 1] or s[q] == 0 or not (585 <= mm_[q] <= 930): continue
        i = tpos5.get(np.datetime64(ends[q], 'ns'))
        if i is None: continue
        d = pd.Timestamp(dd[i])
        if d not in end_x or not ok(d) or DY.roll[d]: continue
        seen.setdefault(d, []).append((h2x[i], int(s[q])))
    E['VEC'] = [(d, *v[0]) for d, v in seen.items()]
    N = m1_dir(); N = N.drop_duplicates('day')
    for r_ in N.itertuples():
        d = pd.Timestamp(r_.day)
        if d not in end_x or DY.roll.get(d, True): continue
        ix = gi[d]; w = np.where(np.isclose(o[ix], r_.e))[0]
        if len(w): E['NOISE'].append((d, h2x[ix[w[0]]], int(r_.dir)))
    return E
if __name__ == '__main__':
    E = entries(); PER = [(2018, 2021), (2022, 2025)]
    nday = {p: sum(1 for d in days if p[0] <= d.year <= p[1] and ok(d)) for p in PER}
    print('per period: hit rate signal side / other side | pts per trade signal / other | qualifying days a month (signal side)', flush=True)
    for nm in ('LONG', 'ON', 'VEC', 'NOISE'):
        R = pd.DataFrame([(d, S, touch(j, end_x[d], dr, S)[0], touch(j, end_x[d], -dr, S)[0]) for d, j, dr in E[nm] for S in STOPS], columns=['day', 'S', 'p', 'q'])
        R['yr'] = R.day.dt.year; R['hg'] = (R.day.map(DY.g2p) >= 2 / 3).values; print(f'\n{nm} ({len(E[nm])} days)', flush=True)
        for S in STOPS:
            line = f'  stop {S:>4} (no-edge {S / (S + TG):3.0%}) '
            for a, b in PER:
                t = R[(R.S == S) & (R.yr >= a) & (R.yr <= b)]
                line += f'| {a}-{b % 100}: {np.mean(t.p >= 7.4):3.0%} / {np.mean(t.q >= 7.4):3.0%}  {t.p.mean():+5.2f} / {t.q.mean():+5.2f}  {(t.p >= 7.4).sum() / nday[(a, b)] * 21:4.1f}/mo '
            t = R[(R.S == S) & (R.yr >= 2022)]; line += f'| 22-25 gamma top third {t[t.hg].p.mean():+5.2f}, rest {t[~t.hg].p.mean():+5.2f}'
            print(line, flush=True)
