"""Nov 8 2026: EXITS for the trend-hold version. DECLARED BEFORE RESULTS. Every RTH 15m flip 09:45-15:30 (<= 4 a day, no overlap), and the Rule A subset; NQ points, 1 contract.
  X15    hold until the 15m vector flips back (stop 50, flat 15:55) - the current trend-hold
  X5     exit when the 5m vector (built on 5m RTH bars) flips against the trade, at the next 5m open (stop 50, flat 15:55)
  XRUN   X15, plus a volatility runner stop: once the trade is +1 x 15m ATR, trail at the best price - 2 x 15m ATR (checked on 5m bars)
  XVS    X15 with a volatility stop: 2 x 15m ATR, kept between 30 and 80 pts, instead of 50
  XCUT   X15, but in high volatility (15m ATR above its 20-day median) a loser is cut at -1 x 15m ATR (take the L fast when vol is high)
  X14    X15 with no entries after 14:00
Report: points per day (all sessions), trades a day, Sharpe, worst month, max drawdown, per day by period 18-21 / 22-23 / 24-25 / 2026."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
r5 = np.asarray(vec(H[['open', 'high', 'low', 'close']]))                                   # 5m vector on RTH 5m bars (state at each 5m close)
pc_ = B.close.shift(1); atr15 = pd.concat([B.high - B.low, (B.high - pc_).abs(), (B.low - pc_).abs()], axis=1).max(axis=1).rolling(14).mean()
atr_med = atr15.rolling(26 * 20, min_periods=26 * 5).median(); A15 = pd.Series(atr15.values, index=B.end.values); AM15 = pd.Series(atr_med.values, index=B.end.values)
FL['atr'] = [A15.get(t5[j], np.nan) * k5[j] for j in FL.j.values]; FL['hiv'] = [A15.get(t5[j], np.nan) > AM15.get(t5[j], np.nan) for j in FL.j.values]
LA = pd.read_csv('live_data/nq_rule_a_trades.csv', parse_dates=['day', 'entry']); RAK = set(zip(LA.entry.values.astype('datetime64[ns]'), LA.dir.values))
FL['ruleA'] = [(t5[j], d) in RAK for j, d in zip(FL.j.values, FL.dir.values)]
def simx(j, d, mode, atr, hiv):
    e = o5[j]; kk = k5[j]; S = 50.0; best = 0.0
    if mode == 'XVS' and np.isfinite(atr): S = float(np.clip(2 * atr, 30, 80))
    if mode == 'XCUT' and hiv and np.isfinite(atr): S = min(50.0, atr)
    for q in range(j, len(H)):
        if d5[q] != d5[j] or m5[q] >= 955: return (c5[q - 1] - e) * d * kk, q - 1
        lo = ((l5[q] if d > 0 else h5[q]) - e) * d * kk; hi = ((h5[q] if d > 0 else l5[q]) - e) * d * kk
        stop = -S
        if mode == 'XRUN' and np.isfinite(atr) and best >= atr: stop = max(stop, best - 2 * atr)
        if lo <= stop: return stop, q
        best = max(best, hi)
        nxt = q + 1 < len(H) and d5[q + 1] == d5[j]
        if not nxt: continue
        if mode == 'X5':
            if r5[q] == -d: return (o5[q + 1] - e) * d * kk, q
        elif m5[q] % 15 == 10:
            st = reg15.get(t5[q] + np.timedelta64(5, 'm'))
            if st is not None and st == -d: return (o5[q + 1] - e) * d * kk, q
    return None
ALLD = pd.DatetimeIndex(sorted(H.day.unique()))
def book(X, mode):
    rows = []
    for day, g in X.groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy: continue
            res = simx(int(r.j), int(r.dir), mode, r.atr, r.hiv)
            if res is None: continue
            rows.append((day, res[0] - 1.0)); busy = res[1]; n += 1
    T = pd.DataFrame(rows, columns=['day', 'pts']); P = T.groupby('day').pts.sum().reindex(ALLD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum(); cum = P.cumsum()
    per = np.select([P.index < '2022-01-01', P.index < '2024-01-01', P.index < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
    return dict(day=P.mean(), n=len(T) / len(ALLD), sh=P.mean() / P.std() * np.sqrt(252), worst=M.min(), dd=(cum - cum.cummax()).min(), m50=(M >= 50).mean(),
                per=[P[per == p].mean() for p in ('18-21', '22-23', '24-25', '2026')])
for nm, X in (('every RTH flip', FL), ('Rule A entries', FL[FL.ruleA])):
    print(f'\n{nm} ({len(X)} signals)   pts/day trades/day Sharpe worst mo  max DD months>=50 | per day 18-21 / 22-23 / 24-25 / 2026', flush=True)
    for mode in ('X15', 'X5', 'XRUN', 'XVS', 'XCUT', 'X14'):
        Xm = X[X.me <= 840] if mode == 'X14' else X; r = book(Xm, 'X15' if mode == 'X14' else mode)
        print(f'  {mode:<6} {r["day"]:>+8.2f} {r["n"]:>10.2f} {r["sh"]:>6.2f} {r["worst"]:>+8.0f} {r["dd"]:>+7.0f} {r["m50"]:>10.0%} | ' + ' / '.join(f'{v:+.2f}' for v in r['per']), flush=True)
