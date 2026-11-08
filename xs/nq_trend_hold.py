"""Nov 5 2026: the TREND-HOLD version (every RTH 15m flip 09:45-15:30, hold until the vector flips back, stop 50, flat 15:55, <= 4 a day; +3.07 pts/day in nq_points_day)
with volatility / VWAP / cold-streak sizing and a volatility stop. DECLARED BEFORE RESULTS. NQ points per contract after a 1 pt cost; size is a fraction of 1 contract.
  D0  base (should reproduce +3.07 / day)
  D1  x0.5 when Yang-Zhang 20d vol is in the top third of its year (prior close) - the prop system's rule
  D1r x0.5 when YZ vol is in the BOTTOM third instead (trend-holding may prefer high vol; shown as the opposite case, not picked after the fact)
  D2  x0.5 when the entry is stretched past session VWAP (distance in 15m ATR above the 2018-21 top-third cut)
  D3  x0.5 while the last 20 trades (1 lot) are net negative
  D4  D1 + D2 ladder (1 / 0.5 / 0.3 by the count) x D3 - the prop system's full sizing on this signal
  D5  volatility stop: 2 x 15m ATR in NQ points, kept between 30 and 80, instead of 50
  D6  D4 + D5
Per day: mean (flat days = 0), share >= +7.5, share < 0; per month: mean, share >= 50, worst; max drawdown of the running total; by year; coin-flip on D0 / D4 / D6."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
sys.path.insert(0, '/home/claude/fartpie/tools'); import account_signals as AS
q_, y_ = AS.data('2016-06-01'); yz_, yzp_, *_ = AS.series(q_, y_)
yzd = pd.Series(yzp_.reindex(pd.DatetimeIndex(d5) - pd.Timedelta(days=1), method='ffill').values, index=np.arange(len(H)))   # percentile at the prior close
tpv = (H.high + H.low + H.close) / 3 * H.volume; vw5 = (tpv.groupby(H.day).cumsum() / H.volume.groupby(H.day).cumsum()).values
pc = B.close.shift(1); atr15 = pd.concat([B.high - B.low, (B.high - pc).abs(), (B.low - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
atr_at = pd.Series(atr15.values, index=B.end.values)
FL['atr'] = [atr_at.get(t5[j]) for j in FL.j.values]; FL['atrpts'] = FL.atr * k5[FL.j.values]
FL['vwd'] = (o5[FL.j.values] - vw5[np.maximum(FL.j.values - 1, 0)]) * FL.dir / FL.atr
FL['yzp'] = yzd.reindex(FL.j.values).values
cut_vw = FL[FL.day < '2022-01-01'].vwd.quantile(2 / 3)
def simv(j, d, S, flat):
    e = o5[j]; kk = k5[j]
    for q in range(j, len(H)):
        if d5[q] != d5[j] or m5[q] >= flat: return (c5[q - 1] - e) * d * kk, q - 1
        if (((l5[q] if d > 0 else h5[q]) - e) * d * kk) <= -S: return -S, q
        if m5[q] % 15 == 10:
            st = reg15.get(t5[q] + np.timedelta64(5, 'm'))
            if st is not None and st == -d and q + 1 < len(H) and d5[q + 1] == d5[j]: return (o5[q + 1] - e) * d * kk, q
    return None
def runs(X, size_fn, volstop=False):
    rows = []; hist = []
    for day, g in X.groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            if n >= 4 or r.j <= busy: continue
            S = float(np.clip(2 * r.atrpts, 30, 80)) if volstop and np.isfinite(r.atrpts) else 50.0
            res = simv(int(r.j), int(r.dir), S, 955)
            if res is None: continue
            net = res[0] - 1.0; cold = len(hist) >= 20 and sum(hist[-20:]) < 0
            rows.append((day, net, size_fn(r, cold))); hist.append(net); busy = res[1]; n += 1
    T = pd.DataFrame(rows, columns=['day', 'net', 'size']); T['pts'] = T.net * T['size']; return T
ALLD = pd.DatetimeIndex(sorted(H.day.unique()))
def stats(T):
    P = T.groupby('day').pts.sum().reindex(ALLD).fillna(0.0); M = P.groupby(P.index.to_period('M')).sum(); cum = P.cumsum()
    per = np.select([P.index < '2022-01-01', P.index < '2024-01-01', P.index < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
    return dict(day=P.mean(), ge=(P >= 7.5).mean(), neg=(P < 0).mean(), mo=M.mean(), mo50=(M >= 50).mean(), worst=M.min(), dd=(cum - cum.cummax()).min(), sh=P.mean() / P.std() * np.sqrt(252),
                per=[P[per == p].mean() for p in ('18-21', '22-23', '24-25', '2026')], yr=P.groupby(P.index.year).sum(), size=T['size'].mean())
one = lambda r, cold: 1.0
yzh = lambda r: np.isfinite(r.yzp) and r.yzp >= 2 / 3
vwx = lambda r: np.isfinite(r.vwd) and r.vwd > cut_vw
ladder = lambda r, cold: (1.0, .5, .3)[min(int(yzh(r)) + int(vwx(r)), 2)] * (.5 if cold else 1.0)
V = {'D0 base': (one, False), 'D1 x0.5 high vol': (lambda r, c: .5 if yzh(r) else 1.0, False), 'D1r x0.5 low vol': (lambda r, c: .5 if (np.isfinite(r.yzp) and r.yzp <= 1 / 3) else 1.0, False),
     'D2 x0.5 VWAP stretched': (lambda r, c: .5 if vwx(r) else 1.0, False), 'D3 x0.5 cold streak': (lambda r, c: .5 if c else 1.0, False), 'D4 full sizing': (ladder, False),
     'D5 vol stop': (one, True), 'D6 full sizing + vol stop': (ladder, True)}
rng = np.random.default_rng(9)
print(f'flips {len(FL)} | VWAP-stretch cut {cut_vw:.2f} ATR | 15m ATR median {FL.atrpts.median():.0f} NQ pts -> vol stop median {np.clip(2 * FL.atrpts, 30, 80).median():.0f}', flush=True)
print(f'\n{"NQ points per contract":<28} {"pts/day":>8} {"avg size":>8} {"days>=7.5":>9} {"days<0":>7} {"pts/mo":>7} {"mo>=50":>7} {"worst mo":>9} {"max DD":>7} {"Sharpe":>6} | per day 18-21 / 22-23 / 24-25 / 2026 | coin-flip pts/day', flush=True)
for nm, (fn, vs) in V.items():
    s = stats(runs(FL, fn, vs)); cf = ''
    if nm.split()[0] in ('D0', 'D4', 'D6'): cf = f'{np.median([stats(runs(FL.assign(dir=rng.choice([-1, 1], len(FL))), fn, vs))["day"] for _ in range(8)]):+.2f}'
    print(f'  {nm:<26} {s["day"]:>+8.2f} {s["size"]:>8.2f} {s["ge"]:>9.0%} {s["neg"]:>7.0%} {s["mo"]:>+7.1f} {s["mo50"]:>7.0%} {s["worst"]:>+9.0f} {s["dd"]:>+7.0f} {s["sh"]:>6.2f} | ' + ' / '.join(f'{v:+.2f}' for v in s['per']) + f' | {cf}', flush=True)
    print(f'      by year: ' + ' '.join(f'{y} {v:+.0f}' for y, v in s['yr'].items()), flush=True)
