"""Nov 8 2026: RATE EVERY NQ FLIP BY ITS CONTEXT. DECLARED BEFORE RESULTS. Every RTH 15m flip 09:45-15:30, 2018 - Oct 2026 (QQQ 5m -> NQ points, 1 pt cost),
each simulated on its own (no daily cap): TH = hold until the vector flips back, stop 50, flat 15:55;  F75 = 75 / 50, flat 15:55.
Context, all known at the entry (daily items at the prior close):
  time      signal minute: 09:45-10:15 / 10:15-11:30 / 11:30-14:00 lunch / 14:00-15:30
  vol       Yang-Zhang 1-day / 5-day / 20-day and 5m realized variance 1 / 5 / 20 days, each as a 1-year percentile; RV1 / RV20 (shock vs grind); VIX, VXN 1-year percentile
  chop      15m ATR at the flip / its 20-day median (compression); today's range so far / 20-day average daily range
  breakout  flip bar range / 15m ATR (size of the break); where the flip bar closed in its range, in the trade's direction (acceptance); body / range
  volume    flip bar volume / the same 15m slot's 20-day average (RVOL)
  vwap      entry distance from session VWAP in 15m ATR, signed with the trade
  drift     today's open-to-entry move / 20-day ADR, signed; gap signed; prior day's return signed
  z         QQQ z-score of the prior close vs 5 / 21 / 63 / 252 days, signed with the trade (+ = trading with the stretch)
  room      distance to the nearest prior-day / prior-week / 20-day high-low level ahead of the trade, in daily ATR
  ruleA     the flip is a Rule A trade (AM only)
Output: per feature, terciles (or the named buckets) with TH and F75 points per flip in 2018-21 / 22-23 / 24-25 / 2026; a feature is CONSISTENT when its best-minus-worst
bucket has the same sign in all four periods. Saves the table to /tmp/claude-0/ml/nq_flip_features.parquet for the sizing models."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
os.makedirs('/tmp/claude-0/ml', exist_ok=True)
# ---- daily RTH bars from the 5m file
D = H.groupby('day').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), v=('volume', 'sum'))
lr = np.log(H.close).diff(); lr[H.day != H.day.shift(1)] = np.nan; RV = (lr ** 2).groupby(H.day).sum()
ov = np.log(D.o / D.c.shift(1)); oc = np.log(D.c / D.o); rs = np.log(D.h / D.c) * np.log(D.h / D.o) + np.log(D.l / D.c) * np.log(D.l / D.o)
def yzn(n):
    if n == 1: return ov ** 2 + rs
    k = .34 / (1.34 + (n + 1) / (n - 1)); return ov.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
DF_ = pd.DataFrame({'yz1': pct(yzn(1)), 'yz5': pct(yzn(5)), 'yz20': pct(yzn(20)), 'rv1': pct(RV), 'rv5': pct(RV.rolling(5).mean()), 'rv20': pct(RV.rolling(20).mean()),
                    'shock': RV / RV.rolling(20).mean(), 'adr': (D.h - D.l).rolling(20).mean(), 'prev_ret': np.log(D.c / D.c.shift(1)),
                    'z5': (D.c - D.c.rolling(5).mean()) / D.c.rolling(5).std(), 'z21': (D.c - D.c.rolling(21).mean()) / D.c.rolling(21).std(),
                    'z63': (D.c - D.c.rolling(63).mean()) / D.c.rolling(63).std(), 'z252': (D.c - D.c.rolling(252).mean()) / D.c.rolling(252).std(),
                    'pdh': D.h, 'pdl': D.l, 'h20': D.h.rolling(20).max(), 'l20': D.l.rolling(20).min(), 'pc': D.c})
wk = D.index.to_period('W-FRI'); W = D.groupby(wk).agg(wh=('h', 'max'), wl=('l', 'min')); W.index = W.index.to_timestamp('D', 'end').normalize()
DF_['pwh'] = W.wh.shift(1).reindex(D.index, method='ffill'); DF_['pwl'] = W.wl.shift(1).reindex(D.index, method='ffill')
PRIOR = DF_.shift(1)                                                                              # everything known at the prior close
VX = yf.download(['^VIX', '^VXN'], start='2016-06-01', interval='1d', progress=False, auto_adjust=True)['Close'].ffill(); VX.index = VX.index.normalize()
PRIOR['vix'] = pct(VX['^VIX']).shift(1).reindex(PRIOR.index, method='ffill'); PRIOR['vxn'] = pct(VX['^VXN']).shift(1).reindex(PRIOR.index, method='ffill')
# ---- intraday pieces
tpv = (H.high + H.low + H.close) / 3 * H.volume; vw5 = (tpv.groupby(H.day).cumsum() / H.volume.groupby(H.day).cumsum()).values
pc_ = B.close.shift(1); B['atr'] = pd.concat([B.high - B.low, (B.high - pc_).abs(), (B.low - pc_).abs()], axis=1).max(axis=1).rolling(14).mean()
B['atr_med'] = B.atr.rolling(26 * 20, min_periods=26 * 5).median()
Bv = H.set_index('dt').volume.groupby(pd.Grouper(freq='15min')).sum(); B['vol'] = Bv.reindex(B.dt).values
B['slot'] = B.dt.dt.hour * 60 + B.dt.dt.minute; B['day'] = B.dt.dt.normalize()
B['rvol'] = B.vol / B.groupby('slot').vol.transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
bi = pd.Series(np.arange(len(B)), index=B.end.values)
dayo = H.groupby('day').open.first(); hi_so = H.groupby('day').high.cummax().values; lo_so = H.groupby('day').low.cummin().values
LA = pd.read_csv('live_data/nq_rule_a_trades.csv', parse_dates=['day', 'entry']); RAK = set(zip(LA.entry.values.astype('datetime64[ns]'), LA.dir.values))
rows = []
for f in FL.itertuples():
    j, d = int(f.j), int(f.dir); k = int(bi[t5[j]]) if t5[j] in bi.index else None
    if k is None: continue
    day = pd.Timestamp(d5[j]); p = PRIOR.loc[day] if day in PRIOR.index else None
    if p is None or not np.isfinite(B.atr.values[k]): continue
    th = sim(j, d, None, 50, 955, True); f75 = sim(j, d, 75, 50, 955, False)
    if th is None or f75 is None: continue
    e = o5[j]; atr = B.atr.values[k]; rng_ = B.high.values[k] - B.low.values[k]
    clv = ((B.close.values[k] - B.low.values[k]) / rng_ if rng_ > 0 else .5); clv = clv if d > 0 else 1 - clv
    lv = [p.pdh, p.pdl, p.pwh, p.pwl, p.h20, p.l20]; ahead = [x for x in lv if np.isfinite(x) and (x - e) * d > 0]
    room = min(abs(x - e) for x in ahead) / p.adr if ahead and p.adr > 0 else 3.0
    so_far = (hi_so[j - 1] - lo_so[j - 1]) if j > 0 and d5[j - 1] == d5[j] else 0.0
    rows.append(dict(day=day, j=j, dir=d, me=int(f.me), ruleA=int((t5[j], d) in RAK), th=th[0] - 1.0, f75=f75[0] - 1.0,
                     yz1=p.yz1, yz5=p.yz5, yz20=p.yz20, rv1=p.rv1, rv5=p.rv5, rv20=p.rv20, shock=p.shock, vix=p.vix, vxn=p.vxn,
                     comp=atr / B.atr_med.values[k] if B.atr_med.values[k] > 0 else np.nan, used=so_far / p.adr if p.adr > 0 else np.nan,
                     brk=rng_ / atr, acc=clv, body=abs(B.close.values[k] - B.open.values[k]) / rng_ if rng_ > 0 else 0, rvol=B.rvol.values[k],
                     vwd=(e - vw5[j - 1]) * d / atr, drift=(e - dayo.loc[day]) * d / p.adr if p.adr > 0 else np.nan, gap=(dayo.loc[day] / p.pc - 1) * d * 100,
                     prev=p.prev_ret * d * 100, z5=p.z5 * d, z21=p.z21 * d, z63=p.z63 * d, z252=p.z252 * d, room=room, dow=day.dayofweek))
X = pd.DataFrame(rows); X['per'] = np.select([X.day < '2022-01-01', X.day < '2024-01-01', X.day < '2026-01-01'], ['18-21', '22-23', '24-25'], '2026')
X['tod'] = np.select([X.me <= 615, X.me <= 690, X.me <= 840], ['09:45-10:15', '10:15-11:30', '11:30-14:00'], '14:00-15:30')
X.to_parquet('/tmp/claude-0/ml/nq_flip_features.parquet')
PER = ['18-21', '22-23', '24-25', '2026']
print(f'{len(X)} flips | TH {X.th.mean():+.2f} pts/flip, F75 {X.f75.mean():+.2f} | per period TH ' + ' / '.join(f'{X[X.per == p].th.mean():+.1f}' for p in PER), flush=True)
RES = []
def table(name, lab):
    out = []
    for v in sorted(lab.dropna().unique(), key=str):
        g = X[lab == v]; out.append((v, len(g), g.th.mean(), [g[g.per == p].th.mean() for p in PER], g.f75.mean(), [g[g.per == p].f75.mean() for p in PER]))
    best, worst = max(out, key=lambda o: o[2]), min(out, key=lambda o: o[2])
    cons_th = all(np.sign(b - w) == np.sign(best[2] - worst[2]) for b, w in zip(best[3], worst[3]) if np.isfinite(b) and np.isfinite(w))
    bf, wf = max(out, key=lambda o: o[4]), min(out, key=lambda o: o[4])
    cons_f = all(np.sign(b - w) == np.sign(bf[4] - wf[4]) for b, w in zip(bf[5], wf[5]) if np.isfinite(b) and np.isfinite(w))
    RES.append((name, best[2] - worst[2], cons_th, str(best[0]), str(worst[0]), bf[4] - wf[4], cons_f))
    print(f'  -- {name}', flush=True)
    for v, n, m, pp, m75, pp75 in out:
        print(f'    {str(v):<14} n{n:>4} | TH {m:+6.1f} (' + ' / '.join(f'{x:+.1f}' for x in pp) + f') | F75 {m75:+6.1f} (' + ' / '.join(f'{x:+.1f}' for x in pp75) + ')', flush=True)
T3 = lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low', 'mid', 'high']).astype(str).where(s.notna())
table('time of day', X.tod); table('Rule A (AM flips only)', X.ruleA.where(X.me <= 690).map({1: 'Rule A', 0: 'not Rule A'})); table('day of week', X.dow.map(dict(enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']))))
for c in ['yz1', 'yz5', 'yz20', 'rv1', 'rv5', 'rv20', 'shock', 'vix', 'vxn', 'comp', 'used', 'brk', 'acc', 'body', 'rvol', 'vwd', 'drift', 'gap', 'prev', 'z5', 'z21', 'z63', 'z252', 'room']:
    table(c, T3(X[c]))
print('\nRANKED by trend-hold spread (best - worst bucket, pts per flip); * = same direction in all four periods', flush=True)
for name, sp, ct, b, w, sp75, cf in sorted(RES, key=lambda r: (-int(r[2]), -abs(r[1]))):
    print(f'  {"*" if ct else " "} {name:<24} TH spread {sp:+6.1f} (best {b}, worst {w}) | F75 spread {sp75:+6.1f} {"*" if cf else ""}', flush=True)
