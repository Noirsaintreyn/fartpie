"""Nov 8 2026: the rest of Rey's NQ ideas, on every RTH 15m flip 2018 - Oct 2026 held to the 15m vector exit (NQ points per flip). DECLARED BEFORE RESULTS.
  ACCEPT   wait one bar: enter only if the next 15m bar closes beyond the flip bar's close in the trade's direction (accepted), at the open after it; vs skip (fake-out)
  PIVOTS   breakout through a cluster: number of 15m / 1h / daily swing pivots (2 bars each side, last 20 sessions) the flip bar's range took out on the breakout side
  CORRZ    QQQ vs VXN 20-day return correlation, z-scored over a year (prior close)
  RSI      15m RSI(14) at the flip, signed with the trade (high = buying into overbought / selling into oversold)
  RSIDIV   divergence: the flip bar makes a 20-bar extreme in the trade's direction while RSI is below its level at the previous such extreme
  STOCH    15m stochastic RSI(14) signed
  OVN      the overnight session's direction (Globex close-to-close is not in the file, so the 04:00-09:30 premarket move of QQQ) agrees with the flip
Each: buckets with points per flip by period 18-21 / 22-23 / 24-25 / 2026 and the coin-flip-free spread check (same sign in all four periods)."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
X = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features.parquet')
PER = ['18-21', '22-23', '24-25', '2026']
bi = pd.Series(np.arange(len(B)), index=B.end.values); kb = np.array([bi.get(t5[j], -1) for j in X.j.values]); X = X[kb >= 0].copy(); X['kb'] = kb[kb >= 0]
c = B.close; d_ = c.diff(); up = d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean(); rsi = 100 - 100 / (1 + up / dn)
srsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
X['rsi'] = np.where(X.dir > 0, rsi.values[X.kb], 100 - rsi.values[X.kb]); X['stoch'] = np.where(X.dir > 0, srsi.values[X.kb], 1 - srsi.values[X.kb])
hh, ll = B.high.rolling(20).max(), B.low.rolling(20).min(); div = []
for k, dr in zip(X.kb.values, X.dir.values):
    if k < 40: div.append(np.nan); continue
    if dr > 0 and B.high.values[k] >= hh.values[k - 1]:
        prev = np.where(B.high.values[k - 40:k - 1] >= hh.values[k - 41:k - 2])[0]; div.append(float(rsi.values[k] < rsi.values[k - 40 + prev[-1]]) if len(prev) else np.nan)
    elif dr < 0 and B.low.values[k] <= ll.values[k - 1]:
        prev = np.where(B.low.values[k - 40:k - 1] <= ll.values[k - 41:k - 2])[0]; div.append(float(rsi.values[k] > rsi.values[k - 40 + prev[-1]]) if len(prev) else np.nan)
    else: div.append(np.nan)
X['rsidiv'] = div
def swings(df, w=2):
    h, l = df.high.values, df.low.values; sh = np.zeros(len(df), bool); sl = np.zeros(len(df), bool)
    for i in range(w, len(df) - w):
        sh[i] = h[i] == h[i - w:i + w + 1].max(); sl[i] = l[i] == l[i - w:i + w + 1].min()
    return pd.DataFrame({'t': df.end.values if 'end' in df else df.index.values, 'h': h, 'l': l, 'sh': sh, 'sl': sl})
H1 = B.set_index('dt')[['open', 'high', 'low', 'close']].resample('1h', offset='30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna(); H1['end'] = H1.index + pd.Timedelta('1h')
DY = H.groupby('day').agg(high=('high', 'max'), low=('low', 'min')); DY['end'] = DY.index + pd.Timedelta('16h')
SW = [swings(B), swings(H1.reset_index(drop=True)), swings(DY.reset_index(drop=True))]
piv = []
for k, dr, day in zip(X.kb.values, X.dir.values, X.day.values):
    t = B.end.values[k]; t0 = t - np.timedelta64(30, 'D'); lo, hi = B.low.values[k], B.high.values[k]; n = 0
    for S in SW:
        m = (S.t.values < t - np.timedelta64(45, 'm')) & (S.t.values >= t0)                     # pivot confirmed (2 bars later) well before the flip bar
        lv = S.h.values[m & S.sh.values] if dr > 0 else S.l.values[m & S.sl.values]
        n += int(((lv >= lo) & (lv <= hi)).sum())
    piv.append(n)
X['pivots'] = piv
VXq = yf.download(['QQQ', '^VXN'], start='2016-06-01', interval='1d', progress=False, auto_adjust=True)['Close'].ffill(); rr = np.log(VXq).diff()
cz = rr['QQQ'].rolling(20).corr(rr['^VXN']); czz = ((cz - cz.rolling(252, min_periods=120).mean()) / cz.rolling(252, min_periods=120).std()).shift(1); czz.index = czz.index.normalize()
X['corrz'] = czz.reindex(pd.DatetimeIndex(X.day), method='ffill').values
Hall = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_5m_*.parquet')[0]); Hall['day'] = Hall.dt.dt.normalize()
pre = Hall[Hall.m < 570].groupby('day').close.last(); opn = Hall[Hall.m >= 570].groupby('day').open.first(); pcl = Hall[Hall.m < 960].groupby('day').close.last().shift(1)
ovn = np.sign(opn - pcl.reindex(opn.index)); X['ovn'] = np.where(np.sign(ovn.reindex(pd.DatetimeIndex(X.day)).values) == X.dir.values, 'agrees', 'disagrees')
acc_pts, acc_ok = [], []
for j, d, k in zip(X.j.values, X.dir.values, X.kb.values):
    if k + 1 >= len(B) or B.dt.values[k + 1].astype('datetime64[D]') != B.dt.values[k].astype('datetime64[D]'): acc_ok.append(np.nan); acc_pts.append(np.nan); continue
    ok = (B.close.values[k + 1] - B.close.values[k]) * d > 0; acc_ok.append(float(ok))
    j2 = pos5.get(B.end.values[k + 1]); res = sim(int(j2), int(d), None, 50, 955, True) if (ok and j2 is not None) else None
    acc_pts.append(res[0] - 1.0 if res else np.nan)
X['accept'] = acc_ok; X['th_late'] = acc_pts
print(f'{len(X)} flips | trend-hold per flip {X.th.mean():+.2f}', flush=True)
a = X[X.accept == 1]; f = X[X.accept == 0]
print(f'\nACCEPT  next bar holds beyond the flip close: {len(a)} flips | immediate entry {a.th.mean():+.1f}, entry after the confirm bar {a.th_late.mean():+.1f} per flip | fake-outs {len(f)}: immediate entry {f.th.mean():+.1f}', flush=True)
for nm, g in (('accepted, immediate', a.th), ('accepted, late entry', a.th_late), ('fake-out, immediate', f.th)):
    print(f'    {nm:<22} ' + ' / '.join(f'{g[X.loc[g.index].per == p].mean():+.1f}' for p in PER), flush=True)
def table(name, lab):
    out = []
    for v in sorted(lab.dropna().unique(), key=str):
        g = X[lab == v]; out.append((v, len(g), g.th.mean(), [g[g.per == p].th.mean() for p in PER]))
    best, worst = max(out, key=lambda o: o[2]), min(out, key=lambda o: o[2])
    cons = all(np.sign(b - w) == np.sign(best[2] - worst[2]) for b, w in zip(best[3], worst[3]) if np.isfinite(b) and np.isfinite(w))
    print(f'  -- {name}  spread {best[2] - worst[2]:+.1f} (best {best[0]}, worst {worst[0]}) {"* every period" if cons else ""}', flush=True)
    for v, n, m, pp in out: print(f'    {str(v):<12} n{n:>4} TH {m:+6.1f} (' + ' / '.join(f'{x:+.1f}' for x in pp) + ')', flush=True)
T3 = lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low', 'mid', 'high']).astype(str).where(s.notna())
print('', flush=True)
table('PIVOTS taken out (0 / 1-2 / 3+)', pd.Series(np.select([X.pivots == 0, X.pivots <= 2], ['0', '1-2'], '3+'), index=X.index))
table('CORRZ', T3(X.corrz)); table('RSI signed', T3(X.rsi)); table('STOCH RSI signed', T3(X.stoch))
table('RSI divergence at a new extreme', X.rsidiv.map({1.0: 'divergence', 0.0: 'no divergence'})); table('OVN premarket agrees', X.ovn)
