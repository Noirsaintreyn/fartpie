"""Nov 4 2026: last round for the NQ NY AM prop trade. DECLARED BEFORE RESULTS. QQQ proxy in NQ points, 1 NQ, target 75 / stop 50, 1 pt cost, up to 2 trades a day, flat 12:00.
One-month eval (rules as nq_month.py): pass rate for start dates 2022 - Aug 2026, and for start dates in 2026 only; coin-flip direction baseline (10 draws).
  BASE   15m QQQ vector flip, bar closing 09:30-11:30, any day (the best one-month version so far)
  VIXD   BASE, only trades against the DAILY VIX vector (VIX daily bear -> longs only, VIX bull -> shorts only)        VXND  same with VXN
  VIX1H  trigger = a 1h ^VIX vector flip (bar closing 10:30 / 11:30), trade NQ the OPPOSITE way (Yahoo 1h: only the last ~2 years)   VXN1H  same with ^VXN
  EMA    trigger = 15m QQQ EMA 9 / 21 cross (bar closing 09:30-11:30), either direction                                 EMAF  BASE, longs only above the 15m EMA 50 / shorts only below"""
_s = open('xs/nq_month.py').read(); exec(_s[:_s.index("SETS = {")])
def evalw(R, usd, cost, a, b, n=21, target=3000, mll=2000, dll=1000):
    v = (R.assign(p=(R.pts - cost) * usd).groupby('day').p.sum()).reindex(days).fillna(0.0); idx = v.index; v = v.values; res = {'pass': 0, 'fail': 0, 'expire': 0}
    for s in range(len(v) - n):
        if not (pd.Timestamp(a) <= idx[s] < pd.Timestamp(b)): continue
        bal, peak, floor, best, out = 0.0, 0.0, -mll, 0.0, 'expire'
        for k in range(s, s + n):
            day = max(v[k], -dll); bal += day; best = max(best, day)
            if bal <= floor: out = 'fail'; break
            if bal >= target and best <= .5 * bal: out = 'pass'; break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        res[out] += 1
    tot = max(sum(res.values()), 1); return res['pass'] / tot, tot
def dvec(sym):
    D = yf.download(sym, start='2016-01-01', interval='1d', progress=False, auto_adjust=False); D.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in D.columns]
    D = D[['open', 'high', 'low', 'close']].dropna(); D.index = D.index.normalize(); return pd.Series(np.asarray(vec(D)), index=D.index)
def at_prior(s, dd): return s.reindex(pd.DatetimeIndex(dd) - pd.Timedelta(days=1), method='ffill').values
def h1_trig(sym):
    Y = yf.download(sym, period='730d', interval='1h', progress=False, auto_adjust=False)
    if not len(Y): return pd.DataFrame(columns=['day', 'j', 'dir', 'k'])
    Y.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in Y.columns]; Y = Y.reset_index(); Y['dt'] = pd.to_datetime(Y.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    Y = Y[['dt', 'open', 'high', 'low', 'close']].dropna().reset_index(drop=True); r = np.asarray(vec(Y)); out = []
    for k in range(2, len(Y)):
        if r[k] == r[k - 1] or r[k - 1] == 0: continue
        end = Y.dt.iloc[k] + pd.Timedelta('1h'); me = end.hour * 60 + end.minute
        if not (600 < me <= 690): continue
        j = np.searchsorted(t5, np.datetime64(end))
        if j < len(H5) and d5[j] == np.datetime64(end.normalize()): out.append((end.normalize(), int(j), -int(r[k])))
    print(f'  {sym} 1h bars {len(Y)} ({Y.dt.min():%Y-%m-%d} - {Y.dt.max():%Y-%m-%d}), NY AM flips {len(out)}', flush=True)
    if not out: return pd.DataFrame(columns=['day', 'j', 'dir', 'k'])
    T = pd.DataFrame(out, columns=['day', 'j', 'dir']); T['k'] = NQs.reindex(pd.DatetimeIndex(T.day) - pd.Timedelta(days=1), method='ffill').values / o5[T.j.values]; return T
ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().values
cl = H15.close.values; e9, e21, e50 = ema(cl, 9), ema(cl, 21), ema(cl, 50); EM = []
for k in range(1, len(H15)):
    s0, s1 = np.sign(e9[k - 1] - e21[k - 1]), np.sign(e9[k] - e21[k])
    if s0 == s1 or s0 == 0 or not (570 < H15.me.iloc[k] <= 690): continue
    j = np.searchsorted(t5, np.datetime64(H15.end.iloc[k]))
    if j < len(H5) and d5[j] == np.datetime64(H15.dt.iloc[k].normalize()): EM.append((H15.dt.iloc[k].normalize(), int(j), int(s1)))
EM = pd.DataFrame(EM, columns=['day', 'j', 'dir']); EM['k'] = NQs.reindex(pd.DatetimeIndex(EM.day) - pd.Timedelta(days=1), method='ffill').values / o5[EM.j.values]
e50_at = pd.Series(e50, index=pd.DatetimeIndex(H15.end)); AL['e50'] = e50_at.reindex(pd.DatetimeIndex(t5[AL.j.values]), method='ffill').values; AL['px'] = o5[AL.j.values]
vixd, vxnd = dvec('^VIX'), dvec('^VXN')
SETS = {'BASE (15m flip, any day)': AL, 'VIXD (against the daily VIX vector)': AL[at_prior(vixd, AL.day) == -AL.dir.values],
        'VXND (against the daily VXN vector)': AL[at_prior(vxnd, AL.day) == -AL.dir.values], 'VIX1H trigger (opposite)': h1_trig('^VIX'), 'VXN1H trigger (opposite)': h1_trig('^VXN'),
        'EMA 9/21 cross trigger': EM, 'EMAF (BASE with EMA50 side)': AL[np.sign(AL.px - AL.e50) == AL.dir]}
rng = np.random.default_rng(41)
print('one-month eval, 1 NQ 75/50, up to 2 a day | pass rate for starts 2022-26 and 2026 only, coin flip in brackets', flush=True)
for nm, Cc in SETS.items():
    if not len(Cc): print(f'  {nm:<38} no data'); continue
    R = run_n(Cc, 75, 50, 2); post = R[R.day >= P0]; p1, n1 = evalw(R, 20, 1.0, '2022-01-01', '2027-01-01'); p2, n2 = evalw(R, 20, 1.0, '2026-01-01', '2027-01-01'); cf1, cf2 = [], []
    for _ in range(10):
        Z = Cc.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); Rz = run_n(Z, 75, 50, 2); cf1.append(evalw(Rz, 20, 1.0, '2022-01-01', '2027-01-01')[0]); cf2.append(evalw(Rz, 20, 1.0, '2026-01-01', '2027-01-01')[0])
    y26 = R[R.day >= '2026-01-01']; mo = monthly(R, 20, 1.0)
    print(f'  {nm:<38} trades 2022+ {len(post):>4} (first {R.day.min():%Y-%m}) win {post.pts.gt(0).mean():.0%} avg {post.pts.mean() - 1:+5.1f} | pass 2022-26 {p1:.0%} ({np.median(cf1):.0%}) | pass 2026 {p2:.0%} ({np.median(cf2):.0%}), 2026 trades {len(y26)} avg {y26.pts.mean() - 1:+.1f} | month $ mean {mo.mean():+,.0f} up {mo.gt(0).mean():.0%}', flush=True)
