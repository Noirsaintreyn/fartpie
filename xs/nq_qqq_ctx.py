"""Nov 4 2026: the context retried with QQQ instead of SPY and VXN instead of VIX, 2022 - Sep 2026 only, one-month eval. DECLARED BEFORE RESULTS.
Signal sets (from nq_month.py): S0x2 = 15m flip on negative-gamma days, up to 2 trades a day; ALLx2 = 15m flip any day, up to 2 a day. Size B: 1 NQ, 75/50 (one win ~ $1,480).
Context at the PRIOR close: QQQ vs 200d | QQQ 20d z-score (< -1.5 / normal / > 1.5) | QQQ 20d return up / down | VXN 1-year percentile terciles | VXN vs its 20d average (rising > +10% / falling)
| VXN/VIX > 1.25 (Nasdaq fear premium) | QQQ gap at 09:30 with / against the trade.
Each cell: trades, win, avg pts in 2022-23 and 2024-26 (a cell counts only if it beats the set's own average in BOTH, n >= 25 each), 1-month eval pass vs coin flip, monthly $."""
_s = open('xs/nq_month.py').read(); exec(_s[:_s.index("SETS = {")])
Y2 = yf.download(['QQQ', '^VXN', '^VIX'], start='2016-01-01', interval='1d', progress=False, auto_adjust=True)['Close'].ffill(); Y2.index = Y2.index.normalize()
q, vx = Y2['QQQ'], Y2['^VXN']
XQ = pd.DataFrame({'qbull': q > q.rolling(200).mean(), 'qz': (q - q.rolling(20).mean()) / q.rolling(20).std(), 'qr20': q / q.shift(20) - 1,
                   'vxnp': vx.rolling(252, min_periods=120).rank(pct=True), 'vxn_rel': vx / vx.rolling(20).mean() - 1, 'vxn_vix': vx / Y2['^VIX']})
def ctx(Cc):
    Cc = Cc.reset_index(drop=True); P = XQ.reindex(pd.DatetimeIndex(Cc.day) - pd.Timedelta(days=1), method='ffill').reset_index(drop=True); Cc = pd.concat([Cc, P], axis=1)
    pc = np.array([c5[np.searchsorted(t5, np.datetime64(d)) - 1] for d in Cc.day]); op = np.array([o5[np.searchsorted(t5, np.datetime64(d + pd.Timedelta(hours=9, minutes=30)))] for d in Cc.day])
    Cc['QQQ200'] = np.where(Cc.qbull, 'QQQ > 200d', 'QQQ < 200d'); Cc['QZ20'] = pd.cut(Cc.qz, [-99, -1.5, 1.5, 99], labels=['QQQ stretched down', 'QQQ normal', 'QQQ stretched up']).astype(str)
    Cc['QR20'] = np.where(Cc.qr20 < 0, 'QQQ 20d down', 'QQQ 20d up'); Cc['VXNP'] = pd.cut(Cc.vxnp, [-1, 1 / 3, 2 / 3, 2], labels=['VXN low', 'VXN mid', 'VXN high']).astype(str)
    Cc['VXNTR'] = np.where(Cc.vxn_rel > .10, 'VXN rising >10%', np.where(Cc.vxn_rel < -.10, 'VXN falling >10%', 'VXN flat'))
    Cc['VXNVIX'] = np.where(Cc.vxn_vix > 1.25, 'VXN/VIX > 1.25', 'VXN/VIX <= 1.25'); Cc['GAP'] = np.where(np.sign(op - pc) == Cc.dir, 'with the gap', 'against the gap'); return Cc
rng = np.random.default_rng(31); lab, usd, T, S, cost = 'B 1 NQ 75/50', 20, 75, 50, 1.0
for sn, (Cc, pdn) in (('S0x2 (neg gamma, 2 a day)', (AL[AL.neg], 2)), ('ALLx2 (any day, 2 a day)', (AL, 2))):
    Cc = ctx(Cc[Cc.day >= P0]); R = run_n(Cc, T, S, pdn).reset_index(drop=True)
    R = R.merge(Cc.drop_duplicates('day')[['day', 'QQQ200', 'QZ20', 'QR20', 'VXNP', 'VXNTR', 'VXNVIX']], on='day', how='left')
    R['h'] = np.where(R.day < '2024-01-01', '22-23', '24-26'); base = {h: (R[R.h == h].pts - cost).mean() for h in ('22-23', '24-26')}
    e0 = evalm(R, usd, cost); mo0 = monthly(R, usd, cost)
    print(f'\n== {sn}, {lab}: trades {len(R)} avg {R.pts.mean() - cost:+.1f} pts (22-23 {base["22-23"]:+.1f} / 24-26 {base["24-26"]:+.1f}) | 1-month pass {e0["pass"]:.0%} | month $ mean {mo0.mean():+,.0f} up {mo0.gt(0).mean():.0%}', flush=True)
    keep = []
    for col in ('QQQ200', 'QZ20', 'QR20', 'VXNP', 'VXNTR', 'VXNVIX'):
        for v, g in R.groupby(col):
            a, b = g[g.h == '22-23'], g[g.h == '24-26']; ok = len(a) >= 25 and len(b) >= 25 and (a.pts - cost).mean() > base['22-23'] and (b.pts - cost).mean() > base['24-26']
            print(f'  {col:<7} {v:<20} n{len(g):>4} win {g.pts.gt(0).mean():.0%} avg {g.pts.mean() - cost:+6.1f} | 22-23 n{len(a):>3} {(a.pts - cost).mean():+6.1f} / 24-26 n{len(b):>3} {(b.pts - cost).mean():+6.1f}' + ('   <- both' if ok else ''), flush=True)
            if ok: keep.append((col, v))
    for col, v in keep:
        days_ok = set(R[R[col] == v].day); sub = Cc[Cc.day.isin(days_ok)]; Rs = run_n(sub, T, S, pdn); e = evalm(Rs, usd, cost); mo = monthly(Rs, usd, cost); cf = []
        for _ in range(10): Z = sub.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); cf.append(evalm(run_n(Z, T, S, pdn), usd, cost)['pass'])
        print(f'    {col}={v}: trades {len(Rs)} ({len(Rs) / len(mo):.1f}/month) | 1-month pass {e["pass"]:.0%} fail {e["fail"]:.0%} expire {e["expire"]:.0%} (coin-flip {np.median(cf):.0%}) | month $ mean {mo.mean():+,.0f} median {mo.median():+,.0f} up {mo.gt(0).mean():.0%} worst {mo.min():+,.0f}', flush=True)
# ---------------------------------------------------------------- volatility models (asked Nov 4): HAR, Garman-Klass, GJR-GARCH. Do they find better moves, or only bigger ones?
# HAR-RV on QQQ 5m RTH realized variance (log HAR, coefficients refit every month on data before it); GK = prior-day Garman-Klass; GJR-GARCH(1,1) fitted on 2016-2021 daily returns,
# then run forward with fixed parameters (no 2022+ data in the fit). Each forecast -> percentile vs its own trailing year -> terciles. Then ADAPTIVE targets: target = a x HAR daily range in NQ pts, stop = 2/3 of it.
from arch import arch_model
rth5 = (m5 >= 570) & (m5 < 960); lr = np.diff(np.log(c5), prepend=np.nan); lr[~rth5] = np.nan; lr[np.r_[True, d5[1:] != d5[:-1]]] = np.nan
RV = pd.Series(lr ** 2).groupby(pd.DatetimeIndex(d5)).sum(min_count=10).dropna(); lrv = np.log(RV)
Hd = pd.DataFrame({'d': lrv, 'w': lrv.rolling(5).mean(), 'm': lrv.rolling(22).mean()}).dropna(); Hd['y'] = lrv.shift(-1)
har = pd.Series(np.nan, index=Hd.index)
for mo_, g in Hd.groupby(Hd.index.to_period('M')):
    tr = Hd[(Hd.index < g.index[0]) & Hd.y.notna()]
    if len(tr) < 250: continue
    A = np.c_[np.ones(len(tr)), tr[['d', 'w', 'm']].values]; b = np.linalg.lstsq(A, tr.y.values, rcond=None)[0]
    har.loc[g.index] = np.exp(np.c_[np.ones(len(g)), g[['d', 'w', 'm']].values] @ b)      # forecast of NEXT day's RV, made at day's close
Dq = yf.download('QQQ', start='2015-06-01', interval='1d', progress=False, auto_adjust=False); Dq.columns = [c[0] if isinstance(c, tuple) else c for c in Dq.columns]; Dq.index = Dq.index.normalize()
gk = .5 * np.log(Dq.High / Dq.Low) ** 2 - (2 * np.log(2) - 1) * np.log(Dq.Close / Dq.Open) ** 2
ret = 100 * np.log(Dq['Adj Close']).diff().dropna(); fit = arch_model(ret[ret.index < '2022-01-01'], vol='GARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
f2 = arch_model(ret, vol='GARCH', p=1, o=1, q=1, dist='t').fix(fit.params); gjr = (f2.conditional_volatility.shift(-1) ** 2)   # sigma^2 for the NEXT day from info through today
VOL = pd.DataFrame({'HAR': har, 'GK': gk, 'GJR': gjr}).sort_index()
VP = VOL.apply(lambda s: s.rolling(252, min_periods=120).rank(pct=True))
print('\n== VOLATILITY MODELS (S0x2 trades, 2022-26, 1 NQ 75/50): forecast tercile at the prior close -> win, avg, avg |pts| (size of moves)', flush=True)
Cc = ctx(AL[(AL.day >= P0) & AL.neg]); R = run_n(Cc, 75, 50, 2).reset_index(drop=True); R['net'] = R.pts - 1.0
pv = VP.reindex(pd.DatetimeIndex(R.day) - pd.Timedelta(days=1), method='ffill').reset_index(drop=True); R = pd.concat([R, pv], axis=1)
for mdl in ('HAR', 'GK', 'GJR'):
    for v, g in R.groupby(pd.cut(R[mdl], [-1, 1 / 3, 2 / 3, 2], labels=['low', 'mid', 'high'])):
        a, b = g[g.day < '2024-01-01'], g[g.day >= '2024-01-01']
        print(f'  {mdl:<4} {str(v):<5} n{len(g):>4} win {g.pts.gt(0).mean():.0%} avg {g.net.mean():+6.1f} | |move| {g.pts.abs().mean():5.1f} | 22-23 {a.net.mean():+6.1f} / 24-26 {b.net.mean():+6.1f}', flush=True)
print('\n  ADAPTIVE targets from HAR (target = a x forecast daily range in NQ pts, stop = 2/3 target), same S0x2 signals, 1 NQ', flush=True)
hr = har.reindex(pd.DatetimeIndex(Cc.day) - pd.Timedelta(days=1), method='ffill').values
nq_lv = NQs.reindex(pd.DatetimeIndex(Cc.day) - pd.Timedelta(days=1), method='ffill').values; rng_pts = np.sqrt(hr) * nq_lv * 1.6      # ~ expected high-low range (Parkinson factor)
print(f'  HAR expected daily range, 2022-26: median {np.nanmedian(rng_pts):.0f} NQ pts (10%-90% {np.nanpercentile(rng_pts, 10):.0f}-{np.nanpercentile(rng_pts, 90):.0f})', flush=True)
for a_ in (.25, .4, .6):
    out = []
    for (day, g), rp in zip(Cc.sort_values('j').groupby('day'), pd.Series(rng_pts, index=Cc.index).groupby(Cc.sort_values('j').day).first().values):
        if not np.isfinite(rp): continue
        Rg = run_n(g, a_ * rp, a_ * rp * 2 / 3, 2); Rg['T'] = a_ * rp; out.append(Rg)
    Ra = pd.concat(out); Ra['net'] = Ra.pts - 1.0; e = evalm(Ra, 20, 1.0); mo = monthly(Ra, 20, 1.0)
    print(f'  a={a_}: median target {Ra["T"].median():.0f} pts | trades {len(Ra)} win {Ra.pts.gt(0).mean():.0%} avg {Ra.net.mean():+.1f} pts (22-23 {Ra[Ra.day < "2024"].net.mean():+.1f} / 24-26 {Ra[Ra.day >= "2024"].net.mean():+.1f}) | 1-month pass {e["pass"]:.0%} | month $ mean {mo.mean():+,.0f} up {mo.gt(0).mean():.0%}', flush=True)
