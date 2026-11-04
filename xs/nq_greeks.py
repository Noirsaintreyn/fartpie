"""Nov 4 2026: (1) volatility SIZING instead of filtering, (2) options Greeks incl. vanna / charm paired with HAR, for the NQ NY AM trade. DECLARED BEFORE RESULTS.
Trades: QQQ 15m vector flip 09:30-11:30, up to 2 a day, 1 NQ base, 75/50, 1 pt cost (QQQ proxy -> NQ points). Sets: ALLx2 (any day) and S0x2 (negative gamma days).
(1) SIZING by the HAR forecast tercile at the prior close (no other change; one-month eval with the consistency rule, starts 2022-26 and 2026):
    Z1 low x1.5 / mid x1 / high x0.5     Z2 low x2 / others x1     Z3 low x1 / others x0.5 (de-risk the noisy days)     Z4 inverse: low x0.5 / high x1.5
(2) GREEKS at the prior close (QQQ, data ends Dec 2025, so 2022-2025 only), each as 1-year percentile terciles: net vanna, net charm, net DEX (call + put delta exposure),
    ATM IV, 25-delta skew, VRP = ATM IV^2 - HAR forecast variance (implied over expected realized). Pairs: vanna sign x VRP high/low, vanna tercile x HAR tercile.
    A cell counts if it beats the set average in BOTH 2022-23 and 2024-25 with n >= 20 each."""
_s = open('xs/nq_month.py').read(); exec(_s[:_s.index("SETS = {")])
_q = open('xs/nq_qqq_ctx.py').read(); exec(_q[_q.index("from arch import arch_model"):_q.index("VOL = pd.DataFrame")])
HP = har.rolling(252, min_periods=120).rank(pct=True)
def prior(s, dd): return s.reindex(pd.DatetimeIndex(dd) - pd.Timedelta(days=1), method='ffill').values
def evald(v_series, a, b, n=21, target=3000, mll=2000, dll=1000):
    v = v_series.reindex(days).fillna(0.0); idx = v.index; v = v.values; res = {'pass': 0, 'fail': 0, 'expire': 0}
    for s in range(len(v) - n):
        if not (pd.Timestamp(a) <= idx[s] < pd.Timestamp(b)): continue
        bal, peak, floor, best, out = 0.0, 0.0, -mll, 0.0, 'expire'
        for k in range(s, s + n):
            day = max(v[k], -dll); bal += day; best = max(best, day)
            if bal <= floor: out = 'fail'; break
            if bal >= target and best <= .5 * bal: out = 'pass'; break
            peak = max(peak, bal); floor = min(max(floor, peak - mll), 0.0)
        res[out] += 1
    t = max(sum(res.values()), 1); return {k: v_ / t for k, v_ in res.items()}
SZ = {'Z0 flat 1 NQ': (1, 1, 1), 'Z1 low 1.5 / mid 1 / high 0.5': (1.5, 1, .5), 'Z2 low 2 / others 1': (2, 1, 1), 'Z3 low 1 / others 0.5': (1, .5, .5), 'Z4 inverse low 0.5 / high 1.5': (.5, 1, 1.5)}
rng = np.random.default_rng(51)
print('(1) SIZING by HAR tercile | 1-month eval pass (starts 2022-26 / 2026) | monthly $ 2022-26', flush=True)
TR = {}
for sn, (Cc, pdn) in (('ALLx2', (AL, 2)), ('S0x2', (AL[AL.neg], 2))):
    R = run_n(Cc, 75, 50, pdn); R['hp'] = prior(HP, R.day); R['t'] = pd.cut(R.hp, [-1, 1 / 3, 2 / 3, 2], labels=[0, 1, 2]).astype(float); TR[sn] = (Cc, R)
    for zn, (a, b_, c_) in SZ.items():
        mult = R.t.map({0.0: a, 1.0: b_, 2.0: c_}).fillna(1.0); dv = ((R.pts - 1.0) * 20 * mult).groupby(R.day).sum()
        e1, e2 = evald(dv, '2022-01-01', '2027-01-01'), evald(dv, '2026-01-01', '2027-01-01'); mo = dv[dv.index >= P0].groupby(dv[dv.index >= P0].index.to_period('M')).sum().reindex(pd.period_range(P0, dpost[-1], freq='M')).fillna(0)
        cf = []
        if zn in ('Z0 flat 1 NQ', 'Z1 low 1.5 / mid 1 / high 0.5', 'Z2 low 2 / others 1'):
            for _ in range(10):
                Z = Cc.copy(); Z['dir'] = rng.choice([-1, 1], len(Z)); Rz = run_n(Z, 75, 50, pdn); tz = pd.cut(prior(HP, Rz.day), [-1, 1 / 3, 2 / 3, 2], labels=[0, 1, 2]).astype(float)
                cf.append(evald(((Rz.pts - 1.0) * 20 * pd.Series(tz).map({0.0: a, 1.0: b_, 2.0: c_}).fillna(1.0).values).groupby(Rz.day).sum(), '2022-01-01', '2027-01-01')['pass'])
        print(f'  {sn:<6} {zn:<30} pass 2022-26 {e1["pass"]:.0%} (fail {e1["fail"]:.0%}){f" coin-flip {np.median(cf):.0%}" if cf else "":<16} | pass 2026 {e2["pass"]:.0%} | month $ mean {mo.mean():+,.0f} median {mo.median():+,.0f} up {mo.gt(0).mean():.0%} worst {mo.min():+,.0f}', flush=True)
# ---------------------------------------------------------------- (2) Greeks
G1 = pd.read_csv(f'{ROOT}/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date'); G2 = pd.read_csv(f'{ROOT}/options_greeks_daily_v2_QQQ.csv', parse_dates=['date']).set_index('date')
GR = pd.DataFrame({'vanna': G2.net_vanna, 'charm': G2.net_charm, 'dex': G1.call_dex + G1.put_dex, 'iv': G1.atm_iv, 'skew': G1.skew_25d}).sort_index()
GR['vrp'] = GR.iv ** 2 - har.reindex(GR.index) * 252                                 # annualised implied variance minus HAR's expected realised variance
GP = GR.apply(lambda s: s.rolling(252, min_periods=120).rank(pct=True))
lab3 = lambda x: pd.cut(x, [-1, 1 / 3, 2 / 3, 2], labels=['low', 'mid', 'high']).astype(str)
for sn in ('ALLx2', 'S0x2'):
    Cc, R = TR[sn]; R = R[(R.day >= P0) & (R.day < '2026-01-01')].copy(); R['net'] = R.pts - 1.0; R['h'] = np.where(R.day < '2024-01-01', '22-23', '24-25')
    base = {h: R[R.h == h].net.mean() for h in ('22-23', '24-25')}
    for k in ('vanna', 'charm', 'dex', 'iv', 'skew', 'vrp'): R[k] = lab3(prior(GP[k], R.day))
    R['vanna_sign'] = np.where(prior(GR.vanna, R.day) > 0, 'vanna +', 'vanna -'); R['har'] = lab3(R.hp)
    R['vanna x VRP'] = R.vanna_sign + ' / VRP ' + R.vrp; R['vanna x HAR'] = 'vanna ' + R.vanna + ' / HAR ' + R.har
    print(f'\n(2) GREEKS, {sn} 2022-25: trades {len(R)} avg {R.net.mean():+.1f} pts (22-23 {base["22-23"]:+.1f} / 24-25 {base["24-25"]:+.1f}); vanna > 0 on {np.mean(prior(GR.vanna, R.day) > 0):.0%} of trade days', flush=True)
    for col in ('vanna', 'charm', 'dex', 'iv', 'skew', 'vrp', 'vanna_sign', 'vanna x VRP', 'vanna x HAR'):
        for v, g in R.groupby(col):
            a, b = g[g.h == '22-23'], g[g.h == '24-25']; ok = len(a) >= 20 and len(b) >= 20 and a.net.mean() > base['22-23'] and b.net.mean() > base['24-25']
            if len(g) < 15: continue
            print(f'  {col:<11} {v:<26} n{len(g):>4} win {g.pts.gt(0).mean():.0%} avg {g.net.mean():+6.1f} | 22-23 n{len(a):>3} {a.net.mean():+6.1f} / 24-25 n{len(b):>3} {b.net.mean():+6.1f}' + ('   <- both' if ok else ''), flush=True)
