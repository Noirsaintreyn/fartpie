"""Nov 2 2026: our own 'expected move after a flip' model. Target = how far the trade runs in our favour (peak gain from entry to exit,
from 15m highs) and, separately, the trade's return. Features: options ratio; overnight-vol block (news share, HAR overnight forecast,
overnight momentum, gap burst); volatility-vector states (realized, gap, smoothed); stress regime flags. Walk-forward by year (LightGBM, small).
Checks: does it predict the peak (rank correlation), does predicted peak / predicted return rank trades, and slot ranking with 40 matched runs."""
import lightgbm as lgb
exec(open('/home/claude/fartpie/xs/ovn_vol.py').read().split("TK = taken(Ls)")[0].replace("print(", "(lambda *a, **k: None)("))
src = open('/home/claude/fartpie/xs/vec_on_vol.py').read(); exec(src[src.index('def vec_series'):src.index('F = pd.concat(F)')].replace('for tk in Ls.tk.unique():', 'F = []\nfor tk in Ls.tk.unique():'))
F = pd.concat(F); F['d'] = pd.to_datetime(F.d); Ls = Ls.merge(F, on=['tk', 'd'], how='left')
d_ = Ls.d.values; Ls['dn'] = mk.dn.reindex(d_, method='ffill').fillna(False).astype(float).values
Ls['reb'] = (st['A'].reindex(d_, method='ffill').values == 'Rebound').astype(float); Ls['jm0'] = (st['C'].reindex(d_, method='ffill').values == 'JM-0').astype(float)
PK = np.full(len(Ls), np.nan)
for tk, g in Ls.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'open', 'high']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt'); t = H.dt.values; hi = H.high.values; op = H.open.values
    for i, e, x in zip(g.index, g.entry.values, g.exit.values):
        a, b = np.searchsorted(t, e), np.searchsorted(t, x)
        if a < len(t) and b > a: PK[i] = hi[a:b].max() / op[a] - 1
Ls['peak'] = PK
X_ = ['ratio', 'on_share', 'on_har', 'on_mom', 'on_abs5', 'vrv', 'vgap', 'vhar', 'dn', 'reb', 'jm0', 'h1']
Ls['h1'] = Ls.h1.astype(float)
for tgt in ('peak', 'ret'):
    Ls[f'p_{tgt}'] = np.nan; y = Ls[tgt].clip(-.15, .3)
    for Y in range(2020, 2027):
        tr = (Ls.exit < pd.Timestamp(f'{Y}-01-01')) & y.notna(); te = Ls.entry.dt.year == Y
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_lambda=5, verbose=-1)
        mdl.fit(Ls.loc[tr, X_], y[tr]); Ls.loc[te, f'p_{tgt}'] = mdl.predict(Ls.loc[te, X_])
O = Ls[Ls.entry.dt.year >= 2020].copy(); O['h'] = np.where(O.entry.dt.year <= 2021, '2020-21', '2022-26')
print('A) out-of-sample rank correlation (all flips)')
for h, g in O.groupby('h'):
    print(f'  {h}: predicted peak vs actual peak {g.p_peak.corr(g.peak, method="spearman"):+.3f}   predicted peak vs return {g.p_peak.corr(g.ret, method="spearman"):+.3f}   predicted return vs return {g.p_ret.corr(g.ret, method="spearman"):+.3f}')
for p in ('p_peak', 'p_ret'):
    q = O.groupby('h')[p].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)) + 1
    t = O.groupby(['h', q]).agg(peak=('peak', 'mean'), ret=('ret', 'mean')) * 100
    print(f'  by {p} quintile (q1 low .. q5 high): ' + '   '.join(f'{h}: peak ' + ' '.join(f'{v:.1f}' for v in t.loc[h].peak.values) + ' | ret ' + ' '.join(f'{v:+.2f}' for v in t.loc[h].ret.values) for h in ('2020-21', '2022-26')))
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
P2 = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(eq):
    out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
rng = np.random.default_rng(113); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values; Z0 = Z0[Z0.entry >= '2020-01-01']
    return np.array([stat2(dret(simw(Z0[~dl[Z0.index]])).cumsum()) for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<28} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) slot ranking, 40 matched runs, 2020-21 / 2022-26'); B0 = ev(Ls.ratio); line('live (ratio)', B0)
for nm, k in (('predicted peak', Ls.p_peak), ('ratio + predicted peak', pr(Ls.ratio) + pr(Ls.p_peak)), ('predicted return', Ls.p_ret), ('ratio + predicted return', pr(Ls.ratio) + pr(Ls.p_ret))):
    line(nm, ev(k), B0)
