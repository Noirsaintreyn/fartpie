"""Nov 1 2026: every piece of edge rephrased in auction-market terms, a model trained on that, and the extra (non-AMT) edge as a branch.
AMT block (what the auction is doing at the flip):
  value location     pva / cva (below, inside, above prior-day and 5-day value), distance from prior POC in ATR
  value migration    migr (POC rising), dpoc (developing POC above prior POC)
  acceptance         beyond (close past the trigger, ATR), clv (close location), uwick (rejection / excess)
  initiative         delta proxy on the flip bar and the last 4 bars (volume x close location; no tick data for stocks)
  balance -> imbal.  dur / width / durpct of the base, nb (move vs the time-of-day noise band), ib (range extension)
  other timeframe    relvol (participation), v1h / v4h / vD (higher-timeframe auction direction)
Extra-edge branch: options ratio, stress states (A+, Rebound, jump-model bear, panic, VIX fear), earnings distance, SPY vs 200d, QQQ 20d.
Models (walk-forward by year, train on flips that exited before Y): AMT model; context model; one model on both; and a BRANCH model
(separate AMT models for stress and normal regimes). OOS quintiles, then slot ranking with the 40-run matched yardstick."""
import lightgbm as lgb
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
K = ['tk', 'entry', 'h1']
F = pd.read_parquet('/tmp/claude-0/ml/flipq.parquet')[K + ['v15x', 'v1h', 'v4h', 'vD', 'clv', 'uwick', 'beyond', 'relvol', 'dur', 'width', 'durpct', 'nb']]
A = pd.read_parquet('/tmp/claude-0/ml/amt.parquet')[K + ['pva', 'cva', 'migr', 'dpoc', 'vwap', 'ib']]
X = Ls.reset_index(drop=True).merge(F, on=K, how='left').merge(A, on=K, how='left')
for c, m in (('pva', {'below': -1, 'inside': 0, 'above': 1}), ('cva', {'below': -1, 'inside': 0, 'above': 1}), ('migr', {'down': -1, 'up': 1}),
             ('dpoc', {'below prior POC': -1, 'above prior POC': 1}), ('vwap', {'below': -1, 'above': 1}), ('ib', {'below IB low': -1, 'inside IB': 0, 'above IB high': 1})):
    X[c] = X[c].map(m)
DP = {}
for tk in X.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 240) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    rg = (H.high - H.low).replace(0, np.nan); dp = H.volume * ((H.close - H.low) - (H.high - H.close)) / rg; mv = H.volume.rolling(130, min_periods=40).median()
    DP[tk] = pd.DataFrame({'d1': (dp / mv).values, 'd4': (dp.rolling(4).sum() / (4 * mv)).values}, index=H.dt + pd.Timedelta('15min'))
X['delta1'] = [DP[t].d1.asof(e) if t in DP else np.nan for t, e in zip(X.tk, X.entry)]
X['delta4'] = [DP[t].d4.asof(e) if t in DP else np.nan for t, e in zip(X.tk, X.entry)]
L_ = X; src2 = open('/home/claude/fartpie/xs/pre_earn_flip.py').read(); L = X; exec(src2[src2.index('k = np.full'):src2.index("L['k'] = k")]); X['k_earn'] = np.minimum(k, 30)
X['stress_dn'] = mk.dn.reindex(X.entry.dt.normalize().values, method='ffill').fillna(False).astype(float).values
for c, v in (('A', 'Rebound'), ('C', 'JM-0'), ('B', 'Panic'), ('D', 'Fearful')): X[f's_{v}'] = (X[c] == v).astype(float)
X['spy200'] = mk.spy200.reindex(X.entry.dt.normalize().values, method='ffill').values if 'spy200' in mk else np.nan
X['stress'] = ((X.stress_dn > 0) | (X['s_Rebound'] > 0) | (X['s_JM-0'] > 0)).astype(int)
AMT = ['pva', 'cva', 'migr', 'dpoc', 'vwap', 'ib', 'beyond', 'clv', 'uwick', 'delta1', 'delta4', 'dur', 'width', 'durpct', 'nb', 'relvol', 'v1h', 'v4h', 'vD', 'v15x']
CTX = ['ratio', 'stress_dn', 's_Rebound', 's_JM-0', 's_Panic', 's_Fearful', 'k_earn', 'mz20', 'mz50']
y = X.ret.clip(-.15, .25)
def wf(cols, mask=None, name='p'):
    p = pd.Series(np.nan, index=X.index)
    for Y in range(2020, 2027):
        tr = (X.exit < pd.Timestamp(f'{Y}-01-01')); te = X.entry.dt.year == Y
        if mask is not None: tr &= mask; te &= mask
        if tr.sum() < 300 or te.sum() == 0: continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_lambda=5, verbose=-1)
        m.fit(X.loc[tr, cols], y[tr]); p[te] = m.predict(X.loc[te, cols])
    return p
X['p_amt'] = wf(AMT); X['p_ctx'] = wf(CTX); X['p_all'] = wf(AMT + CTX)
X['p_branch'] = wf(AMT + ['ratio'], X.stress == 1).combine_first(wf(AMT + ['ratio'], X.stress == 0))
O = X[X.entry.dt.year >= 2020]; O = O.assign(h=np.where(O.entry.dt.year <= 2021, '2020-21', '2022-26'))
print('A) all flips out of sample: avg trade % by prediction quintile (q5 = best predicted)')
for p in ('p_amt', 'p_ctx', 'p_all', 'p_branch'):
    q = O.groupby('h')[p].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)) + 1
    t = (O.groupby(['h', q]).ret.mean() * 100).unstack().round(2); print(f'  {p}:  ' + '   '.join(f'{h}: ' + ' '.join(f'{v:+.2f}' for v in t.loc[h].values) for h in t.index), flush=True)
P2 = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(eq):
    out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; d = (e / e.cummax() - 1).min(); out += [c, d, c / -d]
    return out
rng = np.random.default_rng(61); DR = [rng.random(len(X)) < .05 if i else np.zeros(len(X), bool) for i in range(41)]
pr = lambda s: s.groupby(X.entry.dt.year).rank(pct=True)
def ev(key):
    Z0 = X.copy(); Z0['ratio'] = key.values; Z0 = Z0[Z0.entry >= '2020-01-01']
    return np.array([stat2(dret(simw(Z0[~dl[Z0.index]])).cumsum()) for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<30} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) slot ranking, 40 matched runs, 2020-21 / 2022-26')
B0 = ev(X.ratio); line('live (ratio)', B0)
for nm, key in (('AMT model', X.p_amt), ('ratio + AMT model', pr(X.ratio) + pr(X.p_amt)), ('ratio + AMT + context', pr(X.ratio) + pr(X.p_amt) + pr(X.p_ctx)),
                ('one model, all features', X.p_all), ('branch model (stress / normal)', X.p_branch), ('ratio + branch model', pr(X.ratio) + pr(X.p_branch))):
    line(nm, ev(key.fillna(key.median())), B0)
X.to_parquet('/tmp/claude-0/ml/amt_model.parquet')
