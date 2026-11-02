"""Nov 2 2026: models built for THIS payoff (few big winners, many small losses) instead of predicting the average return.
 Q   quantile LightGBM: predict the trade's upper tail (90th pct of return) and lower tail (10th pct) -> rank by predicted asymmetry
     q90 + q10 (convexity: big upside without matching downside)
 H   competing risks (discrete-time hazards, logistic): each day a trade is open, P(big up day: close > +2 sigma) vs P(stopped out that day);
     rank flips by predicted big-up hazard / stop hazard over the first 5 days
Features known at entry: options ratio, overnight-vol block, vol-vector states, stress flags, flip quality (close location, beyond trigger,
relative volume, base length/width, noise band), multi-timeframe states. Walk-forward by year. Then slot ranking, 40 matched runs."""
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
exec(open('/home/claude/fartpie/xs/flip_move_model.py').read().split("X_ = ['ratio'")[0].replace("print(", "(lambda *a, **k: None)("))
FQ = pd.read_parquet('/tmp/claude-0/ml/flipq.parquet')[['tk', 'entry', 'h1', 'clv', 'beyond', 'relvol', 'dur', 'width', 'nb', 'v1h', 'v4h', 'vD']]
FQ['h1'] = FQ.h1.astype(float); Ls = Ls.merge(FQ, on=['tk', 'entry', 'h1'], how='left')
X_ = ['ratio', 'on_share', 'on_har', 'on_mom', 'on_abs5', 'vrv', 'vgap', 'vhar', 'dn', 'reb', 'jm0', 'h1', 'clv', 'beyond', 'relvol', 'dur', 'width', 'nb', 'v1h', 'v4h', 'vD']
y = Ls.ret.clip(-.15, .3)
for q in (.1, .9):
    Ls[f'q{int(q * 100)}'] = np.nan
    for Y in range(2020, 2027):
        tr = Ls.exit < pd.Timestamp(f'{Y}-01-01'); te = Ls.entry.dt.year == Y
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=300, subsample=.8, subsample_freq=1, colsample_bytree=.7, verbose=-1)
        m.fit(Ls.loc[tr, X_], y[tr]); Ls.loc[te, f'q{int(q * 100)}'] = m.predict(Ls.loc[te, X_])
Ls['asym'] = Ls.q90 + Ls.q10
# ---- competing risks: daily panel of open trades (first 5 sessions)
rows = []
for i, (tk, e, x, h1) in enumerate(zip(Ls.tk, Ls.entry, Ls.exit, Ls.h1)):
    b = bars(tk)
    if b is None: continue
    dc = b[1]; days = dc[(dc.index > e.normalize())].index[:5]; sd = dc.pct_change().rolling(20).std().shift(1)
    for k, d_ in enumerate(days):
        stop = d_ >= x.normalize(); big = (dc.pct_change().get(d_, 0) > 2 * sd.get(d_, np.inf))
        rows.append((i, k, int(big), int(stop))); 
        if stop: break
Pn = pd.DataFrame(rows, columns=['i', 'k', 'big', 'stop']); Pn = Pn.join(Ls[X_ + ['entry', 'exit']], on='i')
Ls['hz_big'] = np.nan; Ls['hz_stop'] = np.nan
XX = X_ + ['k']
for Y in range(2020, 2027):
    tr = Pn.exit < pd.Timestamp(f'{Y}-01-01'); te_i = Ls.index[Ls.entry.dt.year == Y]
    A = Pn.loc[tr, XX].fillna(Pn.loc[tr, XX].median()); mu, sdv = A.mean(), A.std().replace(0, 1)
    mb = LogisticRegression(C=.1, max_iter=500).fit((A - mu) / sdv, Pn.loc[tr, 'big']); ms = LogisticRegression(C=.1, max_iter=500).fit((A - mu) / sdv, Pn.loc[tr, 'stop'])
    T_ = Ls.loc[te_i, X_].fillna(A[X_].median()); pb = np.zeros(len(T_)); ps = np.zeros(len(T_)); surv = np.ones(len(T_))
    for k in range(5):
        Z = T_.copy(); Z['k'] = k; Z = (Z[XX] - mu) / sdv; b_ = mb.predict_proba(Z)[:, 1]; s_ = ms.predict_proba(Z)[:, 1]
        pb += surv * b_; ps += surv * s_; surv *= (1 - s_)
    Ls.loc[te_i, 'hz_big'] = pb; Ls.loc[te_i, 'hz_stop'] = ps
Ls['hz_ratio'] = Ls.hz_big / Ls.hz_stop
O = Ls[Ls.entry.dt.year >= 2020].copy(); O['h'] = np.where(O.entry.dt.year <= 2021, '2020-21', '2022-26')
print('A) all flips out of sample: avg return % by quintile (q1 low .. q5 high)')
for p in ('asym', 'q90', 'hz_big', 'hz_ratio'):
    q = O.groupby('h')[p].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)) + 1
    t = (O.groupby(['h', q]).ret.mean() * 100).unstack(); print(f'  {p:<9} ' + '   '.join(f'{h}: ' + ' '.join(f'{v:+.2f}' for v in t.loc[h].values) for h in t.index), flush=True)
pr = lambda s: s.groupby(Ls.entry.dt.year).rank(pct=True)
P2 = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(eq):
    out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
rng = np.random.default_rng(131); DR = [rng.random(len(Ls)) < .05 if i else np.zeros(len(Ls), bool) for i in range(41)]
def ev(key_):
    Z0 = Ls.copy(); Z0['ratio'] = key_.fillna(key_.median()).values; Z0 = Z0[Z0.entry >= '2020-01-01']
    return np.array([stat2(dret(simw(Z0[~dl[Z0.index]])).cumsum()) for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<26} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) slot ranking, 40 matched runs, 2020-21 / 2022-26'); B0 = ev(Ls.ratio); line('live (ratio)', B0)
for nm, k in (('asymmetry (q90+q10)', Ls.asym), ('ratio + asymmetry', pr(Ls.ratio) + pr(Ls.asym)), ('big-day hazard', Ls.hz_big), ('ratio + big-day hazard', pr(Ls.ratio) + pr(Ls.hz_big)),
              ('hazard ratio big/stop', Ls.hz_ratio), ('ratio + hazard ratio', pr(Ls.ratio) + pr(Ls.hz_ratio))):
    line(nm, ev(k), B0)
