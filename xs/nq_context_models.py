"""Nov 8 2026: SIZE EVERY NQ FLIP BY ITS CONTEXT - walk-forward only. DECLARED BEFORE RESULTS.
Trades: every RTH 15m flip 09:45-15:30 held to the vector exit (stop 50, flat 15:55, <= 4 a day, no overlap), the +3.07 pts/day version; context table = nq_flip_features.
Test years 2020-2026; each year's model is fit only on flips before Jan 1 of that year (expanding window from 2018). Size per flip: 0 (skip), 0.5, 1, 1.5 or 2 contracts.
  M0  flat 1 contract                                    M0b  flat, no entries after 14:00
  RA  Rule A entries held to the vector exit (reference, flat 1)
  M2  ALIGNMENT COUNT: on the training flips, each feature's best tercile (by mean points) counts as 'aligned' when that tercile beat the worst in every training year;
      score = number of aligned features; size by the score's training terciles: bottom 0.5, middle 1, top 2 ("the more aligned, the more size")
  M3  gradient-boosted trees (LightGBM, small, monotone-free) predicting points; size by predicted terciles 0 / 1 / 2
  M4  quantile trees: predicted 25th percentile of points; size 2 when it is above 0, 1 in the middle third, 0 in the bottom third
  M5  isolation forest on the regime features (vol, VIX / VXN, compression) as a pre-filter (skip the 1/3 most 'normal' days) + M3 sizing on the rest
  M6  epsilon-SVR (epsilon = 20 pts: errors on small trades ignored, so it fits the big moves); size by predicted terciles 0 / 1 / 2
  M7  M2 with the 20-trade lookback on top (x0.5 while the last 20 sized trades are net negative)
Report on 2020 - Oct 2026: points per day (all sessions), average size, points per day per contract of average size, days >= 7.5, months >= 50, worst month, max drawdown,
Sharpe, losing years, and points per day by year."""
_pd = open('xs/nq_points_day.py').read(); exec(_pd[:_pd.index("L = pd.read_csv")])
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
X = pd.read_parquet('/tmp/claude-0/ml/nq_flip_features.parquet').sort_values('j').reset_index(drop=True)
ex = []
for r in X.itertuples():
    res = sim(int(r.j), int(r.dir), None, 50, 955, True); ex.append(res[1] if res else int(r.j))
X['q'] = ex; X['yr'] = X.day.dt.year
FEAT = ['yz1', 'yz5', 'yz20', 'rv1', 'rv5', 'rv20', 'shock', 'vix', 'vxn', 'comp', 'used', 'brk', 'acc', 'body', 'rvol', 'vwd', 'drift', 'gap', 'prev', 'z5', 'z21', 'z63', 'z252', 'room', 'me', 'dow', 'ruleA_am']
X['ruleA_am'] = np.where(X.me <= 690, X.ruleA, np.nan)
REG = ['yz1', 'yz5', 'yz20', 'rv1', 'rv5', 'rv20', 'shock', 'vix', 'vxn', 'comp']
def align_fit(tr):
    rules = []
    for c in FEAT:
        s = tr[c]
        if s.notna().sum() < 60: continue
        if c in ('me', 'dow', 'ruleA_am'):
            cuts = None; lab = s
        else:
            qs = s.quantile([1 / 3, 2 / 3]).values; cuts = qs; lab = pd.Series(np.digitize(s, qs), index=s.index).where(s.notna())
        g = tr.groupby(lab).th.mean()
        if len(g) < 2: continue
        best, worst = g.idxmax(), g.idxmin(); ok = True
        for y, gy in tr.groupby('yr'):
            ly = lab.loc[gy.index]; mb, mw = gy.th[ly == best].mean(), gy.th[ly == worst].mean()
            if np.isfinite(mb) and np.isfinite(mw) and mb <= mw: ok = False; break
        if ok: rules.append((c, cuts, best))
    return rules
def align_score(df, rules):
    sc = np.zeros(len(df))
    for c, cuts, best in rules:
        s = df[c]; lab = s if cuts is None else pd.Series(np.digitize(s, cuts), index=s.index).where(s.notna())
        sc += (lab == best).values
    return sc
def tiers(pred_tr, pred_te, sizes=(0.0, 1.0, 2.0)):
    a, b = np.quantile(pred_tr, [1 / 3, 2 / 3]); return np.where(pred_te <= a, sizes[0], np.where(pred_te <= b, sizes[1], sizes[2]))
P = lambda df: df[FEAT].astype(float).values
SIZE = {k: pd.Series(np.nan, index=X.index) for k in ('M0', 'M0b', 'RA', 'M2', 'M3', 'M4', 'M5', 'M6')}
NRULES = {}
for Y in range(2020, 2027):
    tr, te = X[X.yr < Y], X[X.yr == Y]
    if not len(te): continue
    SIZE['M0'].loc[te.index] = 1.0; SIZE['M0b'].loc[te.index] = np.where(te.me <= 840, 1.0, 0.0); SIZE['RA'].loc[te.index] = te.ruleA.astype(float)
    rules = align_fit(tr); NRULES[Y] = [r[0] for r in rules]
    SIZE['M2'].loc[te.index] = tiers(align_score(tr, rules), align_score(te, rules), (0.5, 1.0, 2.0))
    m3 = lgb.LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=8, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=1).fit(P(tr), tr.th)
    SIZE['M3'].loc[te.index] = tiers(m3.predict(P(tr)), m3.predict(P(te)))
    m4 = lgb.LGBMRegressor(objective='quantile', alpha=.25, n_estimators=200, learning_rate=.03, num_leaves=8, min_child_samples=30, verbose=-1, random_state=1).fit(P(tr), tr.th)
    q_tr, q_te = m4.predict(P(tr)), m4.predict(P(te)); a = np.quantile(q_tr, 1 / 3)
    SIZE['M4'].loc[te.index] = np.where(q_te > 0, 2.0, np.where(q_te > a, 1.0, 0.0))
    R_tr, R_te = tr[REG].fillna(tr[REG].median()), te[REG].fillna(tr[REG].median())
    iso = IsolationForest(n_estimators=300, random_state=1).fit(R_tr); s_tr, s_te = iso.score_samples(R_tr), iso.score_samples(R_te)      # high = normal
    keep = s_te <= np.quantile(s_tr, 2 / 3)
    SIZE['M5'].loc[te.index] = np.where(keep, tiers(m3.predict(P(tr)), m3.predict(P(te))), 0.0)
    imp = pd.DataFrame(P(tr)).fillna(pd.DataFrame(P(tr)).median()); sc = StandardScaler().fit(imp)
    svr = SVR(C=1.0, epsilon=20.0, kernel='rbf', gamma='scale').fit(sc.transform(imp), tr.th.clip(-60, 150))
    te_imp = pd.DataFrame(P(te)).fillna(pd.DataFrame(P(tr)).median())
    SIZE['M6'].loc[te.index] = tiers(svr.predict(sc.transform(imp)), svr.predict(sc.transform(te_imp)))
print('alignment rules picked each year (features whose best tercile beat the worst in every training year):', flush=True)
for Y, r in NRULES.items(): print(f'  {Y}: {", ".join(r) if r else "none"}', flush=True)
ALLD = pd.DatetimeIndex(sorted(H.day.unique())); ALLD = ALLD[ALLD >= '2020-01-01']
def book(size, lookback=False):
    rows = []; hist = []
    for day, g in X[X.yr >= 2020].groupby('day', sort=True):
        busy = -1; n = 0
        for r in g.itertuples():
            sz = size.loc[r.Index]
            if n >= 4 or r.j <= busy or not np.isfinite(sz) or sz <= 0: continue
            if lookback and len(hist) >= 20 and sum(hist[-20:]) < 0: sz *= .5
            rows.append((day, r.th * sz, sz)); hist.append(r.th * sz); busy = r.q; n += 1
    T = pd.DataFrame(rows, columns=['day', 'pts', 'sz']); Pd = T.groupby('day').pts.sum().reindex(ALLD).fillna(0.0)
    M = Pd.groupby(Pd.index.to_period('M')).sum(); cum = Pd.cumsum(); yr = Pd.groupby(Pd.index.year).mean()
    return dict(day=Pd.mean(), sz=T.sz.mean() if len(T) else 0, n=len(T) / len(ALLD), ge=(Pd >= 7.5).mean(), m50=(M >= 50).mean(), worst=M.min(), dd=(cum - cum.cummax()).min(),
                sh=Pd.mean() / Pd.std() * np.sqrt(252), lose=int((Pd.groupby(Pd.index.year).sum() < 0).sum()), yr=yr)
OUT = {k: book(v) for k, v in SIZE.items()}; OUT['M7'] = book(SIZE['M2'], True); OUT['M3+lookback'] = book(SIZE['M3'], True)
print(f'\nWALK-FORWARD 2020 - Oct 2026 (NQ points)   {"pts/day":>8} {"trades/day":>10} {"avg size":>8} {"per contract":>12} {"days>=7.5":>9} {"months>=50":>10} {"worst mo":>9} {"max DD":>7} {"Sharpe":>6} {"losing yrs":>10}', flush=True)
for k, r in OUT.items():
    print(f'  {k:<38} {r["day"]:>+8.2f} {r["n"]:>10.2f} {r["sz"]:>8.2f} {r["day"] / max(r["sz"], 1e-9) / max(r["n"], 1e-9) * r["n"]:>+12.2f} {r["ge"]:>9.0%} {r["m50"]:>10.0%} {r["worst"]:>+9.0f} {r["dd"]:>+7.0f} {r["sh"]:>6.2f} {r["lose"]:>10}', flush=True)
    print('      per day by year: ' + ' '.join(f'{y} {v:+.1f}' for y, v in r['yr'].items()), flush=True)
