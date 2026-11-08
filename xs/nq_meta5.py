"""Nov 8 2026: META-LABELING ON 5-MINUTE VECTOR FLIPS - enough samples for a real model, aimed at qualifying days. DECLARED BEFORE RESULTS.
Why: the 15m system trades 2.8 days a month (1.5 qualifying days); 2 payouts a month needs ~10. 15m flips are too few to train on (488 in 2018-21).
CANDIDATES: every QQQ 5m vector flip 09:35 - 15:00 (entry at the next 5m open). LABEL (triple barrier): target = stop = 0.08 x 20-day ADR (~11 NQ pts in 2018-21, ~21 in
2022-26), 60-minute time limit, then out at market; 1 pt cost. 'good' = target first.
FEATURES (known at entry): direction, minute, weekday, QQQ net GEX sign / 1-year percentile, 5- and 20-day realized vol percentiles, ADR, flip-bar range / 5m ATR, flip-bar
RVOL vs the same time of day (20 days), VWAP z, move since the open and since yesterday's close (ADR units, trade direction), prior-day VA location (one-hot, trade direction),
20-day volume-node percentile, distance to POC / VAH / VAL (ADR), 15m / 1h / daily vector agreement, bars since the last 5m flip, flips so far today, 21-day z, 50-day trend.
MODEL: LightGBM, walk-forward by year (train on all earlier years, predict the next), 2020 - 2026; logistic regression for comparison.
POLICY per day: take a candidate when its predicted probability is in the top X% of the training years' predictions (X = 10 / 20 / 30), one position at a time,
stop for the day after the first target (the day qualifies) or after 2 losses. Report by year: AUC, hit rate, points per trade, trades a day, days a month with a
target hit (qualifying days at 10 micros), points a day at 1 contract."""
_c = open('xs/nq_context_map.py').read(); exec(_c[:_c.rindex("if __name__ == '__main__':")])
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
n = len(H); dd = pd.DatetimeIndex(d5); tod = m5
r5v = np.asarray(vec(H[['open', 'high', 'low', 'close']])); flip = np.r_[False, r5v[1:] != r5v[:-1]] & (tod + 5 >= 575) & (tod + 5 <= 900) & (np.r_[d5[1:], d5[-1]] == d5)
Dd = H.groupby('day').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); kday = pd.Series(k5, index=dd).groupby(level=0).first()
ret = np.log(Dd.c).diff(); rv5 = ret.rolling(5).std(); rv20 = ret.rolling(20).std(); pr = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
ADRq = (Dd.h - Dd.l).rolling(20).mean(); DAY = pd.DataFrame({'rv5p': pr(rv5), 'rv20p': pr(rv20), 'adr': ADRq * kday, 'adrq': ADRq, 'z21': (Dd.c - Dd.c.rolling(21).mean()) / Dd.c.rolling(21).std(),
                                                       'a50': np.sign(Dd.c / Dd.c.rolling(50).mean() - 1), 'pc': Dd.c}).shift(1)
Dd['reg'] = np.asarray(vec(Dd[['o', 'h', 'l', 'c']].rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}))); DAY['vD'] = Dd.reg.shift(1)
O3 = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); g = O3.net_gex.copy(); g.index = g.index.normalize()
DAY['gex'] = g.reindex(DAY.index, method='ffill').shift(1); DAY['gexp'] = pr(g).reindex(DAY.index, method='ffill').shift(1)
tr5 = np.maximum(h5 - l5, np.abs(h5 - np.r_[c5[0], c5[:-1]])); atr5 = pd.Series(tr5).rolling(14).mean().values
vol = H.volume.values.astype(float); vtod = pd.Series(vol).groupby(tod).transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1)).values
tpv = (h5 + l5 + c5) / 3; cv = pd.Series(vol).groupby(dd).cumsum().values; vw = pd.Series(tpv * vol).groupby(dd).cumsum().values / np.maximum(cv, 1)
vsd = np.sqrt(np.maximum(pd.Series(tpv * tpv * vol).groupby(dd).cumsum().values / np.maximum(cv, 1) - vw * vw, 0))
B['reg15'] = B.reg; s15 = pd.Series(B.reg.values, index=pd.DatetimeIndex(B.end.values)); s15 = s15[~s15.index.duplicated(keep='last')]
Q1 = load(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/QQQ_1h_*.parquet')[0]); Q1 = Q1[Q1.dt >= '2017-06-01'].reset_index(drop=True); Q1['reg'] = np.asarray(vec(Q1[['open', 'high', 'low', 'close']]))
s1h = pd.Series(Q1.reg.values, index=pd.DatetimeIndex(Q1.dt.values + np.timedelta64(60, 'm'))); s1h = s1h[~s1h.index.duplicated(keep='last')]
day_open = pd.Series(o5).groupby(dd).transform('first').values; fpos = np.where(flip)[0]
rows = []; last_flip = {}; nflips = {}
for i in fpos:
    if i + 13 >= n: continue
    d = pd.Timestamp(d5[i]); D = DAY.loc[d] if d in DAY.index else None
    if D is None or not np.isfinite(D.adr) or i + 1 >= n or d5[i + 1] != d5[i]: continue
    dr = int(r5v[i]); e = o5[i + 1]; kk = k5[i]; T = .08 * D.adr; S = T; out = None; hit = 0
    for q in range(i + 1, min(i + 13, n)):
        if d5[q] != d5[i] or m5[q] >= 955: out = (c5[q - 1] - e) * dr * kk; break
        lo_ = ((l5[q] if dr > 0 else h5[q]) - e) * dr * kk; hi_ = ((h5[q] if dr > 0 else l5[q]) - e) * dr * kk
        if lo_ <= -S: out = -S; break
        if hi_ >= T: out = T; hit = 1; break
    if out is None: out = (c5[min(i + 12, n - 1)] - e) * dr * kk
    t_end = H.dt.values[i] + np.timedelta64(5, 'm'); a15 = s15.iloc[s15.index.searchsorted(t_end, side='right') - 1] if s15.index.searchsorted(t_end, side='right') > 0 else np.nan
    a1h = s1h.iloc[s1h.index.searchsorted(t_end, side='right') - 1] if s1h.index.searchsorted(t_end, side='right') > 0 else np.nan
    poc, vah, val = LVd[days_[dix[d] - 1]] if dix.get(d, 0) > 0 else (np.nan, np.nan, np.nan); adrq = D.adrq
    cp = composite(d) if dix.get(d, 0) >= 20 else None; npct = np.nan
    if cp is not None:
        b0, c_ = cp; kb = int(np.floor(np.log(e) / LB)) - b0
        if 0 <= kb < len(c_) and c_[kb] > 0: npct = (c_[c_ > 0] < c_[kb]).mean()
    va = 0 if e < val else (1 if e < poc else (2 if e <= vah else 3)); va = va if dr > 0 else 3 - va
    nf = nflips.get(d, 0); nflips[d] = nf + 1; lf = last_flip.get(d); last_flip[d] = i
    rows.append(dict(i=i, day=d, dir=dr, pts=out - 1.0, y=hit, tgt=T, minute=tod[i] + 5, wd=d.dayofweek, gexneg=float(D.gex < 0) if np.isfinite(D.gex) else np.nan, gexp=D.gexp,
                     rv5p=D.rv5p, rv20p=D.rv20p, adr=D.adr, rng=(h5[i] - l5[i]) / atr5[i] if atr5[i] > 0 else np.nan, rvol=vol[i] / vtod[i] if vtod[i] > 0 else np.nan,
                     vz=(e - vw[i]) / vsd[i] * dr if vsd[i] > 0 else np.nan, mopen=(e - day_open[i]) / adrq * dr, mpc=(e - D.pc) / adrq * dr, va=va, node=npct,
                     dpoc=(e - poc) / adrq * dr, dvah=(e - vah) / adrq * dr, dval=(e - val) / adrq * dr, a15=a15 * dr, a1h=a1h * dr, aD=D.vD * dr,
                     since=(i - lf) if lf is not None else 99, nf=nf, z21=D.z21 * dr, a50=D.a50 * dr))
C5 = pd.DataFrame(rows); C5['yr'] = C5.day.dt.year
FE = ['dir', 'minute', 'wd', 'gexneg', 'gexp', 'rv5p', 'rv20p', 'adr', 'rng', 'rvol', 'vz', 'mopen', 'mpc', 'va', 'node', 'dpoc', 'dvah', 'dval', 'a15', 'a1h', 'aD', 'since', 'nf', 'z21', 'a50']
def policy(G, p, thr, AD):
    """one position at a time (a candidate is skipped if it starts before the last trade's 60-min window ends), stop after the first target or 2 losses"""
    rows = []
    for d, g in G.assign(p=p).groupby('day', sort=True):
        busy = -1; wins = losses = 0
        for r in g.itertuples():
            if wins >= 1 or losses >= 2 or r.i <= busy or r.p < thr: continue
            rows.append((d, r.pts, r.y)); busy = r.i + 12
            if r.y: wins += 1
            elif r.pts <= 0: losses += 1
    T = pd.DataFrame(rows, columns=['day', 'pts', 'y']); P = T.groupby('day').pts.sum().reindex(AD).fillna(0.0); Q = T.groupby('day').y.max().reindex(AD).fillna(0)
    return T, P, Q
if __name__ == '__main__':
    print(f'5m flip candidates {len(C5)} ({len(C5) / C5.day.nunique():.1f} a day); target hit {C5.y.mean():.0%}; avg {C5.pts.mean():+.2f} pts; bracket median 2018-21 {C5[C5.yr < 2022].tgt.median():.0f} / 2022-26 {C5[C5.yr >= 2022].tgt.median():.0f} pts', flush=True)
    PRED = pd.Series(np.nan, index=C5.index); PL = pd.Series(np.nan, index=C5.index); THR = {}
    for Y in range(2020, 2027):
        trn = C5.yr < Y; tst = C5.yr == Y
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=0)
        m.fit(C5.loc[trn, FE], C5.loc[trn, 'y']); PRED[tst] = m.predict_proba(C5.loc[tst, FE])[:, 1]; ptr = m.predict_proba(C5.loc[trn, FE])[:, 1]
        THR[Y] = {x: np.quantile(ptr, 1 - x) for x in (.1, .2, .3)}
        Xf = C5.loc[:, FE].fillna(C5.loc[trn, FE].median()); mu, sd = Xf[trn].mean(), Xf[trn].std().replace(0, 1)
        lr_ = LogisticRegression(C=.1, max_iter=3000).fit(((Xf[trn] - mu) / sd).values, C5.loc[trn, 'y']); PL[tst] = lr_.predict_proba(((Xf[tst] - mu) / sd).values)[:, 1]
        imp = pd.Series(m.feature_importances_, index=FE).sort_values(ascending=False)
        print(f'  {Y}: train {trn.sum()}, test {tst.sum()}, AUC LGBM {roc_auc_score(C5.y[tst], PRED[tst]):.3f} logistic {roc_auc_score(C5.y[tst], PL[tst]):.3f} | top features ' + ', '.join(imp.index[:5]), flush=True)
    te = C5.yr >= 2020; print(f'\n  pooled 2020-26 AUC: LGBM {roc_auc_score(C5.y[te], PRED[te]):.3f}, logistic {roc_auc_score(C5.y[te], PL[te]):.3f}', flush=True)
    q = pd.qcut(PRED[te].rank(method='first'), 5, labels=False); g5 = C5[te].assign(q=q.values)
    print('  by LGBM quintile (2020-26): ' + ' | '.join(f'Q{k + 1} hit {g.y.mean():.0%} avg {g.pts.mean():+.2f}' for k, g in g5.groupby('q')), flush=True)
    AD = pd.DatetimeIndex(sorted(C5.day.unique())); AD = AD[AD >= '2020-01-01']
    print('\n  POLICY (1 contract)            trades/day  hit   pts/trade  pts/day  qualifying days/month | by year: qual days/mo, pts/day', flush=True)
    base_T, base_P, base_Q = policy(C5[te], np.ones(te.sum()), 0, AD)
    for lab, (T_, P_, Q_) in [('every flip (no model)', (base_T, base_P, base_Q))]:
        print(f'  {lab:<30} {len(T_) / len(AD):10.2f} {T_.y.mean():5.0%} {T_.pts.mean():+10.2f} {P_.mean():+8.2f} {Q_.sum() / len(AD) * 21:21.1f} | ' +
              ' '.join(f'{y} {Q_[Q_.index.year == y].mean() * 21:.1f}/{P_[P_.index.year == y].mean():+.1f}' for y in range(2020, 2027)), flush=True)
    for x in (.1, .2, .3):
        thr = np.array([THR[y][x] for y in C5.yr[te]]); pp = PRED[te].values - thr
        T_, P_, Q_ = policy(C5[te], pp, 0, AD)
        print(f'  LGBM top {int(x * 100)}%{"":<21} {len(T_) / len(AD):10.2f} {T_.y.mean():5.0%} {T_.pts.mean():+10.2f} {P_.mean():+8.2f} {Q_.sum() / len(AD) * 21:21.1f} | ' +
              ' '.join(f'{y} {Q_[Q_.index.year == y].mean() * 21:.1f}/{P_[P_.index.year == y].mean():+.1f}' for y in range(2020, 2027)), flush=True)
    C5.assign(p_lgbm=PRED, p_log=PL).to_parquet('/tmp/claude-0/ml/nq_meta5.parquet')
