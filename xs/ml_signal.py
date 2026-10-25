"""Standalone ML signal (Oct 25 2026): gradient-boosted trees (LightGBM) with two heads on every 15m bar
  classifier  P(next ~6.5 hours return > +0.25%)        regressor  expected next ~6.5 hours return
State: bull when P is in the top 20% of the training fold's predictions AND the regressor is positive; bear when P falls below the
training median; otherwise unchanged (hysteresis). The 1h track uses the mean P over each hour. Trades go through the live engine
(xs/signal_lab.run: 15m + 1h tracks, sticky 1h stop, LOCK5/0.5, entries at RTH opens), then the live book (stage 2 of signal_lab).
Env: H_FWD (bars ahead, 26 = one premarket+RTH day), THR (label threshold), Q_ON / Q_OFF (entry / exit quantiles), VAR (trade label).
Walk-forward by year: train on every year before Y (with a gap), predict Y, Y = 2019..2026. Pooled across the 54 live names.
  python3 xs/ml_signal.py BARS_DIR OUT_DIR        (features -> walk-forward -> trades OUT_DIR/<tk>.parquet, var 'ml')"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])   # load, agg, known, run
H_FWD, THR = int(os.getenv('H_FWD', 26)), float(os.getenv('THR', .0025)); Q_ON, Q_OFF = float(os.getenv('Q_ON', .8)), float(os.getenv('Q_OFF', .5)); VAR = os.getenv('VAR', 'ml')
BARS, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
CORE = sorted(set(pd.read_parquet(sys.argv[3] if len(sys.argv) > 3 else 'xs/ext/wide_T.parquet').query('sp and ratio >= 1.5').tk))

def features(H):
    c, h, l, o, v = H.close, H.high, H.low, H.open, H.volume.astype(float)
    lr = np.log(c).diff(); vol = lr.rolling(260, min_periods=60).std()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs())); atr = tr.rolling(14).mean()
    F = pd.DataFrame(index=H.index)
    for k in (1, 4, 16, 64, 130): F[f'r{k}'] = np.log(c / c.shift(k)) / (vol * np.sqrt(k))
    for k in (21, 55, 200): F[f'ema{k}'] = (c - c.ewm(span=k, adjust=False).mean()) / atr
    for k in (20, 55, 260):
        hi, lo = h.rolling(k).max(), l.rolling(k).min(); F[f'dch{k}'] = (c - lo) / (hi - lo)
    F['volr'] = lr.rolling(26).std() / vol
    slot = H.m; F['rvol'] = v / v.groupby(slot).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    F['tod'] = H.m; F['pre'] = (H.m < 570).astype(float)
    day = H.dt.dt.normalize(); rth = H.m >= 570
    pclose = c.where(H.m == 945).ffill().shift(1)                                    # prior RTH close
    F['day_ret'] = np.log(c / pclose) / vol / np.sqrt(26)
    dh, dl = h.groupby(day).cummax(), l.groupby(day).cummin()
    rng = (np.log(h.where(rth).groupby(day).transform('max') / l.where(rth).groupby(day).transform('min')))
    typ = pd.Series(rng.groupby(day).first()).rolling(22, min_periods=5).median().shift(1).reindex(day).values
    F['rng_used'] = np.log(dh / dl) / typ
    F['body'] = (c - o) / atr; F['wick'] = (h - np.maximum(o, c) - (np.minimum(o, c) - l)) / atr
    fwd = np.log(c.shift(-H_FWD) / c)
    return F.astype('float32'), fwd

def main():
    fs = {os.path.basename(f).split('_15m_')[0]: f for f in glob.glob(f'{BARS}/*_15m_*.parquet')}
    D = {}
    for tk in CORE:
        if tk not in fs: continue
        H = load(fs[tk]); F, y = features(H); F['tk_id'] = CORE.index(tk); D[tk] = (H, F, y)
    cols = [c for c in D[CORE[0]][1].columns if c != 'tk_id']
    print(f'{len(D)} names, {sum(len(x[1]) for x in D.values()):,} bars, {len(cols)} features', flush=True)
    P = {tk: np.full(len(x[0]), np.nan) for tk, x in D.items()}; R = {tk: np.full(len(x[0]), np.nan) for tk, x in D.items()}; TH = {}
    for Y in range(2019, 2027):
        t0 = pd.Timestamp(f'{Y}-01-01'); gap = t0 - pd.Timedelta(days=3)
        Xs, ys = [], []
        for tk, (H, F, y) in D.items():
            m = (H.dt < gap).values & y.notna().values & F[cols].notna().all(axis=1).values
            idx = np.where(m)[0][::3]; Xs.append(F[cols].values[idx]); ys.append(y.values[idx])
        X, yy = np.vstack(Xs), np.concatenate(ys)
        pc = dict(n_estimators=300, learning_rate=.05, num_leaves=63, min_child_samples=500, subsample=.7, subsample_freq=1, colsample_bytree=.8, verbose=-1, n_jobs=2)
        clf = lgb.LGBMClassifier(**pc).fit(X, (yy > THR).astype(int))
        reg = lgb.LGBMRegressor(**pc).fit(X, np.clip(yy, -.1, .1))
        ptr = clf.predict_proba(X[::20])[:, 1]; TH[Y] = (np.quantile(ptr, Q_ON), np.quantile(ptr, Q_OFF))
        from sklearn.metrics import roc_auc_score
        aa, ics = [], []
        for tk, (H, F, y) in D.items():
            m = (H.dt.dt.year == Y).values
            if not m.any(): continue
            Xt = F[cols].values[m]; ok = ~np.isnan(Xt).any(axis=1); p = np.full(m.sum(), np.nan); r = np.full(m.sum(), np.nan)
            if ok.any(): p[ok] = clf.predict_proba(Xt[ok])[:, 1]; r[ok] = reg.predict(Xt[ok])
            P[tk][m] = p; R[tk][m] = r
            yt = y.values[m]; k = ok & ~np.isnan(yt)
            if k.sum() > 500 and len(set((yt[k] > THR).astype(int))) == 2:
                aa.append(roc_auc_score((yt[k] > THR).astype(int), p[k])); ics.append(pd.Series(r[k]).corr(pd.Series(yt[k]), method='spearman'))
        print(f'{Y}: trained on {len(yy):,} rows | out-of-sample AUC {np.mean(aa):.3f} | regression rank IC {np.mean(ics):+.3f} | P cut {TH[Y][0]:.3f}/{TH[Y][1]:.3f}', flush=True)
        if Y == 2026:
            imp = pd.Series(clf.booster_.feature_importance('gain'), index=cols).sort_values(ascending=False)
            print('feature importance (classifier, gain share):', ', '.join(f'{k} {v / imp.sum():.0%}' for k, v in imp.head(8).items()), flush=True)
    for tk, (H, F, y) in D.items():
        p, r = P[tk], R[tk]; yr = H.dt.dt.year.values; st = np.zeros(len(H), int); s = 0
        for i in range(len(H)):
            if np.isnan(p[i]): st[i] = s; continue
            on, off = TH[yr[i]]
            if p[i] >= on and r[i] > 0: s = 1
            elif p[i] < off: s = -1
            st[i] = s
        hp = pd.Series(p, index=H.index); hr = H.assign(p=hp.values)
        x = hr.set_index('dt')[['p']]; g = x.groupby([x.index.normalize(), pd.Grouper(freq='1h', origin='start_day', offset='9h30min')]).p.mean().dropna()
        g.index = g.index.get_level_values(1); s1h = np.zeros(len(g), int); s = 0
        for i, (t, v) in enumerate(g.items()):
            on, off = TH.get(t.year, (np.inf, -np.inf))
            s = 1 if v >= on else (-1 if v < off else s); s1h[i] = s
        r1 = pd.DataFrame({'dt': g.index}); s1 = known(H, r1, s1h, '1h')
        rows = []
        for h1t in (False, True):
            T, _ = run(H, st, s1, h1t); rows += [(tk, VAR, h1t, 'trade') + t for t in T]
        pd.DataFrame(rows, columns=['tk', 'var', 'h1', 'kind', 'entry', 'exit', 'ret']).to_parquet(f'{OUT}/{tk}.parquet')
    print('trades written', flush=True)

main()
