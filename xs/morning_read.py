"""MORNING READ: nonlinear analog read of the day for SPY / QQQ / IWM (daily bars, VIX). Usage: python3 morning_read.py [YYYY-MM-DD]
Features known at the open: gap (signed, in 20d-range units), |gap| rank, HAR (log range d/w/m), Yang-Zhang 20d, VIX level / 5d chg / open gap, ret5, ret20, dist50, slope20, ADX, BB-width rank, TTM squeeze, vol1, vol5, pos20.
Analogs: Gaussian-kernel kNN (k=40) on robust-z features over PRIOR days only. Outputs conditional distribution of the day (range, up/down excursion, close, close location, gap fill) vs baseline,
plus a walk-forward track record (what the analog read has and has not predicted) so the story is not oversold."""
import sys, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
TARGET = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else None
def dl(s):
    d = yf.download(s, start='2000-01-01', progress=False, auto_adjust=False); d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]; d.index = pd.to_datetime(d.index).normalize(); return d
vix = dl('^VIX'); FEATS = ['gap', 'gabs', 'har_d', 'har_w', 'har_m', 'yz', 'vix', 'vix5', 'vixgap', 'ret5', 'ret20', 'dist50', 'slope20', 'adx', 'bbw', 'ttm', 'vol1', 'vol5', 'pos20']
def build(sym):
    d = dl(sym); O, H, L, C, V = d.Open, d.High, d.Low, d.Close, d.Volume.astype(float); PC = C.shift()
    rg = H - L; lrg = np.log((rg / O).replace(0, np.nan)); rng20 = rg.rolling(20).mean().shift()
    tr = pd.concat([rg, (H - PC).abs(), (L - PC).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    F = pd.DataFrame(index=d.index); x = {}
    # prior-close features (shifted 1 so row t uses data through t-1)
    P = pd.DataFrame(index=d.index)
    P['har_d'] = lrg; P['har_w'] = np.log((rg / O).rolling(5).mean()); P['har_m'] = np.log((rg / O).rolling(22).mean())
    n = 20; kz = .34 / (1.34 + (n + 1) / (n - 1))
    P['yz'] = np.log(O / PC).rolling(n).var() + kz * np.log(C / O).rolling(n).var() + (1 - kz) * (np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)).rolling(n).mean(); P['yz'] = np.log(P.yz.clip(lower=1e-9))
    v = vix.Close.reindex(d.index).ffill(); P['vix'] = np.log(v); P['vix5'] = np.log(v).diff(5)
    P['ret5'] = C.pct_change(5); P['ret20'] = C.pct_change(20); P['dist50'] = C / C.rolling(50).mean() - 1
    xm = np.arange(20) - 9.5; P['slope20'] = C.rolling(20).apply(lambda y: (xm * (y - y.mean())).sum() / (xm ** 2).sum(), raw=True) / atr
    up = H.diff(); dn = -L.diff(); pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), d.index); mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), d.index)
    pdi = 100 * pdm.rolling(14).mean() / atr; mdi = 100 * mdm.rolling(14).mean() / atr; P['adx'] = (100 * (pdi - mdi).abs() / (pdi + mdi)).rolling(14).mean()
    sd = C.rolling(20).std(); P['bbw'] = (4 * sd / C.rolling(20).mean()).rolling(126, min_periods=60).rank(pct=True); P['ttm'] = (2 * sd < 1.5 * atr.rolling(20).mean()).astype(float)
    P['vol1'] = V / V.rolling(20).mean(); P['vol5'] = V.rolling(5).mean() / V.rolling(20).mean(); P['pos20'] = (C - L.rolling(20).min()) / (H.rolling(20).max() - L.rolling(20).min())
    F = P.shift(1)
    F['gap'] = (O - PC) / rng20; F['gabs'] = F.gap.abs(); F['vixgap'] = np.log(vix.Open.reindex(d.index)) - np.log(vix.Close.shift().reindex(d.index))
    F['gabs_rank'] = F.gabs.rolling(252, min_periods=120).rank(pct=True)
    Y = pd.DataFrame(index=d.index); Y['range'] = rg / O * 100; Y['up'] = (H - O) / O * 100; Y['dn'] = (O - L) / O * 100; Y['ret'] = (C - O) / O * 100; Y['loc'] = (C - L) / rg
    Y['fill'] = np.where(O > PC, (L <= PC).astype(float), (H >= PC).astype(float)); Y['rngx'] = rg / rng20; Y['gapfill_n'] = (O != PC)
    return d, F, Y
def wq(x, w, q): i = np.argsort(x); cw = np.cumsum(w[i]) / w.sum(); return np.interp(q, cw, x[i])
def knn(F, Y, t, k=40, yrs_min=3):
    cols = FEATS; X = F[cols]; past = X.index < t; Xp = X[past].dropna(); 
    if t not in X.index or X.loc[t].isna().any(): cols = [c for c in cols if not np.isnan(X.loc[t, c])] if t in X.index else cols; Xp = X.loc[X.index < t, cols].dropna()
    med = Xp.median(); mad = (Xp - med).abs().median() * 1.4826 + 1e-9; Z = ((Xp - med) / mad).clip(-4, 4); z = ((X.loc[t, cols] - med) / mad).clip(-4, 4)
    dist = np.sqrt(((Z.values - z.values) ** 2).mean(axis=1)); idx = np.argsort(dist)[:k]; h = np.median(dist[idx]) + 1e-9; w = np.exp(-(dist[idx] / h) ** 2)
    return Xp.index[idx], w, cols, z, dist[idx]
def report(sym, t):
    d, F, Y = build(sym)
    if t is None: t = d.index[-1]
    dts, w, cols, z, dist = knn(F, Y.drop(columns=['gapfill_n']), t)
    A = Y.loc[dts]; B = Y[(Y.index < t) & (Y.index >= t - pd.Timedelta(days=365 * 6))].dropna()
    f = F.loc[t]; out = [f'## {sym}  ({t.date()})']
    gr = f.gabs_rank; gs = 'LOW gap (bottom third)' if gr <= .33 else 'HIGH gap (top third)' if gr >= .67 else 'mid gap'
    out.append(f'- Open: gap {f.gap:+.2f} x 20d range, |gap| rank {gr:.2f} -> {gs}; VIX gap {f.vixgap*100:+.1f}%, VIX {np.exp(f.vix):.1f} (5d {f.vix5*100:+.1f}%)')
    tags = [];  tags.append('TTM squeeze ON' if f.ttm == 1 else 'no squeeze'); tags.append(f'BB-width rank {f.bbw:.2f}'); tags.append(f'ADX {f.adx:.0f}'); tags.append(f'20d slope {f.slope20:+.2f} ATR/day'); tags.append(f'vol vs 20d {f.vol1:.2f}x'); tags.append(f'close in 20d range {f.pos20:.2f}')
    out.append('- Prior state: ' + ', '.join(tags))
    mu = lambda c: np.average(A[c], weights=w)
    rows = [('range % of open', 'range'), ('up excursion % (H-O)', 'up'), ('down excursion % (O-L)', 'dn'), ('close-open %', 'ret')]
    out.append(f'- Analogs (k={len(dts)}, nearest {dts[0].date()}, {dts[1].date()}, {dts[2].date()}...; median dist {np.median(dist):.2f}):')
    for lab, c in rows:
        out.append(f'    {lab:24} analog p25/50/75 {wq(A[c].values,w,.25):5.2f}/{wq(A[c].values,w,.5):5.2f}/{wq(A[c].values,w,.75):5.2f}   | baseline {B[c].quantile(.25):5.2f}/{B[c].median():5.2f}/{B[c].quantile(.75):5.2f}')
    pu = np.average(A.ret > 0, weights=w); pb = (B.ret > 0).mean(); fl = np.average(A.fill, weights=w); fb = B.fill.mean()
    out.append(f'    P(close>open) {pu:.0%} vs base {pb:.0%} | closes in top quarter of range {np.average(A["loc"]>.75,weights=w):.0%} vs {(B["loc"]>.75).mean():.0%}, bottom quarter {np.average(A["loc"]<.25,weights=w):.0%} vs {(B["loc"]<.25).mean():.0%} | gap fill {fl:.0%} vs {fb:.0%}')
    ratio = wq(A.range.values, w, .5) / B.range.median(); out.append(f'    SIZE READ: expected range {ratio:.2f}x a normal day' + (' (big day)' if ratio > 1.25 else ' (quiet day)' if ratio < .8 else ''))
    return d, F, Y, out, ratio
def track(sym):
    d, F, Y = build(sym); ids = [t for t in F.index if t >= pd.Timestamp('2012-01-01') and not F.loc[t, FEATS].isna().any()][::3]
    pr, rr, pd_, ud, base_r = [], [], [], [], []
    X = F[FEATS]; med = X[X.index < '2012-01-01'].median(); mad = (X[X.index < '2012-01-01'] - med).abs().median() * 1.4826 + 1e-9; Z = ((X - med) / mad).clip(-4, 4)
    Zv = Z.values; ix = {t: i for i, t in enumerate(F.index)}; lr = np.log(Y.range.clip(lower=1e-4)).values; rv = Y.ret.values; ok = ~np.isnan(Zv).any(axis=1) & ~np.isnan(lr)
    har = F[['har_d', 'har_w', 'har_m']].values
    for t in ids:
        i = ix[t]; pm = ok.copy(); pm[i:] = False; pidx = np.where(pm)[0]
        if len(pidx) < 500: continue
        dd = np.sqrt(((Zv[pidx] - Zv[i]) ** 2).mean(axis=1)); j = np.argsort(dd)[:40]; h = np.median(dd[j]) + 1e-9; w = np.exp(-(dd[j] / h) ** 2); pj = pidx[j]
        pr.append(np.average(lr[pj], weights=w)); rr.append(lr[i]); pd_.append(np.average(rv[pj] > 0, weights=w)); ud.append(rv[i] > 0)
        # HAR-only benchmark: linear fit on prior 1000 days
        base_r.append(i)
    pr, rr, pd_, ud = map(np.array, (pr, rr, pd_, ud)); ids = ids[:len(pr)]
    # HAR linear benchmark, expanding refit each year
    from sklearn.linear_model import LinearRegression
    hp = []
    for t in ids:
        i = ix[t]; m = ok.copy(); m[i:] = False; m &= ~np.isnan(har).any(axis=1)
        pass
    cr = np.corrcoef(pr, rr)[0, 1]
    # HAR-only: single fit on <2012
    m0 = ok & (F.index < '2012-01-01') & ~np.isnan(har).any(axis=1); lm = LinearRegression().fit(har[m0], lr[m0]); hh = lm.predict(np.nan_to_num(har[[ix[t] for t in ids]])); ch = np.corrcoef(hh, rr)[0, 1]
    # top/bottom decile direction
    q = np.quantile(pd_, [.1, .9]); lo = ud[pd_ <= q[0]].mean(); hi = ud[pd_ >= q[1]].mean()
    return f'{sym}: n={len(pr)} test days since 2012 | corr(predicted, realized log range): analog {cr:.2f} vs HAR-only {ch:.2f} | direction: analog top-decile P(up) realized {hi:.0%}, bottom-decile {lo:.0%}, all {ud.mean():.0%}'
if __name__ == '__main__':
    allr = {}
    for s in ['QQQ', 'SPY', 'IWM']:
        d, F, Y, out, ratio = report(s, TARGET); allr[s] = (F.loc[TARGET or d.index[-1]], ratio); print('\n'.join(out), '\n')
    print('## TRACK RECORD (walk-forward, every 3rd day since 2012)')
    for s in ['QQQ', 'SPY', 'IWM']: print(' ', track(s))
