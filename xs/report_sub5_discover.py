"""DISCOVERY on the sub-$5 vector flips: which entry-time features separate good from bad trades, judged strictly out of sample.
Universe: ALL sub-$5 1h bull flips (A, n~6000), Dec23-Nov26. Target: net 40bp return per trade. Everything below is known at the flip-bar close.
SPLIT: TRAIN = entries before 2025-07-01, TEST = after. Steps: (1) univariate Spearman IC on TRAIN with Benjamini-Hochberg; (2) which survive on TEST with the same sign; (3) a shallow gradient-boosted model fit on TRAIN only, scored on TEST (rank IC, tercile spread, hit rate, permutation importance); (4) 8-slot book on TEST using the TRAIN model's top half / top third of flips vs all."""
import sys, pickle, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb'))
E = pd.read_pickle(SC + 'earnmac_T2.pkl'); V = pd.read_pickle(SC + 'sub5_T.pkl')[['tk', 'et', 'vwx']]; T = E[E.sub5].merge(V, on=['tk', 'et'], how='left').sort_values(['tk', 'et']).reset_index(drop=True); COST = .004; K = 8
mac = yf.download(['IWM', '^VIX'], start='2023-01-01', progress=False, auto_adjust=True)['Close']; iw = mac['IWM'].dropna(); vx = mac['^VIX'].dropna(); iw20 = iw.pct_change(20).shift(1); vxl = vx.shift(1)
def atr14(h, l, c): pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(14).mean().values
rows = []
for tk, g in T.groupby('tk'):
    Yc = M.closed_only(data[tk]); F = M.features(Yc); r = F.reg.values; o, h, l, c, v = (Yc[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume')); at = atr14(h, l, c); idx = Yc.index; day = idx.normalize()
    D = pd.DataFrame({'o': pd.Series(o, index=idx).groupby(day).first(), 'h': pd.Series(h, index=idx).groupby(day).max(), 'l': pd.Series(l, index=idx).groupby(day).min(), 'c': pd.Series(c, index=idx).groupby(day).last(), 'v': pd.Series(v, index=idx).groupby(day).sum()})
    pcd = D.c.shift(1); dtr = np.maximum(D.h - D.l, np.maximum((D.h - pcd).abs(), (D.l - pcd).abs())); datr = dtr.rolling(14).mean().shift(1); e20 = D.c.ewm(span=20, adjust=False).mean().shift(1); gapd = D.o / pcd - 1
    flips = np.where((r[1:] != r[:-1]) & (r[1:] == 1))[0] + 1; prev_ret = np.nan; prev_t = None
    for _, t in g.iterrows():
        i = int(t.fi); p = D.index.searchsorted(day[i]); f = {}
        if p < 22: rows.append({'ix': _}); continue
        f['ret5'] = D.c.iat[p - 1] / D.c.iat[p - 6] - 1; f['ret20'] = D.c.iat[p - 1] / D.c.iat[p - 21] - 1; f['dist60h'] = c[i] / D.h.iloc[max(0, p - 60):p].max() - 1; f['dvol'] = np.log10((D.v.iloc[p - 20:p] * D.c.iloc[p - 20:p]).mean() + 1)
        f['gap2'] = max(gapd.iat[p], gapd.iat[p - 1]); f['ema20_d'] = (c[i] - e20.iat[p]) / datr.iat[p] if datr.iat[p] > 0 else np.nan; f['atr_pct'] = at[i] / c[i]; f['rng_atr'] = (h[i] - l[i]) / at[i]; f['body_atr'] = (c[i] - o[i]) / at[i]
        f['hour'] = t.et.hour; pf = flips[flips < i]; f['bars_since_flip'] = i - pf[-1] if len(pf) else np.nan; f['flips20d'] = int(((r[max(1, i - 140):i + 1] != r[max(0, i - 141):i]) ).sum()); f['px'] = t.epx; f['logpx'] = np.log(t.epx)
        dd = day[i]; f['vix'] = vxl.asof(dd); f['iwm20'] = iw20.asof(dd); f['rel20'] = f['ret20'] - (iw20.asof(dd) if iw20.asof(dd) == iw20.asof(dd) else 0)
        f['prev_lost'] = float(prev_ret < 0) if prev_ret == prev_ret else np.nan; f['days_since_prev'] = (t.et - prev_t) / np.timedelta64(1, 'D') if prev_t is not None else np.nan
        prev_ret = t.ret; prev_t = t.xt; f['ix'] = _; rows.append(f)
X = pd.DataFrame(rows).set_index('ix'); T = T.join(X); T['net'] = (T.ret - COST).clip(-.6, 1.5); T['logdvol'] = T.dvol
FEATS = ['vr', 'lv', 'vwx', 'dte', 'dse', 'surp', 'dil', 'ret5', 'ret20', 'dist60h', 'dvol', 'gap2', 'ema20_d', 'atr_pct', 'rng_atr', 'body_atr', 'hour', 'bars_since_flip', 'flips20d', 'logpx', 'vix', 'iwm20', 'rel20', 'm1', 'prev_lost', 'days_since_prev']
T['surp'] = T.surp.clip(-100, 100); T['dil'] = T.dil.clip(-.5, 2); T['lv'] = T.lv.astype(float)
for c_ in ('udv', 'cl'):
    pass
T = T[T.et >= '2023-12-01']; tr = T[T.et < '2025-07-01']; te = T[T.et >= '2025-07-01']; print('train', len(tr), 'test', len(te))
res = []
for f in FEATS:
    a = tr[[f, 'net']].dropna(); b = te[[f, 'net']].dropna()
    if len(a) < 100 or len(b) < 100: continue
    r1, p1 = spearmanr(a[f], a.net); r2, p2 = spearmanr(b[f], b.net); res.append((f, r1, p1, r2, p2, len(a), len(b)))
R = pd.DataFrame(res, columns=['feat', 'IC_train', 'p_train', 'IC_test', 'p_test', 'n_tr', 'n_te']).sort_values('p_train'); m = len(R); R['bh_ok'] = R.p_train <= (np.arange(1, m + 1) / m * .10); R['same_sign_test'] = np.sign(R.IC_train) == np.sign(R.IC_test)
print('\nUNIVARIATE (Spearman IC vs net return per trade)'); print(R.round(3).to_string(index=False))
# model on TRAIN only
Xtr = tr[FEATS].astype(float); Xte = te[FEATS].astype(float); mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=.04, max_iter=150, min_samples_leaf=60, l2_regularization=5., random_state=0).fit(Xtr, tr.net)
te = te.copy(); te['pred'] = mdl.predict(Xte); ic, p = spearmanr(te.pred, te.net); print(f'\nMODEL (trained on train only): TEST rank IC {ic:+.3f} (p {p:.3g}), n {len(te)}')
te['q'] = pd.qcut(te.pred.rank(method='first'), 3, labels=['bottom third', 'middle', 'top third']); print(te.groupby('q').net.agg(['count', 'mean', 'median', lambda s: (s > 0).mean(), lambda s: (s > .1).mean()]).round(3).rename(columns={'<lambda_0>': 'win', '<lambda_1>': 'P(>10%)'}))
b = te[te.vr >= 1.5].copy(); b['q'] = pd.qcut(b.pred.rank(method='first'), 3, labels=['bottom third', 'middle', 'top third']); print('\nwithin B (vol>=1.5x) only:'); print(b.groupby('q').net.agg(['count', 'mean', 'median', lambda s: (s > 0).mean()]).round(3))
rng = np.random.default_rng(1); imp = []
for f in FEATS:
    d = []
    for _ in range(8):
        Z = Xte.copy(); Z[f] = rng.permutation(Z[f].values); d.append(spearmanr(mdl.predict(Z), te.net)[0])
    imp.append((f, ic - np.mean(d)))
print('\npermutation importance on TEST (drop in rank IC when shuffled):'); print(pd.DataFrame(imp, columns=['feat', 'dIC']).sort_values('dIC', ascending=False).head(10).round(4).to_string(index=False))
def sim(d):
    rws = sorted(d.to_dict('records'), key=lambda r: (r['et'], -(r['vr'] if r['vr'] == r['vr'] else 0))); eq = 1.0; op = []; cv = [1.0]
    def close(p):
        nonlocal eq
        eq += p['sz'] * (p['ret'] - COST); cv.append(eq)
    for r in rws:
        st = []
        for p in op:
            if p['xt'] <= r['et']: close(p)
            else: st.append(p)
        op = st
        if len(op) < K and r['tk'] not in [p['tk'] for p in op]: r['sz'] = eq / K; op.append(r)
    for p in op: close(p)
    cv = np.array(cv); return eq - 1, (cv / np.maximum.accumulate(cv) - 1).min()
yrs = (pd.Timestamp('2026-11-10') - pd.Timestamp('2025-07-01')).days / 365; ann = lambda r: ((1 + r) ** (1 / yrs) - 1) * 100 if r > -1 else -100
print('\nTEST-period 8-slot book (Jul25-Nov26), model chosen on train only:')
for nm, d in (('all sub-$5 flips', te), ('B: vol>=1.5x', te[te.vr >= 1.5]), ('model top half', te[te.pred >= te.pred.median()]), ('model top third', te[te.q == 'top third']), ('B & model top half', b[b.pred >= b.pred.median()]), ('B & E+V (hand rule)', te[(te.vr >= 1.5) & ~(te.dte <= 3) & ~(te.vwx > 1)])):
    r, dd = sim(d); print(f'   {nm:24s} n {len(d):5d}  return {r * 100:+7.0f}% (ann {ann(r):+5.0f}%)  maxDD {dd * 100:5.0f}%  mean/trade {(d.ret - COST).mean() * 100:+.2f}%')
