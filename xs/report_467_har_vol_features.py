"""Report 467 (Nov 24 2026): Rey - 'ATRP over HAR and stuff like that'. Does volatility RELATIVE to its own forecast (HAR) and other vol-state features separate the flips that run, beyond plain ATRP?
DECLARED BEFORE RESULTS. Data: daily OHLCV of the 1,656-name wide universe (scratchpad/ext/wide_daily_ohlcv.pkl) at each flip's close (structure_small flips, px < $20, 2019-2026Q1; same target definitions as reports 465/466).
Daily range variance proxy rv = ln(H/L)^2 / (4 ln 2) (Parkinson). HAR forecast of next-day log rv = pooled OLS on [log rv_t, log mean rv(5d), log mean rv(22d)] fitted on TRAIN (2019-21) rows only, frozen; HARv = exp(forecast/2) = forecast daily range-vol.
Features (all at the flip close): ATRP_HAR = ATRP / HARv (current ATR vs forecast); SURP = today's range / yesterday's HAR forecast (volatility surprise); RV5_22 = rv5/rv22 (short vs long); RV22_252 = rv22 / rv252 (vol regime vs own year); VOV = std of log rv over 22d (vol of vol); GK22 = Garman-Klass vol 22d; CC_RANGE = |close-to-close| 22d / range 22d (efficiency); GAPABS20 = mean |overnight gap| 20d (overnight activity); UPSH = share of up-days 22d; DVOL_Z = log dollar volume 5d vs 60d; plus HARv itself and ATRP for reference.
Test: AUC per feature (train | test) for BIG (r >= +30%), r >= +10% and REACH (peak >= +10%); direction read from train must hold in test; PASS = |AUC-.5| >= .04 in both halves, same side, test z >= 3.3 (13 features x 3 targets). Then incremental: does adding the passing NEW features (sign from train) to the report-466 six-feature score raise test P(r >= 10%) and the median in the top 10% group (train cut), vs the six-feature score and ATRP alone?"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import rankdata
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
W = pd.read_pickle(SP + 'wide_daily_ohlcv.pkl'); O = pd.read_pickle(SP + 'structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O = O[O.px < 20].copy().reset_index(drop=True)
O['tr'] = (O.d.dt.year <= 2021).values; O['mon'] = O.d.dt.strftime('%Y-%m'); O['era3'] = np.select([O.d.dt.year <= 2021, O.d.dt.year == 2022], ['2019-21', '2022'], '2023-26')
def feats(df):
    df = df[~df.index.duplicated()].sort_index(); h, l, o, c, v = df.high.astype(float), df.low.astype(float), df.open.astype(float), df.close.astype(float), df.volume.astype(float)
    rv = (np.log(h / l) ** 2 / (4 * np.log(2))).replace(0, np.nan); lr = np.log(rv); F = pd.DataFrame(index=df.index)
    F['lr1'] = lr; F['lr5'] = np.log(rv.rolling(5).mean()); F['lr22'] = np.log(rv.rolling(22).mean()); F['lr_next'] = lr.shift(-1)
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2; F['GK22'] = np.sqrt(gk.rolling(22).mean().clip(lower=0))
    F['VOV'] = lr.rolling(22).std(); F['RV5_22'] = rv.rolling(5).mean() / rv.rolling(22).mean(); F['RV22_252'] = rv.rolling(22).mean() / rv.rolling(252, min_periods=120).mean()
    cc = np.log(c / c.shift(1)); F['CC_RANGE'] = cc.abs().rolling(22).mean() / np.sqrt(rv).rolling(22).mean(); F['GAPABS20'] = np.log(o / c.shift(1)).abs().rolling(20).mean()
    F['UPSH'] = (cc > 0).rolling(22).mean(); dv = np.log((c * v).clip(lower=1)); F['DVOL_Z'] = dv.rolling(5).mean() - dv.rolling(60).mean(); F['rng_today'] = np.sqrt(rv)
    return F
FT = {tk: feats(W[tk]) for tk in O.tk.unique() if tk in W}
allf = pd.concat([f.assign(tk=tk) for tk, f in FT.items()]); allf['d'] = allf.index; trn_mask = allf.d.dt.year.between(2019, 2021) & allf[['lr1', 'lr5', 'lr22', 'lr_next']].notna().all(axis=1)
X = allf[trn_mask][['lr1', 'lr5', 'lr22']].values; y = allf[trn_mask].lr_next.values; Xb = np.c_[np.ones(len(X)), X]; beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
print('HAR fit (train, pooled): n', len(X), 'coef (const, d, w, m):', np.round(beta, 3), 'R2', round(1 - ((y - Xb @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3))
rows = []
for tk, g in O.groupby('tk'):
    if tk not in FT: continue
    F = FT[tk]; hf = beta[0] + beta[1] * F.lr1 + beta[2] * F.lr5 + beta[3] * F.lr22; F = F.assign(HARv=np.exp(hf / 2), HARv_prev=np.exp(hf.shift(1) / 2))
    m = F.reindex(g.d.values); m.index = g.index; rows.append(m)
M = pd.concat(rows); O = O.join(M[['GK22', 'VOV', 'RV5_22', 'RV22_252', 'CC_RANGE', 'GAPABS20', 'UPSH', 'DVOL_Z', 'HARv', 'HARv_prev', 'rng_today']])
O['ATRP_HAR'] = O.ATRP / O.HARv; O['SURP'] = O.rng_today / O.HARv_prev; O['PEAKv'] = O.PEAK
NEW = ['ATRP_HAR', 'SURP', 'RV5_22', 'RV22_252', 'VOV', 'GK22', 'CC_RANGE', 'GAPABS20', 'UPSH', 'DVOL_Z', 'HARv']; REF = ['ATRP']
def auc(x, y):
    m = ~np.isnan(x) & np.isfinite(x); x, y = x[m], y[m]
    if y.sum() < 15 or (1 - y).sum() < 15 or np.std(x) == 0: return np.nan, np.nan
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1; a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0); return a, (a - .5) / np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
T = {'BIG>=30%': (O.r >= .3).values.astype(float), 'r>=10%': (O.r >= .1).values.astype(float), 'REACH>=10%': (O.PEAKv >= .1).values.astype(float)}; tr = O.tr.values; res = {}
print('flips', len(O), 'train', int(tr.sum()), 'test', int((~tr).sum()))
for tgt, yv in T.items():
    print(f'\n=== {tgt}: AUC train | test (test z), same side, PASS ===')
    for f in REF + NEW:
        x = O[f].astype(float).values; a1, z1 = auc(x[tr], yv[tr]); a2, z2 = auc(x[~tr], yv[~tr]); ok = (a1 - .5) * (a2 - .5) > 0 and abs(a1 - .5) >= .04 and abs(a2 - .5) >= .04 and abs(z2) >= 3.3; res[(tgt, f)] = (a1, a2, ok)
        print(f'  {f:10s} {a1:.3f} | {a2:.3f} (z {z2:+5.1f}) {"same" if (a1 - .5) * (a2 - .5) > 0 else "FLIPS"} {"PASS" if ok else ""}')
# incremental: add passing NEW features (BIG or REACH) to the 466 six-feature score
src = open('xs/report_465_mover_features.py').read(); 
add = [f for f in NEW if any(res[(t, f)][2] for t in ('BIG>=30%', 'REACH>=10%'))]; print('\nnew features passing (BIG or REACH):', add)

# ---- stage 2 (declared after stage 1 results, before running): do the new vol features ADD to the structure_small part of the report-466 score? ----
# base5 = ATRP(+), F11(+), VOLX(+), LTURN(+), LMC(-) (the 466 features available here; runway needs SEC data); NEWV = base5 + GAPABS20(+, overnight gap activity) + SURP(+) + ATRP_HAR(-, ATR low vs forecast) + VOV(+). Train-ECDF ranks, frozen cuts, top 10% group, test 2022+.
def score(feats):
    R = pd.DataFrame(index=O.index)
    for f, s in feats.items():
        x = O[f].astype(float); ref = np.sort(x[tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
b5 = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
variants = {'ATRP only': {'ATRP': 1}, 'base5': b5, 'base5 + GAPABS20': {**b5, 'GAPABS20': 1}, 'base5 + SURP': {**b5, 'SURP': 1}, 'base5 + ATRP_HAR(-)': {**b5, 'ATRP_HAR': -1}, 'base5 + VOV': {**b5, 'VOV': 1}, 'NEWV (all four added)': {**b5, 'GAPABS20': 1, 'SURP': 1, 'ATRP_HAR': -1, 'VOV': 1}}
te = O[~tr].copy(); trn = O[tr]; print('\n=== stage 2: top 10% group (train cut), test 2022+; px<10 | px<20 ===')
for lab, fs in variants.items():
    sc = score(fs); O['s_'] = sc; out = []
    for sub, nm in ((lambda D: D[D.px < 10], 'px<10'), (lambda D: D, 'px<20')):
        T_ = sub(O[~tr]).dropna(subset=['s_']); cut = np.quantile(sub(O[tr]).s_.dropna(), .9); a = T_[T_.s_ >= cut]
        out.append(f"{nm}: n {len(a):4d} REAL {(a.r >= .1).mean():4.0%} REACH {(a.PEAKv >= .1).mean():4.0%} mean {a.r.mean() * 100:+5.1f}% med {a.r.median() * 100:+5.1f}% [2022 {(a[a.era3 == '2022'].PEAKv >= .1).mean():.0%} | 23-26 {(a[a.era3 == '2023-26'].PEAKv >= .1).mean():.0%} REACH]")
    print(f'{lab:24s} ' + ' || '.join(out))
