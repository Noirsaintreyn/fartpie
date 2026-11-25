"""Report 477 (Nov 24 2026): Rey - 'test both' (high-ATRP high-probability group on the DAILY vector and on the 1h vector), with the manual-exit rule he described: SELL at +10% if it gets there, otherwise ride to the vector's own bear flip.
DECLARED BEFORE RESULTS.
DAILY (report 466/468 setup): flips 2019-26Q1 from structure_small, px<20/<10/<5, base5 score (ATRP,F11,VOLX,LTURN,-LMC) with frozen train-ECDF (2019-21), group = top 30% and top 10% by score; test 2022+ only is reported. Trades reconstructed from daily OHLCV (entry next open, exit = the open reproducing r, 98-100% reconcile as in 468).
1h (live book setup): flips from earnmac_T2 (Dec 2023-Nov 2026), base rules E+V (skip <=3 sessions before earnings, skip >1 ATR above VWAP), no volume gate; score = mean rank of ATRP (daily ATR14 / close, day before the flip, +), VOLX (flip-bar volume ratio, +) and LDV (log 20d avg dollar volume, - = small/illiquid), ECDF cuts frozen from H1 (Dec23-Mar25), group = top 30% by score, test H2 (Apr25-Nov26) reported with H1 beside it. Same-day entry = next bar open; the path is the 1h bars from entry to the vector exit.
Metrics per timeframe x price bucket x group: n, trades/yr, REACH (touch +10% before the vector exit), days to reach (median), win rate / mean / median of 'SELL +10% else vector exit' (costs: 30bp daily, 40bp 1h; fill assumed exactly at +10%, no gap benefit), the same for the plain vector exit for comparison, and the average days held under each rule."""
import numpy as np, pandas as pd, pickle, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
# ---------------- DAILY ----------------
src = open('xs/report_467_har_vol_features.py').read(); exec(src[:src.index("# ---- stage 2")].replace("print(f'\\n=== {tgt}", "(lambda *a, **k: None)(f'\\n=== {tgt}").replace("print(f'  {f:10s}", "(lambda *a, **k: None)(f'  {f:10s}"))
def score(feats):
    R = pd.DataFrame(index=O.index)
    for f, s in feats.items():
        x = O[f].astype(float); ref = np.sort(x[tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
O['s_'] = score({'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}); CD = .003
def recon(r):
    df = W[r.tk]; df = df[~df.index.duplicated()]; ix = df.index; j = ix.searchsorted(pd.Timestamp(r.d))
    if j >= len(ix) or ix[j] != pd.Timestamp(r.d) or j + 2 >= len(ix): return None
    e = float(df.open.iloc[j + 1]); tgt = 1 + r.r + .003; ope = df.open.values[j + 1:j + 400] / e; hi = np.maximum.accumulate(df.high.values[j + 1:j + 400] / e)
    for k in range(1, len(ope)):
        if abs(ope[k] - tgt) < .002 and abs((hi[k - 1] - 1) - r.PEAK) < .002:
            hv = df.high.values[j + 1:j + 1 + k] / e; h10 = np.argmax(hv >= 1.10) if (hv >= 1.10).any() else -1
            return k, h10
    return None
D_rows = []
for cap in (20, 10, 5):
    for pct, nm in ((.7, 'top30%'), (.9, 'top10%')):
        S = O[(O.px < cap) & (~O.tr)]; cut = np.quantile(O[(O.px < cap) & O.tr].s_.dropna(), pct); G = S[S.s_ >= cut].copy(); out = [recon(r) for r in G.itertuples()]; G['ok'] = [o is not None for o in out]; G['hold'] = [o[0] if o else np.nan for o in out]; G['t10'] = [o[1] if o else np.nan for o in out]; K_ = G[G.ok]
        reach = K_.t10 >= 0; alt = np.where(reach, .10 - CD, K_.r); days_alt = np.where(reach, K_.t10 + 1, K_.hold); yrs = (K_.d.max() - K_.d.min()).days / 365.25
        D_rows.append(('DAILY', f'sub-${cap}', nm, len(K_), len(K_) / yrs, reach.mean(), (K_.t10[reach] + 1).median(), (alt > 0).mean(), alt.mean(), np.median(alt), days_alt.mean(), (K_.r > 0).mean(), K_.r.mean(), K_.r.median(), K_.hold.mean()))
# ---------------- 1H ----------------
import midsmall_screener as M
data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); E = pd.read_pickle(SC + 'earnmac_T2.pkl'); T = E[E.epx < 20].copy().reset_index(drop=True); T['et'] = pd.to_datetime(T.et); C1 = .004; rows = []
for tk, g in T.groupby('tk'):
    Y = M.closed_only(data[tk]); h, l, c, o, v = (Y[x].values.astype(float) for x in ('high', 'low', 'close', 'open', 'volume')); ix = Y.index; F = M.features(Y); day = ix.normalize()
    dd = Y.groupby(day).agg(high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), vol=('volume', 'sum')); pcd = dd.close.shift(); tr_ = np.maximum(dd.high - dd.low, np.maximum((dd.high - pcd).abs(), (dd.low - pcd).abs())); atrp = (tr_.rolling(14).mean() / dd.close).shift(1); ldv = np.log((dd.close * dd.vol).rolling(20).mean().shift(1))
    for r in g.itertuples():
        fi, ei = int(r.fi), int(r.ei); xi = ix.get_indexer([r.xt])[0]
        if fi < 100 or xi < 0 or xi <= ei: continue
        hv = h[ei:xi] / r.epx; t10 = int(np.argmax(hv >= 1.10)) if (hv >= 1.10).any() else -1; d0 = day[fi]
        rows.append(dict(tk=tk, et=r.et, xt=r.xt, epx=r.epx, ret=r.ret, vr=r.vr, dte=r.dte, vwx=float(F.vwx.iat[fi]), ATRP=atrp.get(d0, np.nan), LDV=ldv.get(d0, np.nan), t10=t10, hold_d=(r.xt - r.et) / np.timedelta64(1, 'D'), t10_d=((ix[ei + t10] - r.et) / np.timedelta64(1, 'D') + .0) if t10 >= 0 else np.nan))
H = pd.DataFrame(rows); H = H[(~(H.dte <= 3)) & (~(H.vwx > 1))].copy(); H['tr'] = H.et < pd.Timestamp('2025-04-01')
def sc1(D):
    R = pd.DataFrame(index=D.index)
    for f, s in (('ATRP', 1), ('vr', 1), ('LDV', -1)):
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
for cap in (20, 10, 5):
    S = H[H.epx < cap].copy(); S['s_'] = sc1(S)
    for pct, nm in ((.7, 'top30%'), (.9, 'top10%')):
        cut = np.quantile(S[S.tr].s_.dropna(), pct); G = S[(S.s_ >= cut) & (~S.tr)]; reach = G.t10 >= 0; alt = np.where(reach, .10 - C1, G.ret - C1); yrs = (G.et.max() - G.et.min()).days / 365.25 or 1
        D_rows.append(('1H', f'sub-${cap}', nm, len(G), len(G) / yrs, reach.mean(), G.t10_d[reach].median(), (alt > 0).mean(), alt.mean(), np.median(alt), np.where(reach, G.t10_d.fillna(0) + 0, G.hold_d).mean(), ((G.ret - C1) > 0).mean(), (G.ret - C1).mean(), (G.ret - C1).median(), G.hold_d.mean()))
R = pd.DataFrame(D_rows, columns=['tf', 'px', 'group', 'n', 'per_yr', 'REACH', 'med_days_to10', 'win_sell10', 'mean_sell10', 'med_sell10', 'avg_hold_sell10', 'win_vec', 'mean_vec', 'med_vec', 'avg_hold_vec'])
pd.options.display.width = 250; fm = {c_: '{:.0%}'.format for c_ in ('REACH', 'win_sell10', 'win_vec')}; fm.update({c_: '{:+.1%}'.format for c_ in ('mean_sell10', 'med_sell10', 'mean_vec', 'med_vec')}); fm.update({c_: '{:.1f}'.format for c_ in ('per_yr', 'med_days_to10', 'avg_hold_sell10', 'avg_hold_vec')})
print(R.to_string(index=False, formatters=fm))
