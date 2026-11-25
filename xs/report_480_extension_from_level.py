"""Report 480 (Nov 25 2026): Rey - 'whether the flip bar is already extended from the vector level'. DAILY flips (structure_small, px<20/<10/<5), group = base5 top 30% (train-ECDF, test 2022+). EXT_L = (flip close - daily vector level at the flip bar)/flip close and EXT_ATR = (close - level)/ATR14. Mechanism: the vector exit fires when the close crosses back through the level, so a flip that closes far above it carries a built-in loss of about EXT_L if it fails (a spike-day flip), while a flip right on the level has a tight, cheap invalidation; but a close far from the level also signals real thrust. Read: quartiles of EXT_L cut on TRAIN rows of the same bucket (frozen): REACH (>=+10%), win, mean, median, p10 of r (vector exit, 30bp), test only, plus H split by year. PASS = monotone worse (or better) in the top quartile in test AND the same sign in 2022/2023-26 halves; otherwise descriptive."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'xs')
src = open('xs/report_467_har_vol_features.py').read(); exec(src[:src.index("# ---- stage 2")].replace("print(f'\\n=== {tgt}", "(lambda *a, **k: None)(f'\\n=== {tgt}").replace("print(f'  {f:10s}", "(lambda *a, **k: None)(f'  {f:10s}"))
from vector_pine import run_state_machine
def score(feats):
    R = pd.DataFrame(index=O.index)
    for f, s in feats.items():
        x = O[f].astype(float); ref = np.sort(x[tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
O['s_'] = score({'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1})
ext = {}; extatr = {}
for tk, g in O.groupby('tk'):
    df = W[tk]; df = df[~df.index.duplicated()]
    if len(df) < 100: continue
    reg, lev, _, fl = run_state_machine(df.assign(datetime=df.index)); lev = pd.Series(lev, index=df.index); h, l, c = df.high.values, df.low.values, df.close.values; pc = np.r_[c[0], c[:-1]]; atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), index=df.index).rolling(14).mean()
    for i, r in zip(g.index, g.itertuples()):
        d = pd.Timestamp(r.d)
        if d in lev.index: L_ = lev.loc[d]; cc = df.close.loc[d]; ext[i] = (cc - L_) / cc; extatr[i] = (cc - L_) / atr.loc[d] if atr.loc[d] > 0 else np.nan
O['EXT_L'] = pd.Series(ext); O['EXT_ATR'] = pd.Series(extatr); print('flips with level', O.EXT_L.notna().mean().round(2)); pd.options.display.width = 250
for cap in (20, 10, 5):
    S = O[(O.px < cap) & O.EXT_L.notna()]; trn = S[S.tr]; qcut = np.quantile(trn[trn.s_ >= np.quantile(trn.s_.dropna(), .7)].EXT_L, [.25, .5, .75]); G = S[(~S.tr) & (S.s_ >= np.quantile(trn.s_.dropna(), .7))].copy(); G['q'] = np.searchsorted(qcut, G.EXT_L) + 1
    print(f'\n=== sub-${cap} top30% test n={len(G)} | EXT_L quartile cuts (train group) {np.round(qcut, 3)} | median EXT_ATR {G.EXT_ATR.median():.1f}')
    print('  quartile  n   mean EXT_L | REACH  win  mean   median  p10   | 2022 mean / 2023-26 mean')
    for q, x in G.groupby('q'):
        print(f'  Q{q} {"low " if q == 1 else "high" if q == 4 else "    "} {len(x):4d}  {x.EXT_L.mean():.1%}     | {(x.PEAK >= .1).mean():.0%}  {(x.r > 0).mean():.0%}  {x.r.mean():+.1%}  {x.r.median():+.1%}  {x.r.quantile(.1):+.0%} | {x[x.d.dt.year == 2022].r.mean():+.1%} / {x[x.d.dt.year >= 2023].r.mean():+.1%}')
