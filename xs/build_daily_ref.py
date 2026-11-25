"""Nov 25 2026: frozen reference for the live sub-$5 DAILY flip alert (daily_small.py). Train rows (flip date <= 2021) of structure_small with px < 5: ECDF tables (101 quantiles) for ATRP, F11, VOLX, LTURN, LMC and the score cut = 70th percentile of the train score (same definition as reports 482/483). Frozen: never refit on newer data."""
import numpy as np, pandas as pd, json
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); S = O[O.px < 5].copy(); tr = S[S.d.dt.year <= 2021]
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}; qs = np.linspace(0, 1, 101); ref = {f: np.nanquantile(tr[f].astype(float), qs).tolist() for f in B}   # Nov 25 2026: VOLX dropped (report 484 ablation)
def score(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float).values; r = np.interp(x, ref[f], qs); r = np.where(np.isnan(x), np.nan, r); R.append(r if s > 0 else 1 - r)
    return np.nanmean(np.vstack(R), axis=0)
cut = float(np.nanquantile(score(tr), .7)); print('train rows', len(tr), 'cut', round(cut, 4), '| share of ALL px<5 flips >= cut:', round(float((score(S) >= cut).mean()), 3), '| test share', round(float((score(S[S.d.dt.year >= 2022]) >= cut).mean()), 3))
T = S[(S.d.dt.year >= 2022) & (score(S) >= cut)]; bt = dict(n=int(len(T)), mean=float(T.r.mean()), win=float((T.r > 0).mean()), median=float(T.r.median()), p10=float(T.r.quantile(.1)), reach=float((T.PEAK >= .1).mean())); print('test group', bt)
json.dump(dict(bt=bt, ref=ref, signs=B, cut=cut, built='2026-11-25', train='flips <= 2021, px < 5, structure_small'), open('daily_base5_ref.json', 'w'))
