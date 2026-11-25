"""Report 482 (Nov 25 2026): Rey - 'look back and see if there is an obvious bottom ... a way to measure over/under reaction ... if we dump 20% but that was an overreaction the vector flip could be rated better'. DAILY flips (structure_small), base5 top-30% group (train-ECDF, test 2022+), px<20/<10/<5, return = vector exit r (30bp), REACH (peak >= +10%).
DECLARED BEFORE RESULTS. Mechanism: a sell-off larger than the news justifies (no earnings / bad 8-K in the window, a volume climax, a close well off the low) is an overreaction that mean-reverts; a sell-off WITH a justifying event is repricing and does not. Measures from daily bars up to the flip day:
 DROP  = deepest drawdown of the close from the running 20-day high within the last 20 days (negative number); DD20 = flip close vs 20-day high; LOWAGE = sessions since the lowest close of the last 20 (0 = flip day is the low); BOUNCE = flip close / lowest low of the last 10 days - 1; CLIMAX = max volume of the last 10 days / mean volume of the 20 days before.
 EXPL = an earnings 8-K (2.02) or bad 8-K (3.01, 1.02, 1.03, 2.05, 2.06, 4.02) known inside the 20-day window (drop is 'explained'); UNEXPL = none.
Tests: (1) DROP quartiles (cuts frozen from train rows of the same bucket); (2) deep drops (DROP <= -20%): UNEXPL vs EXPL; (3) fresh bottom (LOWAGE <= 3) vs older; (4) CAPITULATION = DROP <= -20% & UNEXPL & CLIMAX >= 3 & LOWAGE <= 5. Read REACH / win / mean / median / p10, by era (2022 vs 2023-26), n shown. A pass = the capitulation set beats the rest of the group on mean AND REACH in both eras with n>=30; otherwise descriptive."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
O['tr'] = O.d.dt.year <= 2021; W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl'); it_ = K['items'].astype(str).str.split(',')
K['EXPLE'] = it_.apply(lambda l: '2.02' in l or any(x in l for x in ('3.01', '1.02', '1.03', '2.05', '2.06', '4.02'))); ev = {t: x.known.values for t, x in K[K.EXPLE].groupby('ticker')}
F = {k: np.full(len(O), np.nan) for k in ('DROP', 'DD20', 'LOWAGE', 'BOUNCE', 'CLIMAX', 'EXPL')}
for tk, g in O.groupby('tk'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index; c = df.close.values; hi = df.high.values; lo = df.low.values; v = df.volume.values.astype(float); a = ev.get(tk)
    for i, d in zip(g.index, g.d):
        j = ix.searchsorted(pd.Timestamp(d))
        if j < 60 or j >= len(ix) or ix[j] != pd.Timestamp(d): continue
        w = slice(j - 19, j + 1); run_hi = pd.Series(hi[w]).cummax().values; dd = c[w] / run_hi - 1; k = int(np.argmin(c[w]))
        F['DROP'][i] = dd.min(); F['DD20'][i] = c[j] / hi[w].max() - 1; F['LOWAGE'][i] = 19 - k; F['BOUNCE'][i] = c[j] / lo[j - 9:j + 1].min() - 1; F['CLIMAX'][i] = v[j - 9:j + 1].max() / max(v[j - 29:j - 9].mean(), 1)
        F['EXPL'][i] = float(a is not None and ((a <= np.datetime64(ix[j])) & (a > np.datetime64(ix[j - 19]))).any())
for k, x in F.items(): O[k] = x
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+.1%} median {x.r.median():+.1%} p10 {x.r.quantile(.1):+.0%}' if len(x) else 'n 0'
eras = lambda x: f'2022 {x[x.d.dt.year == 2022].r.mean():+.1%} ({(x.d.dt.year == 2022).sum()}) / 2023-26 {x[x.d.dt.year >= 2023].r.mean():+.1%} ({(x.d.dt.year >= 2023).sum()})'
for cap in (20, 10, 5):
    S = O[(O.px < cap) & O.DROP.notna()].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); trn = S[S.tr & (S.s_ >= cut)]; G = S[(~S.tr) & (S.s_ >= cut)].copy(); G['cap'] = (G.DROP <= -.2) & (G.EXPL == 0) & (G.CLIMAX >= 3) & (G.LOWAGE <= 5)
    qc = np.quantile(trn.DROP, [.25, .5, .75]); G['q'] = np.searchsorted(qc, G.DROP) + 1
    print(f'\n######## sub-${cap} top30% test 2022+: {len(G)} flips | DROP quartile cuts (train) {np.round(qc, 2)} | median DROP {G.DROP.median():.0%}, share DROP<=-20% {(G.DROP <= -.2).mean():.0%}')
    print('  (1) DROP quartile (Q1 deepest drop):'); [print(f'    Q{q} mean DROP {x.DROP.mean():.0%}  ', st(x), '|', eras(x)) for q, x in G.groupby('q')]
    deep = G[G.DROP <= -.2]; print('  (2) deep drops (<=-20%):', st(deep)); print('      UNEXPLAINED (no earnings/bad 8-K in window):', st(deep[deep.EXPL == 0]), '|', eras(deep[deep.EXPL == 0])); print('      EXPLAINED:', st(deep[deep.EXPL == 1]), '|', eras(deep[deep.EXPL == 1]))
    print('  (3) fresh bottom LOWAGE<=3:', st(G[G.LOWAGE <= 3]), '| older:', st(G[G.LOWAGE > 3])); print('      bounce off the low (BOUNCE>=20%):', st(G[G.BOUNCE >= .2]), '| <20%:', st(G[G.BOUNCE < .2]))
    print('  (4) CAPITULATION (drop<=-20%, unexplained, volume climax>=3x, low within 5d):', st(G[G.cap]), '|', eras(G[G.cap])); print('      rest of the group:', st(G[~G.cap]), '|', eras(G[~G.cap]))
