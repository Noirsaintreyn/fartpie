"""Nov 18 2026 report 309. After 308: trend days are increasingly FRONT-LOADED (share of |net| done by 10:30: 0.29-0.41 in 2018-24, .46 in 2025, .50 in 2026). AMT 'drive' day theory: a one-sided, high-efficiency opening hour means initiative participants (not responsive) are in control; the move continues because the other side has no inventory to defend.
DECLARED: at 10:30 ET, if the first-hour move |fh| >= k x ADR14 AND first-hour efficiency (|net|/path over 09:30-10:30, 5m closes) in the top third (cut = train), go WITH it to 15:55. k fixed a priori at 0.35 (sensitivity .25/.5). Stop 0.25 ADR from entry (ADR-scaled per Rey), cost 1 pt. Failure mode: exhaustion/gap-fill days (report 154: 30-60 pt gaps fill 55-75%).
Train 2019-21 | test 2018 + 2022-Aug 2026; pts per trade (1 NQ ct), t on trades, by-year."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D = pd.read_parquet(SP + 'd305.parquet'); N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); R = {}
for d, x in N.groupby('day'):
    if d not in D.index: continue
    x = x.sort_values('m'); a = x[x.m <= 625]; r = x[x.m > 625]
    if len(a) < 12 or len(r) < 40: continue
    path = a.close.diff().abs().sum() + abs(a.close.iloc[0] - a.open.iloc[0]); net = a.close.iloc[-1] - a.open.iloc[0]; R[d] = dict(fh_eff=abs(net) / path if path else np.nan, ent=a.close.iloc[-1], rows=r[['high', 'low', 'close']].values, fin=r.close.iloc[-1])
rows = []
for d, v in R.items():
    row = D.loc[d]; s = np.sign(row.fh)
    if s == 0 or np.isnan(row.adr): continue
    rows.append(dict(day=d, s=s, fh_adr=abs(row.fh) / row.adr, fh_eff=v['fh_eff'], ent=v['ent'], adr=row.adr, fin=v['fin'], arr=v['rows'], gap=0))
X = pd.DataFrame(rows).set_index('day'); X['tr'] = (X.index >= '2019-01-01') & (X.index < '2022-01-01'); ce = X.loc[X.tr, 'fh_eff'].quantile(2 / 3); print(f'days {len(X)}; train first-hour efficiency top-third cut {ce:.2f}')
def trade(r, stopk=.25, cost=1.0):
    stop = r.ent - r.s * stopk * r.adr; ex = r.fin
    for h, l, c in r.arr:
        if (r.s == 1 and l <= stop) or (r.s == -1 and h >= stop): ex = stop; break
    return (ex - r.ent) * r.s - cost
for k in (.25, .35, .5):
    S = X[(X.fh_adr >= k) & (X.fh_eff >= ce)].copy(); S['pts'] = S.apply(trade, axis=1); S0 = X[(X.fh_adr >= k)].copy(); S0['pts'] = S0.apply(trade, axis=1)
    f = lambda w: f'{w.pts.mean():+6.1f} (t{w.pts.mean() / (w.pts.std() / np.sqrt(len(w))):+.1f}, n{len(w)}, win {100*(w.pts>0).mean():.0f}%)'
    print(f'k={k}: drive (eff top third): train {f(S[S.tr])} | test {f(S[~S.tr])}')
    print(f'      no efficiency filter:   train {f(S0[S0.tr])} | test {f(S0[~S0.tr])}')
k = .35; S = X[(X.fh_adr >= k) & (X.fh_eff >= ce)].copy(); S['pts'] = S.apply(trade, axis=1); print('by year pts/trade:', S.groupby(S.index.year).pts.agg(['mean', 'size']).round(1).T.to_string())
# direction placebo: same days, opposite side
P = S.copy(); P['s'] = -P.s; P['pts'] = P.apply(trade, axis=1); print('placebo (fade the drive) train', round(P[P.tr].pts.mean(), 1), '| test', round(P[~P.tr].pts.mean(), 1))
Dd = D.join(S.pts.rename('drive')).fillna({'drive': 0}); print('\ndaily corr of drive-trade pts with stack pts: ALL %.2f | M1 %.2f | V15a %.2f | RA %.2f | ON+ %.2f' % tuple(Dd[['drive', k]].corr().iloc[0, 1] for k in ('ALL', 'M1', 'V15a', 'RA', 'ON+')))
print('drive trade by month 2026:', S[S.index.year == 2026].groupby(S[S.index.year == 2026].index.to_period('M')).pts.agg(['mean', 'size']).round(1).T.to_string())
w = S[S.index.year == 2026]; print('2026 drive: win%% %.0f, mean win %.1f, mean loss %.1f; share stopped %.0f%%' % (100 * (w.pts > 0).mean(), w[w.pts > 0].pts.mean(), w[w.pts <= 0].pts.mean(), 100 * (w.pts < -.25 * w.adr.mean() * .9).mean()))
