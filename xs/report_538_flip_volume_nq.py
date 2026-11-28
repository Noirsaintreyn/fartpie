"""Report 538 (Nov 28 2026): Rey: "does volume at flip help". Small caps already answered (IDEOLOGY: flip-bar volume ratio top tercile +7.2pp huge-run rate, t 1.9, same direction both eras, the one durable tilt). NQ side tested here.
DECLARED BEFORE RESULTS. NQ 1H neg-gamma vector flips (vx engine, 1:1, cost 1pt). VOLR = volume of the 1H flip bar / mean volume of the SAME clock hour over the prior 20 sessions (removes the intraday volume shape). Hypothesis (mechanism: a flip on heavy participation = real commitment, not a thin pivot): top third (train cut 2019-21) beats bottom third; also ratio of flip-bar volume to its own bar-range (effort/result not tested). Arms: ALL and NEG; TOP vs BOT vs MID. TRAIN 2019-21 | TEST 2018+2022-26; pass = same sign both eras and day-clustered t>=2 on the TOP-minus-BOT contrast. 4 contrasts.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vx_trades import vx
T = vx(); T['yr'] = T.day.dt.year; T['ts'] = T.day + pd.to_timedelta(T.m, unit='m')
b = pd.read_parquet('/home/claude/nqdrive/nq5_drive_et_ext.parquet'); b = b[b.dt >= '2017-06-01'].set_index('dt')
g = b.volume.resample('60min', label='right', closed='left').sum(); g = g[g > 0]; hr = g.index.hour; key = pd.Series(g.index.hour, index=g.index)
mean20 = g.groupby(hr).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()); VR = (g / mean20).rename('VOLR')
# entry bar = open of the 5m bar after the 1H flip bar closes; the flip bar ends at the entry bar open (label='right' index = end time)
T = pd.merge_asof(T.sort_values('ts'), VR.reset_index().rename(columns={'dt': 'ts'}).sort_values('ts'), on='ts', direction='backward', tolerance=pd.Timedelta(minutes=5)).dropna(subset=['VOLR'])
G = pd.read_csv('xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); G.index = G.index.normalize(); T['neg'] = T.neg.astype(bool)
tr = T.yr.between(2019, 2021); te = ~tr; q = T.loc[tr, 'VOLR'].quantile([.33, .67]); T['TOP'] = T.VOLR > q.iloc[1]; T['BOT'] = T.VOLR <= q.iloc[0]
def cell(S):
    if len(S) < 5: return f'n{len(S):4d} (few)'
    dm = S.groupby('day').pts.sum(); t = dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 2 and dm.std() > 0 else np.nan; return f'n{len(S):4d} {S.pts.mean():+6.1f} (t{t:+.1f}) w{(S.pts>0).mean()*100:2.0f}%'
print(f'trades with volume ratio {len(T)}; train cuts VOLR {q.iloc[0]:.2f} / {q.iloc[1]:.2f}; mean VOLR by era train {T[tr].VOLR.mean():.2f} test {T[te].VOLR.mean():.2f}\nnet pts/trade | TRAIN | TEST')
for name, U in (('ALL', T), ('NEG', T[T.neg])):
    for lab, m in (('TOP third', U.TOP), ('MID', ~U.TOP & ~U.BOT), ('BOT third', U.BOT)): print(f'{name:4} {lab:10} | {cell(U[m & tr.loc[U.index]])} | {cell(U[m & te.loc[U.index]])}')
    for e, mk in (('train', tr), ('test', te)):
        a, c = U[U.TOP & mk.loc[U.index]], U[U.BOT & mk.loc[U.index]]; dm = pd.concat([a.groupby('day').pts.mean(), c.groupby('day').pts.mean()], axis=1, keys=['a', 'c']).dropna()
        print(f'   {name} TOP-BOT {e}: {a.pts.mean()-c.pts.mean():+.1f} pts (matched-day diff {dm.a.sub(dm.c).mean():+.1f}, n days {len(dm)})')
print('Spearman(VOLR, pts): ALL train %.3f test %.3f | NEG train %.3f test %.3f' % (T[tr].VOLR.corr(T[tr].pts, method='spearman'), T[te].VOLR.corr(T[te].pts, method='spearman'), T[tr & T.neg].VOLR.corr(T[tr & T.neg].pts, method='spearman'), T[te & T.neg].VOLR.corr(T[te & T.neg].pts, method='spearman')))
