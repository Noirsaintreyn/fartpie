import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.argv = ['x', '2']
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/volx/*.parquet')); T = T[(T['var'] == '0/0')]
T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
L = T[~T.h1].copy(); m = L.entry.dt.hour * 60 + L.entry.dt.minute
labs = ['a 09:30 open (flip pre/overnight)', 'b 09:45-11:00', 'c 11:00-12:00', 'd lunch 12-14', 'e 14:00-15:00', 'f power hour 15-16']
L['tod'] = pd.cut(m, [0, 571, 660, 720, 840, 900, 960], right=False, labels=labs)
L['per'] = np.where(L.entry.dt.year >= 2022, '22-26', '19-21'); L['big'] = L.ret > .05
g = L.groupby(['tod', 'per']).agg(avg=('ret', 'mean'), n=('ret', 'size'), win=('ret', lambda s: (s > 0).mean()), runners=('big', 'mean'))
g['avg'] *= 100; g['runners'] *= 100
print('15m flips by entry time: avg trade %, n, win, % that run > +5%'); print(g.unstack().round(2).to_string())
L['rvb'] = pd.cut(L.rv, [0, .8, 1.5, 3, 99], labels=['quiet<0.8', 'normal', 'heavy1.5-3', 'spike>3'])
g = L.groupby(['tod', 'rvb', 'per']).ret.agg(['mean', 'count']); g['mean'] *= 100
print('\navg trade % by entry time x flip-bar volume (19-21 | 22-26), n:'); print(g.unstack().round(2).to_string())
xm = L.exit.dt.hour * 60 + L.exit.dt.minute
L['xtod'] = pd.cut(xm, [0, 571, 660, 720, 840, 900, 960], right=False, labels=['09:30 open', '09:45-11', '11-12', 'lunch', '14-15', '15-16'])
print('\nexits by time of the break:'); print(L.groupby('xtod').ret.agg(share=lambda s: len(s) / len(L), avg=lambda s: s.mean() * 100).round(3).to_string())
