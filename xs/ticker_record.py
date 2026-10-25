"""Ideas from the 23-name study, on the live book (S&P, ratio >= 1.5, 6 slots, 8% breaker):
 TR  a name's own track record as the slot tie-break (walk-forward: mean return of its trades that EXITED before this entry)
 HS  size by the day's HAR expected range (calmer day -> bigger slot, clipped 0.5x-1.5x)
 CAP more slots (8 / 10 / 12) at 1/n each, for reference"""
import numpy as np, pandas as pd
exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])
T = T.sort_values('entry').reset_index(drop=True)
def record(days, mn):
    out = np.full(len(T), np.nan)
    for tk, g in T.groupby('tk'):
        ex, rt, en = g.exit.values, g.ret.values, g.entry.values
        for i, e in zip(g.index, en):
            m = (ex < e) & (ex >= e - np.timedelta64(days, 'D')) if days else (ex < e)
            if m.sum() >= mn: out[i] = rt[m].mean()
    return out
head('TR ticker track record as tie-break')
for days, mn in ((365, 8), (730, 10), (0, 10)):
    rec = record(days, mn); Z = T.copy(); Z['rec'] = np.where(np.isnan(rec), np.nanmedian(rec), rec)
    show(f'record {days or "all"}d (min {mn}) first', Z, key='rec')
    Z['combo'] = Z.groupby('entry').ratio.rank(pct=True) + Z.groupby('entry').rec.rank(pct=True)
    show(f'ratio rank + record rank ({days or "all"}d)', Z, key='combo')
    Z['ratio'] = Z.ratio + np.where(Z.rec < 0, -100, 0); show(f'ratio, but names with a negative record last ({days or "all"}d)', Z)
head('HS size by HAR expected range')
SIG = {}
for tk in set(T.tk):
    B = bars(tk)
    if B is None: continue
    D = B.groupby(B.dt.dt.normalize()).agg(h=('high', 'max'), l=('low', 'min'))
    x = np.log(np.log(D.h / D.l).clip(lower=1e-5))
    s = np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665
    SIG[tk] = s.shift(1)                                                         # forecast for day d uses days < d
sg = np.array([SIG[t].asof(e.normalize()) if t in SIG else np.nan for t, e in zip(T.tk, T.entry)])
med = pd.Series(sg).expanding(500).median().values
T['hs'] = med / sg
show('size = median HAR / name HAR (0.5x-1.5x)', T, size='hs')
T['hs2'] = np.sqrt(T.hs); show('size = sqrt of that', T, size='hs2')
head('CAP more slots')
for n in (8, 10, 12): show(f'{n} slots', T, n=n)
