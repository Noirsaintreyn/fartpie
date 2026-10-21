"""Oct 21 2026: four published effects as tie-breaks / skip rules on the full-S&P screener (ratio >= 1.5, live rules, 6 slots, 8% breaker).
 1 spike  : options/stock volume over the last 3 days vs the stock's own 252d median (Johnson & So: abnormal option volume = bad news)
 2 frog   : information discreteness over 60d = sign(ret60) * (%neg days - %pos days)  (Da, Gurun & Warachka: smooth trends persist)
 3 near52 : prior close / 252d max close (George & Hwang)
 4 max    : largest single-day return in the prior 21 days (Bali, Cakici & Whitelaw: lottery stocks underperform)
All features use data through the prior close. Skip rules use the feature's percentile among all EARLIER trades (no look-ahead)."""
import os, bisect, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/slot_tests.py').read().split('P = [(np.datetime64')[0])
T = T.sort_values('entry').reset_index(drop=True); d = T.entry.dt.normalize()
R = pd.read_csv('xs/ext/optvol_ratio_sp.csv', parse_dates=['date']).pivot_table(index='date', columns='tk', values='r').sort_index()
MED = R.rolling(252, min_periods=60).median().shift(1); SP3 = (R.rolling(3).mean() / MED).shift(1)      # both known before today's open
rets = CL.pct_change()
ID = (np.sign(CL / CL.shift(60) - 1) * ((rets < 0).rolling(60).mean() - (rets > 0).rolling(60).mean())).shift(1)
N52 = (CL / CL.rolling(252, min_periods=120).max()).shift(1); MAXR = rets.rolling(21).max().shift(1)
def look(M): return np.array([M[t].asof(x) if t in M.columns else np.nan for t, x in zip(T.tk, d)], float)
T['spike'] = look(SP3); T['frog'] = look(ID); T['near52'] = look(N52); T['maxr'] = look(MAXR)
for c in ('spike', 'frog', 'near52', 'maxr'): print(f'{c}: coverage {T[c].notna().mean():.0%}', end='  ')
# percentile vs earlier trades only
def prior_pct(col):
    out = np.full(len(T), np.nan); seen = []
    for i, v in enumerate(T[col].values):
        if v == v:
            if len(seen) >= 200: out[i] = bisect.bisect_left(seen, v) / len(seen)
            bisect.insort(seen, v)
    return out
for c in ('spike', 'frog', 'near52', 'maxr'): T['p_' + c] = prior_pct(c)
print()
T['per'] = np.where(T.entry.dt.year >= 2022, '22-26', '19-21')
for c in ('spike', 'frog', 'near52', 'maxr'):
    q = pd.qcut(T[c], 5, labels=False, duplicates='drop'); g = T.groupby([q, 'per']).ret.mean().unstack() * 100
    print(f'avg trade % by {c} quintile (low->high)  19-21: ' + ' '.join(f'{v:+.2f}' for v in g['19-21']) + '   22-26: ' + ' '.join(f'{v:+.2f}' for v in g['22-26']))
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z=None, **kw):
    Z = T if Z is None else Z
    print(f'  {nm:<50}' + ''.join(f'{c:>+9.1%} /{dd:>5.0%}' for c, dd in (sim(Z, a, b, **kw) for a, b in P)), flush=True)
print(f'\n{"S&P, ratio >= 1.5, live rules":<52}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live (tie-break: options ratio)')
NAN = lambda c, keep: T[c].isna() | keep                                     # keep trades with no feature value
show('1 spike: skip top 20% (biggest option-volume spikes)', T[NAN('p_spike', T.p_spike < .8)])
show('1 spike: skip top 10%', T[NAN('p_spike', T.p_spike < .9)])
show('2 frog: skip top 20% jumpiest trends', T[NAN('p_frog', T.p_frog < .8)])
show('2 frog: skip top 33% jumpiest', T[NAN('p_frog', T.p_frog < .67)])
show('3 near52: skip bottom 20% (farthest from 52w high)', T[NAN('p_near52', T.p_near52 >= .2)])
show('3 near52: skip bottom 33%', T[NAN('p_near52', T.p_near52 >= .33)])
show('4 max: skip top 20% (biggest recent 1-day jump)', T[NAN('p_maxr', T.p_maxr < .8)])
show('4 max: skip top 10%', T[NAN('p_maxr', T.p_maxr < .9)])
T['tb_spike'] = -T.spike.fillna(T.spike.median()); T['tb_frog'] = -T.frog.fillna(0); T['tb_near'] = T.near52.fillna(.9); T['tb_max'] = -T.maxr.fillna(T.maxr.median())
for k, nm in (('tb_spike', 'smallest spike'), ('tb_frog', 'smoothest trend'), ('tb_near', 'closest to 52w high'), ('tb_max', 'smallest recent jump')):
    show(f'tie-break: {nm}', key=k)
