"""Nov 2 2026: follow-up to scan.py (19 two-period patterns vs a chance max of 1). Most of the weak cells share LOW implied vol
(lowest within-year tercile of 30d ATM IV, cross-sectional level). Their big-winner ratios are 0.1-0.5, so unlike earlier filters a
cut here could remove losers without removing jumps. Chosen after seeing both periods -> optimistic; a pass means paper-track only.
 F1 drop lowest IV tercile (primary) | F1q drop lowest IV quintile | F1qq lowest quartile (neighbours)
 F2 same idea with 20d realized vol (covers every trade, not only the 56% with IV)
 F3 drop tight-stop trades in a QQQ uptrend (stopd t1 & qqq bull)"""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
OPT3 = open("/home/claude/fartpie/xs/opt_three.py").read(); exec(OPT3.split("X['h'] = np.where")[0].replace("print(", "(lambda *a, **k: None)("))
exec(OPT3[OPT3.index("HSH = np.array"):OPT3.index("pr = lambda s")])
X['h'] = np.where(X.entry.dt.year <= 2021, '2019-21', '2022-26'); X['yr'] = X.entry.dt.year
RV = {}
for tk in X.tk.unique():
    H = pd.read_parquet(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0], columns=['datetime', 'close'])
    H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute
    c = H[(m >= 570) & (m < 960)].groupby(H.dt.dt.normalize()).close.last(); RV[tk] = c.pct_change().rolling(20).std().shift(1)
X['rv'] = [RV[t].asof(d) for t, d in zip(X.tk, X.d)]
top = X.groupby('h').ret.transform(lambda s: s >= s.quantile(.95))
def pct(c): return X.groupby('yr')[c].rank(pct=True)
ivp, rvp = pct('iv'), pct('rv')
print('single buckets (avg trade / big-winner ratio, 2019-21 | 2022-26):')
for nm, s in (('IV', ivp), ('realized vol', rvp)):
    for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        m = (s > lo) & (s <= hi); row = []
        for h in ('2019-21', '2022-26'):
            z = (X.h == h) & s.notna(); mm = m & z; row.append(f'{X.ret[mm].mean():+.2%} j{(top[mm].sum() / top[z].sum()) / (mm.sum() / z.sum()):.2f} n{mm.sum()}')
        print(f'  {nm:<13} tercile {int(lo * 3) + 1}: ' + '  |  '.join(row))
B0 = ev(X.ratio); print('\nbook, 40 matched runs, 2019-21 / 2022-26'); line('live', B0)
def test(nm, drop):
    d = drop.fillna(False).values.astype(bool); print(f'  {nm}: removes {d.mean():.0%} of flips, {(top.values & d).sum() / top.sum():.0%} of top-5% winners')
    line(f'    {nm}', ev(X.ratio, ~d), B0)
test('F1 drop lowest IV tercile', ivp <= 1 / 3); test('F1q drop lowest IV quintile', ivp <= .2); test('F1qq drop lowest IV quartile', ivp <= .25)
test('F2 drop lowest realized-vol tercile', rvp <= 1 / 3); test('F2q drop lowest realized-vol quintile', rvp <= .2)
sq = X.groupby('yr').stopd.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False)); test('F3 tight stop & QQQ uptrend', (sq == 0) & (X.qqq == 1))
