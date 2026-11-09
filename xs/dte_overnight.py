"""Nov 8 2026: THE OVERNIGHT HALF OF THE DAY. A Lucid trading day runs 18:00 - 17:00 ET, but every system so far trades 09:30 - 16:00 only; days with no RTH signal
can never qualify. The best-documented index-futures anomaly is the OVERNIGHT DRIFT (most of the equity index's return comes between the close and the next open).
DECLARED BEFORE RESULTS. Real NQ futures 5m bars Jan 2014 - Jan 2026; the overnight session of day D = 18:00 ET on the prior evening to 09:25 ET.
  O1 long every night: buy the 18:00 open, sell the 09:25 bar close (before the RTH open); 1 pt cost
  O2 O1 with a hard stop of 0.15 ADR (prop-safe)                    O3 O1, only nights after an RTH up day (close > open)
  O4 O1, only after an RTH down day                                 O5 O1, only when QQQ net GEX at that day's close is negative / positive (2018+)
  O6 O1, only when the daily vector (on RTH closes) is up
Per version: nights, win, points per night, share of nights >= +7.5 (a 50K qualifying session at 10 micros), worst night, max drawdown, by year; daily correlation with the RTH
day-type engine (DTE); and how many sessions a month become qualifying when the overnight trade is added to the DTE at 10 micros overnight."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet')
on = X[(X.m >= 1080) | (X.m < 565)]   # 18:00 - 09:25 bar starts (the 09:25 bar closes at 09:30)
G = on.groupby('day'); N = pd.DataFrame({'o': G.open.first(), 'c': G.close.last(), 'h': G.high.max(), 'l': G.low.min(), 'first_m': G.m.first(), 'n': G.open.size(), 'sym': G.sym.first()})
R = X[X.rth].groupby('day').agg(ro=('open', 'first'), rc=('close', 'last'), rh=('high', 'max'), rl=('low', 'min'), sym=('sym', 'first'))
R['adr'] = (R.rh - R.rl).rolling(20).mean().shift(1)
N = N[(N.first_m >= 1080) & (N.n >= 150)].join(R[['adr', 'sym']].rename(columns={'sym': 'rsym'}), how='inner'); N = N[N.sym == N.rsym]
prev = R[['ro', 'rc']].shift(1); N = N.join(prev.rename(columns={'ro': 'pro', 'rc': 'prc'}))
Gx = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); Gx.index = Gx.index.normalize()
dlist = list(R.index); prevday = pd.Series(dlist[:-1], index=dlist[1:]); N['gex_prev'] = Gx.reindex(prevday.reindex(N.index).values).values
import sys; sys.path.insert(0, '/home/claude/fartpie/xs'); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open('/home/claude/fartpie/xs/signal_lab.py').read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
Rd = R[['ro', 'rh', 'rl', 'rc']].rename(columns={'ro': 'open', 'rh': 'high', 'rl': 'low', 'rc': 'close'}); R['vreg'] = np.asarray(vec(Rd)); N['vprev'] = R.vreg.shift(1).reindex(N.index)
N['pts'] = N.c - N.o - 1.0; N['mae'] = N.l - N.o
S = .15 * N.adr; N['pts_stop'] = np.where(N.mae <= -S, -S - 1.0, N.pts)
DTE = None
try:
    E = {k: pd.read_parquet(f'/tmp/claude-0/ml/dte_{k}.parquet') for k in ('M1', 'M2', 'R1')}
except Exception: E = None
if __name__ == '__main__':
    V = {'O1 long every night': (N.pts, np.ones(len(N), bool)), 'O2 O1 + stop 0.15 ADR': (N.pts_stop, np.ones(len(N), bool)), 'O3 after RTH up day': (N.pts, (N.prc > N.pro).values),
         'O4 after RTH down day': (N.pts, (N.prc < N.pro).values), 'O5a GEX negative': (N.pts, (N.gex_prev < 0).values), 'O5b GEX positive': (N.pts, (N.gex_prev > 0).values),
         'O6 daily vector up': (N.pts, (N.vprev > 0).values), 'O6b daily vector down': (N.pts, (N.vprev < 0).values)}
    yr = N.index.year
    print(f'nights {len(N)} ({N.index[0]:%Y-%m} - {N.index[-1]:%Y-%m}); median overnight range {(N.h - N.l).median():.0f} pts, median ADR {N.adr.median():.0f}', flush=True)
    print(f'  {"version":<24} nights  win  pts/night  >=+7.5  worst  maxDD | 2014-17 / 2018-21 / 2022-26 per night | years +', flush=True)
    for nm, (p, msk) in V.items():
        q = pd.Series(np.where(msk, p, np.nan), index=N.index).dropna(); cum = q.cumsum(); y = q.groupby(q.index.year).sum()
        per = [q[(q.index.year >= a) & (q.index.year <= b)].mean() for a, b in ((2014, 2017), (2018, 2021), (2022, 2026))]
        print(f'  {nm:<24} {len(q):6d} {np.mean(q > 0):4.0%} {q.mean():+9.2f} {np.mean(q >= 7.5):7.0%} {q.min():+6.0f} {(cum - cum.cummax()).min():+6.0f} | ' + ' / '.join(f'{x:+5.1f}' for x in per) + f' | {(y > 0).sum()}/{len(y)}', flush=True)
    print('\n  by year (O1 points per night): ' + ' '.join(f'{y % 100:02d} {v:+.1f}' for y, v in N.pts.groupby(yr).mean().items()), flush=True)
    if E is not None:
        gneg = pd.Series(N.gex_prev.values < 0, index=N.index)
        AD = pd.DatetimeIndex([d for d in R.index if d >= pd.Timestamp('2018-02-01')])
        P = {k: v.groupby('day').pts.sum().reindex(AD).fillna(0.0) for k, v in E.items()}
        GN = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); GN.index = GN.index.normalize()
        gday = (GN.reindex(AD, method='ffill').shift(1) < 0).values
        dte_pts = np.where(gday, P['M1'], P['R1']) + P['M2']; ONp = N.pts_stop.reindex(AD).fillna(0.0).values
        print(f'\n  daily correlation, overnight (O2) vs RTH day-type engine: {np.corrcoef(ONp, dte_pts)[0, 1]:+.2f}', flush=True)
        for lab, ow, rw in (('DTE alone at 10 micros', 0, 1), ('overnight O2 alone at 10 micros', 1, 0), ('DTE + overnight O2, both 10 micros', 1, 1), ('DTE 10 + overnight 5 micros', .5, 1)):
            usd = (ow * ONp + rw * dte_pts) * 20; cum = np.cumsum(usd)
            print(f'  {lab:<36} qualifying sessions (>= $150) {np.mean(usd >= 150) * 21:5.1f}/mo, losing {np.mean(usd < 0) * 21:4.1f}/mo, avg ${usd.mean():+5.0f}/day, max DD ${(cum - np.maximum.accumulate(cum)).min():,.0f}', flush=True)
