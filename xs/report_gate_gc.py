"""GATE IDEA: when the NQ system is in a bad streak, trade GOLD / CRUDE flips instead. Nov 11 2026.
(1) Do NQ bad streaks persist? NQ system daily pts after a bad trailing-10/20-day window vs after a good one (a gate is pointless if they do not).
(2) GOLD.F / CRUDEOIL.F vector flips (15m and 1h bars, US-hours windows, flat at window end; pure exit / 0.25 / 0.40 ADR stops, 1bp cost): bps/day inside vs outside NQ bad streaks, and the fade version. Streak flags use only data through the PRIOR day.
PASS = same sign and |t|>=2 in 2018-21 and 2022-25."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import importlib.util
spec = importlib.util.spec_from_file_location('lf', '/home/claude/fartpie/xs/report_lse_futs.py'); src = open('/home/claude/fartpie/xs/report_lse_futs.py').read()
ns = {}; sys.argv = ['x', 'NONE']; exec(src[:src.index("WIN = {")], ns)
load, trades, tstat, vec = ns['load'], ns['trades'], ns['tstat'], ns['vec']
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TD = pd.read_pickle(SC + 'tiers_days.pkl'); TD.index = pd.to_datetime(TD.index).normalize(); nq = TD.pts.sort_index()
r10 = nq.rolling(10).sum().shift(1); r20 = nq.rolling(20).sum().shift(1)
dd = (nq.cumsum() - nq.cumsum().rolling(40).max()).shift(1)
F = pd.DataFrame({'pts': nq, 'r10': r10, 'r20': r20, 'dd40': dd}).dropna()
F['bad10'] = F.r10 <= 0; F['bad20'] = F.r20 <= 0; F['q10'] = F.r10 <= F.r10.quantile(1 / 3); F['dd'] = F.dd40 <= F.dd40.quantile(1 / 3)
print('== (1) DO NQ BAD STREAKS PERSIST? NQ system pts/day (next day) by flag; n days, mean, t, 2018-21 / 2022-25')
for f in ('bad10', 'bad20', 'q10', 'dd'):
    for v in (True, False):
        s = F[F[f] == v].pts; a = s[s.index < '2022-01-01']; b = s[s.index >= '2022-01-01']
        print(f' {f:6} {str(v):5} n {len(s):4d} mean {s.mean():+6.1f} t {tstat(s):+4.1f} | {a.mean():+6.1f} ({tstat(a):+.1f}) | {b.mean():+6.1f} ({tstat(b):+.1f})')
def resample1h(x):
    y = x.set_index('dt').resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
    y['day'] = y.dt.dt.normalize(); y['m'] = y.dt.dt.hour * 60 + y.dt.dt.minute; return y
CASES = {'GOLD.F': (570, 960), 'CRUDEOIL.F': (570, 960)}
print('\n== (2) GOLD / CRUDE flips by NQ bad-streak flag (bps/day; flat at 16:00 ET; US window 9:30-16:00)')
for s, (lo, hi) in CASES.items():
    x15 = load(s); x1h = resample1h(x15)
    for tf, x in (('15m', x15), ('1h', x1h)):
        for nm, k in (('pure', None), ('stop.25', .25), ('stop.40', .4)):
            T = trades(x, k, lo, hi); T['day'] = pd.to_datetime(T.t).dt.normalize(); d = T.groupby('day').bps.sum()
            J = F.join(d.rename('g'), how='left').fillna({'g': 0.0}); allm = J.g.mean()
            out = []
            for f in ('bad10', 'bad20', 'q10', 'dd'):
                s1 = J[J[f]].g; a = s1[s1.index < '2022-01-01']; b = s1[s1.index >= '2022-01-01']
                ok = len(a) > 30 and len(b) > 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
                out.append(f'{f} n{len(s1)} {s1.mean():+5.1f} (t {tstat(s1):+.1f}; {a.mean():+.1f}/{b.mean():+.1f}){"  PASS" if ok else ""}')
            print(f' {s:10} {tf:3} {nm:7} all-days {allm:+5.1f} bps/day | ' + ' | '.join(out), flush=True)
