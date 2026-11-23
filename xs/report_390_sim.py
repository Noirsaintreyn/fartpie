"""Part 2 of report 390: ratio gate + live book sim on the trade sets built by report_390_persistence_pivots.py (see its docstring)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
r = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio_sp.csv', parse_dates=['date']); r['date'] = r.date.astype('datetime64[ns]'); RS = {}
for tk, g in r.groupby('tk'): s = g.sort_values('date').set_index('date').r; RS[tk] = s.rolling(252, min_periods=126).median().shift(1)
def attach(T):
    T = T.copy(); T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit); T['ratio'] = [RS[t].reindex([e.normalize()], method='ffill').values[0] if t in RS else np.nan for t, e in zip(T.tk, T.entry)]
    T = T[(T.entry >= '2019-01-01') & T.ratio.notna() & (T.ratio >= 1.5) & (T.exit < END)].copy(); T['brank'] = 0.0; T['h1'] = False; return T.reset_index(drop=True)
print('BASE vs live b302 reference: 4253 trades CAGR 23.6% Sharpe 2.60 (train 4.15 | test 1.85)')
base = None
for v in ('BASE', 'k1', 'k2', 'k3'):
    T0 = pd.read_parquet(f'{SP}tr390_{v}.parquet'); T = attach(T0); tr, te = T[T.entry < '2022-01-01'], T[T.entry >= '2022-01-01']
    TK, _ = sim_swap(T, None); z = spread(TK); z = z[z.index >= '2019-01-01']; a, b_ = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print('%-5s flips(all) %6d gated %5d trades %4d | flip mean train %+.2f%% test %+.2f%% | CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f' % (v, len(T0), len(T), len(TK), tr.ret.mean() * 100, te.ret.mean() * 100, cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b_) * 100, sharpe(b_.values)), flush=True)
