"""Nov 22 2026 report 352. Gao-Han-Li-Zhou market intraday momentum and the NY Fed overnight drift, on NQ 5m (RTH 09:30-16:00 ET; overnight from the extended file).  Declared before running.
M1) LAST-30-MIN MOMENTUM: signal = return from the previous RTH close to 15:30 (the rest of the day); trade the last 30 min (15:30 open -> 16:00 close) long if signal > 0 else short; points per contract, 0.75 pt cost.  Variant M1b: signal = first-30-min return (09:30 open -> 10:00 open).  THEORY: hedging demand of short-gamma dealers and leveraged ETFs rebalancing into the close pushes the last 30 min in the direction of the day.
M2) GAMMA FLAG: QQQ net GEX proxy = (call_gex - put_gex)/(call_gex + put_gex) of the PRIOR day (data through 2025-12): is momentum stronger on negative-net-gamma days (the paper's channel)?  Terciles of the flag on 2019-21, shown for train and test.
M3) NQ OVERNIGHT, buy-the-dip (NY Fed): hold long from 16:00 ET (RTH close) to 09:30 next day (full overnight) and the 02:00-03:00 hour, split by the sign / size of that day's RTH return (down day / up day).  Points per contract, 0.75 pt cost per round trip.
Train 2019-21 | test 2018 + 2022-26.  Why they fail: published edges decay; the last-30-min effect is small vs costs on NQ (0.75 pt per ~25k) and the NQ ORB/vector/stack already capture the trend days."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N['dt'] = pd.to_datetime(N.dt); N = N[N.dt >= '2017-06-01'].copy(); N['d'] = N.dt.dt.normalize()
R = N[N.rth & (N.m >= 570) & (N.m <= 955)]; rows = {}
for d, x in R.groupby('d'):
    x = x.sort_values('m'); mm = x.m.values
    if len(x) < 70 or 930 not in mm or 925 not in mm or 600 not in mm: continue
    rows[d] = dict(o=x.open.iloc[0], o30=x[x.m == 600].open.iloc[0], o1530=x[x.m == 930].open.iloc[0], c=x.close.iloc[-1], sym=x.sym.iloc[0])
F = pd.DataFrame(rows).T; F['o'] = F.o.astype(float); F['prevc'] = F.c.astype(float).shift(1); F['ok'] = F.sym == F.sym.shift(1); F = F[F.ok]
F['sig'] = np.sign(F.o1530.astype(float) - F.prevc); F['last'] = F.c.astype(float) - F.o1530.astype(float); F['pnl'] = F.sig * F['last'] - .75
F['sigb'] = np.sign(F.o30.astype(float) - F.o); F['pnlb'] = F.sigb * (F.c.astype(float) - F.o1530.astype(float)) - .75
tr = (F.index >= '2019-01-01') & (F.index < '2022-01-01'); te = (F.index >= '2018-01-01') & ~tr
def f(s, m): v = s[m].dropna(); return f'{v.mean():+6.2f} (t{v.mean() / (v.std() / np.sqrt(len(v))):+4.1f}, n{len(v)}, win {(v > 0).mean():.0%})'
print(f'days {len(F)}; mean |last-30m move| {F["last"].abs().mean():.1f} pts')
print(f'M1  rest-of-day signal -> last 30 min, pts/day net of 0.75: train {f(F.pnl, tr)} | test {f(F.pnl, te)}; gross hit rate (sign agrees) {np.mean(np.sign(F["last"]) == F.sig):.0%}')
print(f'M1b first-30-min signal -> last 30 min:                  train {f(F.pnlb, tr)} | test {f(F.pnlb, te)}')
g = pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date'); g['nge'] = ((g.call_gex - g.put_gex) / (g.call_gex + g.put_gex)).shift(1); F['nge'] = g.nge.reindex(F.index.values).values; G = F.dropna(subset=['nge', 'pnl'])
q = G.loc[(G.index >= '2019-01-01') & (G.index < '2022-01-01'), 'nge'].quantile([1 / 3, 2 / 3]).values; print(f'M2  net-gamma flag (prior day; low = most negative), cuts {q.round(3)}, data to {G.index.max().date()}')
for lab, m in (('low (negative gamma)', G.nge <= q[0]), ('mid', (G.nge > q[0]) & (G.nge < q[1])), ('high (positive gamma)', G.nge >= q[1])): print(f'      {lab:22s} train {f(G.pnl, m & (G.index >= "2019-01-01") & (G.index < "2022-01-01"))} | test {f(G.pnl, m & ((G.index < "2019-01-01") | (G.index >= "2022-01-01")))}')
E_ = N[~N.rth | (N.m > 955)].copy(); E_['t'] = E_.dt; ov = {}
Nn = N.set_index('dt').sort_index()
for d in F.index[:-1]:
    nd = F.index[F.index.get_loc(d) + 1]
    a = Nn.loc[d + pd.Timedelta(hours=16): nd + pd.Timedelta(hours=9, minutes=25)]
    if len(a) < 100 or a.sym.nunique() > 1 or a.sym.iloc[0] != F.sym[d]: continue
    h2 = a.loc[nd + pd.Timedelta(hours=2): nd + pd.Timedelta(hours=2, minutes=55)]
    if len(h2) < 10: continue
    ov[d] = dict(full=a.close.iloc[-1] - a.open.iloc[0], h23=h2.close.iloc[-1] - h2.open.iloc[0], rth=F.c[d] - F.o[d] if False else float(F.c[d]) - float(F.o[d]))
O = pd.DataFrame(ov).T.astype(float); O['dn'] = O.rth < 0; trO = (O.index >= '2019-01-01') & (O.index < '2022-01-01'); teO = (O.index >= '2018-01-01') & ~trO
print(f'\nM3  NQ overnight (16:00 -> next 09:25) and the 02:00-03:00 hour, pts per contract after 0.75 cost, by that day\'s RTH direction (n {len(O)})')
for lab, s in (('full overnight', O.full - .75), ('02-03h hour', O.h23 - .75)):
    for nm, m in (('all days', np.ones(len(O), bool)), ('after DOWN day', O.dn.values), ('after UP day', ~O.dn.values)): print(f'   {lab:15s} {nm:15s} train {f(s, trO & m)} | test {f(s, teO & m)}')
