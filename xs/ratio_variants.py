"""Nov 4 2026: is a daily/spot options ratio better than the live smoothed one? DECLARED BEFORE RESULTS.
27-ticker history (2018-2025) only. Trades = live exits (wide_T base+sticky) for those names, entries 2019-2025, N=3 slots (27 names can't fill 6).
Variants of the rank/gate value, all lagged one day (known before entry):
  live252  = opt_contracts*100/shares, rolling-252 median          (current)
  raw1     = prior-day raw ratio        avg5 / avg20 = rolling means         z = raw1 / rolling-252 median (surge)
  dollar252 = option premium / dollar volume, 252 median            call252 = call share of option volume, 252 median
Two uses: RANK (no gate, ties by value, top-3) and GATE+RANK (value above the name's own 252d median x1.5 equivalent: top 40% of the cross-section each day).
Compared with live252 rank, 40 matched runs, per period, beats-live share."""
import sys
sys.argv = sys.argv[:1] + ['live']
_s = open('xs/uni_slots_ratio.py').read(); exec(_s[:_s.index("base = evN(")])
O = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio.csv', parse_dates=['date']); O = O[O.sh > 0].copy(); O['r'] = O.opt_contracts * 100 / O.sh
def P(col): return O.pivot_table(index='date', columns='tk', values=col).sort_index()
R, DR, CS = P('r'), P('dollar_ratio'), P('call_share')
med = lambda x: x.rolling(252, min_periods=60).median()
V = {'live252': med(R), 'raw1': R, 'avg5': R.rolling(5).mean(), 'avg20': R.rolling(20).mean(), 'surge z': R / med(R), 'dollar252': med(DR), 'call252': med(CS), 'raw dollar1': DR}
V = {k: v.shift(1) for k, v in V.items()}
Y0 = W[W.tk.isin(R.columns) & (W.entry < '2026-01-01')].reset_index(drop=True); d = Y0.entry.dt.normalize().values
def attach(nm, gate):
    Y = Y0.copy(); M = V[nm]; Y['ratio'] = [M[t].asof(pd.Timestamp(x)) if t in M else np.nan for t, x in zip(Y.tk, d)]; Y = Y[Y.ratio.notna()].reset_index(drop=True)
    if gate:
        q = Y.groupby(Y.entry.dt.normalize()).ratio.transform(lambda s: s.rank(pct=True)); Y = Y[(q >= .6) | (Y.groupby(Y.entry.dt.normalize()).ratio.transform('size') < 4)].reset_index(drop=True)
    return Y
print('trades in 27-name set', len(Y0), ' tickers', Y0.tk.nunique())
for gate in (False, True):
    print(f'== {"GATE top40%/day + rank" if gate else "RANK only (no gate)"}  N=3')
    base = evN(attach('live252', gate), 3)
    for nm in V: show(nm, attach(nm, gate), 3, base)
# plain correlation: entry-ratio vs trade return rank IC by period
print('== rank IC (spearman of value vs trade return, within entry-day cross-section approximated pooled)')
from scipy.stats import spearmanr
for nm in V:
    Y = attach(nm, False); ic = []
    for p, (a, b) in {'19-21': ('2019', '2022'), '22-25': ('2022', '2026')}.items():
        z = Y[(Y.entry >= a) & (Y.entry < b)]; ic.append(f'{p} {spearmanr(z.ratio, z.ret)[0]:+.3f}')
    print(f'  {nm:<14}', ' | '.join(ic))
