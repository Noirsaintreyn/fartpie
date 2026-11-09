"""Nov 8 2026: OVERNIGHT DRIFT WITH A STOP ALWAYS ON - which window and stop keep the edge? DECLARED BEFORE RESULTS.
Real NQ futures 5m bars Jan 2014 - Jan 2026. Session day D = 18:00 ET (prior evening) to 09:30. Long only (the drift). Entry at the open of the first bar of the window,
exit at the close of its last bar; stop = k x the 20-day average OVERNIGHT range (18:00 - 09:30 high-low, prior nights), checked on every 5m bar, filled at the stop; 1 pt cost.
Windows (ET): 18:00-09:25 | 18:00-04:00 | 00:00-04:00 | 01:00-05:00 | 02:00-04:00 (the European-open window in Boyarchenko, Larsen & Whelan) | 02:00-09:25 | 03:00-09:25
Stops k = 0.25 / 0.35 / 0.50 / 0.75 (and none for reference). Filters: every night | negative QQQ GEX at the prior close (2018+) | after an RTH down day.
Per cell: nights, win, points per night, nights >= +7.5, worst night, max drawdown, 2014-17 / 2018-21 / 2022-26 points per night."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); on = X[~X.rth & ((X.m >= 1080) | (X.m < 565))].copy()
on['sm'] = np.where(on.m >= 1080, on.m - 1080, on.m + 360)        # minutes since 18:00
R = X[X.rth].groupby('day').agg(ro=('open', 'first'), rc=('close', 'last'), sym=('sym', 'first'))
g = on.groupby('day'); NR = (g.high.max() - g.low.min()); ONADR = NR.rolling(20).mean().shift(1)
Gx = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); Gx.index = Gx.index.normalize()
dl = list(R.index); prevd = pd.Series(dl[:-1], index=dl[1:]); gneg = pd.Series(Gx.reindex(prevd.values).values < 0, index=prevd.index)
gknown = pd.Series(Gx.reindex(prevd.values).notna().values, index=prevd.index); down = (R.rc < R.ro).shift(1)
symok = (g.sym.first().reindex(R.index) == R.sym)
WIN = {'18:00-09:25': (0, 930), '18:00-04:00': (0, 600), '00:00-04:00': (360, 600), '01:00-05:00': (420, 660), '02:00-04:00': (480, 600), '02:00-09:25': (480, 930), '03:00-09:25': (540, 930)}
gi = on.groupby('day').indices; oo, hh, ll, cc, sm = (on[k].values for k in ('open', 'high', 'low', 'close', 'sm'))
def night(d, a, b, k):
    ix = gi.get(d)
    if ix is None or not np.isfinite(ONADR.get(d, np.nan)): return None
    w = ix[(sm[ix] >= a) & (sm[ix] < b)]
    if len(w) < max(3, (b - a) // 5 * .8): return None
    e = oo[w[0]]; S = k * ONADR[d] if k else None
    for i in w:
        if S is not None and ll[i] <= e - S: return -S - 1.0, -S
    return cc[w[-1]] - e - 1.0, (ll[w].min() - e)
if __name__ == '__main__':
    days = [d for d in R.index if d in gi and symok.get(d, False)]
    print(f'nights {len(days)}; median overnight range {NR.median():.0f} pts (2022-25 {NR["2022":].median():.0f})', flush=True)
    print(f'  {"window":<12} {"stop":<5} {"filter":<10} nights  win  pts/night  >=+7.5  worst  maxDD | 14-17 / 18-21 / 22-26', flush=True)
    best = []
    for wn, (a, b) in WIN.items():
        for k in (None, .25, .35, .5, .75):
            res = {d: night(d, a, b, k) for d in days}; res = {d: r for d, r in res.items() if r is not None}
            s = pd.Series({d: r[0] for d, r in res.items()}).sort_index()
            for fl, msk in (('every', None), ('neg gamma', gneg), ('after down', down)):
                q = s if msk is None else s[[bool(msk.get(d, False)) and (fl != 'neg gamma' or bool(gknown.get(d, False))) for d in s.index]]
                if len(q) < 100: continue
                cum = q.cumsum(); per = [q[(q.index.year >= x) & (q.index.year <= y)].mean() for x, y in ((2014, 2017), (2018, 2021), (2022, 2026))]
                print(f'  {wn:<12} {("none" if k is None else f"{k:.2f}"):<5} {fl:<10} {len(q):6d} {np.mean(q > 0):4.0%} {q.mean():+9.2f} {np.mean(q >= 7.5):7.0%} {q.min():+6.0f} {(cum - cum.cummax()).min():+6.0f} | '
                      + ' / '.join(f'{x:+5.1f}' if np.isfinite(x) else '    .' for x in per), flush=True)
                if k is not None: best.append((q.mean() / q.std() * np.sqrt(252) if q.std() > 0 else 0, wn, k, fl, q.mean(), np.mean(q >= 7.5), q.min(), all(np.isfinite(x) and x > 0 for x in per if np.isfinite(x))))
    print('\nTOP 10 WITH A STOP by Sharpe (per-night basis):', flush=True)
    for sh, wn, k, fl, mu, q75, wst, allp in sorted(best, reverse=True)[:10]:
        print(f'  {wn:<12} stop {k:.2f} {fl:<10} Sharpe {sh:+.2f} pts/night {mu:+.2f} >=7.5 {q75:.0%} worst {wst:+.0f} {"all periods +" if allp else ""}', flush=True)
