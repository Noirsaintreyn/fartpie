"""Nov 24 2026 report 421. Rey: 'maybe the options ratio should be higher; swing (252d) and intraday should differ; may need an intraday ratio.  Decide now what's best.'
DECLARED: universe = wide_T 15m flips on PIT S&P names (sp True, h1 False; 168-name pool, 2018-07..2026-08).  Daily ratio r = 100 x option contracts / shares volume (same definition as the live gate).  Gate(N, theta) = rolling-N-day MEDIAN of r (shifted 1 day) >= theta, N in {1, 5, 20, 60, 252} (N=1 is the prior day's own ratio = the 'intraday' ratio), theta in {1.0, 1.5, 2, 3, 4}.  25 gates.
SELECTION on TRAIN only (2019-21): per outcome family, pick the gate with the highest mean net return per trade subject to n_train >= 400.  Families: ALL = every flip with the vector exit (swing-style); SD = flips whose vector exit is the same session (intraday-style).  Then report the pick on TEST (2018 + 2022-26) and by era, vs the live gate (252, 1.5) and vs ungated PIT-S&P flips; month-clustered t of pick minus live gate.
PASS (a pick is adopted): beats the live gate on mean and median in the test half and in >= 3 of 4 eras (n>=30), month-clustered |t| >= 2 on test, and n_test >= 1000 (ALL) / 400 (SD).  Else keep the live gate."""
import numpy as np, pandas as pd
UP = '/mnt/user-data/uploads/fartpie/'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date'])
Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index(); print('ratio matrix', Rr.shape, Rr.index.min().date(), Rr.index.max().date())
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W = W[(~W.h1) & W.sp & (pd.to_datetime(W.entry) >= '2018-07-01') & (pd.to_datetime(W.exit) <= '2026-08-14') & W.tk.isin(Rr.columns)].reset_index(drop=True)
W['entry'] = pd.to_datetime(W.entry); W['exit'] = pd.to_datetime(W.exit); W['d0'] = W.entry.dt.normalize(); W['n'] = W.ret - 4e-4; W['sd'] = W.entry.dt.normalize() == W.exit.dt.normalize(); W['mo'] = W.entry.dt.to_period('M')
e_ = W.entry; W['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test'); W['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26'])
G = {}; di = Rr.index
for N in (1, 5, 20, 60, 252):
    med = Rr.rolling(N, min_periods=max(1, N // 2)).median().shift(1)
    vals = np.full(len(W), np.nan); a = di.searchsorted(W.d0.values, side='right') - 1; ok = a >= 0; cols = {t: i for i, t in enumerate(Rr.columns)}; ci = np.array([cols[t] for t in W.tk]); vals[ok] = med.values[a[ok], ci[ok]]
    for th in (1.0, 1.5, 2, 3, 4): G[(N, th)] = vals >= th
print('flips', len(W), '| ungated train mean %.3f%% test %.3f%%' % (W[W.half == 'train'].n.mean() * 100, W[W.half == 'test'].n.mean() * 100))
live = G[(252, 1.5)]
def stats(d): return f"n{len(d):6d} win {(d.n>0).mean()*100:5.1f} med {d.n.median()*100:6.2f} mean {d.n.mean()*100:5.2f}"
print('\nTRAIN grid (mean net %/trade; n) rows N, cols theta 1.0 1.5 2 3 4:  ALL | SD')
for fam, mask in (('ALL', np.ones(len(W), bool)), ('SD', W.sd.values)):
    print(fam)
    for N in (1, 5, 20, 60, 252):
        s = f'  N={N:3d}: '
        for th in (1.0, 1.5, 2, 3, 4): d = W[(W.half == 'train').values & mask & G[(N, th)]]; s += f"{d.n.mean()*100:6.2f} ({len(d):4d}) " if len(d) else '   -   '
        print(s)
for fam, mask, nmin_t, nmin_s in (('ALL', np.ones(len(W), bool), 400, 1000), ('SD', W.sd.values, 400, 400)):
    best = None
    for k, g in G.items():
        d = W[(W.half == 'train').values & mask & g]
        if len(d) >= nmin_t and (best is None or d.n.mean() > best[1]): best = (k, d.n.mean())
    k = best[0]; print(f'\n=== {fam}: train pick = N{k[0]} theta {k[1]} (train mean {best[1]*100:.2f}%) vs LIVE (252,1.5) ===')
    for lab, g in (('PICK', G[k]), ('LIVE', live), ('UNGATED', np.ones(len(W), bool))):
        d = W[mask & g]; t_ = d[d.half == 'test']; print(f"{lab:8s} train {stats(d[d.half=='train'])} | test {stats(t_)} |", ' '.join(f"{era} {(x.n>0).mean()*100:4.1f}/{x.n.median()*100:5.2f}(n{len(x)})" if len(x := d[d.era == era]) >= 30 else f'{era} n<30' for era in ('18-19', '20-21', '22', '23-26')))
    P = W[mask & G[k]]; L = W[mask & live]; pt = P[P.half == 'test'].groupby('mo').n.mean(); lt = L[L.half == 'test'].groupby('mo').n.mean(); m = (pt - lt).dropna(); print(f'   test PICK minus LIVE monthly mean diff {m.mean()*100:+.3f}pp (t {m.mean()/(m.std(ddof=1)/np.sqrt(len(m))):+.1f}, {len(m)} months)')
