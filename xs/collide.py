"""Nov 2 2026: collision-ranker. Rankers act ONLY when a same-bar group has more eligible flips than free slots; entries, exits, sizes,
universe, breaker unchanged; no delaying, no rotation; options ratio is the fallback; every later decision is replayed.
EXACT features (as in noise_audit.py, all known at the fill):
 entry strength  = (entry-bar open - prior session close) / prior-day ATR(14, daily high-low)      [the +0.61% / +0.53% top-10% result]
 session gap     = (session open / prior close - 1) / 20-day sigma of daily returns
 1h              = the signal came from the 1h track's new flip (h1 = True)
DECLARED BEFORE RESULTS:  A hourly first (all oversubscribed bars) | scope check A1: hourly first only when exactly 1 slot is free
 B entry strength desc (signed) | neighbour B5: by within-year quintile of entry strength, ratio inside the quintile
 C session gap desc (signed)    | neighbour C5: quintile version.   Pass: >= 70% of 40 matched runs beat live return/DD in BOTH
 2019-21 and 2022-26, and the neighbour / scope check >= 60%. Also reported: collision ledger (picked vs displaced, standalone returns),
 paired return / drawdown differences, number of changed selections, share of the gain from the 5 largest changed winners, by year."""
import zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
X['yr'] = X.entry.dt.year; X['h1f'] = X.h1.astype(float)
for c in ('margin', 'gap'): X[c + '_q'] = X.groupby('yr')[c].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(X.tk, X.entry.astype('int64'))], dtype=np.uint64)
DROPS = [np.zeros(len(X), bool)] + [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
IDX = pd.bdate_range('2019-01-01', X.exit.max())
P2 = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(eq):
    eq = eq.reindex(IDX, method='ffill').fillna(0); out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    yrs = eq.groupby(eq.index.year).agg(lambda s: np.exp(s.iloc[-1] - s.iloc[0]) - 1)
    return out, yrs
def sim(ix, rank, scope_one=False, ledger=None):
    """ix: positions into X (after drops). rank(i) -> sort key tuple (smaller first) used ONLY in oversubscribed groups."""
    Z = X.iloc[ix]; o = np.lexsort((Z.tk.values, Z.h1f.values, -Z.ratio.values, Z.entry.values)); ids = Z.index.values[o]
    E = X.entry.values[ids]; op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; i = 0; N = len(ids); taken = []
    while i < N:
        j = i
        while j < N and E[j] == E[i]: j += 1
        e = E[i]
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / 6); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]; free = 6 - len(op); ok = free > 0
        if ok and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): ok = False
        if ok:
            cand = list(ids[i:j]); base = cand[:free]
            if len(cand) > free and rank is not None and (not scope_one or free == 1):
                cand = sorted(cand, key=rank)
                if ledger is not None and set(cand[:free]) != set(base): ledger.append((e, free, len(cand), cand[:free], base))
            for k in cand[:free]: op.append(X.exit.values[k]); pend.append((X.exit.values[k], X.ret.values[k])); taken.append(k)
        i = j
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / 6); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last(), taken
R = X.ratio.values; H1 = X.h1f.values; M = X.margin.fillna(-99).values; G = X.gap.fillna(-99).values; MQ = X.margin_q.fillna(-1).values; GQ = X.gap_q.fillna(-1).values
TKv = X.tk.values
RULES = {'A  hourly first': (lambda k: (-H1[k], -R[k], TKv[k]), False), 'A1 hourly first, only when 1 slot free (scope)': (lambda k: (-H1[k], -R[k], TKv[k]), True),
         'B  entry strength desc': (lambda k: (-M[k], -R[k]), False), 'B5 entry-strength quintile, ratio inside (nbr)': (lambda k: (-MQ[k], -R[k], TKv[k]), False),
         'C  session gap desc': (lambda k: (-G[k], -R[k]), False), 'C5 session-gap quintile, ratio inside (nbr)': (lambda k: (-GQ[k], -R[k], TKv[k]), False)}
def runs(rank, scope=False):
    out, yrs = [], []
    for dl in DROPS:
        ix = np.where(~dl)[0]; eq, _ = sim(ix, rank, scope); s, y = stat2(eq); out.append(s); yrs.append(y)
    return np.array(out), pd.DataFrame(yrs)
print('feature audit')
fb = X.tod == 570
print(f'  first-bar entries {fb.mean():.0%}; corr(entry strength, session gap) first-bar {X[fb].margin.corr(X[fb].gap, method="spearman"):+.2f}, later {X[~fb].margin.corr(X[~fb].gap, method="spearman"):+.2f}')
top = X.groupby(np.where(X.yr <= 2021, 0, 1)).ret.transform(lambda s: s >= s.quantile(.95))
print('  cross-tab: avg trade (big-winner ratio), rows = session-gap tercile, cols = entry-strength tercile, later-session entries only')
L = X[~fb].copy(); L['gt'] = L.groupby('yr').gap.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False)); L['mt'] = L.groupby('yr').margin.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False))
for h, nm in ((L.yr <= 2021, '2019-21'), (L.yr >= 2022, '2022-26')):
    Lh = L[h]; tt = top[Lh.index]; base = tt.mean()
    print(f'   {nm}: ' + ' | '.join('gap t%d: ' % g + ' '.join(f'{Lh[(Lh.gt == g) & (Lh.mt == m)].ret.mean():+.2%}({tt[(Lh.gt == g) & (Lh.mt == m)].mean() / base:.1f})' for m in range(3)) for g in range(3)))
B0, Y0 = runs(None)
print('\nbook replay, 40 matched runs; 2019-21 / 2022-26')
m = np.median(B0[1:], axis=0); print(f'  live (ratio)                                 ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}')
for nm, (rk, sc) in RULES.items():
    A, Y = runs(rk, sc); m = np.median(A[1:], axis=0); w1, w2 = (A[1:, 2] > B0[1:, 2]).mean(), (A[1:, 5] > B0[1:, 5]).mean()
    dr = A[1:, [0, 3]] - B0[1:, [0, 3]]; ddd = A[1:, [1, 4]] - B0[1:, [1, 4]]
    print(f'  {nm:<46} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}   beats live {w1:.0%} / {w2:.0%}' + ('  PASS' if w1 >= .7 and w2 >= .7 else ''))
    print(f'      paired annual-return diff median {np.median(dr[:, 0]):+.1%} / {np.median(dr[:, 1]):+.1%}, 10th pct {np.percentile(dr[:, 0], 10):+.1%} / {np.percentile(dr[:, 1], 10):+.1%}; '
          f'DD diff median {np.median(ddd[:, 0]):+.1%} / {np.median(ddd[:, 1]):+.1%}')
    yd = (Y.iloc[1:] - Y0.iloc[1:]).median(); print('      by year (median paired return diff): ' + ' '.join(f'{int(k)}:{v:+.1%}' for k, v in yd.items()))
    led = []; sim(np.arange(len(X)), rk, sc, led)
    pk = [k for _, _, _, p, b in led for k in set(p) - set(b)]; dp = [k for _, _, _, p, b in led for k in set(b) - set(p)]
    d_ = X.ret.values[pk].mean() - X.ret.values[dp].mean() if pk else np.nan
    gains = np.sort(X.ret.values[pk])[::-1]; tot = X.ret.values[pk].sum() - X.ret.values[dp].sum()
    print(f'      collisions changed {len(led)} (picked {len(pk)} / displaced {len(dp)} trades); standalone avg picked {X.ret.values[pk].mean():+.2%} vs displaced {X.ret.values[dp].mean():+.2%} (diff {d_:+.2%}); '
          f'top-5 picked winners = {gains[:5].sum() / tot:.0%} of the net standalone gain' if tot > 0 else
          f'      collisions changed {len(led)} (picked {len(pk)} / displaced {len(dp)}); standalone avg picked {X.ret.values[pk].mean():+.2%} vs displaced {X.ret.values[dp].mean():+.2%} (diff {d_:+.2%}); net standalone gain <= 0', flush=True)
