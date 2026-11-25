"""Report 486 (Nov 25 2026): Rey - 'knowing ATRP worked, anything else in that direction that helps analyze the better moves?' Reports 466/467 showed GK22, HAR, VOV, GAPABS20, SURP are the same information as ATRP. New, untested direction: is the volatility UPSIDE JUMPS or just two-sided chop?
DECLARED BEFORE RESULTS. Mechanism: the flip pays through gap/jump spikes (news, squeezes). Two stocks with the same ATRP can differ: one moves by repeated up-jumps (active catalyst flow / speculative bid), the other by symmetric chop that mean-reverts. Measures at the flip close (daily bars, last 60 sessions): UPS = share of realized variance from positive-return days (upside semivariance share); MAXR = largest 1-day return; NJ = count of days with return >= +15%; DNJ = count of days <= -15% (control: downside jumps should NOT help).
Sample: structure_small daily flips, ATRP top-30% group inside each bucket (train ECDF cut), px<20/<10/<5, test 2022+. Within the group split by TRAIN-frozen terciles of each measure. Pass (per measure) = top tercile beats bottom tercile on mean AND REACH in both eras (2022, 2023-26) AND month-clustered t of the mean gap >= 2. DNJ is a control (expected: no gain)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O['tr'] = O.d.dt.year <= 2021; W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl')
M = {k: np.full(len(O), np.nan) for k in ('UPS', 'MAXR', 'NJ', 'DNJ')}
for tk, g in O.groupby('tk'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index; c = df.close.values; r = np.r_[np.nan, c[1:] / c[:-1] - 1]
    for i, d in zip(g.index, g.d):
        j = ix.searchsorted(pd.Timestamp(d))
        if j < 61 or j >= len(ix) or ix[j] != pd.Timestamp(d): continue
        x = r[j - 59:j + 1]; v = x ** 2; M['UPS'][i] = v[x > 0].sum() / v.sum() if v.sum() > 0 else np.nan; M['MAXR'][i] = x.max(); M['NJ'][i] = (x >= .15).sum(); M['DNJ'][i] = (x <= -.15).sum()
for k, x in M.items(): O[k] = x
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+6.1%} med {x.r.median():+.1%}'
for cap in (20, 10, 5):
    S = O[(O.px < cap)].dropna(subset=['r', 'PEAK', 'ATRP', 'UPS']).copy(); S['mo'] = S.d.dt.to_period('M').astype(str); ref = np.sort(S[S.tr].ATRP.values); S['a'] = np.searchsorted(ref, S.ATRP.values) / len(ref)
    G = S[S.a >= np.quantile(S[S.tr].a, .7)]; tr = G[G.tr]; te = G[~G.tr]; print(f'\n######## sub-${cap} ATRP top-30% group: train n {len(tr)} | test {st(te)}')
    for m in ('UPS', 'MAXR', 'NJ', 'DNJ'):
        q = np.quantile(tr[m], [1 / 3, 2 / 3]); lo_ = te[te[m] <= q[0]]; hi_ = te[te[m] > q[1]]; mid = te[(te[m] > q[0]) & (te[m] <= q[1])]
        if len(np.unique(q)) < 2: lo_ = te[te[m] <= q[0]]; hi_ = te[te[m] > q[0]]
        ma = hi_.groupby('mo').r.mean(); mb = lo_.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b; t = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 else np.nan
        e = lambda x, y: f"{x[x.d.dt.year == y].r.mean():+.1%}" if y == 2022 else f"{x[x.d.dt.year >= 2023].r.mean():+.1%}"
        print(f'  {m} cuts {np.round(q, 2)}: LOW {st(lo_)} | HIGH {st(hi_)} | gap {d.mean():+.1%} t {t:+.2f} | 2022 hi {e(hi_, 2022)} lo {e(lo_, 2022)} | 2023-26 hi {e(hi_, 2023)} lo {e(lo_, 2023)}')
