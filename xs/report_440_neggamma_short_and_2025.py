"""Nov 24 2026 report 440 = Rey: (a) why did the negative-gamma long flips go bad in 2025, (b) does the SHORT side work in negative gamma (dealer hedging amplifies moves both ways).  DECLARED BEFORE LOOKING.
(a) descriptive only: monthly mean of G20-gated long flips on negative-gamma days in 2025 (is it one month, e.g. the April crash, or broad?).
(b) bear 15m vector flips (short entry at the next open), exit = first bullish vector bar (next open), cost 4bp (large caps, borrow ignored), G20 gate, real gex (prior-close net_gex < 0 = NEG), entries 2018-07..2025-12-30.  Prediction: short flips on NEG days beat short flips on non-NEG days (mean).  One hypothesis -> |t| >= 2, month-clustered, same sign train (2019-21) / test (2018-H2 + 2022-25); by year shown."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
G20 = Rr.rolling(20, min_periods=10).median().shift(1) >= 2
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and glob.glob(UP + f'lse_ohlcv_cache/{f[:-8]}_15m_2018-01-01_*.parquet')); print('names', len(names))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
GX = UP + 'xs/ext/gex/'; names = [t for t in names if os.path.exists(GX + t + '.csv')]
L, S_ = [], []
for tk in names:
    H = rth(pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}), 'ts'); n = len(H); o_ = H.open.values
    day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']])
    g = pd.read_csv(GX + tk + '.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; gprev = g.shift(1).reindex(days, method='ffill').values
    gate = G20[tk].reindex(days, method='ffill').values.astype(float) == 1 if tk in G20 else np.zeros(n, bool)
    for dirn, bucket in ((1, L), (-1, S_)):
        for fb in [i + 1 for i, dd in fls if dd == dirn and i + 2 < n]:
            e = fb + 1; t0 = H.dt[fb - 1]
            if e >= n or t0 < pd.Timestamp('2018-07-01') or t0 > pd.Timestamp('2025-12-30') or not np.isfinite(gprev[fb - 1]) or not gate[fb - 1]: continue
            x = None
            for j in range(e + 1, min(n, e + 6000)):
                if regs[j - 1] == -dirn: x = j; break
            if x is None: continue
            bucket.append(dict(tk=tk, entry=H.dt[e], ret=dirn * (o_[x] / o_[e] - 1) - 4e-4, neg=bool(gprev[fb - 1] < 0), hold=(H.dt[x] - H.dt[e]).total_seconds() / 86400))
L = pd.DataFrame(L); S_ = pd.DataFrame(S_)
for D in (L, S_): D['half'] = np.where((D.entry >= '2019-01-01') & (D.entry < '2022-01-01'), 'train', 'test'); D['yr'] = D.entry.dt.year; D['mon'] = D.entry.dt.to_period('M')
print('(a) 2025, LONG flips on negative-gamma days, by month (G20-gated, X1):')
for m, g in L[(L.yr == 2025) & L.neg].groupby('mon'): print(f"   {m}: n {len(g):3d} mean {g.ret.mean()*100:5.2f}% win {(g.ret>0).mean()*100:3.0f}%")
print('\n(b) SHORT flips (G20-gated) by year: NEG-gamma vs non-NEG')
for y, g in S_.groupby('yr'):
    a, b = g[g.neg], g[~g.neg]; print(f"   {y}: NEG n{len(a):4d} mean {a.ret.mean()*100:6.2f} win {(a.ret>0).mean()*100:3.0f} | non-NEG n{len(b):4d} mean {b.ret.mean()*100:6.2f} win {(b.ret>0).mean()*100:3.0f}")
def contrast(A, B, lab, col='ret'):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); g = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x[col][x.s == 1].sum(), s0=x[col][x.s == -1].sum()))); m1, m0 = A[col].mean(), B[col].mean()
    se = np.sqrt((((g.s1 - g.n1 * m1) / len(A) - (g.s0 - g.n0 * m0) / len(B)) ** 2).sum()); f = lambda x: f"n{len(x):5d} win {(x[col]>0).mean()*100:4.1f} med {x[col].median()*100:6.2f} mean {x[col].mean()*100:6.2f}"
    print(f"  {lab}\n    A {f(A)}\n    B {f(B)}\n    A-B {(m1-m0)*100:+.2f}pp  t {(m1-m0)/se:+.2f} | " + ' | '.join(f"{h} {(A[A.half==h][col].mean()-B[B.half==h][col].mean())*100:+.2f}pp (n {len(A[A.half==h])}/{len(B[B.half==h])})" for h in ('train', 'test')))
print(); contrast(S_[S_.neg], S_[~S_.neg], 'H SHORT flips: A = NEG-gamma days, B = other days; predict A > B')
print(f"\n   all short flips (G20): n {len(S_)} mean {S_.ret.mean()*100:.2f} win {(S_.ret>0).mean()*100:.1f}; NEG shorts alone mean {S_[S_.neg].ret.mean()*100:.2f}; NEG longs {L[L.neg].ret.mean()*100:.2f} (n {int(L.neg.sum())})")
