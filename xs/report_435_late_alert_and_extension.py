"""Nov 24 2026 report 435 = what does a LATE alert cost, and does chasing an extended flip cost more?  DECLARED BEFORE LOOKING.
Trades: 15m bull flips as in 434 (G20-gated primary, ungated robustness), exit X1 (first bearish vector bar, next open), 4bp, no time holds, 25 names, entries 2018-07+.
H_LAG: entering k bars after the flip's own entry (k = 1,2,3; same exit) earns less -- measured on the COMMON set of trades still open after k bars (no survivorship); prediction: mean return falls with k.
H_EXT: EXT = (flip-bar close - vector level) / 15m ATR; high EXT (top third) earns less than low EXT (bottom third) -- chasing.  Prediction: low EXT > high EXT.
PASS (2 hypotheses -> |t| >= 2.2): same sign train/test, month-clustered t on the contrast (paired for LAG: mean difference of per-trade returns, k=3 vs k=0)."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
G20 = Rr.rolling(20, min_periods=10).median().shift(1) >= 2
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and glob.glob(UP + f'lse_ohlcv_cache/{f[:-8]}_15m_2018-01-01_*.parquet')); print('names', len(names))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
rows = []
for tk in names:
    H = rth(pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}), 'ts')
    day = H.dt.dt.normalize().values; regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']]); n = len(H); o_ = H.open.values; c_ = H.close.values
    flips = [i + 1 for i, dd in fls if dd == 1 and i + 2 < n]; gate = G20[tk].reindex(pd.DatetimeIndex(day), method='ffill').values.astype(float) == 1 if tk in G20 else np.zeros(n, bool)
    for fb in flips:
        e = fb + 1
        if e + 3 >= n or H.dt[fb - 1] < pd.Timestamp('2018-07-01'): continue
        x = None
        for j in range(e + 1, min(n, e + 6000)):
            if regs[j - 1] == -1: x = j; break
        if x is None or x <= e + 3 or not np.isfinite(a15[fb - 1]) or a15[fb - 1] <= 0: continue          # common set: still open after 3 bars
        r = {k: o_[x] / o_[e + k] - 1 - 4e-4 for k in range(4)}
        rows.append(dict(tk=tk, entry=H.dt[e], gate=bool(gate[fb - 1]), ext=(c_[fb - 1] - lv[fb - 1]) / a15[fb - 1], **{f'r{k}': r[k] for k in range(4)}))
T = pd.DataFrame(rows); T['ret'] = T.r0; T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['mon'] = T.entry.dt.to_period('M')
def mt(d):   # month-clustered t of a per-trade series
    g = d.groupby('mon').agg(s=('v', 'sum'), n=('v', 'size')); m = d.v.mean(); se = np.sqrt(((g.s - g.n * m) ** 2).sum()) / len(d); return m / se if se > 0 else np.nan
for arm, D in (('PRIMARY: G20-gated', T[T.gate]), ('ROBUSTNESS: all', T)):
    print(f"\n=== {arm} (common set n {len(D)}) ===\n H_LAG mean net return by entry delay (bars of 15m):")
    for k in range(4): print(f"   +{k} bar{'s' if k != 1 else ''}: mean {D[f'r{k}'].mean()*100:5.2f}%  win {(D[f'r{k}']>0).mean()*100:4.1f}  med {D[f'r{k}'].median()*100:5.2f}")
    d = D.assign(v=D.r3 - D.r0); print(f"   delay-3 minus on-time: {d.v.mean()*100:+.2f}pp  month-clustered t {mt(d):+.2f} | " + ' | '.join(f"{h} {d[d.half==h].v.mean()*100:+.2f}pp (t {mt(d[d.half==h]):+.1f})" for h in ('train', 'test')) + ' | ' + ' '.join(f"{e} {d[d.era==e].v.mean()*100:+.2f}" for e in ('18-19', '20-21', '22', '23-26')))
    q1, q2 = D.ext.quantile([1 / 3, 2 / 3]); lo, hi = D[D.ext <= q1], D[D.ext >= q2]
    print(f" H_EXT (EXT terciles at {q1:.2f}/{q2:.2f} ATR): low n{len(lo)} mean {lo.r0.mean()*100:.2f} win {(lo.r0>0).mean()*100:.1f} | mid mean {D[(D.ext>q1)&(D.ext<q2)].r0.mean()*100:.2f} | high n{len(hi)} mean {hi.r0.mean()*100:.2f} win {(hi.r0>0).mean()*100:.1f}")
    dd = pd.concat([lo.assign(s=1), hi.assign(s=-1)]); g = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.r0[x.s == 1].sum(), s0=x.r0[x.s == -1].sum()))); m1, m0 = lo.r0.mean(), hi.r0.mean(); infl = (g.s1 - g.n1 * m1) / len(lo) - (g.s0 - g.n0 * m0) / len(hi); t = (m1 - m0) / np.sqrt((infl ** 2).sum())
    print(f"   low-minus-high {(m1-m0)*100:+.2f}pp  month-clustered t {t:+.2f} | " + ' | '.join(f"{h} {(D[(D.half==h)&(D.ext<=q1)].r0.mean()-D[(D.half==h)&(D.ext>=q2)].r0.mean())*100:+.2f}pp" for h in ('train', 'test')) + ' | ' + ' '.join(f"{e} {(D[(D.era==e)&(D.ext<=q1)].r0.mean()-D[(D.era==e)&(D.ext>=q2)].r0.mean())*100:+.2f}" for e in ('18-19', '20-21', '22', '23-26')))
