"""Nov 24 2026 report 437 = Rey: pair gamma with the day's expected range / S-R location and take only the high-quality flips.  DECLARED BEFORE LOOKING.
Prior work: the range-used idea was already run on the live book (xs/stop_book.py 'rangez': skipping or shrinking stretched entries did not help; the most stretched fifth +0.13% vs +0.31%).  What is NEW here is the pairing with the dealer-gamma regime and the support-location filter.
Trades: 15m bull flips, G20-gated (PRIMARY) / all (ROBUSTNESS), names with gex files (prior-close net_gex, shifted 1 day; entries through 2025-12-30 only: the gex files end there), next-open entry, X1 vector exit, 4bp, no time holds.
USED = (today's high - low so far through the flip bar) / close / mean daily range-fraction of the prior 10 sessions.  LOW <= 0.5, HIGH >= 1.0.  NEG = prior-close net_gex < 0.
SUPPORT = the flip bar's session-to-date low or the prior day's low lies within 0.5 x ATR15 of the lowest low of the 4 bars up to the flip (the flip is a bounce off an established low) -- AT vs NOT.
H1 NEG gamma x room: within NEG flips, LOW used beats HIGH used.
H2 NEG & LOW (the pairing) beats every other flip.
H3 AT support beats NOT (all flips).
PASS (3 hypotheses -> |t| >= 2.4 month-clustered, same sign train 2019-21 / test 2018-H2+2022-25, eras shown)."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
G20 = Rr.rolling(20, min_periods=10).median().shift(1) >= 2
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and glob.glob(UP + f'lse_ohlcv_cache/{f[:-8]}_15m_2018-01-01_*.parquet')); print('names', len(names))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
GX = UP + 'xs/ext/gex/'; names = [t for t in names if os.path.exists(GX + t + '.csv')]; print('names with gex', len(names))
rows = []
for tk in names:
    H = rth(pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}), 'ts'); n = len(H); o_ = H.open.values; c_ = H.close.values; hh = H.high.values; ll = H.low.values
    day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']])
    g = pd.read_csv(GX + tk + '.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; gprev = g.shift(1).reindex(days, method='ffill').values
    gate = G20[tk].reindex(days, method='ffill').values.astype(float) == 1 if tk in G20 else np.zeros(n, bool)
    D = H.groupby(H.dt.dt.normalize()).agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); frac = ((D.h - D.l) / D.c).rolling(10).mean().shift(1); pdl = D.l.shift(1)
    dstart = pd.Series(np.arange(n), index=H.dt).groupby(H.dt.dt.normalize().values).min()
    for fb in [i + 1 for i, dd in fls if dd == 1 and i + 2 < n]:
        e = fb + 1; t0 = H.dt[fb - 1]; d = pd.Timestamp(day[fb - 1])
        if e >= n or t0 < pd.Timestamp('2018-07-01') or t0 > pd.Timestamp('2025-12-30') or fb < 6 or not np.isfinite(gprev[fb - 1]) or not np.isfinite(frac.get(d, np.nan)): continue
        x = None
        for j in range(e + 1, min(n, e + 6000)):
            if regs[j - 1] == -1: x = j; break
        if x is None: continue
        s0 = int(dstart[d]); hi_s, lo_s = hh[s0:fb].max(), ll[s0:fb].min(); used = ((hi_s - lo_s) / c_[fb - 1]) / frac[d]
        lo4 = ll[fb - 4:fb].min(); A = a15[fb - 1]; sup = bool(np.isfinite(A) and A > 0 and (abs(lo4 - lo_s) <= .5 * A or (np.isfinite(pdl.get(d, np.nan)) and abs(lo4 - pdl[d]) <= .5 * A)))
        rows.append(dict(tk=tk, entry=H.dt[e], gate=bool(gate[fb - 1]), ret=o_[x] / o_[e] - 1 - 4e-4, neg=bool(gprev[fb - 1] < 0), used=used, sup=sup))
T = pd.DataFrame(rows); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['mon'] = T.entry.dt.to_period('M'); T.to_parquet(SP + 'g437.parquet'); print(len(T), 'trades,', int(T.gate.sum()), 'gated,', int(T.neg.sum()), 'neg')
def contrast(A, B, lab):
    col = 'ret'; dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); g = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x[col][x.s == 1].sum(), s0=x[col][x.s == -1].sum()))); m1, m0 = A[col].mean(), B[col].mean()
    se = np.sqrt((((g.s1 - g.n1 * m1) / len(A) - (g.s0 - g.n0 * m0) / len(B)) ** 2).sum()); f = lambda x: f"n{len(x):5d} win {(x[col]>0).mean()*100:4.1f} med {x[col].median()*100:5.2f} mean {x[col].mean()*100:5.2f}"
    print(f"  {lab}\n    A {f(A)}\n    B {f(B)}\n    A-B {(m1-m0)*100:+.2f}pp  t {(m1-m0)/se:+.2f} | " + ' | '.join(f"{h} {(A[A.half==h][col].mean()-B[B.half==h][col].mean())*100:+.2f}pp (n {len(A[A.half==h])}/{len(B[B.half==h])})" for h in ('train', 'test')) + ' | ' + ' '.join(f"{e} {(A[A.era==e][col].mean()-B[B.era==e][col].mean())*100:+.2f}" for e in ('18-19', '20-21', '22', '23-26')))
for arm, D in (('PRIMARY: G20-gated', T[T.gate]), ('ROBUSTNESS: all flips', T)):
    print(f"\n=== {arm} (n {len(D)}; neg-gamma {int(D.neg.sum())}; win {(D.ret>0).mean()*100:.1f} mean {D.ret.mean()*100:.2f}) ===")
    N = D[D.neg]; contrast(N[N.used <= .5], N[N.used >= 1.0], 'H1 within NEG-gamma flips: A = LOW range used (<=0.5), B = HIGH (>=1.0)')
    pair = D.neg & (D.used <= .5); contrast(D[pair], D[~pair], 'H2 PAIR: A = NEG & LOW-used, B = every other flip')
    contrast(D[D.sup], D[~D.sup], 'H3 flip AT support (bounce off the day/prior-day low): A = AT, B = NOT')
    print(f"  (context) neg-gamma alone vs not: {D[D.neg].ret.mean()*100:.2f} vs {D[~D.neg].ret.mean()*100:.2f}; LOW-used vs HIGH-used (all): {D[D.used<=.5].ret.mean()*100:.2f} vs {D[D.used>=1].ret.mean()*100:.2f}")
