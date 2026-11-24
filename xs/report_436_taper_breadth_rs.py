"""Nov 24 2026 report 436 = Rey's 1,2,3: (1) volume TAPER as a signal exit, (2) idiosyncratic vs market-wide flips, (3) relative strength at the flip.  DECLARED BEFORE LOOKING.
Trades: 15m bull flips as in 434/435 (25 names, G20-gated PRIMARY, ungated robustness), next-open entry, 4bp, no time holds.  Baseline exit X1 = first bearish vector bar.
H1 TAPER EXIT: RV = bar volume / mean volume of the same 15m time-slot over the prior 20 sessions (removes the intraday U-shape).  Exit at the next open after the first bar j >= entry+3 where mean(RV of the last 4 bars) <= 0.7 AND close_j < highest close since entry; otherwise the vector flip.  Prediction: exit-with-taper beats X1 per trade (mean of the paired difference > 0).
H2 BREADTH: share of the 25 names whose 15m vector is bull at the flip bar's close (other names only).  IDIO = share <= 0.30, MKT = share >= 0.70.  Prediction: IDIO > MKT (stock-specific demand persists; market-wide flips are beta).  Two-sided t.
H3 RELATIVE STRENGTH: stock 20-session return minus QQQ's at the flip.  LEADER = top third, LAGGARD = bottom third (cut within the sample).  Prediction: LEADER > LAGGARD.
PASS (3 hypotheses -> |t| >= 2.4, month-clustered, same sign in train 2019-21 and test 2018-H2+2022-26, eras shown)."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
G20 = Rr.rolling(20, min_periods=10).median().shift(1) >= 2
names = sorted(f[:-8] for f in os.listdir(SP + 'lse_5m') if f.endswith('.parquet') and glob.glob(UP + f'lse_ohlcv_cache/{f[:-8]}_15m_2018-01-01_*.parquet')); print('names', len(names))
def rth(df, col):
    df['dt'] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)

Hs, RG = {}, {}
for tk in names:
    H = rth(pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}), 'ts'); regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']])
    H['slot'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H['rv'] = H.volume / H.groupby('slot').volume.transform(lambda x: x.rolling(20, min_periods=10).mean().shift(1)); Hs[tk] = (H, regs, fls); RG[tk] = pd.Series(regs, index=H.dt)
B = pd.DataFrame(RG); BULL = (B == 1).where(B.notna()); breadth_all = BULL.sum(axis=1); cnt = B.notna().sum(axis=1)
Q = Hs['QQQ'][0].set_index('dt').close if 'QQQ' in Hs else None; q20 = (Q / Q.shift(520) - 1) if Q is not None else None
rows = []
for tk in names:
    H, regs, fls = Hs[tk]; n = len(H); o_ = H.open.values; c_ = H.close.values; rv = H.rv.values; day = H.dt.dt.normalize().values
    gate = G20[tk].reindex(pd.DatetimeIndex(day), method='ffill').values.astype(float) == 1 if tk in G20 else np.zeros(n, bool); r20 = (H.close / H.close.shift(520) - 1).values
    for fb in [i + 1 for i, dd in fls if dd == 1 and i + 2 < n]:
        e = fb + 1; t0 = H.dt[fb - 1]
        if e + 3 >= n or t0 < pd.Timestamp('2018-07-01') or fb < 520: continue
        x = None
        for j in range(e + 1, min(n, e + 6000)):
            if regs[j - 1] == -1: x = j; break
        if x is None: continue
        xt = x; hi = c_[e - 1]
        for j in range(e, x):
            hi = max(hi, c_[j])
            if j >= e + 3 and np.nanmean(rv[j - 3:j + 1]) <= .7 and c_[j] < hi: xt = min(x, j + 1); break
        oth = (breadth_all.get(t0, np.nan) - (1 if regs[fb - 1] == 1 else 0)) / max(cnt.get(t0, 0) - 1, 1)
        rs = r20[fb - 1] - (q20.get(t0, np.nan) if q20 is not None else np.nan) if tk != 'QQQ' else np.nan
        rows.append(dict(tk=tk, entry=H.dt[e], gate=bool(gate[fb - 1]), r0=o_[x] / o_[e] - 1 - 4e-4, rt=o_[xt] / o_[e] - 1 - 4e-4, hold0=(H.dt[x] - H.dt[e]).total_seconds() / 3600, holdt=(H.dt[xt] - H.dt[e]).total_seconds() / 3600, breadth=oth, rs=rs))
T = pd.DataFrame(rows); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['mon'] = T.entry.dt.to_period('M'); T.to_parquet(SP + 'g436.parquet'); print(len(T), 'trades,', int(T.gate.sum()), 'gated')
def mt(v, mon):
    d = pd.DataFrame(dict(v=v.values, mon=mon.values)); g = d.groupby('mon').agg(s=('v', 'sum'), n=('v', 'size')); m = d.v.mean(); se = np.sqrt(((g.s - g.n * m) ** 2).sum()) / len(d); return m / se if se > 0 else np.nan
def contrast(A, B, col, lab):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); g = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x[col][x.s == 1].sum(), s0=x[col][x.s == -1].sum()))); m1, m0 = A[col].mean(), B[col].mean()
    se = np.sqrt((((g.s1 - g.n1 * m1) / len(A) - (g.s0 - g.n0 * m0) / len(B)) ** 2).sum()); f = lambda x: f"n{len(x):5d} win {(x[col]>0).mean()*100:4.1f} med {x[col].median()*100:5.2f} mean {x[col].mean()*100:5.2f}"
    print(f"  {lab}\n    A {f(A)}\n    B {f(B)}\n    A-B {(m1-m0)*100:+.2f}pp  t {(m1-m0)/se:+.2f} | " + ' | '.join(f"{h} {(A[A.half==h][col].mean()-B[B.half==h][col].mean())*100:+.2f}pp" for h in ('train', 'test')) + ' | ' + ' '.join(f"{e} {(A[A.era==e][col].mean()-B[B.era==e][col].mean())*100:+.2f}" for e in ('18-19', '20-21', '22', '23-26')))
for arm, D in (('PRIMARY: G20-gated', T[T.gate]), ('ROBUSTNESS: all flips', T)):
    print(f"\n=== {arm} (n {len(D)}) ===")
    d = D.rt - D.r0; print(f" H1 TAPER EXIT vs vector exit: mean {D.rt.mean()*100:.2f} vs {D.r0.mean()*100:.2f}; win {(D.rt>0).mean()*100:.1f} vs {(D.r0>0).mean()*100:.1f}; median {D.rt.median()*100:.2f} vs {D.r0.median()*100:.2f}; hold {D.holdt.mean():.1f}h vs {D.hold0.mean():.1f}h; exits changed {(d!=0).mean()*100:.0f}% of trades")
    print(f"    paired diff {d.mean()*100:+.2f}pp  t {mt(d, D.mon):+.2f} | " + ' | '.join(f"{h} {d[D.half==h].mean()*100:+.2f}pp (t {mt(d[D.half==h], D.mon[D.half==h]):+.1f})" for h in ('train', 'test')) + ' | ' + ' '.join(f"{e} {d[D.era==e].mean()*100:+.2f}" for e in ('18-19', '20-21', '22', '23-26')))
    contrast(D[D.breadth <= .30].assign(ret=lambda x: x.r0), D[D.breadth >= .70].assign(ret=lambda x: x.r0), 'ret', 'H2 BREADTH: A = IDIO (<=0.30 of other names bull), B = MKT (>=0.70); predict A > B')
    Dr = D.dropna(subset=['rs']); q1, q2 = Dr.rs.quantile([1 / 3, 2 / 3]); contrast(Dr[Dr.rs >= q2].assign(ret=lambda x: x.r0), Dr[Dr.rs <= q1].assign(ret=lambda x: x.r0), 'ret', f'H3 REL STRENGTH vs QQQ (20 sessions; cuts {q1*100:.1f}% / {q2*100:.1f}%): A = LEADER, B = LAGGARD; predict A > B')
