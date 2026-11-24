"""Nov 24 2026 report 434 = Rey: in a trend book, do (H1) flips with THIN volume overhead (low-volume node / air pocket) and (H2) flips whose lead-in volume SUPPORTS the move do better than flips into a heavy node / a tapering move?
DECLARED BEFORE LOOKING.  Trades: 15m vector bull flips (plain, 4bp cost, next-open entry), exit X1 = first bearish vector bar (next open); no time holds; 33 names with 5m + 15m history, entries 2018-07+.  PRIMARY arm = options-ratio gate G20 (20d median ratio, shifted 1d, >= 2) -- the live model's gate, all gamma states (negative-gamma subsample too small).  ROBUSTNESS arm = ungated flips.
FEATURES (known at the flip bar's close; no look-ahead):
 H1 OVERHEAD = volume per $ in [close, close + 0.5 x ATR14d] from the PRIOR 5 sessions' 5m bars (volume placed at hlc3) / that profile's average volume per $ over its own [low, high].  Zero outside the 5d range (price discovery).  THIN = ratio <= 0.5, HEAVY = ratio >= 1.0 (middle excluded from the contrast).  Prediction: THIN > HEAVY.
 H2a SUPPORT = signed volume share over the 12 bars up to the flip = sum(vol x sign(close-open)) / sum(vol).  SUPPORTED >= +0.10, UNSUPPORTED <= -0.10.  Prediction: SUPPORTED > UNSUPPORTED.
 H2b VOLTREND = mean vol of last 4 bars / mean vol of the 8 before.  EXPANDING >= 1.2, TAPERING <= 0.8.  Prediction: EXPANDING > TAPERING.
PASS RULE (3 hypotheses -> |t| >= 2.4, month-clustered Welch t on the contrast of mean net return): same sign in train (2019-21) and test (2018-H2 + 2022-26), |t| >= 2.4 on the full sample, and not carried by one era (era table shown).  Exploratory only: anything else."""
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
    F = rth(pd.read_parquet(SP + f'lse_5m/{tk}.parquet'), 'ts'); F['d'] = F.dt.dt.normalize(); F['px'] = (F.high + F.low + F.close) / 3
    days = sorted(F.d.unique()); dly = F.groupby('d').agg(h=('high', 'max'), l=('low', 'min')); atr = (dly.h - dly.l).rolling(14).mean().shift(0)    # ATR of sessions up to and including the day, used only for the NEXT day
    prof = {}
    for k, d in enumerate(days):
        if k < 14: continue
        prev = days[k - 5:k]; X = F[F.d.isin(prev)]
        if len(X) < 100: continue
        o = np.argsort(X.px.values); p = X.px.values[o]; cv = np.r_[0, np.cumsum(X.volume.values[o])]; prof[d] = (p, cv, float(atr.loc[days[k - 1]]))
    day = H.dt.dt.normalize().values; regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']]); n = len(H); o_ = H.open.values; c_ = H.close.values; v_ = H.volume.values.astype(float)
    flips = [i + 1 for i, dd in fls if dd == 1 and i + 2 < n]; gate = G20[tk].reindex(pd.DatetimeIndex(day), method='ffill').values.astype(float) == 1 if tk in G20 else np.zeros(n, bool)
    for fb in flips:
        e = fb + 1                                  # entry bar = next open;  flip bar = fb-1.. (flip detected at close of bar fb-1)
        if e >= n or H.dt[fb - 1] < pd.Timestamp('2018-07-01') or fb < 14: continue
        d = pd.Timestamp(day[fb - 1])
        if d not in prof: continue
        p, cv, A = prof[d]
        if not np.isfinite(A) or A <= 0 or len(p) < 50: continue
        px = c_[fb - 1]; lo, hi = px, px + .5 * A; vol_w = cv[np.searchsorted(p, hi, 'right')] - cv[np.searchsorted(p, lo, 'left')]; rng = max(p[-1] - p[0], 1e-9); avg = cv[-1] / rng; ratio = (vol_w / (hi - lo)) / avg
        w = slice(fb - 13, fb - 0); sv = (np.sign(c_[w] - H.open.values[w]) * v_[w]).sum() / max(v_[w].sum(), 1); vt = v_[fb - 4:fb].mean() / max(v_[fb - 12:fb - 4].mean(), 1)
        x = None
        for j in range(e + 1, min(n, e + 6000)):
            if regs[j - 1] == -1: x = j; break
        if x is None: continue
        rows.append(dict(tk=tk, entry=H.dt[e], ret=o_[x] / o_[e] - 1 - 4e-4, hold=(H.dt[x] - H.dt[e]).total_seconds() / 3600, gate=bool(gate[fb - 1]), over=ratio, sup=sv, vt=vt))
T = pd.DataFrame(rows); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26']); T['mon'] = T.entry.dt.to_period('M')
T.to_parquet(SP + 'g434.parquet'); print(len(T), 'trades;', int(T.gate.sum()), 'gated')
def tcl(a, b):
    """month-clustered t of mean(a)-mean(b): cluster-mean difference using month-level sums"""
    d = pd.concat([a.assign(s=1), b.assign(s=-1)]); g = d.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.ret[x.s == 1].sum(), s0=x.ret[x.s == -1].sum())))
    m1, m0 = a.ret.mean(), b.ret.mean(); N1, N0 = len(a), len(b); infl = (g.s1 - g.n1 * m1) / N1 - (g.s0 - g.n0 * m0) / N0; se = np.sqrt((infl ** 2).sum()); return (m1 - m0) / se if se > 0 else np.nan
def rep(D, lab, col, hi_cut, lo_cut, hi_is_good_if_high):
    A = D[D[col] >= hi_cut] if hi_is_good_if_high else D[D[col] <= lo_cut]; B = D[D[col] <= lo_cut] if hi_is_good_if_high else D[D[col] >= hi_cut]   # A = predicted better
    def f(x): return f"n{len(x):5d} win {(x.ret>0).mean()*100:4.1f} med {x.ret.median()*100:5.2f} mean {x.ret.mean()*100:5.2f}"
    print(f"  {lab}\n    better-side {f(A)}\n    worse-side  {f(B)}")
    out = [f"    contrast {(A.ret.mean()-B.ret.mean())*100:+.2f}pp  month-clustered t {tcl(A, B):+.2f}"]
    for h in ('train', 'test'): a, b = A[A.half == h], B[B.half == h]; out.append(f"{h} {(a.ret.mean()-b.ret.mean())*100:+.2f}pp (t {tcl(a, b):+.1f}, n {len(a)}/{len(b)})")
    print('  |  '.join(out)); print('    by era:  ' + '  '.join(f"{e} {(A[A.era==e].ret.mean()-B[B.era==e].ret.mean())*100:+.2f}pp (n {len(A[A.era==e])}/{len(B[B.era==e])})" for e in ('18-19', '20-21', '22', '23-26')))
for arm, D in (('PRIMARY: G20-gated flips', T[T.gate]), ('ROBUSTNESS: all flips', T)):
    print(f"\n=== {arm}  (n {len(D)}, win {(D.ret>0).mean()*100:.1f}, med {D.ret.median()*100:.2f}, mean {D.ret.mean()*100:.2f}) ===")
    rep(D, 'H1 THIN overhead (<=0.5) vs HEAVY (>=1.0): predict THIN better', 'over', 1.0, .5, False)
    rep(D, 'H2a SUPPORTED (>=+0.10) vs UNSUPPORTED (<=-0.10): predict SUPPORTED better', 'sup', .10, -.10, True)
    rep(D, 'H2b EXPANDING (>=1.2) vs TAPERING (<=0.8): predict EXPANDING better', 'vt', 1.2, .8, True)
