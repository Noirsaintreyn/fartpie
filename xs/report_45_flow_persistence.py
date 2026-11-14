"""REPORT 45. Nov 14 2026 (declared before results): signed-volume persistence (order-flow imbalance over the last 26 and 130 15m bars, lag-1 autocorrelation of signed volume over 130 bars; the order-splitting mechanism) as the SLOT TIE-BREAK or SIZE TILT for the live screener. Universe = xs/ext/wide_T.parquet, sp and options ratio >= 1.5 (the live universe, 54 names over time), live rules as in xs/site_backtest_wide.py (6 slots, 8% breaker on realized P&L). Tracks tested separately: 15m (h1 False) and 1h (h1 True). Features at entry (known before it): IMB130 = ER20 on daily closes to the prior close; IMB26 = ER over the last 26 15m bars; ACS130 = ER over the last 20 1h bars. Variants: tie-break by ratio (live), IMB130, IMB26, ACS130, composite rank; size tilt x1.25 / x0.75 on PIT tercile. Realized-P&L equity (log, 1/6 per slot, 2 bp cost) so variants compare on one basis. Hypothesis fixed in advance: HIGH ER first/bigger. PASS = beats live tie-break in BOTH 2019-21 and 2022-26 on return with drawdown no worse."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
N = 6; BRK = .08; COST = 2e-4
T0 = pd.read_parquet('xs/ext/wide_T.parquet'); T0 = T0[T0.sp & (T0.ratio >= 1.5)].copy(); T0['entry'] = pd.to_datetime(T0.entry); T0['exit'] = pd.to_datetime(T0.exit); T0 = T0.reset_index(drop=True)
def er(c, n): return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
F = {}
for tk in sorted(T0.tk.unique()):
    f = glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'close', 'volume']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').set_index('dt')
    sv = np.sign(H.close.diff()) * H.volume; vv = H.volume
    i26 = sv.rolling(26).sum() / vv.rolling(26).sum(); i130 = sv.rolling(130).sum() / vv.rolling(130).sum(); ac = sv.rolling(130).corr(sv.shift(1))
    h1 = H.close.resample('60min', offset='30min').last().dropna()
    F[tk] = (i130.groupby(i130.index.normalize()).last().shift(1), i26, ac, H.index, h1.index)
rows = []
for tk, g in T0.groupby('tk'):
    if tk not in F: continue
    ed, e15, e1, i15, i1 = F[tk]
    for ix, e in zip(g.index, g.entry):
        j = i15.searchsorted(e, 'left') - 1; k = j
        rows.append((ix, ed.get(e.normalize(), np.nan), e15.iloc[j] if j >= 0 else np.nan, e1.iloc[k] if k >= 0 else np.nan))
X = pd.DataFrame(rows, columns=['ix', 'IMB130', 'IMB26', 'ACS130']).set_index('ix'); T0 = T0.join(X).dropna(subset=['IMB130', 'IMB26', 'ACS130']); print('trades with features', len(T0), 'names', T0.tk.nunique())
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
print('\nPER TRADE by ER tercile (expanding PIT), mean return % | win % | n | high-low t')
T0 = T0.sort_values('entry')
for c in ('IMB130', 'IMB26', 'ACS130'):
    v = T0[c].values; o = np.full(len(v), np.nan)
    for i in range(300, len(v)): q1, q2 = np.quantile(v[:i], [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
    T0['T_' + c] = o
for trk, lab in ((False, '15m track'), (True, '1h track')):
    for c in ('IMB130', 'IMB26', 'ACS130'):
        line = f' {lab} {c:5s}'
        for e, (a, b) in (('19-21', ('2019', '2022')), ('22-26', ('2022', '2027'))):
            s = T0[(T0.h1 == trk) & (T0.entry >= a) & (T0.entry < b)]; lo = s[s['T_' + c] == 0]; hi = s[s['T_' + c] == 2]
            line += f' | {e}: low {lo.ret.mean() * 100:+.2f}% ({(lo.ret > 0).mean():.0%},{len(lo)}) high {hi.ret.mean() * 100:+.2f}% ({(hi.ret > 0).mean():.0%},{len(hi)}) t {tt(hi.ret, lo.ret):+.1f}'
        print(line)
T0['rk'] = (T0.IMB130.rank(pct=True) + T0.IMB26.rank(pct=True) + T0.ACS130.rank(pct=True)) / 3
def book(T, key, mult=None):
    T = T.sort_values(['entry'] + key[0], ascending=[True] + key[1], kind='stable'); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; out = []
    for e, x, r, m in zip(T.entry.values, T.exit.values, T.ret.values, (mult if mult is not None else np.ones(len(T)))):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / N); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= N: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-BRK): continue
        op.append(x); pend.append((x, (r - COST) * m)); out.append((pd.Timestamp(x), (r - COST) * m))
    return out
def stat(out, a, b):
    s = pd.Series({x: 0.0 for x, _ in out}); d = pd.Series([np.log1p(r / N) for _, r in out], index=[x.normalize() for x, _ in out]).groupby(level=0).sum()
    d = d[(d.index >= a) & (d.index < b)]; days = pd.bdate_range(a, min(pd.Timestamp(b), d.index.max() + pd.Timedelta(days=1))); d = d.reindex(days).fillna(0); eq = d.cumsum(); y = len(d) / 252
    return np.exp(d.sum() / y) - 1, np.exp((eq - eq.cummax()).min()) - 1, d.mean() / d.std() * 252 ** .5, len([1 for x, _ in out if a <= str(x) < b])
VAR = {'ratio (live)': (['ratio'], [False]), 'IMB130 high first': (['IMB130'], [False]), 'IMB26 high first': (['IMB26'], [False]), 'ACS130 high first': (['ACS130'], [False]), 'composite high first': (['rk'], [False]), 'IMB130 LOW first (control)': (['IMB130'], [True])}
for trk, lab in ((None, 'BOTH tracks (live book)'), (False, '15m track only'), (True, '1h track only')):
    T = T0 if trk is None else T0[T0.h1 == trk]; print(f'\nBOOK, {lab}: CAGR / maxDD / Sharpe / trades')
    for nm, k in VAR.items():
        o = book(T, k); a1 = stat(o, '2019-01-01', '2022-01-01'); a2 = stat(o, '2022-01-01', '2027-01-01'); print(f'  {nm:28s} 19-21 {a1[0]:+6.1%} {a1[1]:6.1%} S{a1[2]:.2f} n{a1[3]} | 22-26 {a2[0]:+6.1%} {a2[1]:6.1%} S{a2[2]:.2f} n{a2[3]}')
    for c in ('IMB130', 'IMB26', 'ACS130'):
        mult = np.where(T['T_' + c] == 2, 1.25, np.where(T['T_' + c] == 0, 0.75, 1.0)); Ts = T.sort_values(['entry', 'ratio'], ascending=[True, False], kind='stable'); mult = pd.Series(mult, index=T.index).loc[Ts.index].values
        o = book(T, (['ratio'], [False]), mult=pd.Series(np.where(T['T_' + c] == 2, 1.25, np.where(T['T_' + c] == 0, 0.75, 1.0)), index=T.index).loc[T.sort_values(['entry', 'ratio'], ascending=[True, False], kind='stable').index].values)
        a1 = stat(o, '2019-01-01', '2022-01-01'); a2 = stat(o, '2022-01-01', '2027-01-01'); print(f'  size tilt x1.25/x0.75 on {c:5s}      19-21 {a1[0]:+6.1%} {a1[1]:6.1%} S{a1[2]:.2f} n{a1[3]} | 22-26 {a2[0]:+6.1%} {a2[1]:6.1%} S{a2[2]:.2f} n{a2[3]}')
