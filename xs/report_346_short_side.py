"""Nov 22 2026 report 346. Rey: QQQ weekly vector is bearish (closest to the .com break?), the algo is buy-only; test shorting under the vector (weekly / daily).  Declared before running.
Same vector (xs/vector_pine.run_state_machine, unchanged constants) on QQQ WEEKLY and DAILY bars 1999-2026 (yfinance, split/dividend-adjusted OHLC).  A bear regime = from the bar AFTER the bear flip bar (entry at that bar's open) to the bar after the bull flip (exit at its open); short P&L = -(exit/entry - 1) - 2bp each side (index ETF; borrow ignored for QQQ, stated).
Questions: (1) what did every weekly bear regime do (episodes table: 2000, 2008, 2022, Mar 2026, now); is the current one open and how long; (2) short-only on weekly and on daily regimes: total short return, per-episode, win rate, worst adverse move inside an episode (squeeze); (3) long-only vs long + short equity (long in bull, short in bear) vs buy-and-hold, CAGR / max DD, 2000-2026 and 2019-2026; (4) the daily-vector SHORT on the 54 large caps (2018-26 RTH 15m aggregated to daily): per-trade short return, train 2019-21 | test 2022-26 | 2018, win rate - does it work on single names, not only on the index.
THEORY: the vector is a trailing-stop trend follower; trend persistence is symmetric on an index in a real bear, but equities have positive drift and sharp V-rebounds (Apr 2025 style), so shorts should have a lower win rate and fatter squeezes than longs.  Why it should fail: weekly flips are ~15 events in 27 years, a handful of regimes carry all the P&L (dot-com, 2008), and the March-2026-style whipsaw costs repeatedly."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine; from stress_lib import sharpe
import yfinance as yf
Q = yf.download('QQQ', start='1999-03-10', auto_adjust=True, progress=False); Q.columns = [c[0] if isinstance(c, tuple) else c for c in Q.columns]; Q = Q.dropna(); print(f'QQQ daily {Q.index[0].date()} -> {Q.index[-1].date()} ({len(Q)} bars), last close {Q.close.iloc[-1] if "close" in Q else Q.Close.iloc[-1]:.2f}')
Q.columns = [c.lower() for c in Q.columns]
W = Q.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
def regimes(df, cost=2e-4):
    regs, lv, atr, fl = run_state_machine(df.reset_index(drop=True)); rows = []; o = df.open.values; idx = df.index; i0 = None
    for i, d in fl:
        if d == -1 and i + 1 < len(df): i0 = i + 1
        elif d == 1 and i0 is not None and i + 1 < len(df): rows.append((idx[i0], idx[i + 1], o[i0], o[i + 1], df.high.values[i0:i + 2].max(), 0)); i0 = None
    cur = None
    if i0 is not None: cur = (idx[i0], idx[-1], o[i0], df.close.values[-1], df.high.values[i0:].max())
    elif fl and fl[-1][1] == -1 and fl[-1][0] == len(df) - 1: cur = (idx[-1], idx[-1], df.close.values[-1], df.close.values[-1], df.close.values[-1], 'PENDING')
    E = pd.DataFrame(rows, columns=['t0', 't1', 'p0', 'p1', 'hi', 'x'])
    if len(E): E['short'] = -(E.p1 / E.p0 - 1) - 2 * cost; E['squeeze'] = E.hi / E.p0 - 1; E['days'] = (E.t1 - E.t0).dt.days
    return E, cur, regs
for lab, df in (('WEEKLY', W), ('DAILY', Q)):
    E, cur, regs = regimes(df); print(f'\n=== QQQ {lab} vector: {len(E)} closed bear regimes' + ((f'; BEAR FLIP on the last bar ({cur[0].date()}, close {cur[3]:.2f}); a short would enter at the NEXT open' if len(cur) > 5 else f'; OPEN since {cur[0].date()} entry {cur[2]:.2f}, last {cur[3]:.2f} ({-(cur[3] / cur[2] - 1):+.1%} short), max adverse so far {cur[4] / cur[2] - 1:+.1%}') if cur else '; none open'))
    if lab == 'WEEKLY': print(E[['t0', 't1', 'days', 'p0', 'p1', 'short', 'squeeze']].assign(t0=E.t0.dt.date, t1=E.t1.dt.date).round(3).to_string(index=False))
    else:
        print(f'   short per regime: mean {E.short.mean():+.2%}, median {E.short.median():+.2%}, win {(E.short > 0).mean():.0%}, sum (simple) {E.short.sum():+.0%}; median days {E.days.median():.0f}; worst squeeze in a regime {E['squeeze'].max():+.1%}')
        for a, b in (('1999', '2007'), ('2008', '2018'), ('2019', '2021'), ('2022', '2026')): s = E[(E.t0 >= a) & (E.t0 < f'{int(b) + 1}-01-01')]; print(f'   {a}-{b}: n {len(s)}, mean {s.short.mean():+.2%}, win {(s.short > 0).mean():.0%}, sum {s.short.sum():+.0%}')
    E['lab'] = lab; globals()['E_' + lab] = E
def curve(df, mode, cost=2e-4):
    regs, lv, atr, fl = run_state_machine(df.reset_index(drop=True)); r = df.open.pct_change().shift(-1).fillna(0).values; pos = np.r_[0, regs[:-1]].astype(float)  # regime known at bar close -> position from next bar
    if mode == 'long': pos = np.where(pos == 1, 1, 0)
    elif mode == 'short': pos = np.where(pos == -1, -1, 0)
    c = np.abs(np.diff(np.r_[0, pos])) * cost; return pd.Series(pos * np.r_[r[1:] if False else r] - c, index=df.index)
def stats(s):
    e = (1 + s).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return f'CAGR {e.iloc[-1] ** (1 / y) - 1:6.1%}, maxDD {(e / e.cummax() - 1).min():6.1%}, final x{e.iloc[-1]:.2f}'
print('\n=== QQQ daily vector as a trading rule (position from next open, 2bp per side)')
rD = Q.open.pct_change().shift(-1).fillna(0)
for per in ('1999-2026', '2019-2026'):
    sl = slice('1999', None) if per == '1999-2026' else slice('2019', None)
    print(f' {per}:  buy&hold {stats(Q.close.pct_change().fillna(0)[sl])}')
    for mode in ('long', 'short', 'both'): print(f'   daily vector {mode:5s} {stats(curve(Q, mode)[sl])}')
    for mode in ('long', 'short', 'both'): print(f'   weekly vector {mode:5s} {stats(curve(W, mode)[sl])} (weekly bars)')
# single names
rows = []
for f in sorted(glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')):
    tk = f.split('/')[-1].split('_15m')[0]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); mm = H.index.hour * 60 + H.index.minute; H = H[(mm >= 570) & (mm < 960)]
    Dd = H.groupby(H.index.normalize()).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).dropna(); E, cur, _ = regimes(Dd); E['tk'] = tk; rows.append(E)
S = pd.concat(rows); S['y'] = S.t0.dt.year; trn = (S.t0 >= '2019-01-01') & (S.t0 < '2022-01-01'); tst = S.t0 >= '2022-01-01'
def g(m): s = S.short[m]; return f'{s.mean():+.2%} (t{s.mean() / (s.std() / np.sqrt(len(s))):+4.1f}, n{len(s)}, win {(s > 0).mean():.0%}, med days {S.days[m].median():.0f})'
print(f'\n=== DAILY vector SHORT on the 54 large caps (per bear regime): all {g(S.t0 > "2000")}\n   2018 {g(S.t0 < "2019-01-01")}\n   train 2019-21 {g(trn)}\n   test 2022-26 {g(tst)}\n   by year:', {int(y): round(S.short[S.y == y].mean() * 100, 2) for y in sorted(S.y.unique())})
