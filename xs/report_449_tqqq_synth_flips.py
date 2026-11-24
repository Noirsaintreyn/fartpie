"""Nov 24 2026 report 449. Rey: backtest vector flips / neg-gamma flips traded in TQQQ, signal from QQQ (no TQQQ bars available). DECLARED BEFORE RESULTS.
SYNTHETIC TQQQ = daily-reset 3x of QQQ: each calendar-trading-day return r_d (QQQ close-to-close, first day open-to-close, exit day prev close-to-open gap) -> 3*r_d compounded per day (volatility drag included); SQQQ = -3x same way. Drag = 0.84% expense + 2 x fed funds per year (approx by year: 18 1.9, 19 2.1, 20 0.4, 21 0.1, 22 1.7, 23 5.0, 24 5.1, 25 4.2, 26 3.7 %) accrued per calendar day held; cost 2bp per side. Real TQQQ will differ (tracking error, intraday resets).
SIGNAL QQQ 15m vector (xs/vector_pine) bull flip -> long TQQQ at the next open; exit at the open after the opposite flip (overnight allowed; exits from signals only). Bear flips optionally -> long SQQQ. One position at a time. 2018-07 .. 2026-09.
GATE NEG = QQQ prior-close net_gex < 0 (real OI file); POS = >= 0.
ARMS: A bull flips ungated | B bull NEG | C bull POS | D bull TQQQ + bear SQQQ ungated | E bull+bear both NEG-gated. Benchmarks: buy&hold synthetic TQQQ, QQQ.
PASS (hypothesis: NEG flips follow through more, so B beats C): B-C per-trade TQQQ mean > 0 with month-clustered t >= 2 pooled and same sign in train (2019-21) and test (2018 + 2022+); book: B CAGR / maxDD better than A in train AND test. Report by era 18-19 | 20-21 | 22 | 23-26. Daily-close marks (intraday drawdown not marked => maxDD is a floor)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
df = pd.read_parquet(glob.glob(UP + 'lse_ohlcv_cache/QQQ_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}); df['dt'] = pd.to_datetime(df.ts, utc=True).dt.tz_convert(ET).dt.tz_localize(None) if 'ts' in df else None
m = df.dt.dt.hour * 60 + df.dt.dt.minute; H = df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
import os; TF = os.environ.get('TF', '15m'); print('TIMEFRAME', TF)
if TF != '15m':
    H['d'] = H.dt.dt.normalize(); H['b'] = ((H.dt.dt.hour * 60 + H.dt.dt.minute - 570) // 60) if TF == '1h' else 0
    H = H.groupby(['d', 'b'], sort=True).agg(dt=('dt', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last')).reset_index(drop=True)
n = len(H); o_, c_ = H.open.values, H.close.values
day = H.dt.dt.normalize().values; ud, first = np.unique(day, return_index=True); last_ = np.r_[first[1:] - 1, n - 1]; di = np.searchsorted(ud, day); dayclose = c_[last_]; dayopen = o_[first]
regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']])
g = pd.read_csv(UP + 'xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index().net_gex; gprev = g.shift(1).reindex(pd.DatetimeIndex(day), method='ffill').values
FF = {2018: 1.9, 2019: 2.1, 2020: .4, 2021: .1, 2022: 1.7, 2023: 5.0, 2024: 5.1, 2025: 4.2, 2026: 3.7}; drag_d = lambda y, days: (0.0084 + 2 * FF.get(y, 4) / 100) * days / 365
def daily_parts(e, x):
    """QQQ per-trading-day returns for holding from bar e open to bar x open: list of (day_index, r)"""
    out = []; de, dx = di[e], di[x]
    for d in range(de, dx + 1):
        a = o_[e] if d == de else dayclose[d - 1]; b = dayclose[d] if d < dx else o_[x]
        if d == dx and d == de: b = o_[x]
        out.append((d, b / a - 1))
    return out
trades = []
for i, dd in fls:
    fb = i + 1; e = fb + 1
    if e + 1 >= n or not np.isfinite(gprev[fb - 1]): continue
    t0 = H.dt[e]
    if t0 < pd.Timestamp('2018-07-01') or t0 > pd.Timestamp('2026-09-30'): continue
    x = None
    for j in range(e + 1, n):
        if regs[j - 1] == -dd: x = j; break
    if x is None: continue
    trades.append(dict(side=dd, e=e, x=x, entry=t0, neg=bool(gprev[fb - 1] < 0)))
T = pd.DataFrame(trades); nd = len(ud)
def trade_ret(r):
    parts = daily_parts(r.e, r.x); L = 3 * r.side; v = 1.0; days_held = (pd.Timestamp(ud[di[r.x]]) - pd.Timestamp(ud[di[r.e]])).days
    for d, q in parts: v *= 1 + L * q
    return v - 1 - drag_d(pd.Timestamp(ud[di[r.e]]).year, max(days_held, 0)) - 4e-4, parts
res = [trade_ret(r) for r in T.itertuples()]; T['ret'] = [a for a, b in res]; T['qqq'] = [r.side * (o_[r.x] / o_[r.e] - 1) for r in T.itertuples()]; T['hold'] = [(H.dt[r.x] - H.dt[r.e]).total_seconds() / 86400 for r in T.itertuples()]
T['mon'] = T.entry.dt.to_period('M'); T['era'] = pd.cut(T.entry.dt.year, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test'); PARTS = [b for a, b in res]
def tst(A, B):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.ret[x.s == 1].sum(), s0=x.ret[x.s == -1].sum()))); m1, m0 = A.ret.mean(), B.ret.mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 100, (m1 - m0) / se
f = lambda x: f"n{len(x):4d} win {(x.ret>0).mean()*100:4.1f} mean {x.ret.mean()*100:5.2f}% med {x.ret.median()*100:5.2f}% hold {x.hold.mean():4.1f}d (QQQ {x.qqq.mean()*100:5.2f}%)"
Bl = T[T.side == 1]; A_, B_ = Bl[Bl.neg], Bl[~Bl.neg]; d_, t_ = tst(A_, B_)
print('PER TRADE (bull flips -> TQQQ)'); print(' ALL ', f(Bl)); print(' NEG ', f(A_)); print(' POS ', f(B_)); print(f' NEG-POS {d_:+.2f}pp t {t_:+.2f} | train {(A_[A_.half=="train"].ret.mean()-B_[B_.half=="train"].ret.mean())*100:+.2f} test {(A_[A_.half=="test"].ret.mean()-B_[B_.half=="test"].ret.mean())*100:+.2f}')
print(' by era NEG / POS mean %: ' + ' | '.join(f"{e} {A_[A_.era==e].ret.mean()*100:+.2f}(n{(A_.era==e).sum()}) / {B_[B_.era==e].ret.mean()*100:+.2f}(n{(B_.era==e).sum()})" for e in ('18-19', '20-21', '22', '23-26')))
Br = T[T.side == -1]; print('PER TRADE (bear flips -> SQQQ)'); print(' ALL ', f(Br)); print(' NEG ', f(Br[Br.neg])); print(' POS ', f(Br[~Br.neg]))
def book(sel, nm):
    r = np.zeros(nd); held = np.zeros(nd, bool); S = T[sel].sort_values('e'); last_x = -1
    for k in S.index:
        r_ = T.loc[k]
        if r_.e <= last_x: continue
        last_x = r_.x; L = 3 * r_.side
        for d, q in PARTS[k]: r[d] = (1 + L * q) - 1 if not held[d] else r[d]; held[d] = True
        r[di[r_.e]] -= 2e-4; r[di[r_.x]] -= 2e-4
    dr = np.array([drag_d(pd.Timestamp(ud[d]).year, 1) if held[d] else 0 for d in range(nd)]); r = r - dr; eq = np.cumprod(1 + r); idx = pd.DatetimeIndex(ud); s = pd.Series(r, index=idx); yrs = (idx[-1] - idx[0]).days / 365.25
    def stat(rr):
        e = (1 + rr).cumprod(); return e.iloc[-1] ** (252 / max(len(rr), 1)) - 1, (e / e.cummax() - 1).min(), rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan
    out = {k: stat(s[mask]) for k, mask in (('all', s.index >= '2018-07-01'), ('train', (s.index >= '2019-01-01') & (s.index < '2022-01-01')), ('test', (s.index < '2019-01-01') | (s.index >= '2022-01-01')))}
    exposure = held.mean() * 100; print(f"{nm:34s} exposure {exposure:3.0f}% | " + ' | '.join(f"{k}: CAGR {v[0]*100:6.1f}% DD {v[1]*100:6.1f}% Sh {v[2]:4.2f}" for k, v in out.items()), flush=True)
    return s
print('\nBOOKS (one position at a time, daily marks, daily-reset 3x compounding)')
book(T.side == 1, 'A bull ungated -> TQQQ'); book((T.side == 1) & T.neg, 'B bull NEG -> TQQQ'); book((T.side == 1) & ~T.neg, 'C bull POS -> TQQQ'); book(T.side != 0, 'D bull TQQQ + bear SQQQ'); book(T.neg, 'E bull+bear NEG only')
bh = pd.Series(np.r_[0, [3 * (dayclose[d] / dayclose[d - 1] - 1) - drag_d(pd.Timestamp(ud[d]).year, 1) for d in range(1, nd)]], index=pd.DatetimeIndex(ud)); qq = pd.Series(np.r_[0, dayclose[1:] / dayclose[:-1] - 1], index=pd.DatetimeIndex(ud))
for nm, s in (('B&H synthetic TQQQ', bh), ('B&H QQQ', qq)):
    out = []
    for k, mask in (('all', s.index >= '2018-07-01'), ('train', (s.index >= '2019-01-01') & (s.index < '2022-01-01')), ('test', (s.index < '2019-01-01') | (s.index >= '2022-01-01'))):
        e = (1 + s[mask]).cumprod(); out.append(f"{k}: CAGR {(e.iloc[-1]**(252/mask.sum())-1)*100:6.1f}% DD {((e/e.cummax()-1).min())*100:6.1f}% Sh {s[mask].mean()/s[mask].std()*np.sqrt(252):4.2f}")
    print(f"{nm:34s} exposure 100% | " + ' | '.join(out))
