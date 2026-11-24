"""Nov 24 2026 report 441. Rey: does the negative-gamma vector flip work on QQQ, QQQ+ES, or all the index ETFs (QQQ SPY IWM DIA -> NQ ES RTY YM) as a more efficient prop strategy.
DECLARED BEFORE RESULTS.
DATA  15m RTH bars (lse cache) for QQQ SPY IWM DIA; the only REAL gamma state is QQQ's OI-based net_gex (xs/ext/gex/QQQ.csv). A volume-gamma proxy for SPY/IWM (xs/gex_proxy_theta.py) failed validation vs QQQ real (sign agreement 50%, rank corr -0.30), so it is NOT used. Mechanism for the cross-index gate: index gamma is one market-wide dealer state, so QQQ prior-close net_gex < 0 (NEG) is applied to every index's flips.
RULES vector (xs/vector_pine, live settings) 15m; entry = next bar open after a flip, bars 09:45-14:30; exit = next open after the opposite flip, or 15:45 open (same-session flat: prop accounts can't hold overnight; a session boundary, not a hold parameter); one position per instrument; cost 1bp round trip (index futures); long and short both reported.
TESTS per instrument: long/short x ALL / NEG / POS. PASS (per cell): NEG beats POS with month-clustered |t|>=2 pooled AND same sign in train (2019-21) and test (2018-07..2018, 2022+). Then, ONLY if cells pass: portfolio trades/month and daily-P&L Sharpe for QQQ alone vs QQQ+SPY vs all four (equal-notional, 1bp = 1 unit), to see if more instruments = more efficient.
Eras 18-19 | 20-21 | 22 | 23-26."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
UP = '/mnt/user-data/uploads/fartpie/'; ET = 'America/New_York'
g = pd.read_csv(UP + 'xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').sort_index().net_gex
def rth(df):
    df['dt'] = pd.to_datetime(df.datetime if 'datetime' in df else df.iloc[:, 0], utc=True).dt.tz_convert(ET).dt.tz_localize(None); m = df.dt.dt.hour * 60 + df.dt.dt.minute
    return df[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
rows = []
for tk in ('QQQ', 'SPY', 'IWM', 'DIA'):
    df = pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0])
    if 'datetime' not in df.columns: print(tk, df.columns.tolist())
    H = rth(df.rename(columns={'ts': 'datetime'})); n = len(H); o_ = H.open.values; m_ = (H.dt.dt.hour * 60 + H.dt.dt.minute).values; day = H.dt.dt.normalize().values
    regs, lv, a15, fls = run_state_machine(H[['high', 'low', 'close']]); gprev = g.shift(1).reindex(pd.DatetimeIndex(day), method='ffill').values
    for i, dd in fls:
        fb = i + 1; e = fb + 1
        if e + 1 >= n or not (585 <= m_[e] <= 870) or day[e] != day[fb - 1] or not np.isfinite(gprev[fb - 1]): continue
        x = None
        for j in range(e + 1, n):
            if day[j] != day[e]: x = j - 1; break
            if regs[j - 1] == -dd or m_[j] >= 945: x = j; break
        if x is None: continue
        rows.append(dict(tk=tk, side=dd, entry=H.dt[e], day=pd.Timestamp(day[e]), ret=dd * (o_[x] / o_[e] - 1) - 1e-4, neg=bool(gprev[fb - 1] < 0)))
T = pd.DataFrame(rows); T['yr'] = T.entry.dt.year; T['mon'] = T.entry.dt.to_period('M')
T['era'] = pd.cut(T.yr, [2017, 2019, 2021, 2022, 2100], labels=['18-19', '20-21', '22', '23-26']); T['half'] = np.where((T.entry >= '2019-01-01') & (T.entry < '2022-01-01'), 'train', 'test')
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/t441.pkl'); print('trades', len(T), T.groupby('tk').size().to_dict())
def tstat(A, B):
    dd = pd.concat([A.assign(s=1), B.assign(s=-1)]); gm = dd.groupby('mon').apply(lambda x: pd.Series(dict(n1=(x.s == 1).sum(), n0=(x.s == -1).sum(), s1=x.ret[x.s == 1].sum(), s0=x.ret[x.s == -1].sum()))); m1, m0 = A.ret.mean(), B.ret.mean()
    se = np.sqrt((((gm.s1 - gm.n1 * m1) / len(A) - (gm.s0 - gm.n0 * m0) / len(B)) ** 2).sum()); return (m1 - m0) * 1e4, (m1 - m0) / se
f = lambda x: f"n{len(x):5d} win {(x.ret>0).mean()*100:4.1f} mean {x.ret.mean()*1e4:6.1f}bp"
for side, sl in ((1, 'LONG'), (-1, 'SHORT')):
    print(f'\n=== {sl} flips')
    for tk in ('QQQ', 'SPY', 'IWM', 'DIA'):
        D = T[(T.tk == tk) & (T.side == side)]; A, B = D[D.neg], D[~D.neg]; d, t = tstat(A, B)
        ers = ' '.join(f"{e}:{A[A.era==e].ret.mean()*1e4:+.0f}/{B[B.era==e].ret.mean()*1e4:+.0f}" for e in ('18-19', '20-21', '22', '23-26'))
        tr = [(A[A.half == h].ret.mean() - B[B.half == h].ret.mean()) * 1e4 for h in ('train', 'test')]
        print(f"{tk:4s} ALL {f(D)} | NEG {f(A)} | POS {f(B)} | NEG-POS {d:+.1f}bp t {t:+.2f} (train {tr[0]:+.1f} test {tr[1]:+.1f}) | era NEG/POS bp {ers}")
