"""Nov 1 2026: Zarattini, Aziz & Barbon (2024) 'Beat the Market' intraday momentum, on QQQ 15m bars (2018-2026), as a second return stream.
Noise area per time of day: sigma = mean |close/open - 1| at that time over the prior 14 days. UB = max(open, prev close) x (1 + sigma),
LB = min(open, prev close) x (1 - sigma). Decisions every 30 min from 10:00 (bar closes); long above UB, short below LB; trailing stop
= the band or VWAP (whichever is tighter); flat at the close. Fills at the next 15m open, 1 bp per side. Then correlation / blend with the book."""
import glob, numpy as np, pandas as pd
Q = pd.read_parquet(glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); Q['dt'] = pd.to_datetime(Q.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
m = Q.dt.dt.hour * 60 + Q.dt.dt.minute; Q = Q[(m >= 570) & (m < 960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
Q['d'] = Q.dt.dt.normalize(); Q['tod'] = Q.dt.dt.hour * 60 + Q.dt.dt.minute + 15            # bar close time
Q = Q[Q.groupby('d').dt.transform('size') == 26].reset_index(drop=True)
op = Q.groupby('d').open.first(); pc = Q.groupby('d').close.last().shift(1)
Q['mv'] = (Q.close / Q.d.map(op) - 1).abs()
SG = Q.pivot_table(index='d', columns='tod', values='mv').rolling(14).mean().shift(1)
Q['tp'] = (Q.high + Q.low + Q.close) / 3; Q['vwap'] = (Q.tp * Q.volume).groupby(Q.d).cumsum() / Q.volume.groupby(Q.d).cumsum()
def run(long_only=False, cost=1e-4):
    R = {}
    for d, g in Q.groupby('d'):
        if d not in SG.index or pd.isna(pc.get(d)) or SG.loc[d].isna().all(): continue
        o0 = op[d]; up0, lo0 = max(o0, pc[d]), min(o0, pc[d]); pos = 0; ret = 0.0
        o, c, t, vw = g.open.values, g.close.values, g.tod.values, g.vwap.values
        for k in range(len(g) - 1):
            s = SG.loc[d].get(t[k], np.nan)
            if s != s or t[k] < 600 or t[k] % 30: continue
            ub, lb = up0 * (1 + s), lo0 * (1 - s); want = pos
            if pos == 1 and c[k] < max(ub, vw[k]): want = 0
            if pos == -1 and c[k] > min(lb, vw[k]): want = 0
            if c[k] > ub: want = 1
            elif c[k] < lb and not long_only: want = -1
            if want != pos:
                px = o[k + 1]
                if pos: ret += pos * (px / ent - 1) - cost
                if want: ent = px; ret -= cost
                pos = want
        if pos: ret += pos * (c[-1] / ent - 1) - cost
        R[d] = ret
    return pd.Series(R)
def st(r):
    eq = (1 + r).cumprod(); y = len(r) / 252; return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min(), (r != 0).mean()
B = pd.read_parquet('/tmp/claude-0/ml/vt/series.parquet')['tm:vec']; B.index = pd.to_datetime(B.index)
H = [('2019-21', '2019', '2021'), ('2022-26', '2022', '2026')]
for nm, lo in (('long/short', False), ('long only', True)):
    r = run(lo); r = r[r.index >= '2019-01-01']
    print(f'{nm:<11}' + '   '.join(f'{h}: {c:+.1%}/yr DD {d:.0%} (in market {f:.0%} of days)' for h, (c, d, f) in ((h, st(r.loc[a:b])) for h, a, b in H)))
    J = pd.concat([B, r], axis=1, keys=['book', 'nz']).dropna(); print(f'   corr with the book (daily) {J.book.corr(J.nz):+.2f}')
    for w in (.3, .5):
        x = J.book + w * J.nz
        print(f'   book + {w:.0%} noise strategy: ' + '   '.join(f'{h}: {c:+.1%}/yr DD {d:.0%} ratio {c / -d:.2f}' for h, (c, d, _) in ((h, st(x.loc[a:b])) for h, a, b in H)))
x = B; print('   book alone:              ' + '   '.join(f'{h}: {c:+.1%}/yr DD {d:.0%} ratio {c / -d:.2f}' for h, (c, d, _) in ((h, st(x.loc[a:b])) for h, a, b in H)))
