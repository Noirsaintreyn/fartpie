"""Nov 24 2026 report 431 = report 429 with a CHAIN-BASED gamma proxy (volume-weighted call-minus-put gamma_net from greeks_daily, own 252d PIT percentile) so 2026 can be tested: NEG-proxy = pct <= .33, positive-gamma-proxy = pct >= .5; validated on 2018-2025 against the real net_gex model, then applied to 2026 (gex files end 2025-12-30).  (429: Rey: 'did you try volatility on top of this'.  Base = report 428 candidate: 15m bars, ratio gate G20 (20d median r >= 2; G252 secondary), one trade per negative-gamma episode (first gated long flip), exits X1 vector / X3 positive-gamma-or-vector, 4bp, same 11 names (rerunnable).
DECLARED vol states at the prior close, from the names' own 15m data: HARup = HAR next-day forecast (log RV on lag-day, 5d, 22d means; pooled OLS fit on 2019-21 across the names) > its 22d mean RV (expansion), HARdn = <=; YZup = Yang-Zhang(5d) > Yang-Zhang(20d) from daily OHLC built from the 15m RTH bars, YZdn = <=; N2dn = 15m RV 5d mean <= 20d mean (relief, report 404), N2up = >.
12 comparisons (6 flags x 2 exits) -> Bonferroni |t| >= 2.9 (flag-on vs flag-off WITHIN the base set, episode-clustered); PASS also needs the same sign of the mean difference in train (2019-21) and test (2018+2022-26).  Lead if |t| >= 2 and same sign."""
import os, sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'xs'); from vector_pine import run_state_machine
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; GX = 'xs/ext/gex/'; UP = '/mnt/user-data/uploads/fartpie/'
R = pd.concat([pd.read_csv(UP + f'xs/ext/{f}.csv', parse_dates=['date']).rename(columns={'opt_ratio': 'r'})[['tk', 'date', 'r']] for f in ('optvol_ratio_sp', 'optvol_ratio')] + [pd.read_csv(UP + 'xs/ext/optvol_ratio_live.csv', parse_dates=['date']).assign(r=lambda d: 100 * d.opt_contracts / d.sh)[['tk', 'date', 'r']]]).drop_duplicates(['tk', 'date']); Rr = R.pivot_table(index='date', columns='tk', values='r').sort_index()
GATE = {'G20': Rr.rolling(20, min_periods=10).median().shift(1) >= 2, 'G252': Rr.rolling(252, min_periods=126).median().shift(1) >= 1.5}
GD = pd.read_parquet(SP + 'greeks_daily.parquet'); GD['d'] = pd.to_datetime(GD.d); PCT = {tk: g.set_index('d').sort_index().gamma_net.rolling(252, min_periods=126).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True) for tk, g in GD.groupby('tk')}
names = sorted(tk for tk in PCT if tk in Rr.columns and glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')); print('names', len(names), names)
BAR = {}; DAY = {}
for tk in names:
    H = pd.read_parquet(glob.glob(UP + f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')[0]).rename(columns={'datetime': 'ts'}); H['dt'] = pd.to_datetime(H.ts, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)].sort_values('dt').drop_duplicates('dt').reset_index(drop=True); BAR[tk] = H
    d = H.dt.dt.normalize(); lr = np.log(H.close).groupby(d).diff().fillna(0); D = pd.DataFrame({'o': H.open.groupby(d).first(), 'h': H.high.groupby(d).max(), 'l': H.low.groupby(d).min(), 'c': H.close.groupby(d).last(), 'rv': (lr ** 2).groupby(d).sum() ** .5}); D = D[D.rv > 0]; DAY[tk] = D
def yz(g, n):
    o, h, l, c = (np.log(g[k]) for k in 'ohlc'); so = (o - c.shift(1)).rolling(n).var(); sc = (c - o).rolling(n).var(); rs = ((h - c) * (h - o) + (l - c) * (l - o)).rolling(n).mean(); k = .34 / (1.34 + (n + 1) / (n - 1)); return (so + k * sc + (1 - k) * rs).clip(lower=0) ** .5
X = []; Y = []
for tk, D in DAY.items():
    L = np.log(D.rv); x = np.column_stack([L, np.log(D.rv.rolling(5).mean()), np.log(D.rv.rolling(22).mean())]); y = L.shift(-1).values; tr = (D.index >= '2019-01-01') & (D.index < '2022-01-01'); ok = tr & np.isfinite(x).all(1) & np.isfinite(y); X.append(x[ok]); Y.append(y[ok])
X = np.vstack(X); Y = np.concatenate(Y); beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]; print('HAR coefs', beta.round(3))
VOL = {}
for tk, D in DAY.items():
    L = np.log(D.rv); F = np.exp(beta[0] + beta[1] * L + beta[2] * np.log(D.rv.rolling(5).mean()) + beta[3] * np.log(D.rv.rolling(22).mean())); a, b = yz(D, 5), yz(D, 20)
    VOL[tk] = pd.DataFrame({'HARup': F > D.rv.rolling(22).mean(), 'YZup': a > b, 'N2up': D.rv.rolling(5).mean() > D.rv.rolling(20).mean(), 'ok': F.notna() & a.notna() & b.notna()}).shift(1)
rows = []
for tk in names:
    H = BAR[tk]; g = PCT[tk]; day = H.dt.dt.normalize().values; days = pd.DatetimeIndex(day); gprev = g.shift(1).reindex(days, method='ffill', tolerance=pd.Timedelta(days=5)).values
    gt = {k: v[tk].reindex(days, method='ffill').values.astype(float) == 1 for k, v in GATE.items()}; vf = VOL[tk].reindex(days, method='ffill'); n = len(H); o = H.open.values; newday = np.r_[True, day[1:] != day[:-1]]; neg = np.where(np.isfinite(gprev), gprev <= .33, False)
    regs, lv, atr, fls = run_state_machine(H[['high', 'low', 'close']]); flips = [i + 1 for i, d in fls if d == 1 and i + 2 < n]; ep = np.zeros(n, int); k = 0
    for i in range(n):
        if neg[i] and (i == 0 or not neg[i - 1]): k += 1
        ep[i] = k if neg[i] else 0
    for eid in range(1, k + 1):
        s = np.where(ep == eid)[0][0]
        if H.dt[s] < pd.Timestamp('2018-07-01') or H.dt[s] > pd.Timestamp('2026-10-14'): continue
        for gk in GATE:
            fe = [b for b in flips if ep[b] == eid and gt[gk][b]]
            if not fe: continue
            e = fe[0]
            if not bool(vf['ok'].values[e] == 1): continue
            j1 = j2 = None
            for j in range(e + 1, min(n, e + 3000)):
                if j1 is None and regs[j - 1] == -1: j1 = j
                if j2 is None and newday[j] and np.isfinite(gprev[j]) and gprev[j] >= .5: j2 = j
                if j1 is not None and j2 is not None: break
            if j1 is None: continue
            for xk, j in (('X1', j1), ('X3', min(j1, j2) if j2 is not None else j1)):
                rows.append(dict(tk=tk, ep=f'{tk}{eid}', gate=gk, exit=xk, entry=H.dt[e], ret=o[j] / o[e] - 1 - 4e-4, HARup=bool(vf.HARup.values[e] == 1), YZup=bool(vf.YZup.values[e] == 1), N2up=bool(vf.N2up.values[e] == 1)))
T = pd.DataFrame(rows); e_ = T.entry; T['half'] = np.where((e_ >= '2019-01-01') & (e_ < '2022-01-01'), 'train', 'test'); T['era'] = pd.cut(e_.dt.year, [2017, 2019, 2021, 2022, 2030], labels=['18-19', '20-21', '22', '23-26'])
for c in ('HARup', 'YZup', 'N2up'): T[c.replace('up', 'dn')] = ~T[c]
def row(d, lab): return f"{lab:34s} n{len(d):4d} win {(d.ret>0).mean()*100:5.1f}% med {d.ret.median()*100:5.2f}% mean {d.ret.mean()*100:5.2f}%"
for gk in ('G20', 'G252'):
    for xk in ('X1', 'X3'):
        B = T[(T.gate == gk) & (T.exit == xk)]; print(f'\n=== {gk} {xk} (chain-proxy negative gamma) ===')
        for lab, m in (('2018-2025', B.entry < '2026-01-01'), ('2026 YTD (to Oct 14)', B.entry >= '2026-01-01'), ('2026 May-Oct', B.entry >= '2026-05-01')):
            d = B[m]; print(row(d, lab + ' base')); print(row(d[~d.N2up], lab + ' + vol relief (N2dn)')); print(row(d[d.N2up], lab + ' vol expansion'))
T.to_parquet(SP + 'g431.parquet'); print('\n2026 trades per month (G20 X1):', T[(T.gate == 'G20') & (T.exit == 'X1') & (T.entry >= '2026-01-01')].groupby(T.entry.dt.to_period('M')).size().to_dict())
