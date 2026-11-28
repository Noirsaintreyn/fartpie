"""Report 534 (Nov 28 2026): Rey: test the tweet strategy (@QGFinance, MES 5m: EMA(8)/EMA(9) cross + session VWAP, GEX regime filter, 'hold 5-10 bars') on NQ.
DECLARED BEFORE RESULTS. NQ 5m (nq5_drive_et_ext.parquet, 2017-06..2026-11, ET), signals 09:45-15:30 ET, entry next bar open, cost 1.0 pt/trade (same as reports 230/231), non-overlapping, flat by 15:55.
Long: EMA8 crosses above EMA9 on the bar close AND close > session VWAP (RTH reset, typical price x volume). Short: mirror. (Tweet text is partial; if his VWAP is not session-reset the result can differ.)
Exits: T5 / T10 (time exits, as in the tweet; Rey's research rule forbids time holds so these are the tweet's spec, not a recommendation), SIG (signal exit: opposite EMA cross or close back across VWAP the wrong side), SIG+1R-none (no stops layered).
GEX filter: prior-close QQQ net GEX (xs/ext/gex/QQQ.csv): NEG (<0, 64% of days), LOW3 (bottom third of trailing-252d rank), POS (>=0), ALL.
Split (Rey Nov 17): TRAIN 2019-21 | TEST 2018 + 2022-26. Judge: same sign and day-clustered |t|>=2 in both eras, n>=30; ~24 cells (3 exits x 4 GEX x long/short/both) so read t~3 as the bar. Net pts per trade, win%, PF.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b = pd.read_parquet('/home/claude/nqdrive/nq5_drive_et_ext.parquet'); b = b[b.dt >= '2017-06-01'].reset_index(drop=True)
o, h, l, c, v = [b[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume')]; n = len(b); m = (b.dt.dt.hour * 60 + b.dt.dt.minute).values; day = b.dt.dt.normalize().values; yr = b.dt.dt.year.values
def ema(x, L):
    a = 2 / (L + 1); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)): y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y
e8, e9 = ema(c, 8), ema(c, 9); tp = (h + l + c) / 3; vw = np.full(n, np.nan); rth = (m >= 570) & (m < 960); cp = 0.0; cv = 0.0; pd_ = None
for i in range(n):
    if day[i] != pd_: cp = cv = 0.0; pd_ = day[i]
    if rth[i]: cp += tp[i] * v[i]; cv += v[i]; vw[i] = cp / cv if cv > 0 else np.nan
G = pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); G.index = G.index.normalize(); GR = G.rolling(252, min_periods=120).rank(pct=True)
def gx(d):
    k = G.index.searchsorted(pd.Timestamp(d)) - 1; return (np.nan, np.nan) if k < 0 else (G.iloc[k], GR.iloc[k])
GXD = {d: gx(d) for d in pd.unique(day)}
up = (e8 > e9) & (np.roll(e8, 1) <= np.roll(e9, 1)); dn = (e8 < e9) & (np.roll(e8, 1) >= np.roll(e9, 1))
def run(exit_mode):
    res = []; last = -1
    for i in range(60, n - 3):
        if i <= last or not (585 <= m[i] <= 930) or yr[i] < 2018 or np.isnan(vw[i]): continue
        s = 1 if (up[i] and c[i] > vw[i]) else (-1 if (dn[i] and c[i] < vw[i]) else 0)
        if s == 0: continue
        e = i + 1; pe = o[e]; j = e
        while True:
            if m[j] >= 955 or day[j] != day[e] or j >= n - 2: x = c[j - 1] if day[j] != day[e] else c[j]; break
            if exit_mode in ('T5', 'T10') and j - e + 1 >= (5 if exit_mode == 'T5' else 10): x = c[j]; break
            if exit_mode == 'SIG' and j > e and ((s == 1 and (dn[j] or c[j] < vw[j])) or (s == -1 and (up[j] or c[j] > vw[j]))): x = c[j]; break
            j += 1
        a, r_ = GXD[day[e]]; res.append((day[e], yr[e], s, s * (x - pe) - 1.0, a, r_)); last = j
    return pd.DataFrame(res, columns=['day', 'yr', 's', 'pts', 'gex', 'gr'])
def cell(T):
    if len(T) < 5: return 'n<5'
    dm = T.groupby('day').pts.sum(); t = dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 2 and dm.std() > 0 else np.nan
    w = T.pts[T.pts > 0].sum(); lo = -T.pts[T.pts < 0].sum(); return f'n{len(T):4d} {T.pts.mean():+6.2f} (t{t:+.1f}) win{(T.pts>0).mean()*100:3.0f}% PF{(w/lo if lo>0 else np.nan):.2f}'
for ex in ('T5', 'T10', 'SIG'):
    R = run(ex); tr = R.yr.between(2019, 2021); te = ~tr
    print(f'\n== exit {ex}: trades {len(R)} ({len(R)/ (R.yr.nunique()):.0f}/yr)  net pts per trade after 1.0 pt cost; train 2019-21 | test 2018+2022-26')
    for gl, gm in (('ALL', R.gex.notna()), ('NEG gamma', R.gex < 0), ('LOW3 gamma', R.gr <= 1 / 3), ('POS gamma', R.gex >= 0)):
        for sl, sm in (('both', R.s != 0), ('long', R.s == 1), ('short', R.s == -1)):
            M = gm & sm; print(f'{gl:11} {sl:5} | {cell(R[M & tr])} | {cell(R[M & te])}')
