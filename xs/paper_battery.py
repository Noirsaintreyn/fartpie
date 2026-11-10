"""Nov 10 2026: PUBLISHED INTRADAY EDGES ON REAL NQ (ET 5m RTH, 2014-02..2026-01-30), 1 contract, cost 0.5 pt per round trip, entries at the next bar's open. DECLARED BEFORE RESULTS.
P1 Gao-Han-Li-Zhou intraday momentum: sign of first-30-min return (open->close of bar 6) trades 15:30-16:00 (enter bar 72 open, exit bar 77 close).
P2 same with sign of open->15:25 return.   P3 = P2 only when prior-day QQQ net GEX < 0 (Baltussen-Da-Soebhag-Zhou hedging-demand: effect strongest when dealers are short gamma).   P3b = P2 when GEX > 0.
P4/P5 gap: |gap| > 0.5 sigma(ADR20): fade the gap (P4) or follow it (P5), enter at the open, exit 11:00.
P6/P7 overnight direction (open vs prior close) follow / fade to 10:30.
P8 turn of month (last 2 + first 3 trading days) long open->close.   P9 option-expiry Friday long open->close (3rd Friday).   P10-P14 each weekday long open->close.
P15 compression breakout: prior-day range < 0.7 x its 20-day mean -> trade the break of the first-30-min range (stop opposite side), exit close.   P16 same, prior-day range > 1.3 x mean.
PASS: pts/day > 0 (or < 0 = reversed sign valid only if declared) with t >= 2 in both 2014-21 and 2022-26; ~16 tests, expect ~1 chance pass at one period, ~0.1 both."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, OPEN, PC, CLOSE, NB
C = .5; DI = {d: i for i, d in enumerate(ALLD)}
gx = pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex; gx.index = gx.index.normalize()
rng_ = pd.Series({d: np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2]) for d in ALLD}); adr = rng_.rolling(20).mean().shift(1); prr = rng_.shift(1)
def b(d, j, k): return A[d][0][k][j]
def day_rows():
    R = {k: {} for k in range(1, 17)}; ds = [d for d in ALLD if pd.Timestamp('2014-02-01') <= d <= pd.Timestamp('2026-01-30')]
    for d in ds:
        i = DI[d]; o, h, l, c, v, vw = A[d][0]
        if i < 25 or len(o) < NB or np.isnan(o[0]) or np.isnan(c[NB - 1]) or np.isnan(o[72]): continue
        g = pd.Timestamp(d); gprev = gx[gx.index < g]; gv = gprev.iloc[-1] if len(gprev) else np.nan
        pcl = PC[d] if np.isfinite(PC[d]) else np.nan; gap = (o[0] - pcl) if np.isfinite(pcl) else np.nan; sg = adr[d]
        r1 = c[5] - o[0]; r2 = c[71] - o[0]; late = c[77] - o[72]
        if r1 != 0: R[1][d] = np.sign(r1) * late - C
        if r2 != 0: R[2][d] = np.sign(r2) * late - C
        if r2 != 0 and np.isfinite(gv):
            (R[3] if gv < 0 else R[4])[d] = np.sign(r2) * late - C
        if np.isfinite(gap) and np.isfinite(sg) and abs(gap) > .5 * sg and not np.isnan(o[18]) and not np.isnan(c[16]):
            x = c[16] - o[1] if not np.isnan(o[1]) else np.nan   # enter bar1 open, exit 11:00 (bar 18 open ~ c[17])
            ex = o[18] - o[1]
            R[5][d] = -np.sign(gap) * ex - C; R[6][d] = np.sign(gap) * ex - C
        if np.isfinite(gap) and gap != 0 and not np.isnan(o[13]) and not np.isnan(o[1]):
            ex = o[13] - o[1]; R[7][d] = np.sign(gap) * ex - C; R[8][d] = -np.sign(gap) * ex - C
        R[10 + g.weekday() if g.weekday() < 5 else 0][d] = c[NB - 1] - o[0] - C if g.weekday() < 5 else 0
        pr = prr[d]; ad = adr[d]
        if np.isfinite(pr) and np.isfinite(ad):
            for key, cond in ((15, pr < .7 * ad), (16, pr > 1.3 * ad)):
                if cond:
                    hh, ll = np.nanmax(h[:6]), np.nanmin(l[:6]); pos = 0
                    for j in range(6, NB - 1):
                        if np.isnan(c[j]) or np.isnan(o[j + 1]): continue
                        if c[j] > hh: pos = 1; e = o[j + 1]; break
                        if c[j] < ll: pos = -1; e = o[j + 1]; break
                    if pos:
                        st = ll if pos > 0 else hh; px = c[NB - 1]
                        for j in range(j + 1, NB):
                            if np.isnan(c[j]): continue
                            if (pos > 0 and l[j] <= st) or (pos < 0 and h[j] >= st): px = st; break
                        R[key][d] = (px - e) * pos - C
    return R, ds
R, ds = day_rows(); ds = pd.DatetimeIndex(ds)
# calendar sets
s = pd.Series(1, index=ds); ym = pd.Series(ds.to_period('M'), index=ds); pos_in = s.groupby(ym).cumsum(); n_in = s.groupby(ym).transform('sum'); tom = (pos_in <= 3) | (pos_in > n_in - 2)
for d in ds:
    o = A[d][0][0]; c = A[d][0][3]
    if np.isnan(o[0]) or np.isnan(c[-1]): continue
    if tom[d]: R.setdefault(9, {})[d] = c[Z.NB - 1] - o[0] - C
    if d.weekday() == 4 and 15 <= d.day <= 21: R.setdefault(0, {})[d] = c[Z.NB - 1] - o[0] - C
names = {1: 'P1 first30 -> last30', 2: 'P2 open->15:25 -> last30', 3: 'P3 P2 | GEX<0', 4: 'P3b P2 | GEX>0', 5: 'P4 fade gap >.5s (to 11:00)', 6: 'P5 follow gap', 7: 'P6 follow overnight dir (to 10:30)', 8: 'P7 fade overnight dir', 9: 'P8 turn of month long', 0: 'P9 opex Friday long', 10: 'P10 Mon long', 11: 'P11 Tue long', 12: 'P12 Wed long', 13: 'P13 Thu long', 14: 'P14 Fri long', 15: 'P15 compression ORB', 16: 'P16 expansion ORB'}
print(f'{len(ds)} days')
for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15, 16):
    x = pd.Series(R.get(k, {})); x.index = pd.DatetimeIndex(x.index); x = x.reindex(ds).fillna(0.0); o = []; ok = True; sg = []
    for nm, f in (('14-21', ds < '2022-01-01'), ('22-26', ds >= '2022-01-01')):
        y = x[f]; t = y.mean() / (y.std() / np.sqrt(len(y))) if y.std() > 0 else 0; o.append(f'{nm}: {y.mean():+5.2f}/d t {t:+5.2f}'); sg.append((np.sign(y.mean()), abs(t) >= 2))
    ok = sg[0][0] == sg[1][0] and sg[0][1] and sg[1][1]; n = int(((pd.Series(R.get(k, {}))) != 0).sum())
    print(f'  {names[k]:<34} n {n:5d} | ' + ' | '.join(o) + ('   <== PASS' if ok else ''))
