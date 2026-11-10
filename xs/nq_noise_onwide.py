"""Nov 10 2026: NOISE-AREA MOMENTUM (Zarattini-Aziz-Barbon) ON REAL NQ 5m 2014 - Jan 2026, AND DOES THE OVERNIGHT-RANGE FLAG HELP IT? DECLARED BEFORE RESULTS.
Rey: 'someone runs a momentum break of noise bands; our vector is basically that - try it, more advanced with our models'. The QQQ-proxy version is in xs/nq_noise_area.py /
nq_public_strats.py (about +6 pts/day at 1 contract, Sharpe 1.2, positive every year, but -326 pts worst trade). Here: the same rules on the real NQ file (longer, 2014+), and the one new filter
that held on Rule A (Globex range in the top third of its last 20 sessions, xs/nq_edge_hunt.py).
RULES  sigma(k) = mean over the prior 14 sessions of |close of 5m bar k / that day's 09:30 open - 1|; UB = max(open, prior close) x (1 + sigma), LB = min(open, prior close) x (1 - sigma).
  Decisions at bars closing HH:00 / HH:30 from 10:00 to 15:30, executed at the next 5m open. Flat inside the area; long above UB, short below LB; exit when the close crosses max(UB, VWAP) (long) /
  min(LB, VWAP) (short), and reverse on a break of the opposite band; flat at the 15:55 close. Optional hard stop S pts (checked every 5m bar, filled at the stop). Cost 1 pt a round trip. 1 NQ.
REPORT pts per day (1 contract) 2014-21 | 2022-Jan26: all days, wide-overnight days, narrow-overnight days; t of (wide - narrow). Hard stops: none / 40."""
import numpy as np, pandas as pd, importlib.util, warnings; warnings.filterwarnings('ignore')
sp = importlib.util.spec_from_file_location('h', '/home/claude/fartpie/xs/nq_edge_hunt.py'); h = importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
d = h.load(); D = h.days(d); m = d.index.hour * 60 + d.index.minute; R = d[(m >= 570) & (m < 960)]
DAYS = {}
for day, g in R.groupby(R.index.normalize()):
    if len(g) >= 70:                                  # 2014-20 days have 75 bars (3 missing): put every day on the 78-bar grid (missing bar = previous close, no volume)
        grid = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=78, freq='5min'); g = g.reindex(grid)
        g['c'] = g.c.ffill().bfill(); 
        for x in 'ohl': g[x] = g[x].fillna(g.c)
        g['v'] = g.v.fillna(0); DAYS[day] = g
idx = list(DAYS); K = [k for k in range(78) if (570 + 5 * (k + 1)) % 30 == 0 and 600 <= 570 + 5 * (k + 1) <= 930]     # bars closing HH:00 / HH:30, 10:00..15:30
def move(day):
    g = DAYS[day]; return np.abs(g.c.values[:78] / g.o.values[0] - 1) if len(g) == 78 else None
MV = {dy: move(dy) for dy in idx}
def run(day, prev, hist, S):
    g = DAYS[day]
    if len(g) < 78 or len(hist) < 14: return None
    o, hi, lo, c = (g[x].values for x in 'ohlc'); op = o[0]; pc = prev if np.isfinite(prev) else op; sig = np.mean(hist, axis=0)
    vol = g.v.values.astype(float); vw = np.cumsum((hi + lo + c) / 3 * vol) / np.maximum(np.cumsum(vol), 1)
    pos = 0; ent = 0.0; pnl = 0.0; n = 0; start = 0
    for k in range(78):
        if pos:                                                                         # hard stop on the bar's range (entered at an earlier bar's open)
            if S and k >= start:
                st = ent - S * pos
                if (pos == 1 and lo[k] <= st) or (pos == -1 and hi[k] >= st): pnl += -S - 1; pos = 0; n += 1
        if k in K and k < 77:
            ub = max(op, pc) * (1 + sig[k]); lb = min(op, pc) * (1 - sig[k]); nx = o[k + 1]
            if pos == 1 and c[k] < max(ub, vw[k]): pnl += nx - ent - 1; pos = 0; n += 1
            elif pos == -1 and c[k] > min(lb, vw[k]): pnl += ent - nx - 1; pos = 0; n += 1
            if pos == 0:
                if c[k] > ub: pos, ent, start = 1, nx, k + 1
                elif c[k] < lb: pos, ent, start = -1, nx, k + 1
    if pos: pnl += (c[77] - ent) * pos - 1; n += 1
    return pnl, n
rows = {S: [] for S in (0, 40)}
for i, day in enumerate(idx):
    if i < 15: continue
    hist = [MV[x] for x in idx[i - 14:i] if MV[x] is not None]; prev = D.PC.get(day, np.nan)
    for S in rows:
        r = run(day, prev, hist, S)
        if r: rows[S].append((day, r[0], r[1]))
onr = D.OH - D.OL; W = (onr > onr.shift(1).rolling(20).quantile(2 / 3))
for S, L in rows.items():
    Z = pd.DataFrame(L, columns=['day', 'pts', 'n']).set_index('day'); Z['wide'] = W.reindex(Z.index).fillna(False).values; A = Z.index < '2022-01-01'
    print(f'\n=== noise-area momentum on real NQ, hard stop {S or "none"} | days {len(Z):,} | trades/day {Z.n.mean():.2f} | worst day {Z.pts.min():+.0f} pts ===')
    for lab, mm in (('2014-21', A), ('2022-Jan26', ~A)):
        z = Z[mm]; a, b = z.pts[z.wide], z.pts[~z.wide]; t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        print(f'  {lab:<10} all {z.pts.mean():+6.2f} pts/day (days >= +7.5: {np.mean(z.pts >= 7.5):.0%}, < 0: {np.mean(z.pts < 0):.0%}) | wide {a.mean():+6.2f} (n {len(a)}) | narrow {b.mean():+6.2f} (n {len(b)}) | diff t {t:+.2f}')
    print('  by year all / wide / narrow: ' + ' | '.join(f'{y}: {g.pts.mean():+.1f}/{g.pts[g.wide].mean():+.1f}/{g.pts[~g.wide].mean():+.1f}' for y, g in Z.groupby(Z.index.year)))
