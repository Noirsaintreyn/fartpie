"""Nov 8 2026: DO OUR NQ EDGES CARRY OVER TO ES, GOLD, SILVER, OIL, BONDS AND FX? DECLARED BEFORE RESULTS.
Data: 15m regular-hours bars 2018 - Oct/Nov 2026 of the ETF proxies (London Strategic Edge): SPY (ES), IWM (RTY), DIA (YM), GLD (GC), SLV (SI), USO (CL), TLT (ZB),
FXE (6E), FXY (6J), FXB (6B). No continuous futures intraday history is on hand, so these trade only the 09:30 - 16:00 New York session. QQQ 15m is the calibration
(the NQ results were built on 5m entries; here everything enters at the next 15m open).
Units: R = one stop. Stop = 0.19 x the asset's 20-day average daily range (on NQ that is ~50 points), target = 0.28 x (the 75); cost 0.05 R a trade. 'NQ-eq' = R x 50.
  A  VECTOR 75/50 MORNING  15m vector flips on bars ending 09:45 - 11:30, next open, target / stop (stop first inside a bar), flat at 12:00, <= 2 a day, no overlap
  B  VECTOR TREND-HOLD     flips on bars ending 09:45 - 15:30, held until the opposite 15m flip (next open), stop as above, flat at the close, <= 4 a day
  C  NOISE-AREA MOMENTUM   Zarattini-Aziz-Barbon bounds (14-day average move from the open at that time of day, gap-adjusted), checks HH:00 / HH:30 from 10:00,
                           trailing exit at the bound or VWAP, flat at the close (R = move / the same stop size)
Per asset and strategy: trades a day, win, R per trade, Sharpe (daily), 2018-21 / 2022-26 R per trade, years positive, max drawdown (R)."""
import numpy as np, pandas as pd, sys, os, glob, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
ROOT = '/mnt/user-data/uploads/fartpie/lse_ohlcv_cache'
AS = {'QQQ': 'NQ (calibration)', 'SPY': 'ES', 'IWM': 'RTY', 'DIA': 'YM', 'GLD': 'Gold GC', 'SLV': 'Silver SI', 'USO': 'Crude CL', 'TLT': 'Bonds ZB', 'FXE': 'Euro 6E', 'FXY': 'Yen 6J', 'FXB': 'Pound 6B'}
COST = .05
def prep(sym):
    H = load(glob.glob(f'{ROOT}/{sym}_15m_*.parquet')[0]); H = H[(H.m >= 570) & (H.m < 960)].reset_index(drop=True); H['day'] = H.dt.dt.normalize()
    H['end'] = H.m + 15; D = H.groupby('day').agg(h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), o=('open', 'first'), n=('open', 'size'))
    D = D[D.n >= 20]; H = H[H.day.isin(D.index)].reset_index(drop=True); D['adr'] = (D.h - D.l).rolling(20).mean().shift(1); D['pc'] = D.c.shift(1)
    H['reg'] = np.asarray(vec(H[['open', 'high', 'low', 'close']])); H['adr'] = D.adr.reindex(H.day).values
    tp = (H.high + H.low + H.close) / 3; v = H.volume.astype(float); H['vwap'] = (tp * v).groupby(H.day).cumsum() / v.groupby(H.day).cumsum().clip(lower=1)
    return H, D
def vector(H, mode):
    o, h, l, c, reg, end, day, adr = (H[x].values for x in ('open', 'high', 'low', 'close', 'reg', 'end', 'day', 'adr')); n = len(H); rows = []; busy = -1; cnt = {}
    lo_end, hi_end, maxn = (585, 690, 2) if mode == 'A' else (585, 930, 4)
    for k in range(1, n - 1):
        if reg[k] == reg[k - 1] or reg[k - 1] == 0 or not (lo_end <= end[k] <= hi_end) or day[k + 1] != day[k] or k + 1 <= busy or not np.isfinite(adr[k]) or adr[k] <= 0: continue
        if cnt.get(day[k], 0) >= maxn: continue
        d = int(reg[k]); e = o[k + 1]; S = .19 * adr[k]; T = .28 * adr[k]; res = None
        for q in range(k + 1, n):
            if day[q] != day[k] or (mode == 'A' and end[q] > 720): res = ((c[q - 1] - e) * d / S, q - 1); break
            if ((l[q] if d > 0 else h[q]) - e) * d <= -S: res = (-1.0, q); break
            if mode == 'A' and ((h[q] if d > 0 else l[q]) - e) * d >= T: res = (T / S, q); break
            if mode == 'B' and reg[q] == -d and q + 1 < n and day[q + 1] == day[k]: res = ((o[q + 1] - e) * d / S, q); break
        if res is None: res = ((c[n - 1] - e) * d / S, n - 1)
        rows.append((day[k], res[0] - COST)); busy = res[1]; cnt[day[k]] = cnt.get(day[k], 0) + 1
    return pd.DataFrame(rows, columns=['day', 'R'])
def noise(H, D):
    H = H.copy(); H['p'] = H.groupby('day').cumcount(); op = H.groupby('day').open.transform('first'); H['mv'] = (H.close / op - 1).abs()
    SIG = H.pivot_table(index='day', columns='p', values='mv').rolling(14, min_periods=14).mean().shift(1)
    rows = []; days = list(D.index); gi = H.groupby('day').indices
    for di, d in enumerate(days):
        if di == 0 or d not in SIG.index or d not in gi: continue
        ix = gi[d]; o0 = H.open.values[ix[0]]; pc = D.c.iloc[di - 1]; S = .19 * D.adr.iloc[di]
        if not np.isfinite(S) or S <= 0: continue
        pos = 0; e = None
        for p, i in enumerate(ix[:-1]):
            if H.end.values[i] % 30 != 0 or H.end.values[i] < 600 or H.end.values[i] > 930: continue
            s = SIG.loc[d].get(p, np.nan)
            if not np.isfinite(s): continue
            ub, lb = max(o0, pc) * (1 + s), min(o0, pc) * (1 - s); c = H.close.values[i]; nxt = H.open.values[ix[p + 1]]; vw = H.vwap.values[i]
            if pos == 1 and c < max(ub, vw): rows.append((d, (nxt - e) / S - COST)); pos = 0
            elif pos == -1 and c > min(lb, vw): rows.append((d, (e - nxt) / S - COST)); pos = 0
            if pos == 0 and c > ub: pos, e = 1, nxt
            elif pos == 0 and c < lb: pos, e = -1, nxt
        if pos != 0: rows.append((d, (H.close.values[ix[-1]] - e) * pos / S - COST))
    return pd.DataFrame(rows, columns=['day', 'R'])
def stats(T, days):
    P = T.groupby('day').R.sum().reindex(days).fillna(0.0); cum = P.cumsum(); yr = P.groupby(P.index.year).sum(); ty = pd.DatetimeIndex(T.day).year
    return dict(tpd=len(T) / len(days), win=np.mean(T.R > 0), r=T.R.mean(), sh=P.mean() / P.std() * np.sqrt(252) if P.std() > 0 else np.nan, a=T.R[ty < 2022].mean(), b=T.R[ty >= 2022].mean(),
                yp=f'{(yr > 0).sum()}/{len(yr)}', dd=(cum - cum.cummax()).min(), rpd=P.mean())
if __name__ == '__main__':
    RES = []
    for sym, lab in AS.items():
        H, D = prep(sym); days = pd.DatetimeIndex(D.index[D.adr.notna()]); print(f'\n{sym} -> {lab}: {len(days)} days {days[0]:%Y-%m} - {days[-1]:%Y-%m}', flush=True)
        for nm, T in (('A vector 75/50 AM', vector(H, 'A')), ('B vector trend-hold', vector(H, 'B')), ('C noise momentum', noise(H, D))):
            s = stats(T, days); RES.append((lab, nm, s))
            print(f'  {nm:<20} {s["tpd"]:.2f}/day win {s["win"]:3.0%} {s["r"]:+.3f} R/trade (NQ-eq {s["r"] * 50:+5.1f}) | {s["rpd"]:+.3f} R/day Sharpe {s["sh"]:+.2f} | '
                  f'2018-21 {s["a"]:+.3f} / 2022-26 {s["b"]:+.3f} | years + {s["yp"]} | max DD {s["dd"]:+.1f} R', flush=True)
    print('\nRANKED by Sharpe (both halves positive marked *)', flush=True)
    for lab, nm, s in sorted(RES, key=lambda x: -np.nan_to_num(x[2]['sh'], nan=-9)):
        print(f'  {"*" if s["a"] > 0 and s["b"] > 0 else " "} {lab:<17} {nm:<20} Sharpe {s["sh"]:+.2f}  {s["r"]:+.3f} R/trade  {s["rpd"]:+.3f} R/day  years + {s["yp"]}', flush=True)
