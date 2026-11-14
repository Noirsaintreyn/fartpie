"""Nov 14 2026 ICT-ON-STOCKS (declared before results). Do ICT ideas transfer to the screener's 15m flips (xs/ext/wide_T.parquet, sp, ratio >= 1.5, 15m track)? Features known at entry: (1) PM_UP = share of premarket 15m bars (04:00-09:29 ET, entry day) closing up, and PM_RET = premarket net move / prior RTH close, GAP = 09:30 open / prior close - 1 (the stock 'overnight book': inventory built before the open); tested both directions, no direction assumed; (2) SWP = the 26 bars before entry took out the PRIOR-DAY LOW and closed back above it (stop run + reclaim); (3) ROOM = (prior-day high - entry) / ATR(26 bars) (distance to the next liquidity pool above, clipped at 0). Outcome = the trade's return to its vector exit (ret). Strict split 2019-21 / 2022-26. PASS = same sign both eras and pooled |t| >= 2."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T0 = pd.read_parquet('xs/ext/wide_T.parquet'); T0 = T0[T0.sp & (T0.ratio >= 1.5) & (~T0.h1)].copy().reset_index(drop=True); T0['entry'] = pd.to_datetime(T0.entry); T0['exit'] = pd.to_datetime(T0.exit)
rows = []
for tk, g in T0.groupby('tk'):
    f = glob.glob(f'lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0]); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.drop_duplicates('dt').sort_values('dt').set_index('dt')
    m = H.index.hour * 60 + H.index.minute; PM = H[(m >= 240) & (m < 570)]; RT = H[(m >= 570) & (m < 960)]
    d_rt = RT.groupby(RT.index.normalize()).agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last')); days = d_rt.index
    pmg = {d: x for d, x in PM.groupby(PM.index.normalize())}
    idx = RT.index; hi = RT.high.values; lo = RT.low.values; cl = RT.close.values
    for ix, e, r in zip(g.index, g.entry, g.ret):
        d = e.normalize(); k = days.searchsorted(d)
        if k < 1 or k >= len(days) or days[k] != d: continue
        pd_ = d_rt.iloc[k - 1]; pmb = pmg.get(d)
        if pmb is None or len(pmb) < 4: continue
        j = idx.searchsorted(e)
        if j < 27 or idx[j] != e: continue
        up = (pmb.close > pmb.open).mean(); pr = (pmb.close.iloc[-1] - pmb.open.iloc[0]) / pd_.c; gap = d_rt.iloc[k].o / pd_.c - 1
        atr = (hi[j - 27:j - 1] - lo[j - 27:j - 1]).mean(); win_l = lo[j - 27:j].min(); swp = bool(win_l < pd_.l and cl[j - 1] > pd_.l)
        rows.append(dict(t=e, ret=r, PM_UP=up, PM_RET=pr, GAP=gap, SWP=swp, ROOM=max(pd_.h - cl[j - 1], 0) / atr if atr > 0 else np.nan))
P = pd.DataFrame(rows).sort_values('t').reset_index(drop=True); print('trades', len(P), '| sweep share', round(P.SWP.mean(), 3))
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
ERAS = (('19-21', ('2019', '2022')), ('22-26', ('2022', '2027')), ('pooled', ('2019', '2027')))
print('mean trade return % | low | high | high-low t (expanding PIT terciles)')
for c in ('PM_UP', 'PM_RET', 'GAP', 'ROOM'):
    v = P[c].values; o = np.full(len(v), np.nan)
    for i in range(300, len(v)):
        h = v[:i]; h = h[~np.isnan(h)]
        if not np.isnan(v[i]): q1, q2 = np.quantile(h, [1 / 3, 2 / 3]); o[i] = 0 if v[i] <= q1 else 2 if v[i] >= q2 else 1
    P['T_' + c] = o; line = f' {c:7s}'
    for e, (a, b) in ERAS:
        s = P[(P.t >= a) & (P.t < b)]; lo_ = s[s['T_' + c] == 0].ret; hi_ = s[s['T_' + c] == 2].ret; line += f' | {e}: {lo_.mean() * 100:+.2f} {hi_.mean() * 100:+.2f} (n {len(lo_)}/{len(hi_)}) t {tt(hi_, lo_):+.1f}'
    print(line)
line = ' SWP    '
for e, (a, b) in ERAS:
    s = P[(P.t >= a) & (P.t < b)]; y = s[s.SWP].ret; n_ = s[~s.SWP].ret; line += f' | {e}: no {n_.mean() * 100:+.2f} yes {y.mean() * 100:+.2f} (n {len(n_)}/{len(y)}) t {tt(y, n_):+.1f}'
print(line + '  (sweep of prior-day low then reclaim, vs not)')
