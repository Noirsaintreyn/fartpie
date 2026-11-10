"""Nov 9 2026: VWAP DISTANCE AT THE FLIP + ROTATION + MACRO on the screener's flips (Rey: 'see if the VWAP distance from VWAP flips'; 'I'll try macro and
rotations' discretionally). DECLARED BEFORE RESULTS. Same candidates, outcomes, tests and periods as xs/chop_trend.py (exec'd for the data, features and book).
FEATURES, all known before the entry:
 VWAP (15m RTH bars, Mac): VWD entry - today's VWAP through the bar before entry, / 14-day ATR (flips after the first bar) | PVWD entry - yesterday's full-day VWAP, / ATR |
   WVWD entry - the 5-session rolling VWAP (prior 5 sessions + today so far), / ATR | VREC (flips after the first bar): below VWAP / fresh reclaim (a close below VWAP
   in the last 4 bars, now above) / held above (no close below VWAP in the last 4 bars)
 ROTATION (Yahoo daily, prior close): SECRS20 / SECRS5 the stock's sector ETF return minus SPY's over 20 / 5 days | STKRS20 the stock's 20-day return minus its sector ETF's
 MACRO (Yahoo daily, prior close; same for every flip that day): VIX level | VIXCH5 VIX 5-day change | TNX20 10-year yield 20-day change | DXY20 dollar index 20-day % change
TEST 1 per trade by bucket (terciles cut on 2019-22), 2019-22 (CHOOSE) vs 2023-26 (JUDGE); a feature counts only if its best / worst order repeats.
TEST 2 the 6-slot book: SKIP the worst bucket, or PICK ORDER ratio + feature (stock-level features only; macro is the same for every flip that day). Best on 2019-22,
       judged on 2023-26 against live."""
import os, json, numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_c = open(os.path.join(_HERE, 'chop_trend.py')).read(); exec(_c[:_c.index("\nif __name__ == '__main__':")], globals())
def ydaily(name):
    f = P(f'data/yahoo/daily_v/{name}.parquet')
    if not os.path.exists(f): return None
    Y = pd.read_parquet(f); Y.index = pd.DatetimeIndex(Y.index).as_unit('ns')
    if Y.index.tz is not None: Y.index = Y.index.tz_localize(None)
    return Y.close.astype(float).dropna()
def vwfeat(H, atr, Tk):
    day = H.index.normalize(); tp = ((H.high + H.low + H.close) / 3).values; v = H.volume.values; c = H.close.values; op = H.open.values; pv = tp * v
    cpv = pd.Series(pv).groupby(day.values).cumsum().values; cv = pd.Series(v).groupby(day.values).cumsum().values; vw = cpv / np.where(cv > 0, cv, np.nan)
    cut = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(H)]; days = day[cut[:-1]]; start = dict(zip(days, cut[:-1]))
    dpv, dv = pd.Series(pv, index=H.index).groupby(day).sum(), pd.Series(v, index=H.index).groupby(day).sum()
    pvw = (dpv / dv).shift(1); p5, v5 = dpv.rolling(5).sum().shift(1), dv.rolling(5).sum().shift(1)
    k = H.index.get_indexer(Tk.entry.values); out = pd.DataFrame(index=Tk.index, columns=['VWD', 'PVWD', 'WVWD', 'VREC'], dtype=object)
    for i, (kk, d) in enumerate(zip(k, Tk.day.values)):
        d = pd.Timestamp(d); s = start.get(d); a_ = atr.get(d, np.nan)
        if kk < 0 or s is None or not np.isfinite(a_) or a_ <= 0: continue
        px = op[kk]; row = out.index[i]; out.at[row, 'PVWD'] = (px - pvw.get(d, np.nan)) / a_
        tpv, tv = (cpv[kk - 1], cv[kk - 1]) if kk > s else (0.0, 0.0)
        out.at[row, 'WVWD'] = (px - (p5.get(d, np.nan) + tpv) / (v5.get(d, np.nan) + tv)) / a_
        if kk > s and np.isfinite(vw[kk - 1]):
            out.at[row, 'VWD'] = (px - vw[kk - 1]) / a_
            if px <= vw[kk - 1]: out.at[row, 'VREC'] = 'below VWAP'
            else: out.at[row, 'VREC'] = 'fresh reclaim' if any(c[j] < vw[j] for j in range(max(s, kk - 4), kk)) else 'held above'
    for f in ('VWD', 'PVWD', 'WVWD'): out[f] = pd.to_numeric(out[f], errors='coerce')
    return out
NEW = []
if HAVE15:
    V = []
    for tk in TKS:
        H = bars15(tk)
        if H is None: continue
        g = H.groupby(H.index.normalize()); atr = (g.high.max() - g.low.min()).rolling(14).mean().shift(1)
        V.append(vwfeat(H, atr, T[T.tk == tk]))
    T = T.join(pd.concat(V)); NEW += ['VWD', 'PVWD', 'WVWD', 'VREC']
SEC = {'Communication Services': 'XLC', 'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Energy': 'XLE', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
       'Industrials': 'XLI', 'Technology': 'XLK', 'Basic Materials': 'XLB', 'Real Estate': 'XLRE', 'Utilities': 'XLU'}
try: SMAP = json.load(open(P('data/yahoo/sectors.json')))
except Exception: SMAP = {}
spy = ydaily('SPY')
if spy is not None and SMAP:
    def ret(s, n): return (s / s.shift(n) - 1).shift(1)            # known at the prior close
    r20, r5 = ret(spy, 20), ret(spy, 5); E = {e: ydaily(e) for e in set(SEC.values())}
    E20 = {e: ret(y, 20) for e, y in E.items() if y is not None}; E5 = {e: ret(y, 5) for e, y in E.items() if y is not None}; st20 = {}
    for tk in TKS:                                                 # the stock's own 20-day return (Yahoo daily if cached, else the 15m bars)
        H = None if os.path.exists(P(f'data/yahoo/daily_v/{tk}.parquet')) else bars15(tk); Y, _ = dbars(tk, H)
        if Y is not None: st20[tk] = ret(Y.close, 20)
    S20, S5, K20 = [], [], []
    for tk, d in zip(T.tk, T.day):
        e = SEC.get(SMAP.get(tk, ''), '')
        if e not in E20: S20.append(np.nan); S5.append(np.nan); K20.append(np.nan); continue
        S20.append(E20[e].get(d, np.nan) - r20.get(d, np.nan)); S5.append(E5[e].get(d, np.nan) - r5.get(d, np.nan))
        K20.append(st20[tk].get(d, np.nan) - E20[e].get(d, np.nan) if tk in st20 else np.nan)
    T['SECRS20'], T['SECRS5'], T['STKRS20'] = S20, S5, K20; NEW += ['SECRS20', 'SECRS5', 'STKRS20']
    vix, tnx, dxy = ydaily('VIX'), ydaily('TNX'), ydaily('DX_Y_NYB')
    if vix is not None: T['VIX'] = vix.shift(1).reindex(T.day.values).values; T['VIXCH5'] = (vix - vix.shift(5)).shift(1).reindex(T.day.values).values; NEW += ['VIX', 'VIXCH5']
    if tnx is not None: T['TNX20'] = (tnx - tnx.shift(20)).shift(1).reindex(T.day.values).values; NEW += ['TNX20']
    if dxy is not None: T['DXY20'] = (dxy / dxy.shift(20) - 1).shift(1).reindex(T.day.values).values; NEW += ['DXY20']
MACRO = {'VIX', 'VIXCH5', 'TNX20', 'DXY20'}
if __name__ == '__main__':
    print(f'candidates {len(T):,} | VWAP features: {"yes (15m bars)" if HAVE15 else "no (needs lse_clean - Mac)"} | rotation / macro: {"yes" if "SECRS20" in T else "no (needs the Yahoo daily cache)"}', flush=True)
    for f in NEW: print(f'  {f:<8} coverage {T[f].notna().mean():.0%}', flush=True)
    print('\n=== TEST 1 per trade: bucket | 2019-22 (CHOOSE): n, ret, win, chop-out, runner | 2023-26 (JUDGE) ===', flush=True); SUM = []; SC = {}
    for f in NEW:
        b = bucket(f); ok = b.notna()
        if ok.sum() < 300: continue
        gA, gB = T[ok & (T.per == 'A')].groupby(b[ok & (T.per == 'A')]), T[ok & (T.per == 'B')].groupby(b[ok & (T.per == 'B')])
        mA, mB, cA, cB = gA.ret.mean(), gB.ret.mean(), gA.chop.mean(), gB.chop.mean()
        if len(mA) < 2 or len(mB) < 2: continue
        print(f'  {f}' + ('' if f in ('VREC',) else f'  (cuts {", ".join(f"{q:.3g}" for q in T.loc[T.per == "A", f].astype(float).quantile([1 / 3, 2 / 3]))})'), flush=True)
        for k in sorted(mA.index):
            a, bb = T[ok & (T.per == 'A') & (b == k)], T[ok & (T.per == 'B') & (b == k)]
            print(f'    {k:<14} n {len(a):5,} {a.ret.mean():+.2%} win {a.win.mean():3.0%} chop {a.chop.mean():3.0%} run {a.run.mean():3.0%} | '
                  f'n {len(bb):5,} {bb.ret.mean():+.2%} win {bb.win.mean():3.0%} chop {bb.chop.mean():3.0%} run {bb.run.mean():3.0%}', flush=True)
        hi_, lo_ = mA.idxmax(), mA.idxmin(); jB = mB.get(hi_, np.nan) - mB.get(lo_, np.nan); jC = cB.get(lo_, np.nan) - cB.get(hi_, np.nan)
        SUM.append((f, hi_, lo_, mA[hi_] - mA[lo_], jB, cA[lo_] - cA[hi_], jC)); SC[f] = (b, mA)
    print('\n  SUMMARY (best / worst bucket chosen on 2019-22): ret spread 2019-22 -> same buckets 2023-26 | chop-out gap (worst - best) 2019-22 -> 2023-26', flush=True)
    for f, hi_, lo_, sA, sB, ccA, ccB in sorted(SUM, key=lambda z: -np.nan_to_num(z[4], nan=-9)):
        print(f'    {f:<8} best {str(hi_):<14} worst {str(lo_):<14} {sA:+.2%} -> {sB:+.2%} | chop {ccA:+.0%} -> {ccB:+.0%}  {"HOLDS" if sB > 0 and ccB >= 0 else ""}', flush=True)
    print('\n=== TEST 2 the 6-slot book: CAGR / max DD 2019-22 (CHOOSE) | 2023-26 (JUDGE) ===', flush=True)
    live = daily(taken(T)); print(f'  {"live (ratio order, no skip)":<36} {st2(live)}', flush=True); RES = {}
    for f, (b, mA) in SC.items():
        RES[f'{f}: skip {mA.idxmin()}'] = daily(taken(T[b != mA.idxmin()]))
        if f in MACRO: continue
        sc = b.map(mA).astype(float).fillna(float(mA.median())); X = T.assign(_k=T.groupby('entry').ratio.rank(ascending=False) + sc.groupby(T.entry).rank(ascending=False))
        RES[f'{f}: order ratio + {f}'] = daily(taken(X.sort_values(['entry', '_k', 'tk'], kind='stable')))
    ca = {k: cagr(s, '2019-01-01', '2023-01-01')[0] for k, s in RES.items()}; cl = cagr(live, '2019-01-01', '2023-01-01')[0]; cb = cagr(live, '2023-01-01', '2027-01-01')[0]
    for k in sorted(RES, key=lambda k_: -ca[k_]): print(f'  {k:<36} {st2(RES[k])}', flush=True)
    nb = sum(1 for k, s in RES.items() if ca[k] > cl and cagr(s, '2023-01-01', '2027-01-01')[0] > cb)
    best = max(RES, key=lambda k_: ca[k_]); jb = cagr(RES[best], '2023-01-01', '2027-01-01')[0]
    print(f'  -> {len(RES)} rules; {nb} beat live in BOTH periods | chosen on 2019-22: {best} -> 2023-26 {jb:.1%} vs live {cb:.1%} ({"beats" if jb > cb else "does not beat"} live)', flush=True)
    yr = lambda s: (1 + s).groupby(s.index.year).prod() - 1; yl, yb = yr(live), yr(RES[best])
    print('  by year, live vs chosen: ' + ' | '.join(f'{y} {yl[y]:+.0%} vs {yb[y]:+.0%}' for y in yl.index), flush=True)
