"""LATE/EXTENDED ENTRIES AND EARLY-EXIT RULES on the mid/small-cap 1h vector trades (Rey, Nov 12: "extended from VWAP ... sell if the 2nd or 3rd candle doesn't hold ... or move up a certain amount"). Fixed before results.
Trades = 1h RTH vector bull flips on the cached wide universe (original 44 + IWM-like small-cap proxy + sub-$5 list; see report_universe_expand.py), entry next open, baseline exit = next open after the bear flip. Cost 15 bp (40 bp if entry < $5), same for every variant (round trips unchanged).
ENTRY FILTERS (skip the trade): vwap>1 / vwap>2 = flip-bar close more than 1 / 2 ATR(14,1h) above the session VWAP (RTH, volume-weighted typical price up to that bar); gap>1 = overnight gap above 1 ADR (20d mean range%).
EARLY EXITS (exit at the next open): holdF2/holdF3 = any of the first 2/3 candle closes (entry bar onward) below the flip-bar close; holdE2/holdE3 = same vs the entry open; prog3_.5 / prog3_1 / prog5_.5 / prog5_1 = after 3 or 5 bars the highest high has not reached entry + 0.5 / 1.0 ATR; TP1 / TP2 = take profit at entry + 1 / 2 ATR (fill at the level when the bar's high touches it).
Reported on filters A (all flips), B (flip vol >= 1.5x), C (B + low-vol day), halves Dec23-24 / 2025+. Paired against the unmodified baseline on the same trades; an entry filter's mean is over the trades it keeps, plus how many it removes."""
import sys, os, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import midsmall_screener as M
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data = pickle.load(open(SC + 'uni_cache.pkl', 'rb')); print('names', len(data), flush=True)
def atr14(h, l, c): 
    pc = np.r_[c[0], c[:-1]]; tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(14).mean().values
ROWS = []
for tk, Y in data.items():
    try: Yc = M.closed_only(Y); F = M.features(Yc)
    except Exception: continue
    n = len(F); r = F.reg.values; o, h, l, c, v = (Yc[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume')); at = atr14(h, l, c); day = Yc.index.normalize()
    tp_ = (h + l + c) / 3; vw = (pd.Series(tp_ * v).groupby(day).cumsum() / (pd.Series(v).groupby(day).cumsum() + 1e-9)).values
    dd = pd.DataFrame({'h': Yc.high.groupby(day).max(), 'l': Yc.low.groupby(day).min(), 'o': Yc.open.groupby(day).first(), 'c': Yc.close.groupby(day).last()}); adr = (((dd.h - dd.l) / dd.c).rolling(20).mean().shift()).reindex(day).values; pcl = dd.c.shift().reindex(day).values; dop = dd.o.reindex(day).values
    gap = (dop / pcl - 1) / adr; open_ = None
    for i in range(1, n):
        if r[i] == r[i - 1] or r[i] == 0: continue
        if r[i] == 1 and i + 1 < n and open_ is None: open_ = dict(tk=tk, fi=i, ei=i + 1)
        elif r[i] == -1 and open_ is not None and i + 1 < n: open_['xi'] = i + 1; open_['done'] = True; ROWS.append((open_, tk)); open_ = None
    # (open trades ignored: no baseline exit yet)
    for (t, _) in [x for x in ROWS if x[1] == tk and 'built' not in x[0]]:
        fi, ei, xi = t['fi'], t['ei'], t['xi']; epx = o[ei]; A = at[fi]
        if not (A == A) or A <= 0: t['bad'] = True; t['built'] = True; continue
        t.update(built=True, et=Yc.index[ei], epx=epx, vr=float(F.vrel.iat[fi]) if F.vrel.iat[fi] == F.vrel.iat[fi] else np.nan, lv=bool(F.lv.iat[fi]), base=o[xi] / epx - 1,
                 vwx=(c[fi] - vw[fi]) / A, gap=gap[fi])
        def hold(k, lvl):
            for j in range(ei, min(ei + k, xi)):
                if c[j] < lvl:
                    return o[j + 1] / epx - 1 if j + 1 <= xi else None
            return None
        def prog(M_, a):
            if ei + M_ >= xi: return None
            if h[ei:ei + M_].max() < epx + a * A: return o[ei + M_] / epx - 1
            return None
        def tp(k):
            lv = epx + k * A
            for j in range(ei, xi):
                if h[j] >= lv: return lv / epx - 1
            return None
        for nm, val in (('holdF2', hold(2, c[fi])), ('holdF3', hold(3, c[fi])), ('holdE2', hold(2, epx)), ('holdE3', hold(3, epx)), ('prog3_.5', prog(3, .5)), ('prog3_1', prog(3, 1.)), ('prog5_.5', prog(5, .5)), ('prog5_1', prog(5, 1.)), ('TP1', tp(1.)), ('TP2', tp(2.))):
            t[nm] = val if val is not None else t['base']
T = pd.DataFrame([x[0] for x in ROWS if not x[0].get('bad') and 'et' in x[0]]); T = T[T.et >= pd.Timestamp('2023-12-01')]; print('trades', len(T), flush=True)
cost = np.where(T.epx < 5, .004, .0015)
VARS = ['holdF2', 'holdF3', 'holdE2', 'holdE3', 'prog3_.5', 'prog3_1', 'prog5_.5', 'prog5_1', 'TP1', 'TP2']
FIL = {'A all flips': np.ones(len(T), bool), 'B vol>=1.5x': (T.vr >= 1.5).values, 'C vol>=1.5x + low-vol day': ((T.vr >= 1.5) & T.lv).values}
HALF = {'Dec23-Dec24': (T.et < pd.Timestamp('2025-01-01')).values, '2025+': (T.et >= pd.Timestamp('2025-01-01')).values}
for fn, fm in FIL.items():
    print(f'\n######## filter {fn}')
    for hn, hm in HALF.items():
        m = fm & hm; d = T[m]; cst = cost[m]; b = d.base.values - cst
        print(f'  {hn}: n={m.sum()} baseline mean {b.mean() * 100:+.2f}%/trade win {100 * (b > 0).mean():.1f}%')
        print('     rule        mean %/trade   win%   paired diff vs baseline (t)')
        for vn in VARS:
            x = d[vn].values - cst; df = x - b; t_ = df.mean() / (df.std() / np.sqrt(len(df)) + 1e-12); print(f'     {vn:10s} {x.mean() * 100:+8.2f}    {100 * (x > 0).mean():5.1f}   {df.mean() * 100:+6.2f}% ({t_:+5.2f})')
        for nm, mk in (('skip vwap>1', d.vwx.values > 1), ('skip vwap>2', d.vwx.values > 2), ('skip gap>1', d.gap.values > 1)):
            kp = ~mk; x = (d.base.values - cst)[kp]
            print(f'     {nm:11s} keeps {kp.sum():4d}/{len(d)}: mean {x.mean() * 100:+8.2f}%  win {100 * (x > 0).mean():5.1f}   removed trades averaged {((d.base.values - cst)[mk].mean() * 100 if mk.any() else float("nan")):+.2f}%')
