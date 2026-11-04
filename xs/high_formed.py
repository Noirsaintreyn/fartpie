"""Nov 4 2026: book-level test of 'a high is formed' exits after +1% MFE. DECLARED BEFORE RESULTS. Exit at the next bar open after the trigger; slot frees earlier.
 WK    shooting star: upper wick >= 60% of range, close < open (first 26 bars after entry)      WKf same, whole trade life
 NHP2  confirmed pivot high: the trade's running high bar is followed by 2 bars with lower highs   NHP4 followed by 4 lower-high bars
 TAIL  shooting star then the next bar CLOSES below the star's close (confirmed rejection)
Armed once high >= entry open +1%. 40 matched runs vs live both periods (CAGR = ret, DD, ret/DD) + runners hit."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
NM = ['WK', 'WKf', 'NHP2', 'NHP4', 'TAIL']; rows = []
for tk, G in XL.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]).reset_index(drop=True); o, h, l, c, t = H.open.values, H.high.values, H.low.values, H.close.values, H.dt.values; pos = pd.Series(np.arange(len(H)), index=H.dt.values)
    for i, r in G.iterrows():
        j = pos.get(r.entry)
        if j is None or np.ndim(j): continue
        je = pos.get(r.exit); jx = je if (je is not None and not np.ndim(je)) else None; px = o[j]; off = 0.0
        if jx is not None:
            g = o[jx] / px - 1
            if abs(r.ret - g) < .005: off = r.ret - g
        last = (jx if jx is not None else len(H) - 2) - 1; rec = dict(i=i)
        for nm in NM:
            end = min(j + 26, last) if nm in ('WK', 'NHP2', 'NHP4', 'TAIL') and False else last
            end = min(j + 26, last) if nm == 'WK' else last
            armed = False; hit = None; runhi = -1; runb = -1
            for b in range(j, end + 1):
                if h[b] > runhi: runhi, runb = h[b], b
                if not armed and h[b] / px - 1 >= .01: armed = True
                if not armed: continue
                rg = h[b] - l[b]; star = rg > 0 and (h[b] - max(o[b], c[b])) / rg >= .6 and c[b] < o[b]
                if nm in ('WK', 'WKf') and star: hit = b; break
                if nm == 'TAIL' and b > j and b - 1 >= j:
                    p = b - 1; rp = h[p] - l[p]
                    if rp > 0 and (h[p] - max(o[p], c[p])) / rp >= .6 and c[p] < o[p] and h[p] / px - 1 >= .01 and c[b] < c[p]: hit = b; break
                if nm in ('NHP2', 'NHP4'):
                    k = 2 if nm == 'NHP2' else 4
                    if runb <= b - k and runhi / px - 1 >= .01 and all(h[q] < runhi for q in range(runb + 1, b + 1)) and b - runb == k: hit = b; break
            if hit is not None and hit + 1 <= last + 1 and (jx is None or hit + 1 < jx): rec[nm + '_x'] = t[hit + 1]; rec[nm + '_r'] = o[hit + 1] / px - 1 + off
        rows.append(rec)
F = pd.DataFrame(rows).set_index('i'); Z = XL.join(F); Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26')
B0 = ev(XL); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live', B0)
for nm in NM:
    Y = XL.copy(); m = Z[nm + '_r'].notna(); Y.loc[m.index[m], 'exit'] = Z.loc[m, nm + '_x']; Y.loc[m.index[m], 'ret'] = Z.loc[m, nm + '_r']
    print(f'  {nm}: triggered {m.sum()} ({m.mean():.0%}), runners hit {(Z[m].ret >= .15).sum()}/{(XL.ret >= .15).sum()}')
    line(nm, ev(Y), B0)
