"""Nov 4 2026: in-trade REJECTION SPEED / wick rules - does a fast failed push (hold vs go back) identify trades to leave early? DECLARED BEFORE RESULTS.
Within the first 26 bars (about one RTH day) after the entry, per trade (15m bars, premarket+RTH):
 GB(arm,m): price made +arm% (high vs entry open) and then a CLOSE <= entry open within m bars of the arm bar  -> exit at the next bar open   (arm 1% or 2%, m 4 or 8)
 WK: after +1% MFE, a bar with upper wick >= 60% of its range that closes below its open (shooting star) -> exit at the next bar open
Per-trade accounting (not a book sim): new return = next-open exit; delta = new - live return for triggered trades; total = sum(delta)/6 (a 6-slot book's P&L points), by period;
also runner damage (>=15% trades triggered) and the triggered cohort's live average. A rule is worth a book test only if total > 0 in BOTH periods and runners barely touched."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
RULES = [('GB1/4', .01, 4), ('GB1/8', .01, 8), ('GB2/4', .02, 4), ('GB2/8', .02, 8), ('WK', .01, 0)]
rows = []
for tk, G in XL.groupby('tk'):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = load(f[0]).reset_index(drop=True); o, h, l, c, t = H.open.values, H.high.values, H.low.values, H.close.values, H.dt.values; pos = pd.Series(np.arange(len(H)), index=H.dt.values)
    for i, r in G.iterrows():
        j = pos.get(r.entry)
        if j is None or np.ndim(j): continue
        je = pos.get(r.exit); px = o[j]
        off = 0.0
        if je is not None and not np.ndim(je):
            g = o[je] / px - 1
            if abs(r.ret - g) < .005: off = r.ret - g
        end = min(j + 26, len(H) - 2, (je if (je is not None and not np.ndim(je)) else j + 26)); rec = dict(i=i)
        for nm, arm, m in RULES:
            hit = None; armed = None
            for b in range(j, end):
                if h[b] / px - 1 >= arm and armed is None: armed = b
                if armed is None: continue
                if nm == 'WK':
                    rg = h[b] - l[b]
                    if rg > 0 and (h[b] - max(o[b], c[b])) / rg >= .6 and c[b] < o[b]: hit = b; break
                elif b - armed <= m and c[b] <= px: hit = b; break
                elif b - armed > m: break
            if hit is not None and t[hit + 1] < np.datetime64(r.exit): rec[nm] = o[hit + 1] / px - 1 + off
            else: rec[nm] = np.nan
        rows.append(rec)
Z = XL.join(pd.DataFrame(rows).set_index('i')); Z['per'] = np.where(Z.entry.dt.year <= 2021, '19-21', '22-26'); Z.to_parquet('/tmp/claude-0/ml/reject_Z.parquet')
print('trades', len(Z), ' live avg', {p: f'{g.ret.mean()*100:+.3f}%' for p, g in Z.groupby('per')})
for nm, _, _ in RULES:
    print(f'\n{nm}   triggered | live avg of triggered | new avg | total P&L pts (sum(new-live)/6) | runners(>=15%) hit/ all')
    for p, g in Z.groupby('per'):
        a = g[g[nm].notna()]; d = a[nm] - a.ret
        print(f'  {p}: n{len(a)} ({len(a)/len(g):.0%})  live {a.ret.mean()*100:+.2f}%  new {a[nm].mean()*100:+.2f}%  total {d.sum()/6*100:+.1f} pts   runners {(a.ret>=.15).sum()}/{(g.ret>=.15).sum()}  (runner loss {(d[a.ret>=.15]).sum()/6*100:+.1f} pts, non-runner {(d[a.ret<.15]).sum()/6*100:+.1f} pts)')
