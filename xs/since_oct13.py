"""Nov 4 2026: realized return since 2026-10-13 (backtest data ends 2026-10-21), live vs union 10-slot. Closed trades only (no mark-to-market of open positions)."""
import sys
sys.argv = sys.argv[:1] + ['live']
_s = open('xs/uni_slots_ratio.py').read(); exec(_s[:_s.index("base = evN(")])
def since(Y, N, d0='2026-10-13'):
    sm = mksim(N); HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); out = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl; s = sm(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m])
        eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); e = np.exp(eq - eq.iloc[0]); b = e[e.index < d0].iloc[-1]; out.append(e.iloc[-1] / b - 1)
    return np.array(out), e.index[-1]
for nm, Y, N in (('LIVE S&P 6 slots', book('live', 1.5), 6), ('UNION 10 slots', book('union', 1.5), 10), ('UNION 6 slots', book('union', 1.5), 6)):
    r, last = since(Y, N); print(f'{nm:<20} since 10/13 to {last.date()}: median {np.median(r):+.2%}  p10 {np.percentile(r,10):+.2%}  p90 {np.percentile(r,90):+.2%}')
for nm, Y in (('S&P', book('live', 1.5)), ('UNION', book('union', 1.5))):
    z = Y[(Y.exit >= '2026-10-13')]; print(nm, 'trades closed since 10/13:', len(z), 'avg ret', f'{z.ret.mean():+.2%}', ' still-open/entered since:', int((Y.entry >= '2026-10-13').sum()))
