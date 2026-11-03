"""Nov 3 2026: 2026 YTD (Jan 1 - Sep 15, the last date every backtest file covers) for live vs the 1h-confirmed-stop variants vs QQQ; plus since Oct 13 on the 17 live-parquet names."""
import sys, glob
src = open('/home/claude/fartpie/xs/stop_1h_ev.py').read().split("B0 = ev(XL)")[0]; exec(src)
A, B = pd.Timestamp('2025-12-31'), pd.Timestamp('2026-09-15')
def curve(Xv, i):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Xv.tk, Xv.entry.astype('int64'))], dtype=np.uint64)
    m = np.ones(len(Xv), bool) if i == 0 else ~(((HSH * 2654435761 + i * 40503) % 1000) < 50)
    s = sim(Xv.entry.values[m], Xv.exit.values[m], Xv.ret.values[m], Xv.ratio.values.astype(float)[m], Xv.tk.values[m])
    return np.exp(s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0))
def stats(e):
    w = e[(e.index > A) & (e.index <= B)]; r = w.iloc[-1] / e[e.index <= A].iloc[-1] - 1; seg = np.r_[e[e.index <= A].iloc[-1], w.values]; dd = (seg / np.maximum.accumulate(seg) - 1).min(); return r, dd
print('2026 YTD (Jan 1 - Sep 15): return / max drawdown, median of 21 matched runs (run 0 = no drops)')
for nm in ('L', 'A5', 'Ainf'):
    Xv = build(nm); R = np.array([stats(curve(Xv, i)) for i in range(0, 21)]); print(f'  {nm:<5} {np.median(R[:, 0]):+.1%} / {np.median(R[:, 1]):.1%}   (no-drop run {R[0, 0]:+.1%}; range {R[:, 0].min():+.1%} .. {R[:, 0].max():+.1%})')
Q = pd.read_parquet(glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); dc = 'datetime' if 'datetime' in Q else Q.columns[0]
Q['dt'] = pd.to_datetime(Q[dc], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); mm = Q.dt.dt.hour * 60 + Q.dt.dt.minute
c = Q[(mm >= 570) & (mm < 960)].groupby(Q.dt.dt.normalize()).close.last(); q = c[(c.index <= B)]
print(f'  QQQ   {q.iloc[-1] / q[q.index <= A].iloc[-1] - 1:+.1%} / {((q[q.index > A] / q[q.index > A].cummax()) - 1).min():.1%}   (to {q.index[-1].date()}; file runs to {c.index[-1].date()})')
