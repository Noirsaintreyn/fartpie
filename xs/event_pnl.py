"""Nov 2 2026: where does the book's P&L come from? Each held trade's daily contribution (1/6 weight, open->close path like book_mtm.daily),
tagged by the day: the stock's earnings gap day / FOMC-like day (QQQ 14:00 volume spike, Wed) / jobs-report Friday (approx) /
stock high-volatility day (|daily move| > 2x its prior 20d std, no event) / regular. Share of total P&L vs share of position-days."""
import os, glob, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
L = prep(glob.glob('tm/*.parquet')); L = L[L['var'] == 'vec']; TK = taken(L)
Q = pd.read_parquet(glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/QQQ_15m_*.parquet')[0]); Q['dt'] = pd.to_datetime(Q.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
Q['d'] = Q.dt.dt.normalize(); q2 = Q[Q.dt.dt.strftime('%H:%M') == '14:00'].set_index('d').volume
FOMC = set(q2[(q2 / q2.rolling(60, min_periods=20).median().shift(1) >= 2.5) & (q2.index.dayofweek == 2)].index)
EG = {}
for tk in TK.tk.unique():
    f = f'earn/{tk}.csv'; b = bars(tk)
    if not os.path.exists(f) or b is None: continue
    e = pd.read_csv(f)
    if not len(e): continue
    D = b[1].index; t = pd.to_datetime(e.iloc[:, 0], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    EG[tk] = {d.normalize() if d.hour < 12 else (D[D > d.normalize()][0] if (D > d.normalize()).any() else d.normalize()) for d in t}
rows = []
for tk, e, x, r in TK.itertuples(index=False):
    b = bars(tk)
    if b is None: continue
    op, dc = b; e, x = pd.Timestamp(e), pd.Timestamp(x); px = op.get(e)
    if px is None or px != px: continue
    days = dc[(dc.index >= e.normalize()) & (dc.index < x.normalize())]; v = [px] + list(days.values); idx = list(days.index) + [x.normalize()]; vals = v[1:] + [px * (1 + r)]
    sd = dc.pct_change().rolling(20).std().shift(1)
    for d_, a, b_ in zip(idx, v, vals):
        mv = dc.pct_change().get(d_, np.nan)
        tag = ('earnings' if d_ in EG.get(tk, ()) else 'FOMC' if d_ in FOMC else 'jobs Friday' if (d_.dayofweek == 4 and d_.day <= 7) else
               'high-vol day (no event)' if (mv == mv and sd.get(d_, np.nan) == sd.get(d_, np.nan) and abs(mv) > 2 * sd.get(d_)) else 'regular')
        rows.append((d_, tag, (b_ / a - 1) / 6))
P = pd.DataFrame(rows, columns=['d', 'tag', 'pnl']); P = P[P.d >= '2019-01-01']; P['h'] = np.where(P.d.dt.year <= 2021, '2019-21', '2022-26')
for h, g in P.groupby('h'):
    t = g.groupby('tag').pnl.agg(['size', 'sum', 'mean']); t['share of position-days'] = t['size'] / t['size'].sum(); t['share of P&L'] = t['sum'] / g.pnl.sum()
    t['avg per position-day (x6, %)'] = t['mean'] * 600
    print(f'\n{h}: total book P&L (sum of daily contributions) {g.pnl.sum():+.1%}'); print(t[['share of position-days', 'share of P&L', 'avg per position-day (x6, %)']].round(3).sort_values('share of P&L', ascending=False).to_string())
