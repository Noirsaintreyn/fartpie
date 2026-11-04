"""Nov 4 2026: build tonight's morning report from real data in the sandbox, exactly as the worker would (daily_report.build), for a preview.
Book = the six positions on the site (entries from the site screenshot); bars = Yahoo 15m (premarket+RTH, 60 days); vector/1h states from live_screener.
No Helious/Massive keys here, so the news and economic-calendar parts are empty (the worker has the keys)."""
import sys, os, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import live_screener as LS, daily_report as DR, yfinance as yf
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/ml/preview'
BOOK = [('AMZN', '15m', '2026-10-30 09:30', 232.52), ('NFLX', '15m', '2026-11-02 12:00', 62.22), ('COST', '1h', '2026-10-30 09:30', 1012.99),
        ('AAPL', '15m', '2026-11-04 09:30', 341.23), ('BA', '1h', '2026-10-29 11:30', 173.02), ('META', '1h', '2026-11-03 12:30', 623.21)]
RAT, thru = LS.options_ratio(); last_ratio = RAT.attrs.get('now', RAT.ffill().iloc[-1]); now = LS.now_et()
univ = sorted(t for t in RAT.columns if t not in LS.NO_ETF and last_ratio.get(t, 0) >= 1.5 and LS.in_sp(t, now))
names = sorted(set(univ) | {b[0] for b in BOOK})
raw = yf.download([LS.YF_SYM.get(t, t) for t in names], period='60d', interval='15m', prepost=True, group_by='ticker', progress=False, threads=True)
rows = []
for t in names:
    try: Y = raw[LS.YF_SYM.get(t, t)].dropna(how='all').reset_index()
    except Exception: continue
    ts = pd.to_datetime(Y.iloc[:, 0], utc=True).dt.tz_convert(LS.ET).dt.tz_localize(None)
    H = pd.DataFrame({'dt': ts, 'open': Y['Open'].values, 'high': Y['High'].values, 'low': Y['Low'].values, 'close': Y['Close'].values, 'volume': Y['Volume'].fillna(0).values})
    H.loc[H.volume <= 0, 'volume'] = 1; m = H.dt.dt.hour * 60 + H.dt.dt.minute
    H = H[(m >= 240) & (m < 960) & (H.dt.dt.dayofweek < 5) & (H.dt + pd.Timedelta('15min') <= now)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    H = LS.add_1h(LS.vec(H)); L = H.iloc[-1]
    b = next((x for x in BOOK if x[0] == t), None)
    r = dict(base=t, ticker=t, H=H, last=float(L.close), state15=int(L.reg), trig=float(L.trig), track='15m', note='', ratio=float(last_ratio.get(t, np.nan)))
    if b:
        e = int(np.searchsorted(H.dt.values, np.datetime64(pd.Timestamp(b[2])))); px = b[3]; pk = float(H.high.iloc[e:].max() / px - 1)
        h1 = b[1] == '1h' or (pk >= .02 and (H.s1.iloc[e:] == 1).any())
        r.update(action='HOLD', px=px, e=e, h1=h1, track=b[1], stop=float(H.stop1.iloc[-1] if h1 else H['stop'].iloc[-1]), peak=pk, pnl=float(L.close / px - 1))
    else:
        r.update(action='FLAT', px=None, e=None, h1=False, stop=float(L['stop']), peak=0.0, pnl=0.0)
    rows.append(r)
meta = dict(slots=6, ratios={t: float(last_ratio.get(t, np.nan)) for t in names}, w=1 / 6)
print('rows', len(rows), 'held', [r['base'] for r in rows if r['action'] == 'HOLD'], flush=True)
fs = DR.build(LS, rows, meta, OUT, True)
for f in fs: print('FILE', f)
