"""Nov 25 2026: London Strategic Edge candles as a market-data fallback for the daily sub-$5 alert and the 5m chart (Yahoo rate-limits shared IPs). The API key is read at runtime from LSE_API_KEY (Render env) or, for local dev only, LSE_KEY in a git-ignored .lse_env; it is never printed or written anywhere.
LSE daily bars include extended hours (open = premarket, close = last after-hours print), which does NOT match the regular-session daily bars the research used, so daily bars are rebuilt from 15-minute candles: RTH only (09:30-16:00 ET), open = first RTH bar, close = last RTH bar's close (the 15:45 bar closes at 16:00). Timestamps from LSE are UTC."""
import os, time, requests, pandas as pd
BASE = 'https://api.londonstrategicedge.com/vault'; ET = 'America/New_York'; MAXR = 5000; CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live', 'lse_cache')
def _key():
    k = os.getenv('LSE_API_KEY')
    if k: return k
    for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), '.lse_env'), os.getenv('LSE_ENV_FILE', '')):
        if p and os.path.exists(p):
            try:
                from dotenv import dotenv_values; v = dotenv_values(p); return v.get('LSE_API_KEY') or v.get('LSE_KEY')
            except Exception: pass
    return None
def available(): return bool(_key())
def _get(sym, tf, s, e):
    for k in range(3):
        try:
            r = requests.get(f'{BASE}/candles', headers={'x-api-key': _key()}, params={'symbol': sym, 'timeframe': tf, 'start': s.strftime('%Y-%m-%d'), 'end': e.strftime('%Y-%m-%d')}, timeout=60)
            if r.status_code == 200: return r.json()
            if r.status_code in (429, 503): time.sleep(2 + 3 * k); continue
            return []
        except Exception: time.sleep(1 + k)
    return []
def candles(sym, tf, start, end):
    rows = _get(sym, tf, start, end)
    if len(rows) < MAXR or (end - start).days < 2: return rows
    mid = start + (end - start) / 2; return candles(sym, tf, start, mid) + candles(sym, tf, mid, end)
def frame(sym, tf, start, end):
    r = candles(sym, tf, pd.Timestamp(start), pd.Timestamp(end))
    if not r: return None
    D = pd.DataFrame(r).drop_duplicates('ts'); D['ts'] = pd.to_datetime(D.ts).dt.tz_localize('UTC').dt.tz_convert(ET).dt.tz_localize(None); return D.set_index('ts').sort_index()[['open', 'high', 'low', 'close', 'volume']]
def bars_5m(sym, days=30):
    end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1); return frame(sym, '5m', end - pd.Timedelta(days=days), end)
def daily_rth(sym, years=2.2):
    """RTH daily bars rebuilt from 15m candles; cached per symbol (incremental: only the last few days are re-fetched)"""
    os.makedirs(CACHE, exist_ok=True); f = os.path.join(CACHE, f'{sym}_15m.pkl'); end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1); old = pd.read_pickle(f) if os.path.exists(f) else None
    start = (old.index.max().normalize() - pd.Timedelta(days=4)) if old is not None and len(old) else end - pd.Timedelta(days=int(365 * years))
    new = frame(sym, '15m', start, end)
    if new is None and old is None: return None
    D = pd.concat([x for x in (old, new) if x is not None]); D = D[~D.index.duplicated(keep='last')].sort_index(); D.to_pickle(f)
    t = D.index.hour * 60 + D.index.minute; R = D[(t >= 9 * 60 + 30) & (t < 16 * 60)]
    if not len(R): return None
    G = R.groupby(R.index.normalize()); o = G.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum')); o.index = pd.DatetimeIndex(o.index); return o
