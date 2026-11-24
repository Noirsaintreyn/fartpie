"""WEBULL MARKET-DATA FEED for the live screener (Nov 24 2026). Read-only: quotes + 15m bars, never an order.
Replaces Yahoo as the first source for 15m bars and live prices (Yahoo publishes closed bars late and unevenly);
live_screener.py falls back to Yahoo for anything Webull does not return.
  .env: WEBULL_APP_KEY, WEBULL_APP_SECRET, WEBULL_API_HOST (same as webull_watch.py; Market Data enabled on the key)
        WEBULL_FEED=0 turns it off.  WEBULL_BAR_TS=end if --probe shows a bar stamped with its close time (default: open time, like Yahoo).  WEBULL_SESSIONS=... (optional) trading sessions to request, if your account has pre/post.
  python3 webull_feed.py --probe   print the raw response shape for SPY (bars + snapshot) and what we parse from it - RUN THIS FIRST
  python3 webull_feed.py --selftest offline parser test
  python3 webull_feed.py --lag      time bar publication: seconds from each 15m close until Webull shows that bar (run through one close)
"""
import os, sys, time, json
import pandas as pd
try:
    from dotenv import load_dotenv; load_dotenv('.env'); load_dotenv('discord_bot/.env')
except Exception: pass
ET = 'America/New_York'; LIVE = os.getenv('LIVE_DIR', 'live')
import logging
for _n in ('webull', 'webull.core', 'webull.core.client'): logging.getLogger(_n).setLevel(logging.CRITICAL)   # the SDK logs request headers (incl. your access token) on errors: keep that out of logs/pastes
_DC = {}
def enabled(): return not _DC.get('off') and os.getenv('WEBULL_FEED', '1') != '0' and bool(os.getenv('WEBULL_APP_KEY')) and bool(os.getenv('WEBULL_APP_SECRET'))
def client():
    if 'c' not in _DC:
        from webull.core.client import ApiClient
        from webull.data.data_client import DataClient
        os.environ.setdefault('WEBULL_OPENAPI_TOKEN_DIR', f'{LIVE}/webull_token')
        api = ApiClient(os.environ['WEBULL_APP_KEY'], os.environ['WEBULL_APP_SECRET'], 'us'); api.add_endpoint('us', os.getenv('WEBULL_API_HOST', 'api.webull.com'))
        _DC['c'] = DataClient(api)
    return _DC['c']
def _json(res):
    if res.status_code != 200: raise RuntimeError(f'HTTP {res.status_code} {res.text[:160]}')
    try: return res.json()
    except ValueError: return {}
def _dicts(x):
    """every dict anywhere inside x (response wrappers differ by endpoint)"""
    if isinstance(x, dict):
        yield x
        for v in x.values(): yield from _dicts(v)
    elif isinstance(x, list):
        for v in x: yield from _dicts(v)
def _t(v):
    if v in (None, ''): return None
    try:
        f = float(v); return pd.to_datetime(int(f), unit='ms' if f > 1e11 else 's', utc=True)
    except (TypeError, ValueError):
        t = pd.to_datetime(v, utc=True, errors='coerce'); return None if pd.isna(t) else t
def parse_bars(j):
    """-> {symbol: DataFrame(datetime[UTC], open, high, low, close, volume)} from a batch bars response"""
    out = {}
    def bar_rows(lst):
        R = []
        for b in lst:
            if not isinstance(b, dict): continue
            t = _t(b.get('time') or b.get('timestamp') or b.get('t') or b.get('start_time') or b.get('datetime'))
            try: R.append((t, float(b['open']), float(b['high']), float(b['low']), float(b['close']), float(b.get('volume') or b.get('vol') or 0)))
            except (KeyError, TypeError, ValueError): continue
        return R
    for d in _dicts(j):
        sym = d.get('symbol') or d.get('ticker')
        if not sym: continue
        for k in ('result', 'bars', 'items', 'list', 'data', 'klines'):
            if isinstance(d.get(k), list) and d[k] and isinstance(d[k][0], dict) and ('close' in d[k][0]):
                R = [r for r in bar_rows(d[k]) if r[0] is not None]
                if R: out[str(sym)] = pd.DataFrame(R, columns=['datetime', 'open', 'high', 'low', 'close', 'volume']).sort_values('datetime').drop_duplicates('datetime', keep='last').reset_index(drop=True)
    return out
def parse_prices(j):
    out = {}
    for d in _dicts(j):
        sym = d.get('symbol') or d.get('ticker')
        if not sym: continue
        for k in ('price', 'last', 'last_price', 'latest_price', 'close'):
            if d.get(k) not in (None, ''):
                try: out[str(sym)] = float(d[k]); break
                except (TypeError, ValueError): pass
    return out
def _bars_call(dc, part, count, kw):
    """Webull's batch-bars body wants `symbols` as a JSON array (a comma string gave HTTP 400 'Request body not readable'); fall back to the string form"""
    try: return _json(dc.market_data.get_batch_history_bar(list(part), 'US_STOCK', 'M15', count=str(count), real_time_required=True, **kw))
    except Exception as e:
        if '400' not in str(e): raise
        return _json(dc.market_data.get_batch_history_bar(','.join(part), 'US_STOCK', 'M15', count=str(count), real_time_required=True, **kw))
def bars15(symbols, count=400, chunk=20):
    """15m bars for many symbols, batched. {sym: df}; symbols Webull does not return are simply missing"""
    out = {}; dc = client(); kw = {}
    if os.getenv('WEBULL_SESSIONS'): kw['trading_sessions'] = os.getenv('WEBULL_SESSIONS')
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        for k in range(3):
            try: out.update(parse_bars(_bars_call(dc, part, count, kw))); break
            except Exception as e:
                if 'NOT_SUBSCRIBED' in str(e) or 'HTTP Status: 403' in str(e):
                    _DC['off'] = True; print('  ! webull: STOCK QUOTES not subscribed (403) - feed off for this run, using Yahoo'); return out
                if k == 2: print(f'  ! webull bars {part[0]}..: {str(e)[:80]}')
                else: time.sleep(0.6 * (k + 1))
    return out
def prices(symbols, chunk=50):
    out = {}; dc = client()
    for i in range(0, len(symbols), chunk):
        try: out.update(parse_prices(_json(dc.market_data.get_snapshot(','.join(symbols[i:i + chunk]), 'US_STOCK'))))
        except Exception as e:
            if 'NOT_SUBSCRIBED' in str(e) or 'HTTP Status: 403' in str(e): _DC['off'] = True
            print(f'  ! webull snapshot: {str(e)[:80]}'); break
    return out
def to_screener_frame(df):
    """screener's fallback frame shape: Open/High/Low/Close/Volume capitalised, tz-aware index (what the Yahoo path produces)"""
    if os.getenv('WEBULL_BAR_TS', 'start') == 'end': df = df.assign(datetime=df.datetime - pd.Timedelta('15min'))   # if --probe shows bars stamped at their CLOSE
    Y = df.set_index('datetime').rename(columns=str.capitalize)[['Open', 'High', 'Low', 'Close', 'Volume']]
    return Y

def selftest():
    j = {'data': [{'symbol': 'SPY', 'result': [{'time': '1763996100000', 'open': '1', 'high': '2', 'low': '0.5', 'close': '1.5', 'volume': '10'}, {'time': 1763997000000, 'open': 1.5, 'high': 2, 'low': 1, 'close': 1.8, 'volume': 5}]},
                    {'symbol': 'AAPL', 'bars': [{'t': '2026-11-24T14:30:00Z', 'open': 3, 'high': 4, 'low': 2, 'close': 3.5}]}]}
    B = parse_bars(j); assert set(B) == {'SPY', 'AAPL'} and len(B['SPY']) == 2 and B['SPY'].close.iloc[-1] == 1.8, B
    assert B['AAPL'].datetime.iloc[0] == pd.Timestamp('2026-11-24 14:30', tz='UTC')
    assert parse_prices({'data': [{'symbol': 'SPY', 'price': '600.5'}, {'symbol': 'QQQ', 'last': 500}]}) == {'SPY': 600.5, 'QQQ': 500.0}
    assert list(to_screener_frame(B['SPY']).columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
    print('selftest ok')
def probe():
    if not (os.getenv('WEBULL_APP_KEY') and os.getenv('WEBULL_APP_SECRET')):
        sys.exit('WEBULL_APP_KEY / WEBULL_APP_SECRET are not in ~/fartpie/.env (or discord_bot/.env). Add them there yourself, then rerun. Until then the screener just keeps using Yahoo.')
    dc = client()
    j = _bars_call(dc, ['SPY', 'QQQ'], 5, {})
    print('RAW BARS:', json.dumps(j)[:900]); B = parse_bars(j); print('PARSED:', {k: (len(v), str(v.datetime.iloc[-1]), float(v.close.iloc[-1])) for k, v in B.items()})
    s = _json(dc.market_data.get_snapshot('SPY,QQQ', 'US_STOCK')); print('RAW SNAP:', json.dumps(s)[:600]); print('PARSED PRICES:', parse_prices(s))
def lag():
    """prints, for each 15m close, how many seconds until SPY's finished bar appears in Webull (and in Yahoo for comparison)"""
    import yfinance as yf
    seen = None
    while True:
        now = pd.Timestamp.now(tz=ET); tgt = now.floor('15min'); t0 = tgt
        if seen == tgt: time.sleep(5); continue
        w = y = None
        while (w is None or y is None) and (pd.Timestamp.now(tz=ET) - t0).total_seconds() < 120:
            lastbar = tgt - pd.Timedelta('15min')                     # bar that opened 15m before the close we are timing
            if w is None:
                try:
                    B = bars15(['SPY'], count=3).get('SPY')
                    if B is not None and (B.datetime.dt.tz_convert(ET) >= lastbar).any(): w = (pd.Timestamp.now(tz=ET) - t0).total_seconds()
                except Exception: pass
            if y is None:
                try:
                    h = yf.Ticker('SPY').history(period='1d', interval='15m')
                    if len(h) and h.index[-1].tz_convert(ET) >= lastbar: y = (pd.Timestamp.now(tz=ET) - t0).total_seconds()
                except Exception: pass
            time.sleep(1)
        print(f'{tgt:%H:%M} close: webull bar after {w}s   yahoo after {y}s', flush=True); seen = tgt
if __name__ == '__main__':
    a = sys.argv[1:]
    if '--selftest' in a: selftest()
    elif '--probe' in a: probe()
    elif '--lag' in a: lag()
    else: print(__doc__)
