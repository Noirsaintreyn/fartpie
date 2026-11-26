"""Daily option volume from the Webull OpenAPI (Nov 26 2026, Rey: 'wire the Webull api for live data'): replaces the Mac-only TWS/Theta pull.
Keys are YOURS and only read from the environment (never printed or logged): WEBULL_APP_KEY, WEBULL_APP_SECRET, optional WEBULL_REGION (default 'us'), WEBULL_TOKEN_DIR (where the 2FA token lives).
Needs the paid Webull options market-data subscription. Run after the close (screener_service does it at ~16:12 ET once the screener has finished).
  python3 tools/webull_optvol.py --probe AAPL     print the raw field NAMES of one contract + one snapshot (no values of keys) so we can confirm the response layout
  python3 tools/webull_optvol.py --test AAPL MSFT print the call/put volume totals, write nothing
  python3 tools/webull_optvol.py                  every name in the live ratio file -> appends today's row to live/optvol_webull.csv (tk,date,opt_contracts,sh)
Volume = sum of snapshot volume over calls+puts for contracts expiring within WB_MAX_DTE days (default 120) and strikes within +-WB_BAND (default 35%) of the last price: the part of the chain that carries nearly all volume; it is an approximation of the old all-expiry total."""
import os, sys, json, time, datetime as dt, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); os.chdir(ROOT)
OUT = 'live/optvol_webull.csv'; ETF = {'GLD', 'SLV', 'TLT', 'QQQ', 'SPY', 'IWM', 'IEF', 'DIA'}
MAXDTE = int(os.getenv('WB_MAX_DTE', '120')); BAND = float(os.getenv('WB_BAND', '0.35')); SLEEP = float(os.getenv('WB_SLEEP', '0.15'))
def client():
    from webull.core.client import ApiClient; from webull.data.data_client import DataClient
    k, s = os.getenv('WEBULL_APP_KEY'), os.getenv('WEBULL_APP_SECRET')
    if not k or not s: sys.exit('set WEBULL_APP_KEY and WEBULL_APP_SECRET (your Webull OpenAPI app key/secret) in the environment')
    reg = os.getenv('WEBULL_REGION', 'us'); c = ApiClient(k, s, reg)
    if os.getenv('WEBULL_TOKEN_DIR'): c.set_token_dir(os.getenv('WEBULL_TOKEN_DIR'))
    return DataClient(c)
def items(js):
    """first list of dicts found in a response body (the layout is confirmed with --probe)"""
    if isinstance(js, list): return [x for x in js if isinstance(x, dict)]
    if isinstance(js, dict):
        for k in ('data', 'items', 'contracts', 'list', 'result'):
            if k in js:
                r = items(js[k])
                if r: return r
        for v in js.values():
            if isinstance(v, list) and v and isinstance(v[0], dict): return v
    return []
def pick(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ''): return d[k]
    return None
SYMK = ('option_symbol', 'symbol', 'optionSymbol', 'ticker'); VOLK = ('volume', 'vol', 'total_volume', 'tradeVolume')
def contracts(D, tk, spot):
    out, key = [], None; end = (dt.date.today() + dt.timedelta(days=MAXDTE)).isoformat()
    for _ in range(60):
        kw = dict(underlying_symbols=tk, status='LISTING', end_date=end, strike_price_gte=round(spot * (1 - BAND), 2), strike_price_lte=round(spot * (1 + BAND), 2), pagination_key=key)
        r = D.instrument.list_option_contracts(**kw)
        if r.status_code != 200: raise RuntimeError(f'contracts {tk} http {r.status_code}')
        js = r.json(); out += [pick(x, SYMK) for x in items(js) if pick(x, SYMK)]
        key = js.get('pagination_key') if isinstance(js, dict) else None
        if not key: break
        time.sleep(SLEEP)
    return out
def volume(D, syms):
    tot = 0.; miss = 0
    for i in range(0, len(syms), 20):
        r = D.option_market_data.get_option_snapshot(','.join(syms[i:i + 20]), 'US_OPTION')
        if r.status_code != 200: raise RuntimeError(f'snapshot http {r.status_code}')
        for x in items(r.json()):
            v = pick(x, VOLK)
            try: tot += float(v)
            except Exception: miss += 1
        time.sleep(SLEEP)
    return tot, miss
def last_px_sh(tk):
    f = f'live/{tk}_15m.parquet'
    if not os.path.exists(f): return None, None
    H = pd.read_parquet(f, columns=['datetime', 'close', 'volume']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; H = H[(m >= 570) & (m < 960)]; d = H.dt.dt.normalize().max(); T = H[H.dt.dt.normalize() == d]
    return float(T.close.iloc[-1]), (d, float(T.volume.sum()))
def main():
    a = [x for x in sys.argv[1:] if not x.startswith('--')]; D = client()
    if '--probe' in sys.argv:
        tk = a[0] if a else 'AAPL'; r = D.instrument.list_option_contracts(underlying_symbols=tk, status='LISTING'); print('contracts http', r.status_code)
        js = r.json(); print('top-level keys:', list(js.keys()) if isinstance(js, dict) else type(js).__name__); it = items(js); print('n items', len(it)); print('contract fields:', sorted(it[0].keys()) if it else None)
        s = pick(it[0], SYMK) if it else None
        if s:
            r2 = D.option_market_data.get_option_snapshot(s, 'US_OPTION'); print('snapshot http', r2.status_code); j2 = r2.json(); i2 = items(j2); print('snapshot fields:', sorted(i2[0].keys()) if i2 else (list(j2.keys()) if isinstance(j2, dict) else j2))
            print('volume key found:', [k for k in VOLK if i2 and k in i2[0]])
        return
    test = '--test' in sys.argv
    if a: tks = a
    else: tks = sorted(set(pd.read_csv('live_data/optvol_ratio_live.csv', usecols=['tk']).tk) - ETF)
    today = None; rows = []; fails = []
    for i, tk in enumerate(tks):
        try:
            px, shd = last_px_sh(tk)
            if px is None: px, shd = float(__import__('yfinance').Ticker(tk).fast_info['last_price']), None
            syms = contracts(D, tk, px); v, miss = volume(D, syms)
            if test: print(f'{tk}: {len(syms)} contracts, volume {v:,.0f} (no-volume items {miss})')
            elif v > 0 and shd: rows.append(dict(tk=tk, date=shd[0], opt_contracts=v, sh=shd[1]))
        except Exception as e: fails.append(tk); print(f'{tk} failed: {str(e)[:90]}', flush=True)
        if (i + 1) % 20 == 0: print(f'{i + 1}/{len(tks)} names', flush=True)
    if test: print('failed:', fails); return
    if not rows: sys.exit(f'no rows (failed {len(fails)})')
    N = pd.DataFrame(rows); N['date'] = pd.to_datetime(N.date)
    if os.path.exists(OUT): N = pd.concat([pd.read_csv(OUT, parse_dates=['date']), N]).drop_duplicates(['tk', 'date'], keep='last')
    N.sort_values(['tk', 'date']).to_csv(OUT, index=False); print(f'{OUT}: +{len(rows)} rows, failed {len(fails)}, date {N.date.max():%Y-%m-%d}')
if __name__ == '__main__': main()
