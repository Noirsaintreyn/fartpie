"""DAILY OPTIONS CHAIN LOGGER from the Webull OpenAPI (Nov 10 2026). Read-only: never places an order.
Why: our GEX / DEX / vanna / charm / ATM-IV percentiles come from QQQ options at the PRIOR close. Webull gives no history, so we log the chain ourselves every day and build the history forward.
What it saves per run, under live/options/<SYM>/:
  <YYYY-MM-DD>_raw.json.gz   every contract + every snapshot field exactly as Webull returned it (so nothing is lost if a field turns out to matter later)
  <YYYY-MM-DD>.csv           flat table: underlying spot, expiry, strike, type, plus every numeric/text field from the snapshot
SETUP (once): same .env as webull_watch.py (WEBULL_APP_KEY, WEBULL_APP_SECRET, WEBULL_API_HOST). The API application needs Market Data enabled.
RUN:
  python3 webull_options_log.py --probe             one call: prints the field names Webull returns for a few QQQ contracts (run this first)
  python3 webull_options_log.py                     log QQQ now (run it at ~16:10 ET so open interest + closes are the day's; Webull OI usually updates overnight, so also see --oi-refresh)
  python3 webull_options_log.py --symbols QQQ,SPY --days 45 --band 0.12
  python3 webull_options_log.py --oi-refresh        08:30 ET next morning: re-pull and save <date>_oi.csv with the prior-close open interest (what the model needs at 09:30)
SCHEDULE (Mac): launchd / cron  10 16 * * 1-5  and  30 8 * * 1-5  (ET).
Limits: snapshot = 20 option symbols per request, so a 45-day +-12% QQQ chain is a few hundred calls; the script paces itself and retries."""
import argparse, gzip, json, os, sys, time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__)); LIVE = f'{ROOT}/live'; OPT = f'{LIVE}/options'
def now_et(): return pd.Timestamp.now(tz='America/New_York').tz_localize(None)
def log(m):
    line = f'{now_et():%Y-%m-%d %H:%M:%S} {m}'; print(line, flush=True)
    os.makedirs(LIVE, exist_ok=True)
    with open(f'{LIVE}/webull_options_log.log', 'a') as f: f.write(line + '\n')
def client():
    from webull.core.client import ApiClient
    from webull.data.data_client import DataClient
    os.environ.setdefault('WEBULL_OPENAPI_TOKEN_DIR', f'{LIVE}/webull_token')
    api = ApiClient(os.environ['WEBULL_APP_KEY'], os.environ['WEBULL_APP_SECRET'], 'us'); api.add_endpoint('us', os.getenv('WEBULL_API_HOST', 'api.webull.com'))
    return DataClient(api)
def js(res, what, tries=4):
    for k in range(tries):
        if res.status_code == 200:
            try: return res.json()
            except ValueError: return {}
        time.sleep(1.5 * (k + 1))
        if k == tries - 1: raise RuntimeError(f'{what}: HTTP {res.status_code} {res.text[:300]}')
def rows_of(x):
    """Webull wraps lists differently by endpoint; find the first list of dicts."""
    if isinstance(x, list): return x
    if isinstance(x, dict):
        for k in ('data', 'items', 'list', 'contracts', 'result', 'records'):
            if k in x:
                r = rows_of(x[k])
                if r: return r
        for v in x.values():
            if isinstance(v, list) and v and isinstance(v[0], dict): return v
    return []
def spot(dc, sym):
    j = js(dc.market_data.get_snapshot(sym, 'US_STOCK'), f'stock snapshot {sym}'); r = rows_of(j)
    d = r[0] if r else (j if isinstance(j, dict) else {})
    for k in ('price', 'last', 'close', 'last_price', 'latest_price'):
        if k in d and d[k] not in (None, ''): return float(d[k])
    raise RuntimeError(f'no price in stock snapshot: {str(d)[:300]}')
def contracts(dc, sym, last, days, band):
    lo, hi = last * (1 - band), last * (1 + band); end = (now_et() + pd.Timedelta(days=days)).strftime('%Y-%m-%d'); out = []; key = None
    while True:
        j = js(dc.instrument.list_option_contracts(underlying_symbols=sym, status='LISTING', end_date=end, strike_price_gte=lo, strike_price_lte=hi, pagination_key=key), f'contracts {sym}')
        out += rows_of(j); key = j.get('pagination_key') if isinstance(j, dict) else None
        if not key: break
        time.sleep(.3)
    return out
def osym(c):
    for k in ('option_symbol', 'symbol', 'optionSymbol'):
        if c.get(k): return c[k]
def snap(dc, syms):
    out = []
    for i in range(0, len(syms), 20):
        out += rows_of(js(dc.option_market_data.get_option_snapshot(','.join(syms[i:i + 20]), 'US_OPTION'), 'option snapshot')); time.sleep(.25)
    return out
def run(a):
    dc = client()
    for sym in a.symbols.split(','):
        d = now_et().strftime('%Y-%m-%d') if not a.date else a.date; os.makedirs(f'{OPT}/{sym}', exist_ok=True)
        last = spot(dc, sym); log(f'{sym} spot {last}')
        cs = contracts(dc, sym, last, a.days, a.band); log(f'{sym}: {len(cs)} contracts (<= {a.days}d, +-{a.band:.0%})')
        if not cs: log('no contracts returned - check Market Data permission on the API app'); continue
        if a.probe:
            ss = snap(dc, [osym(c) for c in cs[:5] if osym(c)]); print('CONTRACT FIELDS:', sorted(cs[0].keys())); print('SNAPSHOT FIELDS:', sorted(ss[0].keys()) if ss else 'none'); print(json.dumps(ss[:2], indent=1)[:2500]); continue
        ss = snap(dc, [osym(c) for c in cs if osym(c)]); log(f'{sym}: {len(ss)} snapshots')
        by = {osym(c): c for c in cs}; flat = []
        for s in ss:
            c = by.get(osym(s), {}); flat.append({'asof': now_et().strftime('%Y-%m-%d %H:%M:%S'), 'underlying': sym, 'spot': last, **{f'c_{k}': v for k, v in c.items() if not isinstance(v, (dict, list))}, **{k: v for k, v in s.items() if not isinstance(v, (dict, list))}})
        tag = 'oi' if a.oi_refresh else 'raw'
        with gzip.open(f'{OPT}/{sym}/{d}_{tag}.json.gz', 'wt') as f: json.dump({'asof': flat[0]['asof'] if flat else None, 'spot': last, 'contracts': cs, 'snapshots': ss}, f)
        pd.DataFrame(flat).to_csv(f'{OPT}/{sym}/{d}{"_oi" if a.oi_refresh else ""}.csv', index=False); log(f'{sym}: saved {d} ({len(flat)} rows)')
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--symbols', default='QQQ'); p.add_argument('--days', type=int, default=45); p.add_argument('--band', type=float, default=.12)
    p.add_argument('--probe', action='store_true'); p.add_argument('--oi-refresh', action='store_true'); p.add_argument('--date', default=None); a = p.parse_args()
    try: run(a)
    except Exception as e: log(f'FAILED: {e}'); sys.exit(1)
