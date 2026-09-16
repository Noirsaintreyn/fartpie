"""
Fetches OHLCV candles from the London Strategic Edge /candles endpoint
(live query, NOT the rate-limited /export mechanism - candles aren't
subject to the options exports-per-hour cap). Adaptive date-range
splitting handles the account's max_rows_per_request=5000 cap: if a
request comes back with exactly 5000 rows (likely truncated), the range
is bisected and each half re-fetched, recursively, until every chunk
returns under the cap. Caches to parquet so repeated runs don't re-hit
the API.
"""
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = 'https://api.londonstrategicedge.com/vault'
API_KEY = os.getenv('LSE_API_KEY')
CACHE_DIR = 'lse_ohlcv_cache'
MAX_ROWS = 5000


def _get_candles(symbol, timeframe, start, end):
    r = requests.get(f'{BASE_URL}/candles', headers={'x-api-key': API_KEY},
                      params={'symbol': symbol, 'timeframe': timeframe,
                              'start': start.strftime('%Y-%m-%d'), 'end': end.strftime('%Y-%m-%d')},
                      timeout=60)
    r.raise_for_status()
    return r.json()


def _fetch_range(symbol, timeframe, start, end):
    rows = _get_candles(symbol, timeframe, start, end)
    if len(rows) < MAX_ROWS or (end - start).days < 2:
        return rows
    mid = start + (end - start) / 2
    left = _fetch_range(symbol, timeframe, start, mid)
    right = _fetch_range(symbol, timeframe, mid, end)
    return left + right


def fetch_ohlcv(symbol, timeframe, start, end, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/{symbol}_{timeframe}_{start.date()}_{end.date()}.parquet"
    if use_cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    rows = _fetch_range(symbol, timeframe, start, end)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['ts'])
    df = df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df.to_parquet(cache_path, index=False)
    return df
