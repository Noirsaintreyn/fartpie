"""Pull 5m candles from the LSE vault for the 54 screener names (2018-01 -> 2026-11-17). Key read from the scratchpad .lse_env at runtime (never stored in the repo). 12-day chunks (API caps at 5000 rows)."""
import os, sys, glob, time, json, urllib.request, pandas as pd, concurrent.futures as cf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
tks = sorted({os.path.basename(f).split('_15m_')[0] for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_2018-01-01_*.parquet')}); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); tks = [t for t in tks if t in set(W[W.sp].tk)]
def get(tk, a, b, tries=8):
    for k in range(tries):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=5m&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=90))
        except Exception as e: time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1)); err = e
    print('FAIL', tk, a, b, err, flush=True); return None
def one(tk):
    f = SP + f'lse5m/{tk}.parquet'
    if os.path.exists(f): return tk, -1
    days = pd.date_range('2018-01-01', '2026-11-18', freq='12D'); rows = []
    for a, b in zip(days[:-1], days[1:]):
        d = get(tk, a.date(), (b - pd.Timedelta(days=1)).date()); 
        if d is None: raise RuntimeError(f'failed chunk {tk} {a.date()}')
        rows += d; time.sleep(.25)
        if d and len(d) >= 5000: print('TRUNC', tk, a.date(), len(d), flush=True)
    df = pd.DataFrame(rows); df.to_parquet(f); return tk, len(df)
with cf.ThreadPoolExecutor(3) as ex:
    for tk, n in ex.map(one, tks): print(tk, n, flush=True)
