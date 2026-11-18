"""Pull 15m candles (LSE vault) for the sub-$5 flip names in xs/smallcap_tickers_px5.txt (priority order = flip count), 2019-01 -> 2026-11-17. Key read from the scratchpad .lse_env at runtime (never stored in the repo). 90-day chunks. Usage: python3 pull_lse_15m_small.py N"""
import os, sys, time, json, urllib.request, pandas as pd, concurrent.futures as cf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120; tks = open('xs/smallcap_tickers_px5.txt').read().split()[:N]; os.makedirs(SP + 'lse15s', exist_ok=True)
def get(tk, a, b, tries=8):
    for k in range(tries):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=15m&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=60))
        except Exception as e: time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1)); err = e
    print('FAIL', tk, a, b, err, flush=True); return None
def one(tk):
    f = SP + f'lse15s/{tk}.parquet'
    if os.path.exists(f): return tk, -1
    days = pd.date_range('2019-01-01', '2026-11-19', freq='90D'); rows = []
    for a, b in zip(days[:-1], days[1:]):
        d = get(tk, a.date(), (b - pd.Timedelta(days=1)).date())
        if d is None: return tk, -2
        rows += d; time.sleep(.25)
        if d and len(d) >= 5000: print('TRUNC', tk, a.date(), len(d), flush=True)
    df = pd.DataFrame(rows); df.to_parquet(f); return tk, len(df)
with cf.ThreadPoolExecutor(2) as ex:
    for tk, n in ex.map(one, tks): print(tk, n, flush=True)
