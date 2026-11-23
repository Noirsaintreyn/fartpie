"""Pull daily candles (LSE vault, with volume) for the 110 pool names (xs/ext/wide_T.parquet) that have no 15m cache, 2018-01 -> 2026-11-19, 1-year chunks. Key read from scratchpad .lse_env at runtime (never stored in the repo). Output: scratchpad/lse_daily_pool.parquet."""
import json, time, urllib.request, glob, re, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
w = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); c = {re.search(r'cache/(.+?)_15m', f).group(1) for f in glob.glob('/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/*_15m_2018*')}; miss = sorted(set(w.tk) - c)
def get(tk, a, b):
    for k in range(6):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=1d&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=60))
        except Exception as e: time.sleep(2 * (k + 1))
    return None
rows = []; bad = []
for tk in miss:
    for y in range(2018, 2027):
        d = get(tk, f'{y}-01-01', f'{y}-12-31' if y < 2026 else '2026-11-19')
        if d is None: bad.append((tk, y)); continue
        rows += d
    time.sleep(.2)
df = pd.DataFrame(rows); df.to_parquet(SP + 'lse_daily_pool.parquet'); print(len(df), df.symbol.nunique(), 'failed chunks', len(bad), bad[:5])
