"""Daily candles for the 54 research names from the LSE vault; key read at runtime from the scratchpad .lse_env, never stored here."""
import json, time, urllib.request, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
def get(tk, a, b):
    for k in range(5):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=1d&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=60))
        except Exception as e: time.sleep(3 * (k + 1))
    return None
rows = []; bad = []
for tk in sorted(pd.read_parquet(SP + 'b302.parquet').tk.unique()):
    for a, b in (('2017-06-01', '2020-12-31'), ('2021-01-01', '2026-11-20')):
        d = get(tk, a, b)
        if d is None: bad.append((tk, a)); continue
        for x in d: x['symbol'] = tk
        rows += d; time.sleep(.3)
df = pd.DataFrame(rows).drop_duplicates(['symbol', 'ts']); df.to_parquet(SP + 'lse_daily_54.parquet'); print(len(df), df.groupby('symbol').ts.agg(['min', 'max', 'count']).to_string(), 'bad', bad)
