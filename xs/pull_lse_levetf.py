"""Pull 15m bars for single-stock leveraged ETFs from the LSE vault (key read from scratchpad .lse_env at runtime). 28-day windows stepped 27 days so the exclusive end date never leaves a gap; deduped on ts."""
import sys, time, json, urllib.request, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
def get(tk, a, b, tries=6):
    err = None
    for k in range(tries):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=15m&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); d = json.load(urllib.request.urlopen(r, timeout=90)); return d.get('candles', d) if isinstance(d, dict) else d
        except Exception as e: err = e; time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1))
    print('FAIL', tk, a, str(err)[:80], flush=True); return None
for tk in sys.argv[1:]:
    rows = []
    for a in pd.date_range('2021-06-01', '2026-11-24', freq='27D'):
        d = get(tk, a.date(), (a + pd.Timedelta(days=28)).date())
        if d: rows += d
        time.sleep(.2)
    df = pd.DataFrame(rows).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    if len(df): df.to_parquet(SP + f'{tk}_15m_lse.parquet'); print(tk, len(df), str(df.ts.iloc[0])[:10], str(df.ts.iloc[-1])[:10], flush=True)
    else: print(tk, 'EMPTY', flush=True)
