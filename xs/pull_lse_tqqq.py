"""Pull 15m candles for TQQQ and SQQQ from the LSE vault (key read from the scratchpad .lse_env at runtime, never stored in the repo). 15-day chunks (API caps at 5000 rows)."""
import os, sys, time, json, urllib.request, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
def get(tk, a, b, tries=6):
    err = None
    for k in range(tries):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=15m&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=90))
        except Exception as e: err = e; time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1))
    print('FAIL', tk, a, b, str(err)[:100], flush=True); return None
for tk in sys.argv[1:]:
    days = pd.date_range('2018-01-01', '2026-11-18', freq='15D'); rows = []; bad = 0
    for a, b in zip(days[:-1], days[1:]):
        d = get(tk, a.date(), (b - pd.Timedelta(days=1)).date())
        if d is None: bad += 1; continue
        rows += d; time.sleep(.25)
        if len(d) >= 5000: print('TRUNC', tk, a.date(), len(d), flush=True)
    df = pd.DataFrame(rows); print(tk, len(df), 'failed chunks', bad, df.columns.tolist() if len(df) else '', flush=True)
    if len(df): df.to_parquet(SP + f'{tk}_15m_lse.parquet')
