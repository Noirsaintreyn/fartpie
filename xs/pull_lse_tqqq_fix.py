"""Fix report 450 gaps: the vault's end date is exclusive, so the last day of every 15-day chunk was missing. Re-fetch those days (start = that day, end = next day) for TQQQ and SQQQ and merge. Key read from the scratchpad .lse_env at runtime."""
import sys, time, json, urllib.request, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
def get(tk, a, b, tries=6):
    err = None
    for k in range(tries):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe=15m&start={a}&end={b}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=90))
        except Exception as e: err = e; time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1))
    print('FAIL', tk, a, str(err)[:80], flush=True); return None
for tk in sys.argv[1:]:
    old = pd.read_parquet(SP + f'{tk}_15m_lse.parquet'); days = pd.date_range('2018-01-01', '2026-11-18', freq='15D'); rows = []
    for b in days[1:]:
        d0 = (b - pd.Timedelta(days=1)); d = get(tk, d0.date(), b.date())
        if d: rows += d
        time.sleep(.25)
    new = pd.DataFrame(rows); df = pd.concat([old, new]).drop_duplicates(['ts']).sort_values('ts').reset_index(drop=True); print(tk, 'old', len(old), 'new rows', len(new), 'merged', len(df), flush=True); df.to_parquet(SP + f'{tk}_15m_lse.parquet')
