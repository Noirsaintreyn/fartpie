"""Nov 24 2026: resumable LSE vault pull of 5m (all research + gex names) then 1m (gex names), paginated (API caps 5000 rows/call).  Key read from the scratchpad .lse_env at RUNTIME only; output parquet in the scratchpad (not in git).
Usage: python3 xs/pull_intraday_lse.py 5m|1m"""
import os, sys, time, json, glob, urllib.request, concurrent.futures as cf, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; KEY = [l.split('=', 1)[1].strip() for l in open(SP + '.lse_env') if l.startswith('LSE_KEY')][0].strip('"\'')
TF = sys.argv[1]; OUT = SP + f'lse_{TF}/'; os.makedirs(OUT, exist_ok=True)
GEX = sorted(f[:-4] for f in os.listdir('xs/ext/gex')); B = sorted(pd.read_parquet(SP + 'b302.parquet').tk.unique()); NAMES = GEX if TF == '1m' else GEX + [t for t in B if t not in GEX]
END = (pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)).date()
def get(tk, a):
    for k in range(8):
        try:
            r = urllib.request.Request(f'https://api.londonstrategicedge.com/vault/candles?symbol={tk}&timeframe={TF}&start={a}&end={END}', headers={'x-api-key': KEY, 'User-Agent': 'curl/8.5.0', 'Accept': '*/*'}); return json.load(urllib.request.urlopen(r, timeout=90))
        except Exception as e: time.sleep(4 * (k + 1) if '429' in str(e) else 1.5 * (k + 1))
    return None
def pull(tk):
    f = OUT + f'{tk}.parquet'
    if os.path.exists(f): return tk, 'have'
    rows = []; a = '2018-01-01'; last = None
    while True:
        d = get(tk, a); time.sleep(.5)
        if d is None: return tk, 'FAILED'
        if not d: break
        rows += d; lt = d[-1]['ts']
        if lt == last or len(d) < 5000: break
        last = lt; a = lt[:10]
    H = pd.DataFrame(rows).drop_duplicates('ts').sort_values('ts'); H.to_parquet(f); return tk, f'{len(H)} rows {H.ts.min()} -> {H.ts.max()}'
for tk in NAMES:
    print(*pull(tk), flush=True)
print('DONE', TF)
