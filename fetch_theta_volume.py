"""Daily OPTION VOLUME per underlying from ThetaData v3 EOD (Theta Terminal running + logged in). Only what the options-ratio filter needs:
contracts, premium, call contracts per day -> data/theta_vol/<TK>.csv (one small file per ticker, appended month by month, resumable).
No open interest and no per-contract files (the full pull in fetch_theta_options.py is ~65 MB/name; this is ~50 KB).
Run (4 shards in parallel = Theta's 4-request limit):
  cd ~/fartpie && for s in 0 1 2 3; do SHARD=$s NSHARD=4 nohup .venv311/bin/python fetch_theta_volume.py > /tmp/thvol_$s.log 2>&1 & done
Tickers: every name with cached 15m bars (xs/ext/rot_trades) that has no data/theta/<TK>/ folder yet. START/END env vars (default 2018-01 .. today)."""
import os, sys, io, time, glob, requests, pandas as pd
BASE = 'http://127.0.0.1:25503/v3/option/history'
START = pd.Timestamp(os.getenv('START', '2018-01-01')); END = pd.Timestamp(os.getenv('END', str(pd.Timestamp.today().date())))
SH, NS = int(os.getenv('SHARD', '0')), int(os.getenv('NSHARD', '1'))
def get(params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(f'{BASE}/eod', params={**params, 'format': 'csv'}, timeout=600)
            if r.status_code == 200: return pd.read_csv(io.StringIO(r.text)) if r.text.strip() else pd.DataFrame()
            if r.status_code in (472, 404): return pd.DataFrame()
            print(f'   HTTP {r.status_code}: {r.text[:120]}', flush=True)
        except requests.ConnectionError:
            print('   cannot reach Theta Terminal on 127.0.0.1:25503 - is it running and logged in?', flush=True); os._exit(1)
        except Exception as e: print('   error', str(e)[:120], flush=True)
        time.sleep(5 * (k + 1))
    return None
have = {os.path.basename(d.rstrip('/')) for d in glob.glob('data/theta/*/')}
tks = sys.argv[1:] or sorted(os.path.basename(f)[:-8] for f in glob.glob('xs/ext/rot_trades/*.parquet'))
tks = [t for t in tks if os.getenv('FORCE') == '1' or t not in have][SH::NS]
os.makedirs('data/theta_vol', exist_ok=True); t0 = time.time()
for i, tk in enumerate(tks):
    f = f'data/theta_vol/{tk}.csv'; cur = str(pd.Period(pd.Timestamp.today(), 'M'))
    if os.path.exists(f) and os.getenv('PARTIAL') == '1':                     # drop last run's partial current month, it gets re-pulled
        _x = pd.read_csv(f); _x[_x.month.astype(str) != cur].to_csv(f, index=False)
    done = set(pd.read_csv(f).month.astype(str)) if os.path.exists(f) else set()
    for m in pd.period_range(START, END, freq='M'):
        if str(m) in done or (m >= pd.Period(pd.Timestamp.today(), 'M') and os.getenv('PARTIAL') != '1'): continue   # finished months (PARTIAL=1: + this month)
        a, b = max(m.start_time, START), min(m.end_time.normalize(), END)
        e = get({'symbol': tk.replace('-', '.'), 'expiration': '*', 'start_date': a.strftime('%Y%m%d'), 'end_date': b.strftime('%Y%m%d')})
        if e is None: print(f'   gave up {tk} {m}', flush=True); continue
        if len(e):
            e['date'] = e['created'].astype(str).str[:10]; e['c'] = e['right'].astype(str).str.upper().str.startswith('C')
            g = e.groupby('date').apply(lambda x: pd.Series({'opt_contracts': x.volume.sum(), 'opt_premium': (x.volume * x.close * 100).sum(), 'call_c': x.volume[x.c].sum()})).reset_index()
        else: g = pd.DataFrame(columns=['date', 'opt_contracts', 'opt_premium', 'call_c'])
        g['tk'] = tk; g['month'] = str(m)
        if len(g) == 0: g = pd.DataFrame([{'date': None, 'opt_contracts': 0, 'opt_premium': 0, 'call_c': 0, 'tk': tk, 'month': str(m)}])
        g.to_csv(f, mode='a', header=not os.path.exists(f), index=False)
    print(f'{tk} done ({i + 1}/{len(tks)}, {(time.time() - t0) / 60:.0f} min)', flush=True)
print('ALL DONE', flush=True)
