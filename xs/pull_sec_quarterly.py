"""SEC companyfacts -> raw quarterly+annual rows (10-Q/10-K) for the priced universe: rev, ni, oi, gp, cfo, diluted shares (flows) and assets, equity, debt (instants), with filed dates (point in time). Output scratchpad sec_q.parquet."""
import json, time, requests, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; H = {'User-Agent': 'ReyResearch research-client contact@example.com'}
M = json.load(open('sp500_ticker_cik_map.json')); have = set(pd.read_parquet(SP + 'lse_daily_54.parquet').symbol) | set(pd.read_parquet(SP + 'lse_daily_pool.parquet').symbol)
TAGS = {'rev': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax'], 'ni': ['NetIncomeLoss'], 'oi': ['OperatingIncomeLoss'], 'gp': ['GrossProfit'], 'cfo': ['NetCashProvidedByUsedInOperatingActivities'], 'sh': ['WeightedAverageNumberOfDilutedSharesOutstanding'], 'assets': ['Assets'], 'eq': ['StockholdersEquity'], 'debt': ['LongTermDebt', 'LongTermDebtNoncurrent']}
out = []; bad = []
for i, (tk, cik) in enumerate(sorted(M.items())):
    t = 'META' if tk == 'FB' else tk
    if t not in have: continue
    for k in range(4):
        try:
            r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json', headers=H, timeout=60)
            if r.status_code == 200: break
            time.sleep(2 * (k + 1))
        except Exception: time.sleep(2 * (k + 1))
    else: bad.append(t); continue
    g = r.json().get('facts', {}).get('us-gaap', {})
    for kind, tags in TAGS.items():
        for tag in tags:
            for unit, rows in g.get(tag, {}).get('units', {}).items():
                for x in rows:
                    if x.get('form') in ('10-Q', '10-K', '10-K/A', '10-Q/A', '10-KT'): out.append((t, kind, tag, x.get('start'), x['end'], x['filed'], x['form'], x.get('fp'), x['val']))
    time.sleep(.12)
    if i % 20 == 0: print(i, t, len(out), flush=True)
df = pd.DataFrame(out, columns=['tk', 'kind', 'tag', 'start', 'end', 'filed', 'form', 'fp', 'val']); df.to_parquet(SP + 'sec_q.parquet'); print('rows', len(df), 'names', df.tk.nunique(), 'bad', bad)
