"""Pull free public data for small/mid flip filters: SEC insider-transactions bulk sets (Form 4), SEC submissions form list (offerings), FINRA bi-monthly short interest, FRED macro series. Output to scratchpad/ext."""
import requests, pandas as pd, io, zipfile, time, datetime as dt
H = {'User-Agent': 'quant-research-bot research@example.com'}; OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
# 1 insider
tr = []; sub = []; own = []
for y, q in [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2), (2026, 3), (2026, 4)]:
    r = requests.get(f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip', headers=H, timeout=120)
    if r.status_code != 200: print('insider', y, q, r.status_code); continue
    z = zipfile.ZipFile(io.BytesIO(r.content)); sub.append(pd.read_csv(z.open('SUBMISSION.tsv'), sep='\t', dtype=str)); tr.append(pd.read_csv(z.open('NONDERIV_TRANS.tsv'), sep='\t', dtype=str)); own.append(pd.read_csv(z.open('REPORTINGOWNER.tsv'), sep='\t', dtype=str)); print('insider', y, q, len(tr[-1])); time.sleep(0.3)
S = pd.concat(sub); T = pd.concat(tr); O = pd.concat(own); T = T.merge(S[['ACCESSION_NUMBER', 'FILING_DATE', 'ISSUERTRADINGSYMBOL', 'DOCUMENT_TYPE']], on='ACCESSION_NUMBER'); T = T.merge(O[['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_RELATIONSHIP']].drop_duplicates('ACCESSION_NUMBER'), on='ACCESSION_NUMBER', how='left'); T.to_pickle(OUT + 'insider.pkl'); print('insider rows', len(T))
# 2 offerings
D = pd.read_pickle('/mnt/user-data/uploads/fartpie/live/midsmall_bars.pkl')['d']; tick = requests.get('https://www.sec.gov/files/company_tickers.json', headers=H, timeout=30).json(); m = {v['ticker'].upper(): str(v['cik_str']).zfill(10) for v in tick.values()}; rows = []
for tk in D:
    cik = m.get(tk.upper())
    if not cik: continue
    try:
        r = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json', headers=H, timeout=30).json()['filings']['recent']
        for f, a, form in zip(r['filingDate'], r['acceptanceDateTime'], r['form']):
            if f >= '2024-10-01' and form in ('424B5', '424B4', '424B3', 'S-3', 'S-1', 'S-3ASR', 'S-1/A', '424B2', 'SC 13D', 'SC 13G', 'SC 13D/A', 'SC 13G/A', 'EFFECT', '25-NSE', 'RW', 'NT 10-K', 'NT 10-Q'): rows.append(dict(ticker=tk, filed=f, accept=a, form=form))
    except Exception as e: pass
    time.sleep(0.12)
pd.DataFrame(rows).to_csv(OUT + 'forms.csv', index=False); print('forms', len(rows))
# 3 finra short interest
d = dt.date(2025, 1, 1); fr = []
while d <= dt.date(2026, 11, 14):
    for day in (15, 28, 29, 30, 31):
        try: s = dt.date(d.year, d.month, day)
        except ValueError: continue
        for off in range(0, 4):
            sd = s - dt.timedelta(days=off)
            if sd.weekday() >= 5: continue
            r = requests.get(f'https://cdn.finra.org/equity/otcmarket/biweekly/shrt{sd:%Y%m%d}.csv', timeout=60)
            if r.status_code == 200: fr.append(pd.read_csv(io.StringIO(r.text), sep='|')); break
    d = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
SI = pd.concat(fr).drop_duplicates(['symbolCode', 'settlementDate']); SI.to_pickle(OUT + 'shortint.pkl'); print('short interest rows', len(SI), 'dates', SI.settlementDate.nunique())
# 4 fred
ser = {'HY_OAS': 'BAMLH0A0HYM2', 'VIX': 'VIXCLS', 'DGS10': 'DGS10', 'T10Y2Y': 'T10Y2Y', 'USD': 'DTWEXBGS', 'NFCI': 'NFCI', 'DGS2': 'DGS2'}; F = None
for k, sid in ser.items():
    r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}', timeout=60)
    if r.status_code != 200: print('fred', sid, r.status_code); continue
    x = pd.read_csv(io.StringIO(r.text)); x.columns = ['date', k]; x[k] = pd.to_numeric(x[k], errors='coerce'); x['date'] = pd.to_datetime(x['date']); F = x if F is None else F.merge(x, on='date', how='outer')
F.sort_values('date').to_pickle(OUT + 'fred.pkl'); print('fred', F.shape, F.date.min(), F.date.max())
