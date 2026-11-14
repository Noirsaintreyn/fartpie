"""SEC insider-transactions bulk sets 2018Q1-2026Q1 -> open-market purchases (code P, acquired) by officers/directors, one row per transaction. Output scratchpad/ext/insider_P_hist.pkl"""
import requests, pandas as pd, io, zipfile, time
H = {'User-Agent': 'quant-research-bot research@example.com'}; OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; rows = []
for y in range(2018, 2027):
    for q in range(1, 5):
        if y == 2026 and q > 1: break
        r = requests.get(f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip', headers=H, timeout=180)
        if r.status_code != 200: print(y, q, r.status_code, flush=True); continue
        z = zipfile.ZipFile(io.BytesIO(r.content)); S = pd.read_csv(z.open('SUBMISSION.tsv'), sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'FILING_DATE', 'ISSUERTRADINGSYMBOL']); T = pd.read_csv(z.open('NONDERIV_TRANS.tsv'), sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD']); O = pd.read_csv(z.open('REPORTINGOWNER.tsv'), sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_RELATIONSHIP'])
        T = T[(T.TRANS_CODE == 'P') & (T.TRANS_ACQUIRED_DISP_CD == 'A')].merge(S, on='ACCESSION_NUMBER').merge(O, on='ACCESSION_NUMBER'); T = T[T.RPTOWNER_RELATIONSHIP.fillna('').str.contains('Officer|Director', case=False)]; rows.append(T); print(y, q, len(T), flush=True); time.sleep(0.3)
X = pd.concat(rows); X['fd'] = pd.to_datetime(X.FILING_DATE, errors='coerce'); X.to_pickle(OUT + 'insider_P_hist.pkl'); print('total', len(X), X.fd.min(), X.fd.max())
