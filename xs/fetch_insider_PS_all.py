"""All open-market insider buys (P) and sales (S), 2018-2026Q1, from the SEC Form 3/4/5 quarterly datasets -> ext/insider_PS_all.pkl (Rey Nov 26 2026: per-stock insider baseline needs sale history too)."""
import zipfile, io, requests, pandas as pd, time
S = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; H = {'User-Agent': 'Rey research rey@example.com'}; out = []
for y in range(2018, 2027):
    for q in range(1, 5):
        if y == 2026 and q > 1: continue
        for t in range(3):
            try: r = requests.get(f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip', headers=H, timeout=180); break
            except Exception: time.sleep(5)
        if r.status_code != 200: print(y, q, r.status_code, flush=True); continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        tr = pd.read_csv(z.open('NONDERIV_TRANS.tsv'), sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD', 'SHRS_OWND_FOLWNG_TRANS'])
        tr = tr[tr.TRANS_CODE.isin(['P', 'S'])]
        sb = pd.read_csv(z.open('SUBMISSION.tsv'), sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'FILING_DATE', 'ISSUERTRADINGSYMBOL', 'DOCUMENT_TYPE']); sb = sb[sb.DOCUMENT_TYPE.isin(['4', '4/A'])]
        ro = pd.read_csv(z.open('REPORTINGOWNER.tsv'), sep='\t', dtype=str, usecols=['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE']).drop_duplicates('ACCESSION_NUMBER')
        out.append(tr.merge(sb, on='ACCESSION_NUMBER').merge(ro, on='ACCESSION_NUMBER', how='left')); print(y, q, len(out[-1]), flush=True); time.sleep(.3)
pd.concat(out).to_pickle(S + 'ext/insider_PS_all.pkl'); print('done', flush=True)
