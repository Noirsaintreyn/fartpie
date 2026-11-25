import sys, zipfile, io, requests, pandas as pd, time
S = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; H = {'User-Agent': 'Rey research rey@example.com'}; out = []
for y in range(2018, 2027):
    for q in range(1, 5):
        if y == 2026 and q > 1: continue
        r = requests.get(f'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip', headers=H, timeout=120)
        if r.status_code != 200: print(y, q, r.status_code); continue
        z = zipfile.ZipFile(io.BytesIO(r.content)); x = pd.read_csv(z.open('REPORTINGOWNER.tsv'), sep='\t', dtype=str, usecols=lambda c: c in ('ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_TITLE', 'RPTOWNER_RELATIONSHIP')); out.append(x); print(y, q, len(x), flush=True); time.sleep(.3)
pd.concat(out).to_pickle(S + 'ext/f345_owner_titles.pkl')
