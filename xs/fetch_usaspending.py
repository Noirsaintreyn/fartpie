"""Pull US federal awards (contracts and grants/loans) per sub-$5-universe company from api.usaspending.gov (public, no key). Strict name match: normalised recipient name must start with the normalised company name. Output scratchpad ext/usaspending_awards.pkl (tk, recipient, start, amount, agency, group)."""
import requests, json, re, time, sys, os, pandas as pd, concurrent.futures as cf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = 'https://api.usaspending.gov/api/v2/search/spending_by_award/'
CT = json.load(open('data/fund/company_tickers.json')); NM = {}
for v in (CT.values() if isinstance(CT, dict) else CT):
    NM[str(v.get('ticker', '')).upper()] = v.get('title') or v.get('name')
SUF = r'\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|holdings?|group|the|international|intl|technologies|technology|systems|sa|nv|ag)\b'
def norm(s): s = re.sub(r'[^a-z0-9 ]', ' ', str(s).lower()); s = re.sub(SUF, ' ', s); return re.sub(r'\s+', ' ', s).strip()
GROUPS = {'contracts': ['A', 'B', 'C', 'D'], 'grants': ['02', '03', '04', '05'], 'loans': ['07', '08']}
def q(name, codes):
    b = {'filters': {'recipient_search_text': [name], 'time_period': [{'start_date': '2018-01-01', 'end_date': '2026-11-13'}], 'award_type_codes': codes}, 'fields': ['Recipient Name', 'Award Amount', 'Start Date', 'Awarding Agency'], 'limit': 100, 'page': 1, 'sort': 'Start Date', 'order': 'desc'}
    for k in range(4):
        try:
            r = requests.post(U, json=b, timeout=60)
            if r.status_code == 200: return r.json().get('results') or []
            if r.status_code in (429, 502, 503): time.sleep(2 + 3 * k); continue
            return []
        except Exception: time.sleep(1 + k)
    return []
def one(tk):
    nm = NM.get(tk); out = []
    if not nm: return tk, None
    core = norm(nm)
    if len(core) < 4: return tk, []
    for g, codes in GROUPS.items():
        for r in q(nm, codes):
            if norm(r.get('Recipient Name')).startswith(core) and r.get('Start Date'): out.append(dict(tk=tk, recipient=r['Recipient Name'], start=r['Start Date'], amount=r.get('Award Amount'), agency=r.get('Awarding Agency'), group=g))
        time.sleep(.15)
    return tk, out
if __name__ == '__main__':
    P = pd.read_pickle(SP + 'ext/runs515_panel2.pkl'); tks = sorted(P.tk.unique()); rows = []; miss = []
    with cf.ThreadPoolExecutor(4) as ex:
        for i, (tk, o) in enumerate(ex.map(one, tks)):
            if o is None: miss.append(tk)
            else: rows += o
            if i % 100 == 0: print(i, len(rows), flush=True)
    pd.DataFrame(rows).to_pickle(SP + 'ext/usaspending_awards.pkl'); print('done', len(rows), 'awards', len({r['tk'] for r in rows}), 'tickers with awards; no-name', len(miss))
