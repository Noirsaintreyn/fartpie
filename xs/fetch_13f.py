"""Fetch 13F-HR holdings for a DECLARED fund list (declared before any result). Keys everything on filing date. Output scratch ext/f13_holdings.pkl: fund, group, period, filed, cusip, issuer, shares, value."""
import requests, time, re, pandas as pd, xml.etree.ElementTree as ET
H = {'User-Agent': 'quant-research-bot research-script'}; OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
FUNDS = {'CONC': {1067983: 'Berkshire', 1336528: 'Pershing', 1656456: 'Appaloosa', 1061768: 'Baupost', 1040273: 'ThirdPoint', 1079114: 'Greenlight', 1167483: 'TigerGlobal', 1061165: 'LonePine', 1103804: 'Viking', 1135730: 'Coatue', 1056831: 'Fairholme'},
         'QUANT': {1037389: 'Renaissance', 1423053: 'Citadel', 1009207: 'DEShaw', 1603466: 'Point72'}}
def get(u):
    for _ in range(4):
        try:
            r = requests.get(u, headers=H, timeout=40)
            if r.status_code == 200: time.sleep(.15); return r
            if r.status_code in (403, 429): time.sleep(3)
            if r.status_code == 404: return None
        except Exception: time.sleep(2)
    return None
def filings(cik):
    j = get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json').json(); out = []
    blocks = [j['filings']['recent']] + [get('https://data.sec.gov/submissions/' + f['name']).json() for f in j['filings'].get('files', [])]
    for b in blocks:
        for f, a, d, p in zip(b['form'], b['accessionNumber'], b['filingDate'], b['reportDate']):
            if f == '13F-HR' and d >= '2018-10-01': out.append((a, d, p))
    return out
rows = []
for g, d in FUNDS.items():
    for cik, nm in d.items():
        for a, fd, per in filings(cik):
            base = f'https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace("-", "")}/'; idx = get(base + 'index.json')
            if idx is None: continue
            names = [x['name'] for x in idx.json()['directory']['item'] if x['name'].lower().endswith('.xml') and 'primary_doc' not in x['name'].lower()]
            got = False
            for n in names:
                r = get(base + n)
                if r is None or b'infoTable' not in r.content[:3000] + r.content[-500:] and b'informationTable' not in r.content[:3000]: continue
                txt = re.sub(r'xmlns(:\w+)?="[^"]+"', '', r.text); txt = re.sub(r'<(/?)\w+:', r'<\1', txt); txt = re.sub(r'\s\w+:\w+="[^"]*"', '', txt)
                try: root = ET.fromstring(txt)
                except Exception: continue
                for it in root.iter('infoTable'):
                    g_ = lambda t: (it.find(t).text if it.find(t) is not None else None)
                    sh = it.find('shrsOrPrnAmt'); rows.append((nm, g, per, fd, g_('cusip'), g_('nameOfIssuer'), float(sh.find('sshPrnamt').text) if sh is not None else None, float(g_('value') or 0), g_('putCall')))
                got = True; break
            print(nm, per, fd, 'ok' if got else 'NO TABLE', flush=True)
A = pd.DataFrame(rows, columns=['fund', 'grp', 'period', 'filed', 'cusip', 'issuer', 'shares', 'value', 'putcall']); A.to_pickle(OUT + 'f13_holdings.pkl'); print(len(A), A.fund.nunique(), A.period.nunique())
