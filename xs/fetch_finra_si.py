import requests, pandas as pd, datetime as dt, io
S = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; out = []; got = 0
d = dt.date(2018, 12, 1)
cand = []
for y in range(2018, 2027):
    for m in range(1, 13):
        for day in (13, 14, 15, 16, 17, 27, 28, 29, 30, 31):
            try: cand.append(dt.date(y, m, day))
            except ValueError: pass
for c in cand:
    if c < dt.date(2018, 12, 1) or c > dt.date(2026, 11, 20): continue
    if c.weekday() > 4: continue
    r = requests.get(f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{c:%Y%m%d}.csv", timeout=60)
    if r.status_code != 200: continue
    x = pd.read_csv(io.StringIO(r.text), sep='|') if '|' in r.text[:500] else pd.read_csv(io.StringIO(r.text))
    x.columns = [c_.strip() for c_ in x.columns]; x['settle'] = pd.Timestamp(c); out.append(x); got += 1
    if got == 1: print(x.columns.tolist(), len(x))
pd.concat(out).to_pickle(S + 'ext/finra_si.pkl'); print('files', got)
