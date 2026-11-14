"""FINRA short interest 2018-10..2024-12 (earlier fetch started at 2025). Appends to scratch ext/shortint.pkl."""
import pandas as pd, requests, io, datetime as dt
OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; d = dt.date(2018, 10, 1); fr = []
while d <= dt.date(2024, 12, 31):
    for day in (15, 28, 29, 30, 31):
        try: s = dt.date(d.year, d.month, day)
        except ValueError: continue
        for off in range(0, 4):
            sd = s - dt.timedelta(days=off)
            if sd.weekday() >= 5: continue
            r = requests.get(f'https://cdn.finra.org/equity/otcmarket/biweekly/shrt{sd:%Y%m%d}.csv', timeout=60, headers={'User-Agent': 'Mozilla/5.0 research'})
            if r.status_code == 200: fr.append(pd.read_csv(io.StringIO(r.text), sep='|')); break
    d = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
new = pd.concat(fr); old = pd.read_pickle(OUT + 'shortint.pkl'); new = new.reindex(columns=old.columns)
SI = pd.concat([new, old]).drop_duplicates(['symbolCode', 'settlementDate']); SI['settlementDate'] = pd.to_datetime(SI.settlementDate); SI.to_pickle(OUT + 'shortint.pkl'); print('rows', len(SI), 'dates', SI.settlementDate.nunique(), SI.settlementDate.min())
