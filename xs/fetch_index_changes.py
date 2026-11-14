"""S&P 400/600 constituent change tables from Wikipedia (effective date, added ticker) -> scratch ext/index_changes.csv."""
import pandas as pd, requests, io
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; out = []
for n in ('600', '400'):
    h = requests.get(f'https://en.wikipedia.org/wiki/List_of_S%26P_{n}_companies', headers={'User-Agent': 'Mozilla/5.0 research'}).text
    for t in pd.read_html(io.StringIO(h)):
        if t.columns.nlevels == 2 and ('Added', 'Ticker') in t.columns:
            d = pd.DataFrame({'date': pd.to_datetime(t[('Date', 'Date')], errors='coerce'), 'added': t[('Added', 'Ticker')], 'removed': t[('Removed', 'Ticker')]}); d['idx'] = n; out.append(d)
D = pd.concat(out).dropna(subset=['date']); D.to_csv(SC + 'index_changes.csv', index=False); print(len(D), D.date.min(), D.date.max(), D.added.notna().sum())
