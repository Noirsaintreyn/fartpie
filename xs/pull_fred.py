"""FRED market series via fredgraph.csv (no key). Only market-based series that are not revised: DGS10, DGS2, T10Y2Y, T10YIE, DFII10, BAMLH0A0HYM2 (HY OAS), BAMLC0A0CM (IG OAS), DTWEXBGS (broad dollar), DCOILWTICO, VIXCLS, VXNCLS, SP500."""
import io, requests, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; out = {}
for s in ['DGS10', 'DGS2', 'T10Y2Y', 'T10YIE', 'DFII10', 'BAMLH0A0HYM2', 'BAMLC0A0CM', 'DTWEXBGS', 'DCOILWTICO', 'VIXCLS', 'VXNCLS']:
    try:
        r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}', timeout=60); d = pd.read_csv(io.StringIO(r.text)); d.columns = ['date', s]; d['date'] = pd.to_datetime(d.date); out[s] = d.set_index('date')[s].apply(pd.to_numeric, errors='coerce')
    except Exception as e: print('ERR', s, str(e)[:80])
F = pd.DataFrame(out); F = F[F.index >= '2017-01-01']; F.to_parquet(SP + 'fred.parquet'); print(F.shape, F.index.max().date()); print(F.tail(2).T)
