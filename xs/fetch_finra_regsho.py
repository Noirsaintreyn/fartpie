"""FINRA daily short sale volume (Reg SHO, consolidated) 2023-11-01..2026-11-11, filtered to the 753-name 1h universe -> scratchpad finra_sv.pkl (Date, Symbol, ShortVolume, TotalVolume)."""
import io, pickle, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = set(pickle.load(open(SC + 'uni_cache.pkl', 'rb')).keys())
days = pd.bdate_range('2023-11-01', '2026-11-11')
def g(d):
    u = f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt'
    for _ in range(3):
        try:
            r = requests.get(u, timeout=40)
            if r.status_code == 404: return None
            if r.status_code == 200:
                x = pd.read_csv(io.StringIO(r.text), sep='|'); x = x[x.Symbol.isin(U)][['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume']]; return x
        except Exception: pass
    return 'fail'
with ThreadPoolExecutor(10) as ex: R = list(ex.map(g, days))
ok = [x for x in R if isinstance(x, pd.DataFrame)]; fail = sum(1 for x in R if isinstance(x, str)); print('days ok', len(ok), 'holiday/404', sum(x is None for x in R), 'failed', fail)
S = pd.concat(ok); S['Date'] = pd.to_datetime(S.Date.astype(str)); S.to_pickle(SC + 'finra_sv.pkl'); print(S.shape, S.Date.min(), S.Date.max(), S.Symbol.nunique())
