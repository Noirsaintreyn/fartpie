"""FINRA daily short sale volume (Reg SHO consolidated) 2018-12-01..2026-11-13 for the wide daily universe (1,656 names) -> scratchpad ext/finra_sv_wide.pkl (Date, Symbol, ShortVolume, ShortExemptVolume, TotalVolume). Resumable: caches per-day parts."""
import io, os, pickle, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; U = set(pd.read_pickle(X + 'wide_daily_ohlcv.pkl').keys()); days = pd.bdate_range('2018-12-01', '2026-11-13'); os.makedirs(X + 'sv_parts', exist_ok=True)
def g(d):
    fn = X + f'sv_parts/{d:%Y%m%d}.pkl'
    if os.path.exists(fn): return pd.read_pickle(fn)
    u = f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt'
    for _ in range(3):
        try:
            r = requests.get(u, timeout=60)
            if r.status_code == 404: pd.to_pickle(None, fn); return None
            if r.status_code == 200:
                x = pd.read_csv(io.StringIO(r.text), sep='|'); x = x[x.Symbol.isin(U)][['Date', 'Symbol', 'ShortVolume', 'ShortExemptVolume', 'TotalVolume']]; pd.to_pickle(x, fn); return x
        except Exception: pass
    return 'fail'
with ThreadPoolExecutor(10) as ex: R = list(ex.map(g, days))
ok = [x for x in R if isinstance(x, pd.DataFrame)]; print('days ok', len(ok), 'holiday/404', sum(x is None for x in R), 'failed', sum(1 for x in R if isinstance(x, str)))
S = pd.concat(ok); S['Date'] = pd.to_datetime(S.Date.astype(str)); S.to_pickle(X + 'finra_sv_wide.pkl'); print(S.shape, S.Date.min(), S.Date.max(), S.Symbol.nunique())
