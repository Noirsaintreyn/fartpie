"""Nov 22 2026 report 378: regime atlas v2. Reuses the five regimes of report 32 (R1 2018-01..2020-02 pre-COVID, R2 2020-03..2021-12 COVID/stimulus/retail, R3 2022-01..2023-03 tightening/bear, R4 2023-04..2025-03 AI bull/0DTE, R5 2025-04..now tariff/policy).
Rows = the outlets that survived (large-cap 15m vector flips raw and beta-hedged, NQ legs V15a/RA/M1/S1/all main legs in pts x unit), columns = regimes, next to market conditions measured from SPY 15m (annualised realised vol, return, daily-return variance ratio VR5, share of down days) and the vector's own flip supply.  Descriptive: 5 regimes cannot support a significance test, so the atlas lists, it does not claim; the one stat given is the Spearman rank correlation of each outlet's regime mean with each market condition (n=5, so only |rho|=1 would mean anything)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
RG = [('R1 pre-COVID', '2018-01-01', '2020-03-01'), ('R2 COVID/stimulus', '2020-03-01', '2022-01-01'), ('R3 tightening/bear', '2022-01-01', '2023-04-01'), ('R4 AI bull/0DTE', '2023-04-01', '2025-04-01'), ('R5 tariff/policy', '2025-04-01', '2027-01-01')]
def reg(ts):
    ts = pd.Series(pd.to_datetime(ts).values); out = np.full(len(ts), '', dtype=object)
    for n, a, b in RG: out[((ts >= a) & (ts < b)).values] = n
    return out
names = [r[0] for r in RG]
# market conditions from SPY 15m
f = sorted(glob.glob(U + 'lse_ohlcv_cache/SPY_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
dc = H.groupby(H.index.normalize()).close.last(); dr = np.log(dc).diff().dropna(); dr = dr[dr.abs() < 0.2]; mk = {}  # drops 2 bad SPY cache days (May 2026 scale break, +-0.56 log)
for n, a, b in RG:
    x = dr[(dr.index >= a) & (dr.index < b)]; r5 = x.rolling(5).sum().dropna(); vr5 = r5.var() / (5 * x.var()) if len(x) > 20 else np.nan
    mk[n] = dict(days=len(x), ret_pct=x.sum() * 100, vol_ann=x.std() * np.sqrt(252) * 100, VR5=vr5, down_share=(x < 0).mean() * 100)
M = pd.DataFrame(mk).T
print('== MARKET (SPY) BY REGIME'); print(M.round(2).to_string())
# large-cap vector
A = pd.read_parquet(SP + 'A376.parquet'); b2 = pd.read_parquet(SP + 'b302.parquet'); b2['entry'] = pd.to_datetime(b2.entry); A['rg'] = reg(A.entry); b2['rg'] = reg(b2.entry); rows = {}
rows['vector flips n (b302)'] = b2.groupby('rg').size(); rows['flips per month'] = b2.groupby('rg').size() / pd.Series({n: ((pd.Timestamp(min(b, '2026-10-09')) - pd.Timestamp(a)).days / 30.4) for n, a, b in RG})
rows['vector raw mean %/trade'] = b2.groupby('rg').ret.mean() * 100; rows['vector win %'] = b2.groupby('rg').ret.apply(lambda s: (s > 0).mean() * 100)
rows['vector beta-hedged %/trade (to Aug26)'] = A.groupby('rg').res.mean() * 100
# NQ legs
L = pd.read_pickle(SP + 'legs280.pkl'); tot = []
for k, d in L.items():
    d = d.copy(); d['day'] = pd.to_datetime(d.day); d['rg'] = reg(d.day); d['p'] = d.pts * d.u * (d.sz if 'sz' in d else 1) if False else d.pts * d.u; tot.append(d.assign(leg=k))
    if k in ('V15a', 'RA', 'M1', 'S1'): rows['NQ %s pts/trade (x unit)' % k] = d.groupby('rg').p.mean()
T = pd.concat(tot); T['yrs'] = 1; rows['NQ all legs pts/trade'] = T.groupby('rg').p.mean(); yrs = pd.Series({n: (pd.Timestamp(min(b, '2026-02-01')) - pd.Timestamp(a)).days / 365.25 for n, a, b in RG}); rows['NQ all legs pts/year'] = T.groupby('rg').p.sum() / yrs
Z = pd.DataFrame(rows).T.reindex(columns=names); print('\n== OUTLETS BY REGIME'); print(Z.round(2).to_string())
print('\n== Spearman rho of each outlet row vs market condition across the 5 regimes (n=5)')
from scipy.stats import spearmanr
for r in Z.index:
    if r.startswith('vector flips n'): continue
    print('%-40s' % r, ' '.join('%s %+.1f' % (c, spearmanr(Z.loc[r].values.astype(float), M[c].reindex(names).values.astype(float))[0]) for c in ('ret_pct', 'vol_ann', 'VR5', 'down_share')))
Z.to_csv(SP + 'atlas378.csv'); M.to_csv(SP + 'atlas378_mkt.csv')
