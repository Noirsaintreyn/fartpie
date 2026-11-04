"""Nov 4 2026: sector-ROTATION gate on the live S&P book. DECLARED BEFORE RESULTS. A trade is allowed only if its sector is 'on' per the model at the prior close; allowed trades go through the same 6-slot book (ratio tie-break, 8% breaker).
Sector -> ETF: Tech/Software XLK, Semis SMH, Comm/Media XLC, Cons Disc XLY, Staples XLP, Financials XLF, Health XLV, Industrials XLI, Energy/Mat XLE. Pool for ranking: those + XLB, XLU, XLRE.
 M1a / M1b  sector 126-day momentum rank in the 12-ETF pool: on if top 4 / top 6     M2  sector ETF close > its 100-day SMA     M3  sector 63-day return > SPY 63-day return
 M4  macro cycle: copper/gold 50-day change > 0 -> cyclicals on (Cons Disc, Industrials, Financials, Semis, Tech, Energy/Mat); else defensives on (Staples, Health, Comm/Media, Tech)
Reports per-trade allowed vs blocked (does the gate block the bad ones?) and 40 matched runs vs live, both periods."""
_s = open('xs/universe_test.py').read(); exec(_s[:_s.index("B0 = ev(W[W.sp]")].replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
import glob, os
MAP = {'Tech/Software': 'XLK', 'Semis': 'SMH', 'Comm/Media': 'XLC', 'Cons Disc': 'XLY', 'Staples': 'XLP', 'Financials': 'XLF', 'Health': 'XLV', 'Industrials': 'XLI', 'Energy/Mat': 'XLE'}
POOL = ['XLK', 'SMH', 'XLC', 'XLY', 'XLP', 'XLF', 'XLV', 'XLI', 'XLE', 'XLB', 'XLU', 'XLRE']
def etf(t):
    d = pd.read_parquet(f'/mnt/user-data/uploads/fartpie/etf_daily/{t}_yf.parquet') if os.path.exists(f'/mnt/user-data/uploads/fartpie/etf_daily/{t}_yf.parquet') else pd.read_parquet(f'/mnt/user-data/uploads/fartpie/etf_daily/{t}.parquet')
    if 'datetime' in d.columns: d = d.set_index('datetime')
    s = d['Adj Close'] if 'Adj Close' in d.columns else d['adjclose']; s.index = pd.to_datetime(s.index).tz_localize(None).normalize().astype('datetime64[ns]'); return s[~s.index.duplicated()]
P = pd.DataFrame({t: etf(t) for t in POOL + ['SPY']}).sort_index().ffill()
mom = P[POOL].pct_change(126); rk = mom.rank(axis=1, ascending=False); r63 = P.pct_change(63)
ON = {'M1a': {s: (rk[e] <= 4) for s, e in MAP.items()}, 'M1b': {s: (rk[e] <= 6) for s, e in MAP.items()}, 'M2': {s: (P[e] > P[e].rolling(100).mean()) for s, e in MAP.items()}, 'M3': {s: (r63[e] > r63['SPY']) for s, e in MAP.items()}}
M = pd.read_parquet('/tmp/claude-0/ml/macro/px.parquet').ffill(); M.index = pd.to_datetime(M.index).astype('datetime64[ns]'); cg = (M['HG=F'] / M['GC=F']); up = (cg.pct_change(50) > 0)
CYC = {'Cons Disc', 'Industrials', 'Financials', 'Semis', 'Tech/Software', 'Energy/Mat'}; DEF = {'Staples', 'Health', 'Comm/Media', 'Tech/Software'}
ON['M4'] = {s: (up if s in CYC else pd.Series(False, index=up.index)) | (~up if s in DEF else pd.Series(False, index=up.index)) for s in MAP}
S = W[W.sp & W.sec.isin(MAP)].copy(); S['d'] = S.entry.dt.normalize(); print('S&P live trades', len(S), 'with mapped sector', S.sec.isin(MAP).sum(), 'of', int(W.sp.sum()))
for mname, d in ON.items():
    flag = np.zeros(len(S), bool)
    for s, g in S.groupby('sec'):
        ser = d[s].shift(1).astype(float); ix = ser.index.searchsorted(g.d.values, side='right') - 1; v = ser.values[np.clip(ix, 0, len(ser) - 1)]; flag[S.index.get_indexer(g.index)] = np.nan_to_num(v) > 0
    S[mname] = flag
S = S.reset_index(drop=True); B0 = ev(S.assign(h1=S.h1)); print('\n40 matched runs, 2019-21 / 2022-26'); line('L live (S&P, mapped sectors)', B0)
for mname in ON:
    A, Bk = S[S[mname]], S[~S[mname]]
    pr = lambda g, lo, hi: f'{g[(g.entry >= lo) & (g.entry < hi)].ret.mean()*100:+.2f}%'
    print(f'  {mname}: allowed {len(A)} ({len(A)/len(S):.0%}), avg trade allowed {pr(A, "2019", "2022")} / {pr(A, "2022", "2027")}, blocked {pr(Bk, "2019", "2022")} / {pr(Bk, "2022", "2027")}; runners allowed {(A.ret >= .15).sum()} blocked {(Bk.ret >= .15).sum()}', flush=True)
    line(f'{mname} gate', ev(A.reset_index(drop=True)), B0)
