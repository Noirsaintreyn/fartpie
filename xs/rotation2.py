"""Nov 4 2026: sector gates round 2. DECLARED BEFORE RESULTS. Gates (known at the prior close): G1 sector 126d momentum top 4 of 12 | G3 sector 63d return > SPY 63d | G5 sector ETF's own DAILY VECTOR bullish (vector_pine on the ETF's daily OHLC)
| G6 sector ETF EMA20 > EMA50 (daily) | G13 = G1 and G3 | G35 = G3 and G5.  Applied to (i) the live S&P book and (ii) the union universe live + Nasdaq-100(>=1.5) + Dow(>=1.0). Baselines = the same universe restricted to mapped sectors, ungated."""
_u = open('xs/universe_test.py').read(); exec(_u[:_u.index("B0 = ev(W[W.sp]")].replace(" & (W.ratio >= 1.5)].copy()", "].copy()").replace("print('unclassified", "pass #").replace("print('parity", "pass #"))
_r = open('xs/rotation_test.py').read(); exec(_r[_r.index("import glob, os"):_r.index("S = W[W.sp & W.sec.isin(MAP)].copy()")])
from vector_pine import run_state_machine
def etf_ohlc(t):
    f = f'/mnt/user-data/uploads/fartpie/etf_daily/{t}_yf.parquet'; d = pd.read_parquet(f); d.index = pd.to_datetime(d.index).tz_localize(None).normalize(); d = d[~d.index.duplicated()].sort_index()
    return d.rename(columns=str.lower)[['open', 'high', 'low', 'close']].dropna()
VEC = {}
for s, e in MAP.items():
    o = etf_ohlc(e if e != 'SMH' else 'SMH'); o['datetime'] = o.index; reg = np.asarray(run_state_machine(o)[0]); VEC[s] = pd.Series(reg == 1, index=o.index)
GATE = {'G1 mom top4': ON['M1a'], 'G3 beats SPY 63d': ON['M3'], 'G5 sector vector bull': VEC, 'G6 sector EMA20>50': {s: (P[e].ewm(span=20, adjust=False).mean() > P[e].ewm(span=50, adjust=False).mean()) for s, e in MAP.items()}}
GATE['G13 G1 & G3'] = {s: GATE['G1 mom top4'][s].reindex(P.index).fillna(False) & GATE['G3 beats SPY 63d'][s].reindex(P.index).fillna(False) for s in MAP}
GATE['G35 G3 & G5'] = {s: GATE['G3 beats SPY 63d'][s].reindex(P.index).fillna(False) & GATE['G5 sector vector bull'][s].reindex(P.index).ffill().fillna(False) for s in MAP}
W = W[W.ratio.notna()]; LIVE = W[W.sp & (W.ratio >= 1.5)]; N15 = W[W.ndx & (W.ratio >= 1.5)]; D1 = W[W.dow & (W.ratio >= 1.0)]
def cat(*fs): return pd.concat(fs).drop_duplicates(['tk', 'entry', 'h1']).sort_values('entry').reset_index(drop=True)
def flag(S, d):
    out = np.zeros(len(S), bool)
    for s, g in S.groupby('sec'):
        ser = d[s].reindex(P.index).ffill().shift(1).astype(float); ix = ser.index.searchsorted(g.entry.dt.normalize().values, side='right') - 1; v = ser.values[np.clip(ix, 0, len(ser) - 1)]; out[S.index.get_indexer(g.index)] = np.nan_to_num(v) > 0
    return out
for uname, Uf in (('LIVE S&P', LIVE), ('UNION live + Nasdaq-100 + Dow(>=1.0)', cat(LIVE, N15, D1))):
    S = Uf[Uf.sec.isin(MAP)].reset_index(drop=True); B0 = ev(S); print(f'\n=== {uname}: {len(S)} trades in mapped sectors; 40 matched runs, 2019-21 / 2022-26'); line('ungated baseline', B0)
    for gn, gd in GATE.items():
        S['f'] = flag(S, gd); A, Bk = S[S.f], S[~S.f]; pr = lambda g, lo, hi: f'{g[(g.entry >= lo) & (g.entry < hi)].ret.mean()*100:+.2f}%'
        print(f'  {gn}: allowed {len(A)} ({len(A)/len(S):.0%}); avg allowed {pr(A, "2019", "2022")} / {pr(A, "2022", "2027")}, blocked {pr(Bk, "2019", "2022")} / {pr(Bk, "2022", "2027")}; runners allowed {(A.ret >= .15).sum()} blocked {(Bk.ret >= .15).sum()}', flush=True)
        line(f'{gn}', ev(A.drop(columns='f').reset_index(drop=True)), B0)
