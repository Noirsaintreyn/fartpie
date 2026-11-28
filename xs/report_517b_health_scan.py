"""517b: live scan, HEALTH sub-$6 names with HIGH R&D intensity (RND top third by 2019-21 HEALTH quantile, latest 10-K) -> current daily vector state, last flip, ATRP. Bars rebuilt from LSE 15m (RTH). Key read at runtime by lse_data (LSE_ENV_FILE)."""
import sys, os, io, contextlib, numpy as np, pandas as pd, concurrent.futures as cf, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); import lse_data; from vector_pine import run_state_machine
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); qh = P[(P.sector == 'HEALTH') & TRm].RND.quantile([.33, .67]); print('HEALTH RND/assets cuts (33/67%):', qh.round(3).tolist())
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); hl = set(S[S.sector == 'HEALTH'].tk); last = {t: D[t].close.dropna().iloc[-1] for t in hl if t in D and len(D[t].dropna(subset=['close']))}
FDl = FD.sort_values('filed').groupby('tk').tail(1).set_index('tk'); FDl = FDl[FDl.filed >= pd.Timestamp('2025-06-01')]
cand = [t for t, px in last.items() if px < 6 and t in FDl.index and FDl.loc[t, 'RND'] > qh.iloc[1]]; lo = [t for t, px in last.items() if px < 6 and t in FDl.index and FDl.loc[t, 'RND'] <= qh.iloc[0]]
print('health names px<6 (as of 11/13) with fresh 10-K:', sum(1 for t, px in last.items() if px < 6 and t in FDl.index), '| RND_hi', len(cand), '| RND_lo', len(lo))
assert lse_data.available()
def one(t):
    try: return t, lse_data.daily_rth(t, years=1.0)
    except Exception: return t, None
with cf.ThreadPoolExecutor(3) as ex: B = dict(ex.map(one, cand))
rows = []
for t in cand:
    q = B.get(t)
    if q is None or len(q) < 150: continue
    reg, lev, atr, _ = run_state_machine(q.assign(datetime=q.index)); fl = np.where((reg[1:] == 1) & (reg[:-1] != 1))[0] + 1
    rows.append(dict(tk=t, px=round(q.close.iat[-1], 2), reg=int(reg[-1]), days_since_flip=(len(q) - 1 - fl[-1]) if len(fl) else None, last_flip=q.index[fl[-1]].date() if len(fl) else None, ATRP=round(atr[-1] / q.close.iat[-1] * 100, 1), RND_assets=round(float(FDl.loc[t, 'RND']), 2), asof=q.index[-1].date()))
R = pd.DataFrame(rows).sort_values(['reg', 'days_since_flip'], ascending=[False, True]); R.to_csv('xs/report_517b_health_scan.csv', index=False)
print('asof', R['asof'].max(), 'scanned', len(R), 'of', len(cand)); print('\nIN REGIME (long) now:'); print(R[R.reg == 1].to_string(index=False)); print('\nOUT of regime, highest ATRP (watch for next flip):'); print(R[R.reg != 1].sort_values('ATRP', ascending=False).head(12).to_string(index=False))
