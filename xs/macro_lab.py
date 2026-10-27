"""Oct 26 2026: macro / positioning context for the vector book, by the protocol (trades the live book admits, by period -> book
rules -> compare with the luck band ~+51..+60% 2019-21, ~+24..+30% 2022-26).
 COT (CFTC Traders in Financial Futures, E-mini S&P 13874A and Nasdaq mini 209742; a Tuesday report is used from the next Monday):
   es_am / nq_am    asset managers' net long / open interest, as a 3-year percentile (COT index)
   es_lev / nq_lev  leveraged funds' (hedge funds / CTAs) net / OI, 3-year percentile
 Markets (prior close): vix_pct (VIX 1-year percentile), vix_ts (VIX / VIX3M, > 1 = stress), credit (HYG/IEF 50-day trend),
   usd (UUP 50-day), rates (10-year yield 50-day change), cu_au (copper / gold 50-day), oil (USO 50-day), breadth (RSP/SPY 50-day),
   spy200 (SPY over its 200-day)
Rules for each: no new entries in the worst third (picked on 2019-21), and 4 slots instead of 6 in the worst third (rest in cash)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
ss = src[src.index('def sim('):src.index("P = [(np.datetime64")].replace('def sim(', 'def sim_cap(').replace("zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.w.values)", "zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.w.values, Z.cap.values)").replace("for e, x, r_, ww in", "for e, x, r_, ww, cp in").replace("if len(op) >= n: continue", "if len(op) >= cp: continue")
exec(ss)
MD = '/tmp/claude-0/ml/macro'
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
T = taken(L); key = set(zip(T.tk, T.entry)); L['inbook'] = [k in key for k in zip(L.tk, L.entry)]; L['half'] = np.where(L.d.dt.year >= 2022, '22-26', '19-21')
C = pd.read_parquet(f'{MD}/tff_all.parquet'); C['code'] = C.CFTC_Contract_Market_Code.astype(str); C['rd'] = pd.to_datetime(C['Report_Date_as_YYYY-MM-DD'])
F = {}
for code, nm in (('13874A', 'es'), ('209742', 'nq')):
    x = C[C.code == code].groupby('rd').sum(numeric_only=True).sort_index(); oi = x.Open_Interest_All
    for grp, col in (('am', 'Asset_Mgr'), ('lev', 'Lev_Money')):
        net = (x[f'{col}_Positions_Long_All'] - x[f'{col}_Positions_Short_All']) / oi
        idx = net.rolling(156, min_periods=52).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
        s = pd.Series(idx.values, index=idx.index + pd.Timedelta(days=6)); F[f'{nm}_{grp}'] = s
P = pd.read_parquet(f'{MD}/px.parquet'); P.index = pd.to_datetime(P.index).tz_localize(None)
r50 = lambda s: s / s.shift(50) - 1
F['vix_pct'] = P['^VIX'].rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True); F['vix_ts'] = P['^VIX'] / P['^VIX3M']
F['credit'] = r50(P.HYG / P.IEF); F['usd'] = r50(P.UUP); F['rates'] = P['^TNX'] - P['^TNX'].shift(50); F['cu_au'] = r50(P['HG=F'] / P['GC=F'])
F['oil'] = r50(P.USO); F['breadth'] = r50(P.RSP / P.SPY); F['spy200'] = P.SPY / P.SPY.rolling(200).mean() - 1
for k, s in F.items():
    s = s.dropna(); s = s[~s.index.duplicated()].sort_index(); s.index = pd.DatetimeIndex(s.index).astype('datetime64[ns]')
    prior = (L.d - pd.Timedelta(days=1)).astype('datetime64[ns]').values
    pos_ = np.searchsorted(s.index.values, prior, side='right') - 1; L[k] = np.where(pos_ >= 0, s.values[np.clip(pos_, 0, None)], np.nan)
B = L[L.inbook]; rows = []
for k in F:
    q = pd.qcut(B[k].rank(method='first'), 3, labels=False); g = B.groupby([q, 'half']).ret.agg(['mean', 'std', 'size']).unstack()
    t = {h: (g[('mean', h)].iloc[-1] - g[('mean', h)].iloc[0]) / np.hypot(g[('std', h)].iloc[0] / np.sqrt(g[('size', h)].iloc[0]), g[('std', h)].iloc[-1] / np.sqrt(g[('size', h)].iloc[-1])) for h in ('19-21', '22-26')}
    rows.append((k, *(g[('mean', h)].iloc[i] * 100 for h in ('19-21', '22-26') for i in (0, 1, 2)), t['19-21'], t['22-26']))
R = pd.DataFrame(rows, columns=['feature', 'low 19-21', 'mid 19-21', 'high 19-21', 'low 22-26', 'mid 22-26', 'high 22-26', 't 19-21', 't 22-26'])
print('admitted trades: avg trade % by third of each macro input (low / mid / high), t-stat of high minus low'); print(R.round(2).to_string(index=False))
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z, cap=False):
    f = sim_cap if cap else sim; R_ = [f(Z, a, b) for a, b in PER]; flag = '  <-- outside the luck band in both' if R_[0][0] > .60 and R_[1][0] > .30 else ''
    print(f'  {nm:<52}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + flag, flush=True)
print(f'\n{"":<54}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
for k in F:
    tr = B[B.half == '19-21']; lo, hi = tr[k].quantile([1 / 3, 2 / 3]); a = R.set_index('feature').loc[k]
    worst_low = a['low 19-21'] < a['high 19-21']; bad = (L[k] < lo) if worst_low else (L[k] > hi); lab = f'{k} {"low" if worst_low else "high"} third'
    show(f'no new entries: {lab}', L[~bad]); z = L.copy(); z['cap'] = np.where(bad, 4, 6); show(f'4 slots: {lab}', z, cap=True)
