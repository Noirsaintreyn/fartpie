"""Nov 14 2026 EMA TREND CONTEXT ON THE FLIPS, LENGTHS x TIMEFRAMES (declared before results; read 'try ema different lengths and time frames' as: is the flip's own trend, measured by an EMA, a filter?). Mechanism: a flip below a long EMA is a bounce inside a downtrend (falling knife, sellers return), a flip above it is a continuation; the tail may come from either, which is the open question. Grid fixed now (10 EMAs): DAILY 10/20/50/100/200; WEEKLY 10/20/40 (weekly closes, last COMPLETED week before the flip bar, no look-ahead); MONTHLY 6/10 (last completed month). For each: ABOVE = flip-bar close above the EMA, and SLOPE = EMA rising vs 3 bars of that timeframe earlier. Also the market: equal-weight universe index above its daily EMA 20/50/100/200. Per era (2019-21 / 2022-26Q1) and on the screener rule (BV>=1.5): n, mean net, median, HUGE per 100, lift vs era base, ex-top-3 mean. PASS = the same state (above or below) has HUGE lift >=1.3 and mean above base in BOTH eras with n>=100. 20+ comparisons, so only a pass in both eras counts."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['d0'] = pd.to_datetime(P.d0); D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl')
CFG = [('D10', 'D', 10), ('D20', 'D', 20), ('D50', 'D', 50), ('D100', 'D', 100), ('D200', 'D', 200), ('W10', 'W', 10), ('W20', 'W', 20), ('W40', 'W', 40), ('M6', 'M', 6), ('M10', 'M', 10)]
try: pd.Series([1], index=pd.date_range('2020', periods=1)).resample('ME'); ME = 'ME'
except Exception: ME = 'M'
cols = {f'{n}_ABOVE': np.full(len(P), np.nan) for n, _, _ in CFG}; cols.update({f'{n}_UP': np.full(len(P), np.nan) for n, _, _ in CFG})
def asof(series, dates): 
    s = series.dropna(); return s.reindex(s.index.union(dates)).ffill().reindex(dates).values
for tk, g in P.groupby('tk'):
    B = D[tk]; B = B[~B.index.duplicated()]; c = B.close; dts = pd.DatetimeIndex(g.d0); cd = c.reindex(dts).values
    for name, tf, n in CFG:
        if tf == 'D': e = c.ewm(span=n, adjust=False, min_periods=n).mean(); ev = e.reindex(dts).values; up = (e > e.shift(3)).reindex(dts).values
        else:
            cc = c.resample('W-FRI' if tf == 'W' else ME).last().dropna(); e = cc.ewm(span=n, adjust=False, min_periods=n).mean(); ev = asof(e, dts); up = asof((e > e.shift(3)).astype(float).where(e.notna() & e.shift(3).notna()), dts)
        ok = ~np.isnan(ev); a = np.where(ok, (cd > ev).astype(float), np.nan); cols[f'{name}_ABOVE'][g.index] = a; cols[f'{name}_UP'][g.index] = np.where(ok, up.astype(float), np.nan)
for k, v in cols.items(): P[k] = v
un = pd.DataFrame({tk: B.close[~B.index.duplicated()] for tk, B in D.items()}).sort_index().pct_change(fill_method=None).clip(-.5, 1).mean(axis=1); ix = (1 + un.fillna(0)).cumprod()
for n in (20, 50, 100, 200): P[f'MKT{n}_ABOVE'] = (ix > ix.ewm(span=n, adjust=False, min_periods=n).mean()).reindex(P.d0).astype(float).values
E1 = P.t < '2022'; names = [f'{n}_ABOVE' for n, _, _ in CFG] + [f'{n}_UP' for n, _, _ in CFG] + [f'MKT{n}_ABOVE' for n in (20, 50, 100, 200)]
def row(z, f, st, base):
    s = z[z[f] == st]; 
    if len(s) < 100: return None
    return dict(n=len(s), mean=s.rn.mean() * 100, med=s.rn.median() * 100, hp=s.HUGE.mean() * 100, lift=s.HUGE.mean() / base, ex3=np.sort(s.rn.values)[:-3].mean() * 100)
for lab, S0 in (('ALL FLIPS', P), ('SCREENER RULE (BV>=1.5)', P[P.BV >= 1.5])):
    print(f'\n######## {lab}'); res = {}
    for en, z in (('2019-21', S0[S0.t < '2022']), ('2022-26Q1', S0[S0.t >= '2022'])):
        base = z.HUGE.mean(); print(f'\n--- {en}: n {len(z)} base mean {z.rn.mean()*100:+.2f}% median {z.rn.median()*100:+.2f} HUGE/100 {base*100:.2f}')
        print(f'  {"filter":12s} | state=1 (above / rising): n   mean%  med%  HUGE/100 lift ex-top3 | state=0 (below / falling): n  mean%  med%  HUGE/100 lift ex-top3')
        for f in names:
            r1 = row(z, f, 1.0, base); r0 = row(z, f, 0.0, base); res[(en, f)] = (r1, r0, z.rn.mean() * 100)
            fm = lambda r: f'{r["n"]:5d} {r["mean"]:+6.2f} {r["med"]:+5.1f} {r["hp"]:5.2f} {r["lift"]:4.2f} {r["ex3"]:+6.2f}' if r else '   n<100                     '
            print(f'  {f:12s} | {fm(r1)} | {fm(r0)}')
    print(f'\n  VERDICT ({lab}): PASS = same state has lift>=1.3 and mean above base in BOTH eras, n>=100')
    for f in names:
        for st, k in (('above/rising', 0), ('below/falling', 1)):
            a = res[('2019-21', f)][k]; b = res[('2022-26Q1', f)][k]
            if a and b and a['lift'] >= 1.3 and b['lift'] >= 1.3 and a['mean'] > res[('2019-21', f)][2] and b['mean'] > res[('2022-26Q1', f)][2]: print(f'    PASS {f} {st}: lift {a["lift"]:.2f}/{b["lift"]:.2f} mean {a["mean"]:+.2f}/{b["mean"]:+.2f}% (n {a["n"]}/{b["n"]})')
