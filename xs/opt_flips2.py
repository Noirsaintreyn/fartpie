"""Nov 9 2026: WHAT SHOULD ALSO WORK - options trades conditioned on what already has edge. DECLARED BEFORE RESULTS.
Trades: xs/opt_flips.py (real ThetaData EOD bid / ask). Context at the entry close, from xs/ext/gex/<TK>.csv (own net GEX + ATM IV, 26 names):
  VRP  = ATM IV / 20-day realized vol (both annualized) -> terciles, thresholds from 2019-21 baseline days.   IVR = ATM IV percentile in its own trailing year.
  GAM  = the stock's OWN net GEX at the prior close: positive vs negative.          TRACK = 1h-track flip vs 15m-track flip (flips only).
Hypotheses (each from something that already works):
  H1 variance premium (Goyal-Saretto): selling puts / put spreads pays more when IV is rich vs realized (VRP top third > bottom third), flips and random days alike.
  H2 dealer gamma (NQ: moves get absorbed in positive gamma): selling premium pays more in positive own gamma; calls on flips pay more in negative own gamma.
  H3 cheap optionality: calls / call spreads on flips pay more when VRP is low.          H4: 1h-track flips (longer holds) suit options better.
Pass = same sign in 2019-21 AND 2022-26 (and for H1 / H2 premium selling, in flips AND random days). Report points (% per trade) at the TOUCH and at the MID."""
import numpy as np, pandas as pd, glob, os
U = '/mnt/user-data/uploads/fartpie'
R = pd.read_parquet('/tmp/claude-0/opt_flips_R.parquet')
T = pd.read_parquet(f'{U}/xs/ext/wide_T.parquet'); T = T[T.sp & (T.ratio >= 1.5)].copy(); T['d0'] = T.entry.dt.normalize()
T = T.sort_values(['tk', 'entry']).drop_duplicates(['tk', 'd0']); H1 = T.set_index(['tk', 'd0']).h1
CTX = []
for f in glob.glob(f'{U}/xs/ext/gex/*.csv'):
    tk = os.path.basename(f)[:-4]; g = pd.read_csv(f, parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')
    r = np.log(g.spot).diff(); r[r.abs() > .3] = np.nan; rv = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    CTX.append(pd.DataFrame({'tk': tk, 'd0': g.index, 'vrp': (g.atm_iv / rv).values, 'ivr': g.atm_iv.rolling(252, min_periods=120).rank(pct=True).values,
                             'gam': np.sign(g.net_gex.shift(1)).values}))
CTX = pd.concat(CTX); R = R.merge(CTX, on=['tk', 'd0'], how='inner')
R['h1'] = [H1.get((t, d), np.nan) if s == 'flip' else np.nan for t, d, s in zip(R.tk, R.d0, R.set)]
q = R[(R.set == 'base') & (R.per == '19-21')].drop_duplicates(['tk', 'd0']); v1, v2 = q.vrp.quantile([1 / 3, 2 / 3]); i1, i2 = 1 / 3, 2 / 3
R['VRP'] = np.where(R.vrp <= v1, 'low', np.where(R.vrp >= v2, 'high', 'mid')); R['IVR'] = np.where(R.ivr <= i1, 'low', np.where(R.ivr >= i2, 'high', 'mid'))
R['GAM'] = np.where(R.gam > 0, 'pos', np.where(R.gam < 0, 'neg', 'na')); R['TRACK'] = np.where(R.h1 == True, '1h', np.where(R.h1 == False, '15m', 'na'))
print(f'trades with context: {len(R)} rows, {R.tk.nunique()} names; VRP tercile cuts {v1:.2f} / {v2:.2f} (IV / realized)', flush=True)
KEY = [('W', 'PCS 0', 'EXP'), ('W', 'PCS 0.5', 'EXP'), ('M', 'PCS 0', 'EXP'), ('M', 'PCS 0.5', 'EXP'), ('M', 'CSP 0.5', 'EXP'),
       ('W', 'CALL', 'VEC'), ('W', 'CDS', 'VEC'), ('M', 'CALL', 'VEC'), ('W', 'CALL', 'EXP'), ('W', 'CDS', 'EXP')]
def cell(t): return f'{t.ret.mean() * 100:+6.2f}% ({len(t):4d})' if len(t) >= 30 else '     .       '
for var, levels in (('VRP', ('low', 'high')), ('IVR', ('low', 'high')), ('GAM', ('neg', 'pos')), ('TRACK', ('15m', '1h'))):
    print(f'\n=== {var}: {levels[0]} vs {levels[1]} | per trade, n | 2019-21 then 2022-26 | "*" = {levels[1]} - {levels[0]} same sign in both periods (and both sets for selling)', flush=True)
    for fill in ('TOUCH', 'MID'):
        for ten, st, ex in KEY:
            g = R[(R.ten == ten) & (R.st == st) & (R.ex == ex) & (R.fill == fill)]; line = f'  {fill:<5} {ten} {st:<7} {ex} '; diffs = []
            for s in (('flip',) if var == 'TRACK' else ('flip', 'base')):
                for per in ('19-21', '22-26'):
                    a = g[(g.set == s) & (g.per == per) & (g[var] == levels[0])]; b = g[(g.set == s) & (g.per == per) & (g[var] == levels[1])]
                    line += f'| {s} {per}: {cell(a)} vs {cell(b)} '; diffs.append(b.ret.mean() - a.ret.mean() if len(a) >= 30 and len(b) >= 30 else np.nan)
            d = np.array(diffs, dtype=float); sell = st.startswith(('PCS', 'CSP'))
            need = d if (sell and var != 'TRACK') else d[:2]
            flag = '*' if np.isfinite(need).all() and (np.all(need > 0) or np.all(need < 0)) else ' '
            print(f'{flag}{line}', flush=True)
