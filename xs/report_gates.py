"""GATES: time of day, weekday, session, macro/earnings calendar, oil/gold/VIX/VXN/DXY/TNX context vs the NQ system (variant Z days/trades from report_tiers). Nov 10 2026.
PASS (gate candidate): same sign and |t|>=2 in both 2018-21 and 2022-26."""
import numpy as np, pandas as pd, warnings, yfinance as yf; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
TD = pd.read_pickle(SC + 'tiers_days.pkl'); TR = pd.read_pickle(SC + 'tiers_trades.pkl'); TD.index = pd.to_datetime(TD.index).normalize()
ND = pd.read_pickle(SC + 'nq_days.pkl'); ND.index = pd.to_datetime(ND.index).normalize(); days = ND.index
FOMC = pd.to_datetime(sum([[f'{y}-{m}' for m in ms] for y, ms in {2018:['01-31','03-21','05-02','06-13','08-01','09-26','11-08','12-19'],2019:['01-30','03-20','05-01','06-19','07-31','09-18','10-30','12-11'],2020:['01-29','03-03','03-15','04-29','06-10','07-29','09-16','11-05','12-16'],2021:['01-27','03-17','04-28','06-16','07-28','09-22','11-03','12-15'],2022:['01-26','03-16','05-04','06-15','07-27','09-21','11-02','12-14'],2023:['02-01','03-22','05-03','06-14','07-26','09-20','11-01','12-13'],2024:['01-31','03-20','05-01','06-12','07-31','09-18','11-07','12-18'],2025:['01-29','03-19','05-07','06-18','07-30','09-17','10-29','12-10'],2026:['01-28','03-18','04-29','06-17','07-29','09-16','10-28','12-09']}.items()], []))
X5 = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X5['day'] = pd.to_datetime(X5.day).dt.normalize(); b = X5[X5.m == 510].set_index('day'); rg = b.high - b.low; ratio = rg / rg.rolling(60, min_periods=30).median().shift(1)
E1 = (ratio >= 2.5).reindex(days).fillna(False)
NFP = pd.DatetimeIndex([d for d in days if d.dayofweek == 4 and d.day <= 7])
cpi = []
sel = ratio[(ratio.index.day >= 9) & (ratio.index.day <= 16)]
for (y, m), g in sel.groupby([sel.index.year, sel.index.month]):
    if g.max() >= 2: cpi.append(g.idxmax())
CPI = pd.DatetimeIndex(cpi)
pos = pd.Series(np.arange(len(days)), index=days)
def shift_flag(dts, k):
    out = pd.Series(False, index=days)
    for d in dts:
        if d in pos.index:
            j = pos[d] + k
            if 0 <= j < len(days): out.iloc[j] = True
    return out
F = pd.DataFrame(index=days)
for i, n in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']): F['dow_' + n] = days.dayofweek == i
F['FOMC day'] = shift_flag(FOMC, 0); F['FOMC day-1'] = shift_flag(FOMC, -1); F['FOMC day+1'] = shift_flag(FOMC, 1); F['CPI day'] = shift_flag(CPI, 0); F['CPI day+1'] = shift_flag(CPI, 1); F['NFP day'] = shift_flag(NFP, 0)
F['any macro (FOMC|CPI|NFP)'] = F['FOMC day'] | F['CPI day'] | F['NFP day']; F['big 8:30 bar (E1)'] = E1
dm = pd.Series(days, index=days); F['3rd Friday (opex)'] = (days.dayofweek == 4) & (days.day >= 15) & (days.day <= 21); F['first 3 trading days of month'] = (days.to_series().groupby([days.year, days.month]).cumcount() < 3).values
F['last 3 trading days of month'] = (days.to_series().groupby([days.year, days.month]).cumcount(ascending=False) < 3).values
# earnings reaction days (mega caps)
cnt = pd.Series(0, index=days); nv = pd.Series(False, index=days)
for t in ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA']:
    try: ed = yf.Ticker(t).get_earnings_dates(limit=100)
    except Exception as ex: print('earnings fail', t, str(ex)[:60]); continue
    for ts in ed.index:
        d = pd.Timestamp(ts).tz_localize(None).normalize(); rd = d + pd.Timedelta(days=1) if pd.Timestamp(ts).hour >= 16 else d
        j = days.searchsorted(rd)
        if j < len(days) and days[j] - rd < pd.Timedelta(days=4) and days[j] >= pd.Timestamp('2017-06-01'):
            cnt.iloc[j] += 1
            if t == 'NVDA': nv.iloc[j] = True
F['mega-cap earnings reaction day (>=1)'] = cnt >= 1; F['>=2 mega-cap reactions same day'] = cnt >= 2; F['NVDA earnings reaction day'] = nv
# cross-asset
def dl(t): s = yf.download(t, start='2016-06-01', end='2026-02-05', progress=False, auto_adjust=True)['Close'].squeeze(); s.index = pd.DatetimeIndex(s.index).tz_localize(None); return s
CL, GC, VIX, VXN, V3M, DXY, TNX = [dl(t) for t in ['CL=F', 'GC=F', '^VIX', '^VXN', '^VIX3M', 'DX-Y.NYB', '^TNX']]
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True)
C = pd.DataFrame({'oil_ret5': CL.pct_change(5), 'gold_ret5': GC.pct_change(5), 'oil_ret1': CL.pct_change(1), 'gold_ret1': GC.pct_change(1), 'vix_pct': pct(VIX), 'vxn_pct': pct(VXN), 'vxn_over_vix_pct': pct(VXN / VIX), 'vix_term(VIX/VIX3M)': VIX / V3M, 'vix_chg1': VIX.pct_change(1), 'vxn_chg1': VXN.pct_change(1), 'dxy_ret5': DXY.pct_change(5), 'tnx_chg5': TNX.diff(5)}).shift(1).reindex(days)
def tt(a, b):
    a, b = a.dropna(), b.dropna()
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + 1e-12) if len(a) >= 8 and len(b) >= 8 else np.nan
D = TD.drop(columns=['vxn_pct']).join(F, how='left').join(C, how='left'); D['half'] = np.where(D.index.year < 2022, 'A', 'B'); D['T'] = D.tier.str[0].astype(int)
print(f'days {len(D)} | mean pts/day {D.pts.mean():+.2f}  (A {D[D.half=="A"].pts.mean():+.2f}, B {D[D.half=="B"].pts.mean():+.2f})\n')
print('=== CALENDAR / EVENT FLAGS (day-level): flagged vs not; t = flagged minus rest ===')
print(f'{"flag":38} | {"n":>4} | A: flagged  rest   t  | B: flagged  rest   t  | terr% / great% flagged (rest) | gate?')
for f in F.columns:
    m = D[f].fillna(False).astype(bool)
    if m.sum() < 25: print(f'{f:38} | {m.sum():4d} | (too few)'); continue
    r = []
    for h in 'AB':
        a, b2 = D[m & (D.half == h)].pts, D[~m & (D.half == h)].pts; r.append((a.mean(), b2.mean(), tt(a, b2)))
    flag = 'YES' if np.sign(r[0][2]) == np.sign(r[1][2]) and min(abs(r[0][2]), abs(r[1][2])) >= 2 else ''
    print(f'{f:38} | {m.sum():4d} | {r[0][0]:+7.1f} {r[0][1]:+6.1f} {r[0][2]:+5.1f} | {r[1][0]:+7.1f} {r[1][1]:+6.1f} {r[1][2]:+5.1f} | {np.mean(D[m].T==1):4.0%}/{np.mean(D[m].T==5):4.0%} ({np.mean(D[~m].T==1):4.0%}/{np.mean(D[~m].T==5):4.0%}) | {flag}')
print('\n=== CROSS-ASSET CONTEXT (known at prior close), thirds cut on 2018-21: mean pts/day low | mid | high and high-low t per half ===')
A_ = D[D.half == 'A']
for f in C.columns:
    c1, c2 = A_[f].quantile([1 / 3, 2 / 3]).values; row = []; ts = []
    for h in 'AB':
        d = D[D.half == h]; lo, hi = d[d[f] <= c1].pts, d[d[f] >= c2].pts; mi = d[(d[f] > c1) & (d[f] < c2)].pts; row.append(f'{h}: {lo.mean():+5.1f} {mi.mean():+5.1f} {hi.mean():+5.1f}'); ts.append(tt(hi, lo))
    flag = 'YES' if np.sign(ts[0]) == np.sign(ts[1]) and min(abs(ts[0]), abs(ts[1])) >= 2 else ''
    print(f'{f:22} cuts {c1:+.3f}/{c2:+.3f} | ' + ' | '.join(row) + f' | hi-lo t A {ts[0]:+.1f} B {ts[1]:+.1f} | {flag}')
# time of day / session (trade-level)
TR['hh'] = TR.m.where(~TR.leg.isin(['ON+', 'ON-']), -1)
def bucket(m, leg):
    if leg in ('ON+', 'ON-'): return 'overnight legs (ON+/ON-)'
    return '09:30-10:00' if m < 600 else '10:00-10:30' if m < 630 else '10:30-11:00' if m < 660 else '11:00-12:00' if m < 720 else '12:00-14:00' if m < 840 else '14:00+'
TR['tb'] = [bucket(m, l) for m, l in zip(TR.m, TR.leg)]; TR['T'] = np.nan
q = TR.pts.quantile(.05); TR['terr'] = TR.pts <= q
print('\n=== TIME OF DAY / SESSION (trade-level, unit-weighted NQ pts): mean, win%, terrible-trade%, n; by half ===')
for tb, g in sorted(TR.groupby('tb'), key=lambda x: x[0]):
    ga, gb = g[g.half == 'A'], g[g.half == 'B']; print(f'{tb:26} n {len(g):5d} | mean {g.pts.mean():+6.2f} win {np.mean(g.pts>0):4.0%} terr {g.terr.mean():4.0%} | A {ga.pts.mean():+6.2f} (t {ga.pts.mean()/(ga.pts.std()/np.sqrt(len(ga))):+.1f}) B {gb.pts.mean():+6.2f} (t {gb.pts.mean()/(gb.pts.std()/np.sqrt(len(gb))):+.1f})')
print('\n  by leg x time bucket (mean pts per trade, n):')
for lg, g in TR[~TR.leg.isin(['ON+', 'ON-'])].groupby('leg'):
    print(f'   {lg:6} ' + ' | '.join(f'{tb[:11]} {x.pts.mean():+6.2f} (n{len(x)})' for tb, x in sorted(g.groupby('tb')) if len(x) >= 15))
# FOMC-day timing: trades entered before vs after 14:00 (news ahead vs already known)
fo = TR.day.isin(FOMC); print('\n=== FOMC day, RTH legs by entry time (before 14:00 = position held through the statement) ===')
R_ = TR[fo & ~TR.leg.isin(['ON+', 'ON-'])]
for lab, g in (('entry < 14:00', R_[R_.m < 840]), ('entry >= 14:00', R_[R_.m >= 840]), ('all other days entry < 14:00', TR[~fo & ~TR.leg.isin(['ON+', 'ON-']) & (TR.m < 840)])):
    print(f'  {lab:30} n {len(g):4d} mean {g.pts.mean():+6.2f} win {np.mean(g.pts>0):4.0%} terr {g.terr.mean():4.0%} | A {g[g.half=="A"].pts.mean():+6.2f} B {g[g.half=="B"].pts.mean():+6.2f}')
D.to_pickle(SC + 'gates_days.pkl')
