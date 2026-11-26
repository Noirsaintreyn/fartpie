"""Report 504 (Nov 26 2026): Rey - 'a lot of the big stocks move together; it is not so much edge on the trades as the time it is trading. Look back at the news/factors that led to bad times and see if that gives a lead.'
DECLARED BEFORE RESULTS. Population: large-cap flips (xs/ext/wide_T.parquet, S&P member, options ratio >= 1.5), 15m and 1h tracks pooled for timing questions, outcome = trade return (bp, before costs). Train 2019-21 / test 2018 + 2022+ (era table too). SE clustered by month.
PART 1 how much is time: share of flip-return variance explained by the entry DAY alone (day fixed effects, adjusted), and whether each signal model's lift over other flips survives removing the day (tagged minus the same day's average).
PART 2 leads for bad times (all prior-close known, tercile cuts frozen on train, top-minus-bottom tercile mean return; PASS = same sign train and test and pooled |t|>=2):
 S1 VIX level rank (own trailing 252d)   - (stress: breakouts whipsaw, gaps against longs)
 S2 VIX 5-day change                      - (vol shock: dealers/forced sellers; flips fire into it and reverse)
 S3 HY credit spread 20d change           - (credit stress leads equity risk-off; funding)
 S4 index drawdown from its high          + two-sided read, declared + (deeper drawdown = more bounce) 
 S5 index below its 200d average          - (downtrend: breakouts fail)
 S6 market breadth (share of names up-regime) +  (broad risk-on carries breakouts)
 S7 index realised vol 20d                -  
 S8 yield-curve (10y-2y) and NFCI financial conditions: exploratory, two-sided
PART 3 gate: drop the worst tercile of any PASSING lead; result by era (mean bp per trade, trades kept, win rate), train vs test. PART 4: the worst months table with their readings. News: no historical headline archive on hand, so 'news' enters only as these measurable shocks (rates, credit, vol), not text."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
UP = '/mnt/user-data/uploads/fartpie/'; SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
W = pd.read_parquet(UP + 'xs/ext/wide_T.parquet'); W = W[W.sp & (W.ratio >= 1.5)].copy(); W['entry'] = pd.to_datetime(W.entry); W['day'] = W.entry.dt.normalize(); W['mon'] = W.entry.dt.strftime('%Y-%m'); W['bp'] = W.ret * 1e4
y = W.entry.dt.year; W['tr'] = y.isin([2019, 2020, 2021]); W['era'] = np.select([y <= 2019, y <= 2021, y == 2022], ['18-19', '20-21', '22'], '23-26'); W = W[W.bp.abs() < 5000].copy()
MS = pd.read_pickle(SP + 'ext/market_state.pkl').sort_index(); MS.index = pd.DatetimeIndex(MS.index)
FR = pd.read_pickle(SP + 'ext/fred.pkl'); FR['date'] = pd.to_datetime(FR.date); FR = FR.drop_duplicates('date').set_index('date').sort_index().ffill()
F = pd.DataFrame(index=MS.index); F['S1'] = FR.VIX.reindex(F.index, method='ffill').rolling(252, min_periods=60).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
v = FR.VIX.reindex(F.index, method='ffill'); F['S2'] = v - v.shift(5); h_ = FR.HY_OAS.reindex(F.index, method='ffill'); F['S3'] = h_ - h_.shift(20); F['S4'] = MS.idx_dd; F['S5'] = (MS.ma200 < 0).astype(float).where(MS.ma200.notna()); F['S6'] = MS.breadth
F['S7'] = np.log(MS.idx).diff().rolling(20).std() * np.sqrt(252); F['S8a'] = FR.T10Y2Y.reindex(F.index, method='ffill'); F['S8b'] = FR.NFCI.reindex(F.index, method='ffill')
F = F.shift(1); F.index.name = 'day'     # prior close known
F.index = F.index.astype('datetime64[ns]'); W['day'] = W.day.astype('datetime64[ns]'); W = pd.merge_asof(W.sort_values('day'), F.reset_index().sort_values('day'), on='day', direction='backward')
print(f'population: {len(W)} flips, {W.tk.nunique()} names, {W.day.min():%Y-%m-%d} to {W.day.max():%Y-%m-%d}; train {int(W.tr.sum())} test {int((~W.tr).sum())}; mean {W.bp.mean():+.1f}bp')
# PART 1
d = W.groupby('day').bp.transform('mean'); n_d = W.groupby('day').bp.transform('size'); sst = ((W.bp - W.bp.mean()) ** 2).sum(); sse = ((W.bp - d) ** 2).sum(); k = W.day.nunique(); r2 = 1 - (sse / (len(W) - k)) / (sst / (len(W) - 1))
mo = W.groupby('mon').bp.transform('mean'); sse_m = ((W.bp - mo) ** 2).sum(); km = W.mon.nunique(); r2m = 1 - (sse_m / (len(W) - km)) / (sst / (len(W) - 1))
print(f'\nPART 1  variance of flip returns explained by the entry day (adj R2): {r2:.1%}; by the entry month: {r2m:.1%}  (flips per day median {W.groupby("day").size().median():.0f})')
MF = pd.read_csv('live_data/model_flips.csv', parse_dates=['entry']); MF = MF[MF.tf == '15m']; D = W[~W.h1].copy(); D['dm'] = D.bp - D.groupby('day').bp.transform('mean')
print(f"{'tag':5s} {'n':>6s} {'raw lift bp':>12s} {'t':>6s} | {'same-day lift bp':>16s} {'t':>6s}")
for tg in ('NEG', 'PCV', 'PCZ', 'SKW', 'CHM'):
    key = set(zip(MF[MF.tag == tg].tk, MF[MF.tag == tg].entry)); D['f'] = [(a, b) in key for a, b in zip(D.tk, D.entry)]
    if D.f.sum() < 30: continue
    def tt(col):
        g = D.groupby(['mon', 'f'])[col].mean().unstack().dropna(); x = (g[True] - g[False]); return x.mean(), x.mean() / (x.std() / np.sqrt(len(x)))
    a, b = tt('bp'), tt('dm'); print(f'{tg:5s} {int(D.f.sum()):6d} {a[0]:+12.1f} {a[1]:+6.1f} | {b[0]:+16.1f} {b[1]:+6.1f}')
# PART 2
print('\nPART 2  leads (top tercile minus bottom tercile of the feature, mean bp per trade; month-clustered t)')
rows = []
def eff(Dd, f):
    if len(Dd) < 100 or (Dd[f + '_t'] == 2).sum() < 30 or (Dd[f + '_t'] == 0).sum() < 30: return (np.nan, np.nan)
    m = smf.ols('bp ~ top + bot', Dd.assign(top=(Dd[f + '_t'] == 2).astype(float), bot=(Dd[f + '_t'] == 0).astype(float))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Dd.mon)[0]})
    c = m.params['top'] - m.params['bot']; se = np.sqrt(m.cov_params().loc['top', 'top'] + m.cov_params().loc['bot', 'bot'] - 2 * m.cov_params().loc['top', 'bot']); return (c, c / se)
dirs = dict(S1='-', S2='-', S3='-', S4='+', S5='-', S6='+', S7='-', S8a='?', S8b='?'); cuts = {}
for f in dirs:
    x = W[f]; q = np.nanquantile(x[W.tr], [1 / 3, 2 / 3]) if f != 'S5' else (.5, .5); cuts[f] = q
    W[f + '_t'] = np.where(x.isna(), np.nan, np.where(x <= q[0], 0, np.where(x >= q[1], 2, 1))) if f != 'S5' else np.where(x.isna(), np.nan, np.where(x > .5, 2, 0))
    Wk = W.dropna(subset=[f + '_t']); a = eff(Wk[Wk.tr], f); b = eff(Wk[~Wk.tr], f); c = eff(Wk, f)
    top = Wk[Wk[f + '_t'] == 2].bp.mean(); bot = Wk[Wk[f + '_t'] == 0].bp.mean(); ok = (np.sign(a[0]) == np.sign(b[0])) and abs(c[1]) >= 2     # S3 (HY spread) only exists from 2023 in the FRED pull: it cannot pass a train/test split
    rows.append((f, dirs[f], ok, c)); print(f"{f:4s} declared {dirs[f]}  top {top:+6.1f}bp  bottom {bot:+6.1f}bp | effect train {a[0]:+6.1f} (t {a[1]:+.1f})  test {b[0]:+6.1f} (t {b[1]:+.1f})  all {c[0]:+6.1f} (t {c[1]:+.1f}) | {'PASS' if ok else '-'}")
leads = [f for f, dr, ok, c in rows if ok]; print('PASSING:', leads)
# PART 3
print('\nPART 3  gate: drop the WORST tercile of each passing lead (worst = the side with lower mean return)')
for f in leads:
    top = W[W[f + '_t'] == 2].bp.mean(); bot = W[W[f + '_t'] == 0].bp.mean(); worst = 2 if top < bot else 0; keep = W[W[f + '_t'] != worst]; drop = W[W[f + '_t'] == worst]
    print(f"\n{f}: drop tercile {worst} ({len(drop)} trades, {len(drop) / len(W):.0%}); mean bp all {W.bp.mean():+.1f} -> kept {keep.bp.mean():+.1f}, dropped {drop.bp.mean():+.1f}; win {W.bp.gt(0).mean():.0%} -> {keep.bp.gt(0).mean():.0%}")
    for e in ('18-19', '20-21', '22', '23-26'):
        a = W[W.era == e]; kk = keep[keep.era == e]; dd = drop[drop.era == e]; print(f"   {e:6s} all {a.bp.mean():+6.1f} (n {len(a)})  kept {kk.bp.mean():+6.1f} (n {len(kk)})  dropped {dd.bp.mean():+6.1f} (n {len(dd)})")
    print(f"   month-clustered t of (kept - dropped): {smf.ols('bp ~ k', W.assign(k=(W[f + '_t'] != worst).astype(float))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(W.mon)[0]}).tvalues['k']:+.1f}")
# PART 4
M = W.groupby('mon').agg(n=('bp', 'size'), bp=('bp', 'mean'), win=('bp', lambda s: (s > 0).mean()), VIX=('S1', 'mean'), VIX5=('S2', 'mean'), HY20=('S3', 'mean'), DD=('S4', 'mean'), B200=('S5', 'mean'), BRD=('S6', 'mean')); M = M[M.n >= 15].sort_values('bp')
print('\nPART 4  worst 12 months (n>=15 flips)'); print(M.head(12).round(3).to_string()); print('\nbest 6 months'); print(M.tail(6).round(3).to_string())
print(f'\nshare of months with mean < 0: {(M.bp < 0).mean():.0%}; bottom-decile months hold {M.head(int(len(M) * .1)).n.sum() / M.n.sum():.0%} of trades and a mean of {np.average(M.head(int(len(M) * .1)).bp, weights=M.head(int(len(M) * .1)).n):+.0f}bp')
M.to_csv('xs/report_504_months.csv')
# PART 5 (declared before running; added because PART 2 found no market feature): REGIME PERSISTENCE of the flip book itself. Mechanism: flips are one trade (breakouts in big stocks) driven by one shared factor; when that factor is working/failing it persists for weeks (Rey's persistence thesis), so the average result of the flips that EXITED in the last 20 calendar days (known at the entry day, no look-ahead) leads the next flips. Direction +. Tercile cuts on train, same PASS rule; also the trailing win rate.
print('\nPART 5  persistence of the flip book (trailing 20 calendar days of EXITED flips)')
X = W[['exit', 'bp']].copy(); X['exit'] = pd.to_datetime(X.exit); X = X.sort_values('exit'); days = np.sort(W.day.unique()); mu, wr, nn = [], [], []
ex = X.exit.values; bpv = X.bp.values; cs = np.concatenate([[0], np.cumsum(bpv)]); cw = np.concatenate([[0], np.cumsum(bpv > 0)])
for d in days:
    hi = np.searchsorted(ex, d, side='left'); lo = np.searchsorted(ex, d - np.timedelta64(20, 'D'), side='left'); n = hi - lo
    mu.append((cs[hi] - cs[lo]) / n if n >= 30 else np.nan); wr.append((cw[hi] - cw[lo]) / n if n >= 30 else np.nan); nn.append(n)
P = pd.DataFrame({'day': days, 'T_mu': mu, 'T_wr': wr}); W = W.drop(columns=[c for c in ('T_mu', 'T_wr') if c in W]).merge(P, on='day', how='left')
for f in ('T_mu', 'T_wr'):
    x = W[f]; q = np.nanquantile(x[W.tr], [1 / 3, 2 / 3]); W[f + '_t'] = np.where(x.isna(), np.nan, np.where(x <= q[0], 0, np.where(x >= q[1], 2, 1))); Wk = W.dropna(subset=[f + '_t'])
    a = eff(Wk[Wk.tr], f); b = eff(Wk[~Wk.tr], f); c = eff(Wk, f); top = Wk[Wk[f + '_t'] == 2].bp.mean(); bot = Wk[Wk[f + '_t'] == 0].bp.mean(); ok = (np.sign(a[0]) == np.sign(b[0])) and abs(c[1]) >= 2 and c[0] > 0
    print(f"{f}  declared +  top {top:+6.1f}bp  bottom {bot:+6.1f}bp | effect train {a[0]:+6.1f} (t {a[1]:+.1f})  test {b[0]:+6.1f} (t {b[1]:+.1f})  all {c[0]:+6.1f} (t {c[1]:+.1f}) | {'PASS' if ok else '-'}   cuts {q.round(1)}")
    if ok or f == 'T_mu':
        keep = Wk[Wk[f + '_t'] != 0]; drop = Wk[Wk[f + '_t'] == 0]; print(f"   gate (skip the worst-trailing tercile): drops {len(drop) / len(Wk):.0%} of trades; mean {Wk.bp.mean():+.1f} -> kept {keep.bp.mean():+.1f} (dropped {drop.bp.mean():+.1f}); win {Wk.bp.gt(0).mean():.0%} -> {keep.bp.gt(0).mean():.0%}")
        for e in ('18-19', '20-21', '22', '23-26'):
            a_ = Wk[Wk.era == e]; k_ = keep[keep.era == e]; d_ = drop[drop.era == e]
            if len(a_): print(f"   {e:6s} all {a_.bp.mean():+6.1f} (n {len(a_)})  kept {k_.bp.mean():+6.1f} (n {len(k_)})  dropped {d_.bp.mean():+6.1f} (n {len(d_)})")
        # worst-month capture: how many of the 12 worst months would the gate have flagged (share of that month's trades dropped)
        Mm = pd.read_csv('xs/report_504_months.csv'); w12 = Mm.head(12).mon.tolist(); sh = Wk[Wk.mon.isin(w12)].groupby('mon').apply(lambda g: (g[f + '_t'] == 0).mean()); print('   share of each worst month dropped by the gate:', sh.round(2).to_dict())
