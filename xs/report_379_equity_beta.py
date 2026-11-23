"""Nov 22 2026 report 379: equity vector, beta vs residual (next step after the atlas, report 378).  Questions: (1) does the beta-hedged edge (ret - beta_prior x SPY return over the hold) survive each YEAR, not only five regimes; (2) how much of the P&L is SPY exposure vs stock-specific residual, per regime, and monthly correlations with SPY; (3) why flips per month rose from ~25 to ~190: more names with data, or more flips per name; (4) book level: the live 6-slot book's daily series vs SPY, rolling 126-day beta, beta-hedged book Sharpe by regime; (5) conditional expectancy by market state (SPY vs its 200d average, trailing 20-day SPY vol tercile).  Descriptive plus week-clustered t per year.  Flip data = A376 (11,774 flips with a sector peer and SPY coverage to 2026-08-15)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
RG = [('R1 pre-COVID', '2018-01-01', '2020-03-01'), ('R2 COVID/stim', '2020-03-01', '2022-01-01'), ('R3 tighten/bear', '2022-01-01', '2023-04-01'), ('R4 AI bull/0DTE', '2023-04-01', '2025-04-01'), ('R5 tariff/policy', '2025-04-01', '2027-01-01')]
def reg(ts):
    ts = pd.Series(pd.to_datetime(ts).values); out = np.full(len(ts), '', dtype=object)
    for n, a, b in RG: out[((ts >= a) & (ts < b)).values] = n
    return out
A = pd.read_parquet(SP + 'A377.parquet'); A['yr'] = A.entry.dt.year; A['mo'] = A.entry.dt.to_period('M'); A['rg'] = reg(A.entry); A['mkt'] = A.bprior * A.spyret; A['wk'] = A.entry.dt.to_period('W').astype(str)
print('== (1) PER YEAR: raw vs beta-hedged mean per flip (%), week-clustered t of hedged')
for y, g in A.groupby('yr'):
    m = sm.OLS(g.res.values, np.ones(len(g))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(g.wk)[0]})
    print('%d n %5d raw %+.2f market part %+.2f hedged %+.2f t %+.1f | mean beta %.2f' % (y, len(g), g.ret.mean() * 100, g.mkt.mean() * 100, g.res.mean() * 100, m.tvalues[0], g.bprior.mean()))
print('\n== (2) DECOMPOSITION by regime: mean raw | market part | residual | market share of raw | monthly corr(mean flip ret, SPY month) | monthly corr(residual, SPY month)')
f = sorted(glob.glob(U + 'lse_ohlcv_cache/SPY_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
dc = H.groupby(H.index.normalize()).close.last(); dr = np.log(dc).diff().dropna(); dr = dr[dr.abs() < .2]; spm = dr.groupby(dr.index.to_period('M')).sum()
mm = A.groupby('mo').agg(raw=('ret', 'mean'), res=('res', 'mean'), mkt=('mkt', 'mean'), n=('ret', 'size')); mm['spy'] = spm.reindex(mm.index); mm = mm.dropna(); mm['rg'] = reg(mm.index.to_timestamp())
for n, a, b in RG:
    g = A[A.rg == n]; q = mm[mm.rg == n]; print('%-16s %+.2f | %+.2f | %+.2f | %3.0f%% | %+.2f | %+.2f  (months %d)' % (n, g.ret.mean() * 100, g.mkt.mean() * 100, g.res.mean() * 100, g.mkt.mean() / g.ret.mean() * 100, q.raw.corr(q.spy), q.res.corr(q.spy), len(q)))
print('all months: corr(raw, SPY) %+.2f, corr(residual, SPY) %+.2f' % (mm.raw.corr(mm.spy), mm.res.corr(mm.spy)))
print('\n== (3) FLIP SUPPLY: names with flips, flips per active name-month, by regime (b302, all 12,360)')
b = pd.read_parquet(SP + 'b302.parquet'); b['entry'] = pd.to_datetime(b.entry); b['rg'] = reg(b.entry); b['mo'] = b.entry.dt.to_period('M')
for n, a, e in RG:
    g = b[b.rg == n]; nm = g.mo.nunique(); act = g.groupby('mo').tk.nunique().mean(); print('%-16s months %2d flips/month %5.1f | names flipping per month %4.1f | flips per flipping name-month %.2f | distinct names %d' % (n, nm, len(g) / nm, act, len(g) / (nm * act), g.tk.nunique()))
first = {t: str(pd.to_datetime(x.entry).min())[:7] for t, x in b.groupby('tk')}; late = sorted([t for t, v in first.items() if v > '2019-06']); print('names whose first flip is after 2019-06 (late data / late listing):', late)
print('\n== (4) BOOK LEVEL (live 6-slot book daily series vs SPY)')
zv = pd.read_pickle(SP + 'zlive372.pkl'); sp = dr.copy(); sp.index = pd.to_datetime(sp.index); X = pd.concat([zv.rename('b'), sp.rename('s')], axis=1).dropna(); X['beta'] = (X.b.rolling(126).cov(X.s) / X.s.rolling(126).var()).shift(1); X['h'] = X.b - X.beta * X.s; X['rg'] = reg(X.index)
print('whole period: beta %.2f, corr %+.2f | book Sharpe %.2f, hedged Sharpe %.2f, hedged ann return %.1f%%' % (X.b.cov(X.s) / X.s.var(), X.b.corr(X.s), X.b.mean() / X.b.std() * np.sqrt(252), X.h.dropna().mean() / X.h.dropna().std() * np.sqrt(252), X.h.dropna().mean() * 252 * 100))
for n, a, e in RG:
    g = X[X.rg == n].dropna()
    if len(g) > 30: print('%-16s book ann %+5.1f%% Sharpe %5.2f | beta %.2f | hedged ann %+5.1f%% Sharpe %5.2f | SPY ann %+5.1f%%' % (n, g.b.mean() * 252 * 100, g.b.mean() / g.b.std() * np.sqrt(252), g.b.cov(g.s) / g.s.var(), g.h.mean() * 252 * 100, g.h.mean() / g.h.std() * np.sqrt(252), g.s.mean() * 252 * 100))
rb = X.beta.dropna(); print('rolling 126d beta: min %.2f median %.2f max %.2f; by year: %s' % (rb.min(), rb.median(), rb.max(), rb.groupby(rb.index.year).mean().round(2).to_dict()))
print('\n== (5) CONDITIONAL EXPECTANCY by market state at the flip (SPY vs 200d mean of daily closes; trailing 20d SPY vol tercile), raw | hedged %')
d2 = dc[dc.index.isin(dr.index) | True]; ma = d2.rolling(200).mean(); vol20 = np.log(d2).diff().where(lambda s: s.abs() < .2).rolling(20).std() * np.sqrt(252)
A['day'] = A.entry.dt.normalize(); pd_ = d2.index.searchsorted(A.day.values) - 1; A['above'] = (d2.values[pd_] > ma.values[pd_]).astype(float); A.loc[np.isnan(ma.values[pd_]), 'above'] = np.nan; A['v20'] = vol20.values[pd_]
for lab, g in (('SPY above 200d', A[A.above == 1]), ('SPY below 200d', A[A.above == 0])): print('%-16s n %5d raw %+.2f hedged %+.2f win %.0f%%' % (lab, len(g), g.ret.mean() * 100, g.res.mean() * 100, (g.ret > 0).mean() * 100))
q1, q2 = A.v20.quantile([1 / 3, 2 / 3])
for lab, g in (('SPY vol low', A[A.v20 <= q1]), ('SPY vol mid', A[(A.v20 > q1) & (A.v20 <= q2)]), ('SPY vol high', A[A.v20 > q2])): print('%-16s n %5d raw %+.2f hedged %+.2f win %.0f%%' % (lab, len(g), g.ret.mean() * 100, g.res.mean() * 100, (g.ret > 0).mean() * 100))
