"""Nov 23 2026 report 391. Leadership-rank swap for the S&P flag (experiment 2 in the full-analysis doc).
Hypothesis: the S&P x gate interaction that exists only in 2022-26 (report 382) is a size/leadership effect: high-ratio flips earn in the names that lead the market by dollar volume, and S&P membership is only a proxy.  Mechanism: index-eligible leaders attract persistent institutional flow, so a break that fires with unusual option activity continues; high-ratio names outside the leadership group are meme/high-vol names that mean-revert.  Would stop working: when leadership rotates to non-index names or when dollar volume stops tracking institutional ownership.
Leadership score LEAD = cross-sectional percentile (within the 168-name pool, each day) of the trailing 60-day mean dollar volume (close x volume), shifted one day.  Volume source: LSE daily candles for 108 pool names + 15m RTH sums for the 58 cached names (source mismatch is a stated caveat).  Pool = wide_T 15m-type rows (h1 False), ratio >= 1.0 table as in report 382.
Tests: (1) trade level, OLS ret ~ GATE x SP and GATE x LEAD3 (top third = 1) and the joint model, week-clustered, train 2019-21 | test 2022-26; does LEAD absorb the SP x GATE term?  (2) portfolio, common pool, 6 slots, 8% breaker, ratio first, 2+2bp: gate only | gate + S&P (live) | gate + LEAD top 50/33/25%.  Pre-registered pass: a LEAD cutoff beats the S&P flag in Sharpe in BOTH halves with comparable trades."""
import numpy as np, pandas as pd, glob, os, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
_p = px_at
def px_at(tk, t): return _p(tk, t) if tk in PX else 1.0
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; U = '/mnt/user-data/uploads/fartpie/'
w = pd.read_parquet(U + 'xs/ext/wide_T.parquet'); w['entry'] = pd.to_datetime(w.entry).astype('datetime64[ns]'); w['exit'] = pd.to_datetime(w.exit); w = w[(w.exit < END) & ~w.h1].copy(); w['brank'] = 0.0; w['GATE'] = (w.ratio >= 1.5).astype(int); w['SP'] = w.sp.astype(int); w['train'] = w.entry < '2022-01-01'; w['wk'] = w.entry.dt.to_period('W').astype(str)
# dollar-volume panel
dv = {}
d = pd.read_parquet(SP + 'lse_daily_pool.parquet'); d['date'] = pd.to_datetime(d.ts).dt.normalize()
for tk, g in d.groupby('symbol'): dv[tk] = (g.set_index('date').close * g.set_index('date').volume)
for f in glob.glob(U + 'lse_ohlcv_cache/*_15m_2018-01-01_*.parquet'):
    tk = os.path.basename(f).split('_15m_')[0]
    if tk in dv or tk not in set(w.tk): continue
    H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = H.index.hour * 60 + H.index.minute; H = H[(m >= 570) & (m < 960)]; dv[tk] = (H.close * H.volume).groupby(H.index.normalize()).sum()
P = pd.DataFrame(dv).sort_index(); P = P[[c for c in P.columns if c in set(w.tk)]]; A = P.rolling(60, min_periods=30).mean().shift(1); L = A.rank(axis=1, pct=True)
print('pool names with a leadership series: %d of %d' % (L.shape[1], w.tk.nunique()))
w['LEAD'] = [L[t].reindex([e.normalize()], method='ffill').values[0] if t in L else np.nan for t, e in zip(w.tk, w.entry)]; w = w[w.LEAD.notna()].copy(); print('rows with LEAD: %d (S&P share %.0f%%; share of S&P rows with LEAD>=0.5: %.0f%%; of non-S&P: %.0f%%)' % (len(w), w.SP.mean() * 100, (w[w.SP == 1].LEAD >= .5).mean() * 100, (w[w.SP == 0].LEAD >= .5).mean() * 100))
w['L3'] = (w.LEAD >= 2 / 3).astype(int)
print('\n== TRADE LEVEL (gated cells), mean flip return, train | test')
for sp in (1, 0):
    for l3 in (1, 0):
        g = w[(w.GATE == 1) & (w.SP == sp) & (w.L3 == l3)]; a, b_ = g[g.train], g[~g.train]; print('  gate, %-8s %-14s n %5d | train %+.2f%% (n %d) | test %+.2f%% (n %d)' % ('S&P' if sp else 'not S&P', 'top-third lead' if l3 else 'lower lead', len(g), a.ret.mean() * 100, len(a), b_.ret.mean() * 100, len(b_)))
for h, DD in (('train', w[w.train]), ('test', w[~w.train])):
    for fm in ('ret ~ SP + GATE + SP:GATE', 'ret ~ L3 + GATE + L3:GATE', 'ret ~ SP + L3 + GATE + SP:GATE + L3:GATE'):
        m = smf.ols(fm, DD).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(DD.wk)[0]}); print('  OLS %s [%s]: ' % (h, fm.replace('ret ~ ', '')) + ' | '.join('%s %+.2f (t %+.1f)' % (k, m.params[k] * 100, m.tvalues[k]) for k in m.params.index if ':' in k))
def go(lab, T):
    T = T.reset_index(drop=True); TK, _ = sim_swap(T, None); z = spread(TK); z = z[z.index >= '2019-01-01']; a, b_ = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print('  %-30s flips %5d trades %5d | CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f' % (lab, len(T), len(TK), cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b_) * 100, sharpe(b_.values)), flush=True)
print('\n== PORTFOLIO, common pool (ratio >= 1.5 always)'); G = w[w.GATE == 1]
go('gate only', G); go('gate + S&P (live filter)', G[G.SP == 1])
for c in (.5, .67, .75): go('gate + LEAD >= %.2f' % c, G[G.LEAD >= c])
go('gate + S&P + LEAD >= 0.5', G[(G.SP == 1) & (G.LEAD >= .5)])
