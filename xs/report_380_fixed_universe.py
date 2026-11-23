"""Nov 22 2026 report 380: fixed-universe backtest of the live 6-slot book (no parameter changes: 6 slots, 8% breaker, 2+2bp, options-ratio >= 1.5 gate, same pick order, same entries/exits as the live backtest = b302 through the report-349 sim).
WHY: flips/month rose 46 -> 183 across regimes while flips per flipping name stayed ~5, so the growth is names entering eligibility.  Diagnostic (this report's header run): the point-in-time S&P + options-ratio >= 1.5 gate is the cause.  Early years: ratio data exist for the names (optvol_ratio_sp.csv from 2018-01-02) but their options-volume/shares ratio was below 1.5 (e.g. NFLX median 0.25 in 2019H1, AMD 0.96, JPM 0.84); 8 names did not exist yet (ABNB, APP, COIN, CRWD, GEV, HOOD, PLTR, UBER price data start 2019-06..2024); AVGO, QCOM, PYPL have no ratio rows in the 2018 files.  So late arrival = failed eligibility (options activity rose over time) + late listings, not missing prices.
UNIVERSES (same dates, same rules): A all 54 (the live backtest); B names with price data from early 2018 (46: drops the 8 late listings); C CORE = names with a gate-passing flip by 2019-12-31 (always-eligible); D = LATE = the names whose first gate-passing flip is after 2019-12-31 (all flips from then on).  C vs D on identical dates isolates composition from market conditions.
CAVEAT: all 54 are today's liquid names (survivorship in the NAME LIST itself, larger than the composition effect); the point-in-time part is only the S&P flag and the ratio gate."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
U = '/mnt/user-data/uploads/fartpie/'
f = sorted(glob.glob(U + 'lse_ohlcv_cache/SPY_15m_2018-01-01_*.parquet'))[-1]; Hs = pd.read_parquet(f); Hs.index = pd.to_datetime(Hs.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); Hs = Hs.sort_index(); dc = Hs.groupby(Hs.index.normalize()).close.last(); spr = np.log(dc).diff().dropna(); spr = spr[spr.abs() < .2]
RG = [('R1 pre-COVID', '2018-01-01', '2020-03-01'), ('R2 COVID/stim', '2020-03-01', '2022-01-01'), ('R3 tighten/bear', '2022-01-01', '2023-04-01'), ('R4 AI bull/0DTE', '2023-04-01', '2025-04-01'), ('R5 tariff/policy', '2025-04-01', '2027-01-01')]
names = sorted(B0.tk.unique()); first = B0.groupby('tk').entry.min(); core = sorted(first[first <= '2019-12-31'].index); late = sorted(set(names) - set(core))
ps = {}
for t in names:
    g = sorted(glob.glob(U + f'lse_ohlcv_cache/{t}_15m_2018-01-01_*.parquet'))[-1]; ps[t] = pd.to_datetime(pd.read_parquet(g, columns=['datetime']).datetime).min()
early = [t for t in names if ps[t] < pd.Timestamp('2018-06-30')]
U_ = {'A all 54 (live)': names, 'B early-listed %d' % len(early): early, 'C CORE eligible by 2019 (%d)' % len(core): core, 'D LATE entrants (%d)' % len(late): late}
print('core names:', core); print('late names:', late)
def run(S):
    TK, _ = sim_swap(B0[B0.tk.isin(S)].reset_index(drop=True), None); z = spread(TK); z = z[z.index >= '2019-01-01']; return TK, z
def expo(TK, z):
    n = np.zeros(len(D));
    for tk, e, x, r, ra in TK:
        a, b = D.searchsorted(e.normalize()), D.searchsorted(x.normalize()); n[a:min(b, len(D) - 1) + 1] += 1
    s = pd.Series(np.minimum(n, 6) / 6, index=D); return s[s.index >= '2019-01-01']
res = {}
for lab, S in U_.items():
    TK, z = run(S); ex = expo(TK, z); X = pd.concat([z.rename('b'), spr.rename('s')], axis=1).dropna(); beta = X.b.cov(X.s) / X.s.var(); mo = (1 + z).groupby(z.index.to_period('M')).prod() - 1; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']; yrs = (z.index[-1] - z.index[0]).days / 365.25
    print('\n%-30s trades %d (%.1f/month) | CAGR %5.1f%% DD %6.1f%% Sharpe %.2f | 2019-21 CAGR %5.1f%% Sh %.2f | 2022-26 CAGR %5.1f%% Sh %.2f | avg slots filled %.0f%% (cash %.0f%%) | beta to SPY %.2f | positive months %.0f%%' % (lab, len(TK), len(TK) / (yrs * 12), cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b) * 100, sharpe(b.values), ex.mean() * 100, 100 - ex.mean() * 100, beta, (mo > 0).mean() * 100))
    out = []
    for n, a0, b0 in RG:
        g = z[(z.index >= a0) & (z.index < b0)]
        if len(g) > 40: out.append('%s CAGR %+.0f%% Sh %.1f' % (n[:2], ((1 + g).prod() ** (252 / len(g)) - 1) * 100, sharpe(g.values)))
    print('   by regime: ' + ' | '.join(out)); res[lab] = (TK, z)
print('\n== PER-NAME CONTRIBUTION in the live book (A): sum of trade return / 6, top 10 and bottom 5, share of total')
TKA = res['A all 54 (live)'][0]; c = pd.Series({tk: 0.0 for tk in names})
for tk, e, x, r, ra in TKA: c[tk] += r / 6
c = c.sort_values(ascending=False); print((c.head(10) * 100).round(1).to_dict()); print((c.tail(5) * 100).round(1).to_dict()); print('top 5 names share of total contribution: %.0f%%; core names share: %.0f%%; late names share: %.0f%%' % (c.head(5).sum() / c.sum() * 100, c[core].sum() / c.sum() * 100, c[late].sum() / c.sum() * 100))
print('\n== SAME DATES: C vs D by year (CAGR %, Sharpe)')
zc, zd = res['C CORE eligible by 2019 (%d)' % len(core)][1], res['D LATE entrants (%d)' % len(late)][1]
for y in range(2019, 2027):
    g1, g2 = zc[zc.index.year == y], zd[zd.index.year == y]; print(y, 'CORE %+6.1f%% Sh %5.2f | LATE %+6.1f%% Sh %5.2f' % (((1 + g1).prod() ** (252 / len(g1)) - 1) * 100, sharpe(g1.values), ((1 + g2).prod() ** (252 / len(g2)) - 1) * 100 if (g2 != 0).any() else 0, sharpe(g2.values) if g2.std() > 0 else 0))
