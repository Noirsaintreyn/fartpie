"""Nov 23 2026 report 382: 2x2 factorial of the two screener filters (S&P membership at the flip x options-ratio >= 1.5) on ONE flip table (xs/ext/wide_T.parquet, 168-name pool), plus reconciliation of wide_T-gated 17,447 rows vs the live 12,360 (b302).
Reconciliation finding (computed here): b302 contains no rows that are not in wide_T; 4,362 of the 17,447 wide_T rows (25%) are NOT in b302, and every one of them is a 1H-vector flip row (h1 = True); the other ~13,085 rows map onto the 12,360 unique (name, entry) 15m flips (an entry bar can carry both a 15m and a 1H row).  So 17,391 vs 12,360 = 15m-vector book vs 15m+1H-vector rows; b301/b302 additionally drop rows missing the seven point-in-time signals.
LIMIT: wide_T only holds rows with ratio >= 1.0, so 'below the gate' means ratio 1.0-1.5; flips with ratio < 1.0 are not in this table.
Tests: (1) trade level: mean flip return in each cell, train 2019-21 | test 2022-26; OLS ret ~ SP + GATE + SP x GATE with week-clustered SE.  (2) portfolio level in a COMMON candidate pool (6 slots, 8% breaker, 2+2bp, higher ratio first, no brank): pool of everything, + S&P only, + gate only, + both, for the 15m-type rows (h1 False) and for all rows."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
_p = px_at
def px_at(tk, t): return _p(tk, t) if tk in PX else 1.0
U = '/mnt/user-data/uploads/fartpie/'; w = pd.read_parquet(U + 'xs/ext/wide_T.parquet'); w['entry'] = pd.to_datetime(w.entry); w['exit'] = pd.to_datetime(w.exit); w = w[w.exit < END].copy(); w['brank'] = 0.0
w['GATE'] = (w.ratio >= 1.5).astype(int); w['SP'] = w.sp.astype(int); w['train'] = w.entry < '2022-01-01'; w['wk'] = w.entry.dt.to_period('W').astype(str)
print('rows %d | 1H rows %d | h1 False rows %d | names %d' % (len(w), w.h1.sum(), (~w.h1).sum(), w.tk.nunique()))
def cells(T, lab):
    print('\n== TRADE LEVEL, %s (mean flip return %%, win %%, n; train | test)' % lab)
    for sp in (1, 0):
        for ga in (0, 1):
            g = T[(T.SP == sp) & (T.GATE == ga)]; a, b = g[g.train], g[~g.train]
            print('  %-12s %-14s n %5d | all %+.2f%% win %.0f%% | train %+.2f%% (n %d) | test %+.2f%% (n %d)' % ('S&P' if sp else 'not S&P', 'ratio>=1.5' if ga else 'ratio 1.0-1.5', len(g), g.ret.mean() * 100, (g.ret > 0).mean() * 100, a.ret.mean() * 100, len(a), b.ret.mean() * 100, len(b)))
    for h, D in (('train', T[T.train]), ('test', T[~T.train])):
        m = smf.ols('ret ~ SP + GATE + SP:GATE', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]})
        print('  OLS %s (pp): ' % h + ' | '.join('%s %+.2f (t %+.1f)' % (k, m.params[k] * 100, m.tvalues[k]) for k in ('Intercept', 'SP', 'GATE', 'SP:GATE')))
P15 = w[~w.h1]; cells(P15, '15m-type rows (h1 False)'); cells(w, 'all rows (15m + 1H)')
def go(lab, T):
    T = T.reset_index(drop=True); TK, _ = sim_swap(T, None); z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print('  %-34s flips %5d trades %5d | CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f' % (lab, len(T), len(TK), cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b) * 100, sharpe(b.values)), flush=True)
for nm, T in (('15m-type rows', P15), ('all rows', w)):
    print('\n== PORTFOLIO, common pool, %s' % nm)
    go('pool: everything (ratio>=1.0)', T); go('+ S&P membership only', T[T.SP == 1]); go('+ ratio>=1.5 only', T[T.GATE == 1]); go('+ both (the live filter)', T[(T.SP == 1) & (T.GATE == 1)])
