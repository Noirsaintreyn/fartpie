"""Part 2 of report 397: tests on the feature table built by report_397_amd_all_books.py (see its docstring)."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; F = pd.read_parquet(SP + 'amd397.parquet'); F['entry'] = pd.to_datetime(F.entry); F = F[F.entry >= '2018-01-01'].copy()
tfs = ('15m', '1h', '4h', 'D'); F['S_HTF'] = F[['S_1h', 'S_4h', 'S_D']].max(axis=1); F['S_any'] = F[[f'S_{t}' for t in tfs]].max(axis=1); F['C_any'] = F[[f'C_{t}' for t in tfs]].max(axis=1); F['nC'] = F[[f'C_{t}' for t in tfs]].sum(axis=1); F['wk'] = F.entry.dt.to_period('W').astype(str)
def per(d, a, bb):
    D = d[a(d) | bb(d)].copy(); D['top'] = a(D).astype(int)
    if D.top.sum() < 20 or (1 - D.top).sum() < 20: return 'n/a (yes n %d)' % D.top.sum()
    m = smf.ols('ret ~ top', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.wk)[0]}); return 'yes %+.2f%% (n %d) no %+.2f%% diff %+.2fpp t %+.1f' % (D[D.top == 1].ret.mean() * 100, D.top.sum(), D[D.top == 0].ret.mean() * 100, m.params['top'] * 100, m.tvalues['top'])
def book(lab, d, periods):
    print('\n=== %s: %d flips; share S_HTF %.1f%%, S_any %.1f%%, C_any %.0f%%; mean flip %+.2f%%' % (lab, len(d), d.S_HTF.mean() * 100, d.S_any.mean() * 100, d.C_any.mean() * 100, d.ret.mean() * 100))
    for nm, a, bb in (('S_HTF (primary, 1h|4h|D sequence)', lambda D: D.S_HTF == 1, lambda D: D.S_HTF == 0), ('S_1h', lambda D: D.S_1h == 1, lambda D: D.S_1h == 0), ('S_4h', lambda D: D.S_4h == 1, lambda D: D.S_4h == 0), ('S_D', lambda D: D.S_D == 1, lambda D: D.S_D == 0), ('S_any', lambda D: D.S_any == 1, lambda D: D.S_any == 0), ('C_any consolidation', lambda D: D.C_any == 1, lambda D: D.C_any == 0), ('2+ TFs consolidating vs none', lambda D: D.nC >= 2, lambda D: D.nC == 0)):
        print('  %-36s' % nm + ' || '.join('%s: %s' % (pl, per(d[(d.entry >= lo) & (d.entry < hi)], a, bb)) for pl, lo, hi in periods), flush=True)
A = F[F.ds == 'A']; book('A large-cap vector, ALL flips (no gate), 54 names', A, (('2018', '2018-01-01', '2019-01-01'), ('2019-21', '2019-01-01', '2022-01-01'), ('2022-26', '2022-01-01', '2030-01-01')))
B = F[F.ds == 'B']; book('B sub-$5 small caps, all flips, 108 names', B, (('2019-21', '2019-01-01', '2022-01-01'), ('2022-26', '2022-01-01', '2030-01-01'))); Bs = B[B.px < 5]; book('B sub-$5 small caps, entry price < $5 (the book)', Bs, (('2019-21', '2019-01-01', '2022-01-01'), ('2022-26', '2022-01-01', '2030-01-01')))
