"""Nov 23 2026 report 381: does the point-in-time options-ratio gate remove survivorship?  Rey: the options data eats some survivorship and the first PIT-universe test came back good.  Same sim (6 slots, 8% breaker, 2+2bp, ratio-rank pick order, NO brank, wide_T rows with exit < END), same 168-name data pool, split by gate state at the flip.  Absolute numbers differ from the live book (wide_T gated has 17,391 flips vs 12,360 in b302; the live book applies extra entry rules not reproduced here); only the comparison across groups is the point."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
_p = px_at
def px_at(tk, t): return _p(tk, t) if tk in PX else 1.0
U = '/mnt/user-data/uploads/fartpie/'; w = pd.read_parquet(U + 'xs/ext/wide_T.parquet'); w['entry'] = pd.to_datetime(w.entry); w['exit'] = pd.to_datetime(w.exit); w = w[w.exit < END].copy(); w['brank'] = 0.0; w['h1'] = False
print('pool names', w.tk.nunique(), '| S&P-flag rows', w.sp.sum(), '| non-S&P rows', (~w.sp).sum())
def go(lab, T):
    T = T.reset_index(drop=True); TK, _ = sim_swap(T, None); z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print('%-44s names %3d flips %5d trades %5d CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f | mean flip ret %+.2f%%' % (lab, T.tk.nunique(), len(T), len(TK), cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b) * 100, sharpe(b.values), T.ret.mean() * 100))
go('GATED: S&P PIT & ratio>=1.5 (the book, no brank)', w[w.sp & (w.ratio >= 1.5)])
go('BELOW GATE: S&P PIT & ratio 1.0-1.5', w[w.sp & (w.ratio < 1.5)])
go('NOT S&P at the time, ratio>=1.5', w[(~w.sp) & (w.ratio >= 1.5)])
go('ALL S&P PIT flips, ratio ignored (pool)', w[w.sp])
