"""Nov 23 2026 report 383: put the 1H-vector flips into the live book the way the live rules say (Rey, Nov 15: each asset counts as ONE signal, 'APP' and 'APP 1h' are one; the 1h flip only enters if a slot is open after the 15m ones).  Pool = wide_T rows with S&P at the flip and ratio >= 1.5 (the live filter), 6 slots, 8% breaker, 2+2bp, higher ratio first.
A: 15m rows only (h1 False) = the live backtest (reproduces 23.7% / Sh 2.61).  B: all rows, no one-slot rule (previous run, 30.3% / 3.05).  C: all rows, 15m rows first within a bar, ONE POSITION PER NAME (a row is dropped if the same name already holds an earlier row, greedy by entry time, 15m before 1H on ties; the drop ignores capacity skips, so it slightly over-restricts).  D: 1H rows only, one per name.  Descriptive; the 1H rows come from the same table so the pool is identical to the factorial test (report 382)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
_p = px_at
def px_at(tk, t): return _p(tk, t) if tk in PX else 1.0
U = '/mnt/user-data/uploads/fartpie/'; w = pd.read_parquet(U + 'xs/ext/wide_T.parquet'); w['entry'] = pd.to_datetime(w.entry); w['exit'] = pd.to_datetime(w.exit); w = w[(w.exit < END) & w.sp & (w.ratio >= 1.5)].copy(); w['brank'] = w.h1.astype(float); w = w.reset_index(drop=True)
def one_per_name(T):
    keep = []; T = T.reset_index(drop=True).sort_values(['entry', 'brank', 'tk'], kind='stable')
    for tk, g in T.groupby('tk'):
        free = pd.Timestamp.min
        for i, r in zip(g.index, g.itertuples()):
            if r.entry >= free: keep.append(i); free = r.exit
    return T.loc[keep]
def go(lab, T):
    T = T.reset_index(drop=True); TK, _ = sim_swap(T, None); z = spread(TK); z = z[z.index >= '2019-01-01']; a, b = z[z.index < '2022-01-01'], z[z.index >= '2022-01-01']
    print('%-44s rows %5d trades %5d | CAGR %5.1f%% DD %6.1f%% Sh %.2f | 2019-21 %5.1f%% Sh %.2f | 2022-26 %5.1f%% Sh %.2f' % (lab, len(T), len(TK), cagr(z) * 100, mdd(z) * 100, sharpe(z.values), cagr(a) * 100, sharpe(a.values), cagr(b) * 100, sharpe(b.values)), flush=True); return z
zA = go('A 15m rows only (live backtest)', w[~w.h1]); zB = go('B all rows, no one-slot rule', w); zC = go('C all rows, 15m first, one slot per name', one_per_name(w)); zD = go('D 1H rows only, one slot per name', one_per_name(w[w.h1]))
print('daily-return correlation A vs D: %.2f; A vs C: %.2f' % (zA.corr(zD), zA.corr(zC)))
