"""Report 458 (Nov 24 2026): separate signal models on the options-ratio flip universe, each an ALERT of its own beside the main screener.
Models (declared before running; prior-close features from data/theta_feat2, ranks = own trailing 252d):
  A  NEG  negative gamma: net dealer gamma < 0 (base model, 15m and 1h)
  B  PCV  put/call VOLUME rank in top third (15m only)
  C  SKW  skew (put IV - call IV) rank in top third (15m and 1h)
  D  CHM  charm rank in top third (15m and 1h)
Universe: xs/ext/wide_T.parquet, S&P member, options ratio >= 1.5, names that have theta features (23 of the 54 live names so far).
Per model and timeframe: n, mean bp per trade, month-clustered t vs 0, win rate, mean hold (trading hours), train 2019-21 / test 2018+2022+, era means, overlap with A, and the lift over ALL gated flips of the same timeframe (month-paired t).
Also: 'any of B/C/D' and 'A or B/C/D' unions, and which timeframe each model's mean per hour of holding favours (does the model catch the quick move)."""
import pandas as pd, numpy as np, os
exec(open('xs/report_457_optfeat_flips.py').read().split("NAMES = dict")[0].split('"""')[2])   # reuse data prep from 457 (everything up to NAMES)
W['hold'] = (pd.to_datetime(W.exit) - W.entry).dt.total_seconds() / 3600
def tm(x):
    m = x.groupby('mon').ret.mean(); return m.mean() * 1e4, (m.mean() / (m.std() / np.sqrt(len(m))) if len(m) > 2 and m.std() > 0 else np.nan)
def lift(a, b):
    d = (a.groupby('mon').ret.mean() - b.groupby('mon').ret.mean()).dropna(); return d.mean() * 1e4, d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
for h1 in [False, True]:
    D = W[(W.h1 == h1) & W.gex.notna()].copy(); tf = '1h' if h1 else '15m'
    D['A'] = D.gex < 0; D['B'] = D.r_pc_vol >= 2 / 3; D['C'] = D.r_skew >= 2 / 3; D['D'] = D.r_charm >= 2 / 3
    D['U'] = D.B | D.C | D.D; D['AU'] = D.A | D.U
    print(f'\n=== {tf}: all gated flips with features n {len(D)}  mean {D.ret.mean() * 1e4:+.1f}bp  hold {D.hold.mean():.1f}h ===')
    for k, lab in [('A', 'A neg gamma'), ('B', 'B put/call volume high'), ('C', 'C skew high'), ('D', 'D charm high'), ('U', 'B or C or D'), ('AU', 'A or B or C or D')]:
        if k == 'B' and h1: continue
        x = D[D[k].fillna(False)]; m, t = tm(x); ll, lt = lift(x, D[~D[k].fillna(False)])
        sp = [tm(x[x.half == h])[0] for h in ['train', 'test']]; spt = [tm(x[x.half == h])[1] for h in ['train', 'test']]
        eras = ' '.join(f'{e} {x[x.era == e].ret.mean() * 1e4:+.0f}({(x.era == e).sum()})' for e in ['18-19', '20-21', '22', '23-26'])
        ov = D[D[k].fillna(False)].A.mean() if k != 'A' else np.nan
        print(f'{lab:24s} n {len(x):5d} mean {m:+6.1f}bp t {t:5.2f} win {(x.ret > 0).mean():.0%} hold {x.hold.mean():4.1f}h  bp/h {x.ret.mean() * 1e4 / max(x.hold.mean(), .1):+5.1f} | train {sp[0]:+.0f} (t{spt[0]:.1f}) test {sp[1]:+.0f} (t{spt[1]:.1f}) | lift vs rest {ll:+.1f}bp t {lt:.2f} | overlap with A {ov:.0%}')
        print(f'{"":24s} eras {eras}')
