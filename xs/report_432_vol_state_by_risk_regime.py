"""Nov 24 2026 report 432. Rey: instead of only volatility relief, identify WHEN relief vs expansion works.
DECLARED hypothesis: relief (N2dn: 15m RV 5d <= 20d mean) helps in RISK-OFF (SPY <= 200d SMA at the prior close = calming after stress); expansion (N2up) helps in RISK-ON (breakout vol in a trend).  Data: report 431 trades (15m, ratio-gated G20/G252, chain-proxy negative gamma, one trade per episode, X1 vector exit and X3 positive-gamma-or-vector exit), 24 names, 2018-07..2026-10-14, 4bp.
Test: 2x2 (regime x vol state), per cell n/win/median/mean; relief-minus-expansion within each regime, Welch t (episodes are one trade each); train (2019-21) vs test (2018 + 2022+) and 2026 shown.  PASS: relief-minus-expansion > 0 in risk-off AND < 0 in risk-on in BOTH halves, |t| >= 2 in at least one regime."""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; T = pd.read_parquet(SP + 'g431.parquet')
M_ = pd.read_parquet(SP + 'macro_daily.parquet'); sp = M_[M_.symbol == 'SPY'].assign(d=lambda x: pd.to_datetime(x.ts).dt.normalize()).drop_duplicates('d').set_index('d').close.sort_index(); on = (sp > sp.rolling(200).mean()).shift(1)
T['on'] = on.reindex(T.entry.dt.normalize(), method='ffill').values; T = T[T.on.notna()].copy(); T['on'] = T.on.astype(bool); T['relief'] = ~T.N2up; e_ = T.entry; T['y26'] = e_ >= '2026-01-01'; print('trades', len(T), '| risk-on share', T.on.mean().round(2), '| 2026 risk-on share', T[T.y26].on.mean().round(2))
def cell(d): return f"n{len(d):4d} win {(d.ret>0).mean()*100:5.1f} med {d.ret.median()*100:5.2f} mean {d.ret.mean()*100:5.2f}"
def welch(a, b): return (a.ret.mean() - b.ret.mean()) / np.sqrt(a.ret.var() / len(a) + b.ret.var() / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
for gk in ('G20', 'G252'):
    for xk in ('X1', 'X3'):
        B = T[(T.gate == gk) & (T.exit == xk)]; print(f'\n=== {gk} {xk} ===')
        for lab, m in (('ALL', B.entry >= '2018-01-01'), ('train 2019-21', (B.entry >= '2019-01-01') & (B.entry < '2022-01-01')), ('test 2018+2022-25', (B.entry < '2019-01-01') | ((B.entry >= '2022-01-01') & (B.entry < '2026-01-01'))), ('2026', B.y26)):
            d = B[m]; s = f'{lab:18s}'
            for reg, rn in ((False, 'RISK-OFF'), (True, 'RISK-ON')):
                r = d[(d.on == reg) & d.relief]; x = d[(d.on == reg) & ~d.relief]; s += f" | {rn}: relief {cell(r)} / expansion {cell(x)} -> diff {(r.ret.mean()-x.ret.mean())*100:+.2f}pp (t {welch(r, x):+.1f})"
            print(s)
