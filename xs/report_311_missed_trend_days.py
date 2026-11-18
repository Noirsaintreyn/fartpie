"""Nov 18 2026 report 311. Lead 1b: 2026 trend days the stack missed. Trend day = RTH efficiency in the train top tercile (cut .16). Per year: share of trend days with stack pts > 0, and pts on trend days split by whether the first hour (open->10:30) already went the day's direction (CONT) or reversed it (REV). Hypothesis (declared): the 2026 capture drop sits in REV trend days (first hour fades the final direction), since the legs follow the first-hour move. Why wrong: capture fell on CONT days as well."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D = pd.read_parquet(SP + 'd305.parquet'); D['y'] = D.index.year
T = D[D.effday >= .16].copy(); T['cont'] = np.sign(T.fh) == np.sign(T.net); T['fhn'] = T.fh.abs() / T.adr
print('trend days by year: n | % with stack>0 | pts/day CONT (share CONT) | pts/day REV | |net|/ADR')
for y, g in T.groupby('y'):
    c, r = g[g.cont], g[~g.cont]; print(f'{y} n{len(g):3d} win{(g.ALL > 0).mean():4.0%} CONT {c.ALL.mean():+6.1f} ({len(c) / len(g):.0%}) REV {r.ALL.mean():+6.1f} n{len(r)}  net/ADR {(g.net.abs() / g.adr).mean():.2f}  capture CONT {c.ALL.sum() / c.net.abs().sum():.2f} REV {r.ALL.sum() / max(r.net.abs().sum(), 1):.2f}')
for k in ('M1', 'V15a', 'ON+', 'RA', 'RULES'):
    print(k, {y: f"C{g[g.cont][k].mean():+.0f}/R{g[~g.cont][k].mean():+.0f}" for y, g in T.groupby('y') if y >= 2024})
