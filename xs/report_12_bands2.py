"""Nov 12 2026: looser filters (no daily-vector requirement) by price band, to keep the trade count up. DECLARED BEFORE RESULTS. Uses bands_T.pkl from report_12_bands.py (B flips: vol >= 1.5x, since 2024-01).
Sets: K = Kalman up | E1 = above 1h EMA200 | E4 = above 4h EMA200 | K&E1 | E1&E4 | K&E1&E4 | any 2 of (K,E1,E4). Same sim / split / costs."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; T = pd.read_pickle(SC + 'bands_T.pkl'); COST = .004
src = open('xs/report_12_bands.py').read(); exec(src[src.index('def book(Z, cost'):src.index("for nm, lo, hi, cost in")])
T['n2'] = T.kal.astype(int) + T.a200.astype(int) + T.a4.astype(int)
for nm, lo, hi in (('$1-2', 1, 2), ('$2-5', 2, 5), ('< $5 (all)', .2, 5)):
    X = T[(T.sig >= lo) & (T.sig < hi)]; print(f'== {nm}: {len(X)} B flips')
    for sn, m in (('B (volume only)', X.vr > 0), ('K', X.kal), ('E1 (1h EMA200)', X.a200), ('E4 (4h EMA200)', X.a4), ('K & E1', X.kal & X.a200), ('E1 & E4', X.a200 & X.a4), ('K & E1 & E4', X.kal & X.a200 & X.a4), ('any 2 of K,E1,E4', X.n2 >= 2), ('A+ (D&K) ref', X.ad & X.kal)): rep(sn, X[m], X)
