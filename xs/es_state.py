"""Run with NZ_SRC=es5_real.parquet: saves ES noise-band state per day/bar (+1 above UB, -1 below LB, 0 inside) for the cross-market confirmation test."""
import sys, numpy as np, pandas as pd, pickle; sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, OPEN, PC, SIG, SI
out = {}
for d in ALLD:
    o, h, l, c, v, vw = A[d][0]; sg = SIG[SI[d]]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; ub = max(op, pc) * (1 + sg); lb = min(op, pc) * (1 - sg)
    out[d] = np.where(c > ub, 1, np.where(c < lb, -1, 0)).astype(int) * np.isfinite(sg)
pickle.dump(out, open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/es_state.pkl', 'wb')); print(len(out), 'days')
