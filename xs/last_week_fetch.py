"""Pull ~170 days of 15m RTH candles for the 54 research names (LSE key read from the scratchpad .lse_env at RUNTIME only) and cache them in the scratchpad for the last-week indicator check (report 367)."""
import os, sys
_s = open('xs/forward_flip_log.py').read(); exec(_s[:_s.index('def feats')])
os.makedirs(SP + 'lastweek', exist_ok=True); ok = 0
for tk in TK:
    f = SP + f'lastweek/{tk}.parquet'
    if os.path.exists(f): ok += 1; continue
    H = bars(tk)
    if H is None or len(H) < 400: print('skip', tk, flush=True); continue
    H.to_parquet(f); ok += 1; print(tk, len(H), H.index.max(), flush=True)
print('done', ok, 'of', len(TK))
