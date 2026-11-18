"""Daily adjusted close + volume for the 753-name universe via yfinance (2015-2026). SURVIVORSHIP: universe is today's list, so delisted names are missing -> flagged in every report."""
import pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
U = [l.strip() for l in open('/mnt/user-data/uploads/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
fr = []
for i in range(0, len(U), 100):
    d = yf.download(U[i:i + 100], start='2015-06-01', end='2026-11-17', auto_adjust=True, progress=False, threads=True)
    for k in ('Close', 'Volume'): pass
    fr.append(d); print(i, d.shape, flush=True)
C = pd.concat([f['Close'] for f in fr], axis=1); V = pd.concat([f['Volume'] for f in fr], axis=1)
C.to_parquet(SP + 'sm_close.parquet'); V.to_parquet(SP + 'sm_vol.parquet'); print(C.shape, C.notna().any().sum(), C.index.min(), C.index.max())
