"""Split history (for share-count basis alignment: SEC shares are as reported at filing, yfinance prices are split-adjusted) for the 753 small caps + the 54 large caps."""
import glob, os, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
U = [l.strip() for l in open('/mnt/user-data/uploads/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
L = sorted({os.path.basename(f)[:-8] for f in glob.glob('data/yahoo/daily/*.parquet')} | {os.path.basename(f)[:-5] for f in glob.glob('data/fund/sec/*.json')}); L = [t for t in L if t.isalpha() or '-' in t]
T = sorted(set(U) | set(L)); out = []
for i in range(0, len(T), 100):
    d = yf.download(T[i:i + 100], start='2015-01-01', end='2026-11-17', auto_adjust=False, actions=True, progress=False, threads=True)
    s = d['Stock Splits'].stack(); s = s[s > 0]; out.append(s.rename('ratio').reset_index().rename(columns={'level_1': 'tk', 'Ticker': 'tk'}))
    if i == 0: pass
    print(i, len(s), flush=True)
S = pd.concat(out); S.columns = ['date', 'tk', 'ratio']; S.to_parquet(SP + 'splits.parquet'); print(S.shape); print(S.sort_values('date').tail(3))
