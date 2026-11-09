"""Nov 9 2026: compact the ThetaData EOD option quotes (data/theta/<TK>/*.parquet) to one file per ticker: data/theta_compact/<TK>.parquet.
Keeps EOD bid/ask/volume for calls and puts with 0-70 days to expiry and strikes within 30% of spot. Spot per day is implied by put-call parity
(strike where call and put mids are closest: S = K + C - P, median over expiries with 3-60 days left), so it is in the same (unadjusted) units as the strikes."""
import sys, glob, os, numpy as np, pandas as pd
tk = sys.argv[1]; out = 'data/theta_compact'; os.makedirs(out, exist_ok=True); parts = []
for f in sorted(glob.glob(f'data/theta/{tk}/{tk}_*.parquet')):
    x = pd.read_parquet(f, columns=['expiration', 'strike', 'right', 'kind', 'created', 'bid', 'ask', 'volume'])
    x = x[x.kind == 'eod']
    if not len(x): continue
    d = pd.to_datetime(x.created.astype(str).str[:10]); e = pd.to_datetime(x.expiration); dte = (e - d).dt.days
    y = pd.DataFrame({'d': d.values, 'e': e.values, 'K': x.strike.astype('float32').values, 'r': np.where(x.right.astype(str).str.upper().str[0] == 'C', 1, -1).astype('int8'),
                      'bid': x.bid.astype('float32').values, 'ask': x.ask.astype('float32').values, 'vol': x.volume.astype('float32').values, 'dte': dte.astype('int16').values})
    parts.append(y[(y.dte >= 0) & (y.dte <= 70)])
X = pd.concat(parts, ignore_index=True).drop_duplicates(['d', 'e', 'K', 'r'], keep='last')
X['mid'] = (X.bid + X.ask) / 2; q = X[(X.dte >= 3) & (X.dte <= 60) & (X.bid > 0)]
m = q[q.r == 1][['d', 'e', 'K', 'mid']].merge(q[q.r == -1][['d', 'e', 'K', 'mid']], on=['d', 'e', 'K'], suffixes=('c', 'p'))
m['df'] = m.midc - m.midp; m['ad'] = m.df.abs(); s = m.loc[m.groupby(['d', 'e']).ad.idxmin()]; S = (s.K + s.df).groupby(s.d).median()
X['S'] = X.d.map(S).astype('float32'); X = X[(X.K / X.S > .7) & (X.K / X.S < 1.3)].drop(columns=['mid'])
X.to_parquet(f'{out}/{tk}.parquet', index=False); print(tk, len(X), X.d.min().date(), X.d.max().date(), flush=True)
