"""Report 199: biggest 2026 vector-flip returns on sub-$5 names (daily bars, wide universe 1,656 tickers, entry next open after the bull flip, exit open after the bear flip or last close if still open, no costs shown). Flags a trade as DISCONTINUITY if any single day inside it has open/prev close >2 or <0.5 (possible reverse split / reorganisation / bad adjustment)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]; B = B[B.index >= '2025-06-01']
    if len(B) < 80 or B.index[-1] < pd.Timestamp('2026-10-01'): continue
    reg, _, _, _ = run_state_machine(B.assign(datetime=B.index)); reg = np.asarray(reg); idx = B.index; o = B.open.values; c = B.close.values; n = len(B)
    for i in range(1, n - 1):
        if reg[i] == 1 and reg[i - 1] != 1 and idx[i] >= pd.Timestamp('2026-01-01'):
            e = i + 1; j = i + 1
            while j < n and reg[j] != -1: j += 1
            open_ = j >= n - 1; x = n - 1 if open_ else j + 1
            exitpx = c[n - 1] if open_ else o[x]; gap = o[e:x + 1] / np.r_[o[e - 1] if False else c[e - 1], c[e:x]][:len(o[e:x + 1])] if x + 1 > e else np.array([1.])
            g = o[e + 1:x + 1] / c[e:x] if x > e else np.array([1.])
            rows.append(dict(tk=tk, entry=idx[e], px=o[e], exit=idx[min(x, n - 1)], exitpx=exitpx, ret=exitpx / o[e] - 1, open=open_, maxgap=g.max() if len(g) else 1, mingap=g.min() if len(g) else 1, disc=bool(len(g) and (g.max() > 2 or g.min() < .5)), peak=c[e:x + 1].max() / o[e] - 1))
R = pd.DataFrame(rows); R = R[R.px < 5].sort_values('ret', ascending=False)
print('2026 flips with entry px<$5:', len(R), '| discontinuity-flagged:', int(R.disc.sum()))
pd.set_option('display.width', 200); f = lambda d: d.assign(ret=(d.ret * 100).round(0), peak=(d.peak * 100).round(0), px=d.px.round(2), exitpx=d.exitpx.round(2))[['tk', 'entry', 'px', 'exit', 'exitpx', 'ret', 'peak', 'open', 'disc']]
print('\nTOP 20 (all):'); print(f(R.head(20)).to_string(index=False))
C = R[~R.disc]; print('\nTOP 15 after dropping discontinuity-flagged:'); print(f(C.head(15)).to_string(index=False))
print('\nFlagged rows:'); print(f(R[R.disc].head(10)).to_string(index=False))
print('\nmedian ret clean %.1f%%, win %.0f%%, >=100%%: %d of %d' % (C.ret.median() * 100, (C.ret > 0).mean() * 100, (C.ret >= 1).sum(), len(C)))
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/penny2026_199.pkl')
