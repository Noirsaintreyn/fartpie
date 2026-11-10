"""Nov 10 2026: THE NOISE ENGINE ON ANOTHER MARKET (ES) - SAME RULES, SAME 14-SESSION SIGMA BANDS, STOP = 0.2% OF THE OPEN (NQ's 40 pts is about 0.2%), COST 0.25 ES PT (= $12.50) vs 0.5 NQ PT ($10). DECLARED BEFORE RESULTS.
Run as: NZ_SRC=... NZ_COST=... INST=ES python inst_noise.py. Prints pts/day (instrument points), t, mean/std (unit-free) both periods, all trades and long-only, dollars per contract per day; saves the daily series for the cross-market combination."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import DAYS, OPEN
INST = os.environ.get('INST', 'NQ'); PV = {'NQ': 20.0, 'ES': 50.0}[INST]; AD = pd.DatetimeIndex([d for d in DAYS if d >= pd.Timestamp('2018-02-01')])
out = {}
for lab, side in (('both', 0), ('long only', 1)):
    rows = [r for d in DAYS for r in Z.noise(d, stop=0.002 * OPEN[d], side=side)]; T = pd.DataFrame(rows, columns=['d', 'em', 'pos', 'pts', 'mae']); T['d'] = pd.to_datetime(T.d)
    s = T.groupby('d').pts.sum().reindex(pd.DatetimeIndex(DAYS)).fillna(0.0); out[lab] = s
    res = []
    for nm, f in (('14-21', s.index < '2022-01-01'), ('22-26', s.index >= '2022-01-01')):
        x = s[f]; t = x.mean() / (x.std() / np.sqrt(len(x))); res.append(f'{nm}: {x.mean():+5.2f} pts/d (${x.mean() * PV:+6.0f}) t {t:+5.2f} m/s {x.mean() / x.std():.3f}')
    print(f'{INST} {lab:<10} trades {len(T):5d} | ' + ' | '.join(res), flush=True)
pd.DataFrame(out).to_pickle(f'/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/inst_{INST}.pkl')
