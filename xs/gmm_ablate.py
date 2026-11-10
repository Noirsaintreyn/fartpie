"""Nov 10 2026: FROM arXiv 2605.04004 / 2605.11423 (Mesfin 2026, MNQ): (A) GMM intraday regime classification -> 60-minute hold, (B) VVG reversal-day classifier. DECLARED BEFORE RESULTS.
(A) Decision points every 30 min 10:00-14:30 ET. State features at the decision bar (all in noise-sigma units s = open x sigma_end): open->now return, close-VWAP, range so far, 1-hour return, gap, prior-day range, bar volume ratio vs 14-day same-time mean, noise-band state (+1 above UB, -1 below LB), HAR-style log prior 5d range. Standardise, fit a Gaussian mixture (K = 4, 6, 8; diag covariance) on all earlier years, assign each earlier sample a cluster; per cluster mean of the forward 60-min move (next-bar open -> +12 bars) and its t on the training set; a cluster trades the sign of its mean if |t| >= 1.5 and n >= 60. Walk-forward: refit each January, trade the next year, cost 0.5 pt, one position per decision (30-min decisions overlap in the 60-min hold: 1 contract each decision, reported as pts per day summed). Controls: 100 random-sign draws over the same trained clusters. PASS = pts/day > 0, t >= 2 both 2018-21 and 2022-26, and above the random-sign 95th percentile.
(B) VVG: flag a day when |gap|, |first-30-min return| and first-bar volume ratio vs 20-day mean are ALL above their expanding 75th percentile. Report share flagged, reversal-from-peak frequency, mean giveback (peak-to-close, in points), next-day return vs unflagged, and the reversal trade (fade the first-30-min direction from 10:00 to the close) pts/day and t, both periods."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, DAYS, GRID, NB, OPEN, PC, SIG, SI
from sklearn.mixture import GaussianMixture
C = .5; DI = {d: i for i, d in enumerate(ALLD)}
V_ = pd.DataFrame({d: A[d][0][4] for d in ALLD}).T; FVR = (V_ / V_.rolling(14, min_periods=10).mean().shift(1)).values
rng_ = pd.Series({d: np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2]) for d in ALLD}); LR5 = np.log(rng_.rolling(5).mean().shift(1)); PRR = rng_.shift(1)
X, Y, DAY = [], [], []
for d in DAYS:
    if d not in DI or d < pd.Timestamp('2014-04-01'): continue
    i = DI[d]; o, h, l, c, v, vw = A[d][0]; sg = SIG[SI[d]]; s_ = OPEN[d] * sg[-1]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op
    if not np.isfinite(s_) or i < 25 or not np.isfinite(LR5.iloc[i]): continue
    for jb in range(5, 60, 6):                                                   # bar closes 10:00 .. 14:30 on 30-min steps
        if jb + 14 >= NB or np.isnan(c[jb]) or np.isnan(o[jb + 1]) or np.isnan(o[jb + 13]) or not np.isfinite(sg[jb]): continue
        ub, lb = max(op, pc) * (1 + sg[jb]), min(op, pc) * (1 - sg[jb]); fv = FVR[i][jb]
        X.append([(c[jb] - op) / s_, (c[jb] - vw[jb]) / s_, (np.nanmax(h[:jb + 1]) - np.nanmin(l[:jb + 1])) / s_, (c[jb] - c[max(jb - 12, 0)]) / s_, (op - pc) / s_, PRR.iloc[i] / s_, fv if np.isfinite(fv) else 1.0, 1.0 if c[jb] > ub else -1.0 if c[jb] < lb else 0.0, LR5.iloc[i]])
        Y.append(o[jb + 13] - o[jb + 1]); DAY.append(d)
X = np.nan_to_num(np.array(X)); Y = np.array(Y); DAY = pd.DatetimeIndex(DAY); yr = DAY.year.values; print(len(Y), 'samples', flush=True)
ALLDAYS = pd.DatetimeIndex([d for d in DAYS if d >= pd.Timestamp('2018-02-01')])
def ev(pts, days, label, extra=''):
    s = pd.Series(pts, index=pd.DatetimeIndex(days)).groupby(level=0).sum().reindex(ALLDAYS).fillna(0.0); o = []; sg = []
    for nm, f in (('18-21', ALLDAYS < '2022-01-01'), ('22-26', ALLDAYS >= '2022-01-01')):
        x = s[f]; t = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0; o.append(f'{nm}: {x.mean():+6.2f}/d t {t:+5.2f}'); sg.append(x.mean() > 0 and t >= 2)
    print(f'  {label:<34} ' + ' | '.join(o) + f'{extra}' + ('   <== PASS' if all(sg) else ''), flush=True); return s

names = ['ret','vwd','rng','ret1h','gap','prr','vol','band','har']
def run(cols, K=6):
    P, D, K_ = [], [], []
    for Yr in range(2018, 2027):
        tr = yr < Yr; te = yr == Yr
        if te.sum() == 0: continue
        Xs = X[:, cols]; mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-9; g = GaussianMixture(K, covariance_type='diag', random_state=1, n_init=2).fit((Xs[tr] - mu) / sd)
        ctr = g.predict((Xs[tr] - mu) / sd); cte = g.predict((Xs[te] - mu) / sd); sign = np.zeros(K)
        for k in range(K):
            yk = Y[tr][ctr == k]
            if len(yk) >= 60 and yk.std() > 0 and abs(yk.mean() / (yk.std() / np.sqrt(len(yk)))) >= 1.5: sign[k] = np.sign(yk.mean())
        m = sign[cte] != 0; P += list((sign[cte] * Y[te] - C)[m]); D += list(DAY[te][m])
    return np.array(P), pd.DatetimeIndex(D)
import pandas as pd
G = pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/G_noise.pkl'); G['day'] = pd.to_datetime(G.day).dt.normalize(); NZd = G.groupby('day').pts.sum().reindex(ALLDAYS).fillna(0.0)
for lab, cols in (('all 9', list(range(9))), ('no band', [0,1,2,3,4,5,6,8]), ('no band, no ret/ret1h', [1,2,4,5,6,8]), ('only ret, vwd, band', [0,1,7]), ('only context (gap, prr, vol, har)', [4,5,6,8]), ('only rng, ret1h, vol', [2,3,6])):
    P, D = run(cols); s_ = ev(P, D, f'K=6 {lab}', f' | trades {len(P)}'); print(f'      daily corr with noise engine {np.corrcoef(s_.values, NZd.values)[0,1]:+.2f}', flush=True)
