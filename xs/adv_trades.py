"""Nov 10 2026: ADVANCED MODELS ON NQ, ALL JUDGED WALK-FORWARD. DECLARED BEFORE RESULTS.
Data: ET 5m NQ RTH (nz_core), cost 0.5 pt per round trip, 1 contract. Test years 2018..2026-Jan: every model is refit each January on ALL earlier data (2014-02 on) and scored only on the next year, so every number is out of sample. Periods 2018-21 | 2022-26.
M1 ROCKET: 300 random convolutional kernels (lengths 5/7/9, dilation 1/2) over the last 24 5m bars, 4 channels (return, flip-free volume ratio, range, close-VWAP; all scaled by the day's noise sigma), PPV + max features -> ridge classifier on the SIGN of the next-30-minute return. Decisions every 30 min 11:30-15:00; trade the next open, exit 30 min later; trade only the top 10% |score| (threshold from the training scores).
M2 PATH SIGNATURE: depth-3 signature of the 3-d path (time, price, cumulative volume ratio) over the same 24 bars (39 terms) -> ridge classifier, same decisions and trade rule.
M3 GBM META-LABEL on the noise-area engine's trades (LightGBM, 8 leaves, 200 trees, lr .03, min 100 per leaf): features known at the signal bar (time of day, side, band distance in sigma, VWAP distance, return since open in sigma, range so far in sigma, gap in sigma, volume ratio (bar and 3 bars), wick, bars beyond the band, prior-day range / ADR, 12-bar variance ratio and efficiency ratio, time since the day's high / low). Target = trade points > 0. Keep the trades whose predicted probability is above the TRAINING median (and, second row, the top third).
M4 HMM REGIME: 3-state Gaussian HMM on daily features (log HAR range, close-to-close return, overnight gap / sigma, log last-day range / ADR) refit yearly; the FILTERED state at 09:30 (no smoothing) -> engine points per day by state (noise engine) and the system.
PASS for M1/M2: points per day > 0 with t >= 2 in both periods. M3: filtered engine mean/std and points per day against the unfiltered engine, both periods. M4: engine pts/day by state, both periods, same sign and |t| >= 2 for the best vs worst state."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
import nz_core as Z
from nz_core import A, ALLD, DAYS, GRID, NB, OPEN, CLOSE, PC, SIG, SI, noise
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
COST = .5; W = 24; DI = {d: i for i, d in enumerate(ALLD)}; D18 = [d for d in DAYS if d >= pd.Timestamp('2018-02-01')]
V_ = pd.DataFrame({d: A[d][0][4] for d in ALLD}).T; FVR = (V_ / V_.rolling(14, min_periods=10).mean().shift(1)).values

import lightgbm as lgb
def trade_feats(d, em, pos, pts):
    o, h, l, c, v, vw = A[d][0]; jb = em - 1; i = DI[d]; sg = SIG[SI[d]]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op
    if jb < 1 or not np.isfinite(sg[jb]) or np.isnan(c[jb]): return None
    s_ = op * sg[jb]; ub, lb = max(op, pc) * (1 + sg[jb]), min(op, pc) * (1 - sg[jb]); rng_ = h[jb] - l[jb]; hh = np.nanmax(h[:jb + 1]); ll = np.nanmin(l[:jb + 1]); thi = int(np.nanargmax(h[:jb + 1])); tlo = int(np.nanargmin(l[:jb + 1]))
    beyond = (c[:jb + 1] > ub) if pos > 0 else (c[:jb + 1] < lb); acc = 0
    for q in range(jb, -1, -1):
        if q < len(beyond) and beyond[q]: acc += 1
        else: break
    r = np.diff(np.log(c[:jb + 1][~np.isnan(c[:jb + 1])])); x = r[-72:]; vr = np.nan
    if len(x) >= 24 and x.var() > 0: y_ = np.convolve(x, np.ones(6), 'valid'); vr = y_.var() / (6 * x.var())
    cc = c[:jb + 1][~np.isnan(c[:jb + 1])][-25:]; er = abs(cc[-1] - cc[0]) / np.abs(np.diff(cc)).sum() if len(cc) > 2 else np.nan
    pr = ALLD[i - 1] if i else d; prr = (np.nanmax(A[pr][0][1]) - np.nanmin(A[pr][0][2])) if i else np.nan
    return dict(tod=GRID[jb], side=pos, dist=((c[jb] - ub) if pos > 0 else (lb - c[jb])) / s_, vwd=pos * (c[jb] - vw[jb]) / s_, ret=pos * (c[jb] - op) / s_, rngs=(hh - ll) / s_, gap=(op / pc - 1) / sg[-1], rv=FVR[i][jb], rv3=np.nanmean(FVR[i][max(0, jb - 2):jb + 1]), wick=((h[jb] - c[jb]) / rng_ if pos > 0 else (c[jb] - l[jb]) / rng_) if rng_ > 0 else np.nan, acc=acc, pdr=prr / s_, vr=vr, er=er, sinceH=jb - thi, sinceL=jb - tlo)
rows = []
for d in DAYS:
    for (dd_, em, pos, pts, mae) in noise(d):
        f = trade_feats(d, em, pos, pts)
        if f: rows.append(dict(day=d, pts=pts, **f))
G = pd.DataFrame(rows); G['day'] = pd.to_datetime(G.day); FC = [c for c in G.columns if c not in ('day', 'pts')]; print(f'\nM3: {len(G)} noise trades, features {len(FC)}', flush=True)

G.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/G_noise.pkl')
