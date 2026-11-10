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
rs = np.random.default_rng(9)
for K in (4, 6, 8):
    P, D, NZ_, RND = [], [], [], []
    for Yr in range(2018, 2027):
        tr = yr < Yr; te = yr == Yr
        if te.sum() == 0: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; g = GaussianMixture(K, covariance_type='diag', random_state=1, n_init=2).fit((X[tr] - mu) / sd)
        ctr = g.predict((X[tr] - mu) / sd); cte = g.predict((X[te] - mu) / sd); sign = np.zeros(K)
        for k in range(K):
            yk = Y[tr][ctr == k]
            if len(yk) >= 60 and yk.std() > 0 and abs(yk.mean() / (yk.std() / np.sqrt(len(yk)))) >= 1.5: sign[k] = np.sign(yk.mean())
        P += list(sign[cte] * Y[te] - C * (sign[cte] != 0)); D += list(DAY[te]); NZ_.append((cte, te, sign))
    P = np.array(P); D = pd.DatetimeIndex(D); m = P != -C * 0; take = np.array([abs(p) >= 0 for p in P])
    keep = np.concatenate([(s_[c_] != 0) for c_, te, s_ in NZ_]); base = ev(P[keep], D[keep], f'GMM K={K} trained cluster signs', f' | trades {int(keep.sum())}')
    rnd = []
    for _ in range(100):
        pp = []; dd = []
        for c_, te, s_ in NZ_:
            sgn = s_ * 0; 
            for k in np.where(s_ != 0)[0]: sgn[k] = rs.choice([-1.0, 1.0])
            pp += list(sgn[c_] * Y[te] - C * (sgn[c_] != 0)); dd += list(DAY[te])
        pp = np.array(pp); dd = pd.DatetimeIndex(dd); ss = pd.Series(pp, index=dd).groupby(level=0).sum().reindex(ALLDAYS).fillna(0.0); rnd.append(ss[ALLDAYS >= '2022-01-01'].mean())
    print(f'      random-sign 22-26 mean pts/day: median {np.median(rnd):+.2f}, 95th pct {np.percentile(rnd, 95):+.2f}', flush=True)
# ---- B: VVG
rows = []
for d in DAYS:
    if d not in DI or d < pd.Timestamp('2014-04-01'): continue
    i = DI[d]; o, h, l, c, v, vw = A[d][0]; pc = PC[d]
    if i < 25 or not np.isfinite(pc) or np.isnan(c[5]) or np.isnan(o[6]) or np.isnan(c[NB - 1]): continue
    v20 = np.nanmean([A[ALLD[q]][0][4][0] for q in range(i - 20, i)])
    rows.append(dict(day=d, gap=abs(o[0] - pc), r30=abs(c[5] - o[0]), dir30=np.sign(c[5] - o[0]), vol=v[0] / v20, ent=o[6], cl=c[NB - 1], pk=np.nanmax(h[6:]) if np.sign(c[5] - o[0]) >= 0 else np.nanmin(l[6:]), full=(c[NB - 1] - o[0])))
V = pd.DataFrame(rows).set_index('day')
for cc in ('gap', 'r30', 'vol'): V[cc + '_q'] = V[cc].expanding(min_periods=250).quantile(.75).shift(1)
V['flag'] = (V.gap > V.gap_q) & (V.r30 > V.r30_q) & (V.vol > V.vol_q); F = V[V.flag]; print(f'\n(B) VVG: flagged {int(V.flag.sum())} of {len(V)} days ({V.flag.mean():.1%}); per year ' + ' '.join(f'{y}:{n}' for y, n in F.groupby(F.index.year).size().items()))
rev = ((F.dir30 > 0) & (F.cl < F.pk)) | ((F.dir30 < 0) & (F.cl > F.pk)); give = np.where(F.dir30 > 0, F.pk - F.cl, F.cl - F.pk); print(f'  reversal-from-peak frequency {np.mean(F.cl != F.pk):.1%}; mean peak-to-close giveback {np.mean(give):.1f} pts')
pts = pd.Series(-F.dir30 * (F.cl - F.ent) - C, index=F.index); pts = pts.reindex(ALLDAYS).fillna(0.0)
for nm, f in (('14-21', pts.index < '2022-01-01'), ('22-26', pts.index >= '2022-01-01')): print(f'  fade-30-min trade {nm}: {pts[f].mean():+.2f} pts/day (trades {int((pts[f] != 0).sum())}, mean per trade {pts[f][pts[f] != 0].mean():+.1f})')
