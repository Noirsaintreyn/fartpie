"""Nov 10 2026: COMPOSITE OF EVERYTHING THAT HAD EDGE, AS A SIZE MULTIPLIER ON THE LIVE INTRADAY LEGS. DECLARED BEFORE RESULTS.
Legs RA, M1, R1, RULES, V15a (live sz, weights UW). Features at entry (all known then): HARf, YZr, VIX, IV, gexp, dexp, vanna, charm, vflow, volume-profile / TPO (gp, dpoc, dtpoc, vaw, pocpos, sp, bal), HMM regime state (filtered 3-state, 0 bear..2 bull), noise-band state WITH (+1) / inside (0) and VWAP distance in trade direction (RA, V15a only; others 0).
FIT on one period only: for each feature the sign of its Spearman correlation with the trade's points (normalised by that leg's std), keep |rho| >= 0.02. Composite = sum of sign x z-score (z from the fit period). Multiplier m = clip(1 + 0.25 x z(composite), 0.5, 1.5) applied to sz. Evaluate on the OTHER period: pts/day (sz x u weighted), std, mean/std, worst month; both directions (fit 2018-21 -> test 2022-26, and fit 2022-26 -> test 2018-21). 200 shuffles of the feature signs give the chance range of the mean/std uplift. Pass = uplift above the shuffle 95th percentile in BOTH directions."""
import sys, io, contextlib, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
a = open('adv_ctx.py').read(); exec(a[:a.index('import lightgbm as lgb')])                       # builds M (day-level context)
from hmmlearn.hmm import GaussianHMM
r = open('adv_regime.py').read(); exec(r[r.index('Dd = pd.DataFrame'):r.index('G = pd.read_pickle')]); HM = states(3)
import nz_core as Z
from nz_core import A, SIG, SI, OPEN, PC
BM = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb'))); OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'RULES': 1.0, 'V15a': .3}
def band(d, m):
    if d not in A or d not in SI: return 0.0, 0.0
    jb = int((m - 570) // 5) - 1; o, h, l, c, v, vw = A[d][0]
    if jb < 1 or jb >= len(c) or np.isnan(c[jb]) or not np.isfinite(SIG[SI[d]][jb]): return 0.0, 0.0
    sg = SIG[SI[d]]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; ub, lb = max(op, pc) * (1 + sg[jb]), min(op, pc) * (1 - sg[jb])
    return (1.0 if c[jb] > ub else -1.0 if c[jb] < lb else 0.0), (c[jb] - vw[jb]) / (op * sg[-1])
rows = []
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n not in UW: continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); B = BM[n].copy(); B['day'] = pd.to_datetime(B.day).dt.normalize(); B = B[B.sz > 0][['day', 'm', 'sz']].rename(columns={'sz': 'szb'}); t = t.merge(B, on=['day', 'm']); t['sz'] = t.szb
    t = t[(t.day >= '2018-02-01') & (t.day <= '2026-01-30')].reset_index(drop=True); t['leg'] = n; t['u'] = UW[n]
    if 'dir' in t.columns:
        bs = [band(d, m) for d, m in zip(t.day, t.m)]; t['bandw'] = [b[0] * dr for b, dr in zip(bs, t.dir)] ; t['bandw'] = (t.bandw > 0).astype(float); t['vwdd'] = [b[1] * dr for b, dr in zip(bs, t.dir)]
    else: t['bandw'] = 0.0; t['vwdd'] = 0.0
    rows.append(t[['day', 'm', 'pts', 'sz', 'leg', 'u', 'bandw', 'vwdd']])
T = pd.concat(rows).reset_index(drop=True); T['hmm'] = T.day.map(HM); T = T.join(M, on='day')
FE = [c for c in T.columns if c not in ('day', 'm', 'pts', 'sz', 'leg', 'u')]; T = T.dropna(subset=['hmm']).reset_index(drop=True); T[FE] = T[FE].apply(pd.to_numeric, errors='coerce'); print(len(T), 'trades; features', len(FE), flush=True)
T['npts'] = T.pts / T.groupby('leg').pts.transform('std'); P1 = (T.day < '2022-01-01').values; P2 = ~P1
def build(fit, signs=None, rng=None):
    mu, sd = T.loc[fit, FE].mean(), T.loc[fit, FE].std().replace(0, np.nan); Z_ = ((T[FE] - mu) / sd).fillna(0.0).clip(-3, 3)
    if signs is None: signs = pd.Series({c: (np.sign(T.loc[fit, [c, 'npts']].corr(method='spearman').iloc[0, 1]) if abs(T.loc[fit, [c, 'npts']].corr(method='spearman').iloc[0, 1]) >= .02 else 0.0) for c in FE})
    comp = (Z_ * signs).sum(axis=1); cf = comp[fit]; z = (comp - cf.mean()) / cf.std(); return np.clip(1 + .25 * z, .5, 1.5), signs
def score(mult, test):
    x = T[test]; w = x.sz * x.u * (mult[test] if mult is not None else 1.0); s = (x.pts * w).groupby(x.day).sum(); days = pd.DatetimeIndex(sorted(set(x.day))); s = s.reindex(days).fillna(0)
    mo = s.groupby(s.index.to_period('M')).sum(); return s.mean(), s.std(), s.mean() / s.std(), mo.min()
rng = np.random.default_rng(5)
for lab, fit, test in (('fit 2018-21 -> test 2022-26', P1, P2), ('fit 2022-26 -> test 2018-21', P2, P1)):
    mult, signs = build(fit); b = score(None, test); c = score(mult, test)
    print(f'\\n{lab}: kept {int((signs != 0).sum())}/{len(FE)} features; positive: ' + ', '.join(signs[signs > 0].index) + ' | negative: ' + ', '.join(signs[signs < 0].index))
    print(f'  base      pts/d {b[0]:+6.2f} std {b[1]:6.1f} m/s {b[2]:.3f} worst month {b[3]:+6.0f}'); print(f'  composite pts/d {c[0]:+6.2f} std {c[1]:6.1f} m/s {c[2]:.3f} worst month {c[3]:+6.0f}')
    null = []
    for _ in range(200):
        sh = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=len(FE)) , index=FE); mm, _s = build(fit, signs=sh); null.append(score(mm, test)[2])
    print(f'  m/s uplift {c[2] - b[2]:+.3f}; shuffled-sign m/s 95th pct {np.percentile(null, 95) - b[2]:+.3f} (median {np.median(null) - b[2]:+.3f})' + ('   <== above chance' if c[2] > np.percentile(null, 95) else ''), flush=True)
