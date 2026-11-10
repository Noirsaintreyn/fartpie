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
def sample_days():
    X, y, meta = [], [], []
    for d in DAYS:
        if d not in DI: continue
        o, h, l, c, v, vw = A[d][0]; i = DI[d]; sg = SIG[SI[d]][-1]
        if not np.isfinite(sg): continue
        sc = OPEN[d] * sg
        for jb in range(W + 12 - 1, NB - 8):
            t = GRID[jb] + 5
            if t % 30 or t < 690 or t > 900: continue
            lo = jb - W + 1; seg = slice(lo, jb + 1)
            if np.isnan(c[seg]).any() or np.isnan(o[jb + 1]) or jb + 7 >= NB or np.isnan(o[jb + 7]): continue
            r = np.diff(c[lo - 1:jb + 1]) / sc; vr = np.clip(np.nan_to_num(FVR[i][seg], nan=1.0) - 1, -2, 5); rg = (h[seg] - l[seg]) / sc; vd = (c[seg] - vw[seg]) / sc
            X.append(np.stack([r, vr, rg, vd])); y.append(o[jb + 7] - o[jb + 1]); meta.append((d, jb))
    return np.array(X), np.array(y), meta
def kernels(rng, n=300):
    K = []
    for _ in range(n):
        L = int(rng.choice([5, 7, 9])); dil = int(rng.choice([1, 2])); w = rng.normal(size=(4, L)); w -= w.mean(axis=1, keepdims=True); K.append((w, dil, rng.uniform(-.5, .5)))
    return K
def rocket(X, K):
    out = np.zeros((len(X), 2 * len(K)), np.float32)
    for q, (w, dil, b) in enumerate(K):
        L = w.shape[1]; span = (L - 1) * dil + 1; P = X.shape[2] - span + 1
        conv = sum(np.einsum('nct,c->nt', np.stack([X[:, :, p + np.arange(0, span, dil)] for p in range(P)], axis=2)[:, :, :, 0] if False else np.stack([X[:, :, p:p + span:dil] for p in range(P)], axis=2)[:, :, :, l], w[:, l]) for l in range(L)) if False else None
        win = np.stack([X[:, :, p:p + span:dil] for p in range(P)], axis=2)                 # N, C, P, L
        conv = np.einsum('ncpl,cl->np', win, w) + b; out[:, 2 * q] = (conv > 0).mean(axis=1); out[:, 2 * q + 1] = conv.max(axis=1)
    return out
def signature(X):
    N = len(X); price = np.cumsum(X[:, 0, :], axis=1); vol = np.cumsum(X[:, 1, :], axis=1) / 10; tm = np.tile(np.linspace(0, 1, X.shape[2]), (N, 1)); P = np.stack([tm, price, vol], axis=2)            # N, W, 3
    S1 = np.zeros((N, 3)); S2 = np.zeros((N, 3, 3)); S3 = np.zeros((N, 3, 3, 3))
    for k in range(1, P.shape[1]):
        dx = P[:, k] - P[:, k - 1]; d2 = np.einsum('ni,nj->nij', dx, dx) / 2; d3 = np.einsum('nij,nk->nijk', d2, dx) / 3
        S3 = S3 + np.einsum('nij,nk->nijk', S2, dx) + np.einsum('ni,njk->nijk', S1, d2) + d3; S2 = S2 + np.einsum('ni,nj->nij', S1, dx) + d2; S1 = S1 + dx
    return np.concatenate([S1, S2.reshape(N, -1), S3.reshape(N, -1)], axis=1)
def evaluate(label, pts, days, show=True):
    s = pd.Series(pts, index=pd.DatetimeIndex(days)).groupby(level=0).sum().reindex(pd.DatetimeIndex(D18)).fillna(0.0); res = []; ok = True
    for nm, f in (('18-21', lambda x: x < '2022-01-01'), ('22-26', lambda x: x >= '2022-01-01')):
        x = s[f(s.index)]; t = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else 0; res.append((x.mean(), t, x.std())); ok &= x.mean() > 0 and t >= 2
    print(f'  {label:<46} ' + ' | '.join(f'{n}: {m:+6.2f}/d (t {t:+5.2f}, std {sd:5.1f}, m/s {m / sd if sd else 0:.3f})' for n, (m, t, sd) in zip(('18-21', '22-26'), res)) + ('   <== PASS' if ok else ''), flush=True)
    return s
print('building samples...', flush=True); X, y, meta = sample_days(); days = np.array([m[0] for m in meta]); yr = np.array([d.year for d in days]); print(f'{len(y)} decision samples, label sign balance {np.mean(y > 0):.3f}', flush=True)
def ctx(meta):
    C = []
    for d, jb in meta:
        o, h, l, c, v, vw = A[d][0]; i = DI[d]; sg = SIG[SI[d]]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; s_ = op * sg[-1]
        ub, lb = max(op, pc) * (1 + sg[jb]), min(op, pc) * (1 - sg[jb]); pr = ALLD[i - 1]; prr = np.nanmax(A[pr][0][1]) - np.nanmin(A[pr][0][2])
        C.append([GRID[jb] / 390, (c[jb] - op) / s_, (c[jb] - vw[jb]) / s_, (c[jb] - ub) / s_, (c[jb] - lb) / s_, (op - pc) / s_, prr / s_, np.nanmax(h[:jb + 1]) - np.nanmin(l[:jb + 1]) and (np.nanmax(h[:jb + 1]) - np.nanmin(l[:jb + 1])) / s_, FVR[i][jb], float(c[jb] > ub), float(c[jb] < lb)])
    return np.nan_to_num(np.array(C, float), nan=0.0, posinf=0, neginf=0)
rng = np.random.default_rng(3); K = kernels(rng); FR = np.nan_to_num(rocket(X, K)); FS = np.nan_to_num(signature(X)); CX = ctx(meta); print('features built', flush=True)
FR2 = np.hstack([FR, np.repeat(CX, 5, axis=1)]); FS2 = np.hstack([FS, np.repeat(CX, 5, axis=1)])
for nm, F in (('M1 ROCKET', FR), ('M1+CTX ROCKET fed known context', FR2), ('M2 signature', FS), ('M2+CTX signature fed context', FS2), ('CTX only (ridge)', CX)):
    pts = []; dd = []
    for Y in range(2018, 2027):
        tr = yr < Y; te = yr == Y
        if te.sum() == 0 or tr.sum() < 3000: continue
        sc = StandardScaler().fit(F[tr]); m = RidgeClassifier(alpha=1000.0 if F.shape[1] > 100 else 10.0).fit(sc.transform(F[tr]), y[tr] > 0)
        st_tr = np.abs(m.decision_function(sc.transform(F[tr]))); thr = np.quantile(st_tr, .9); s_te = m.decision_function(sc.transform(F[te]))
        sel = np.abs(s_te) >= thr; dr = np.sign(s_te[sel]); pts += list(dr * y[te][sel] - COST); dd += list(days[te][sel])
    evaluate(nm + ': top 10% score', np.array(pts), dd)
    # sensitivity: all decisions (no threshold) and top 25%
    for q, lab in ((0.0, 'every decision'), (.75, 'top 25%')):
        pts = []; dd = []
        for Y in range(2018, 2027):
            tr = yr < Y; te = yr == Y
            if te.sum() == 0 or tr.sum() < 3000: continue
            sc = StandardScaler().fit(F[tr]); m = RidgeClassifier(alpha=1000.0 if F.shape[1] > 100 else 10.0).fit(sc.transform(F[tr]), y[tr] > 0); thr = np.quantile(np.abs(m.decision_function(sc.transform(F[tr]))), q); s_te = m.decision_function(sc.transform(F[te])); sel = np.abs(s_te) >= thr
            pts += list(np.sign(s_te[sel]) * y[te][sel] - COST); dd += list(days[te][sel])
        evaluate(nm + ': ' + lab, np.array(pts), dd)
# ---- M3 GBM meta-label on the noise engine
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
keep_med = np.zeros(len(G), bool); keep_top3 = np.zeros(len(G), bool); yrG = G.day.dt.year.values
for Y in range(2018, 2027):
    tr = yrG < Y; te = yrG == Y
    if te.sum() == 0 or tr.sum() < 400: continue
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=8, min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=5.0, verbose=-1).fit(G.loc[tr, FC], G.pts[tr] > 0)
    ptr = mdl.predict_proba(G.loc[tr, FC])[:, 1]; pte = mdl.predict_proba(G.loc[te, FC])[:, 1]; keep_med[te] = pte >= np.median(ptr); keep_top3[te] = pte >= np.quantile(ptr, 2 / 3)
G18 = G[G.day >= '2018-02-01']; k18 = np.where(G.day >= '2018-02-01')[0]
evaluate('noise engine, all trades (base)', G18.pts.values, G18.day.values)
evaluate('M3 keep P(win) >= training median', G.pts.values[keep_med & (G.day >= '2018-02-01').values], G.day.values[keep_med & (G.day >= '2018-02-01').values])
evaluate('M3 keep top third', G.pts.values[keep_top3 & (G.day >= '2018-02-01').values], G.day.values[keep_top3 & (G.day >= '2018-02-01').values])
evaluate('M3 DROP the bottom third (keep top two thirds)', G.pts.values[(G.day >= '2018-02-01').values & ~(~keep_med & ~keep_top3 & (G.day >= '2018-02-01').values)], G.day.values[(G.day >= '2018-02-01').values & ~(~keep_med & ~keep_top3 & (G.day >= '2018-02-01').values)])
# ---- M4 HMM
from hmmlearn.hmm import GaussianHMM
Dn = pd.DataFrame(index=pd.DatetimeIndex(ALLD)); rg = pd.Series({d: np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2]) for d in ALLD}); Dn['lr'] = np.log(rg / pd.Series(OPEN)).shift(1).values; Dn['ret'] = np.log(pd.Series(CLOSE) / pd.Series(OPEN)).shift(1).values
Dn['gap'] = (pd.Series(OPEN) / pd.Series(PC) - 1).values; Dn['lr5'] = Dn.lr.rolling(5).mean().values; Dn = Dn.dropna(); Dn = Dn[Dn.index >= '2014-04-01']
NZs = G.groupby('day').pts.sum().reindex(Dn.index).fillna(0.0); st = pd.Series(np.nan, index=Dn.index)
for Y in range(2018, 2027):
    tr = Dn.index < f'{Y}-01-01'; te = Dn.index.year == Y
    if te.sum() == 0: continue
    mu, sd = Dn[tr].mean(), Dn[tr].std(); Xtr = ((Dn[tr] - mu) / sd).values; h = GaussianHMM(3, covariance_type='full', n_iter=100, random_state=1).fit(Xtr)
    order = np.argsort(h.means_[:, 0]); mp = {k: i for i, k in enumerate(order)}                  # 0 = calmest .. 2 = most volatile
    Xte = ((Dn[te] - mu) / sd).values; fp = np.zeros((len(Xte), 3)); alpha = h.startprob_.copy()
    for k in range(len(Xte)):                                                                    # filtered (forward-only) probabilities
        lik = np.exp(h._compute_log_likelihood(Xte[k:k + 1])[0]); alpha = (alpha @ h.transmat_ if k else alpha) * lik; alpha = alpha / alpha.sum(); fp[k] = alpha
    st[te] = [mp[a] for a in fp.argmax(axis=1)]
print('\nM4 HMM filtered state (0 calm .. 2 volatile) at 09:30: noise engine pts/day by state, 2018-21 | 2022-26')
for nm, f in (('18-21', lambda x: x < '2022-01-01'), ('22-26', lambda x: x >= '2022-01-01')):
    m_ = f(Dn.index) & st.notna().values; print(f'  {nm}: ' + ' | '.join(f'state {k}: {NZs[m_ & (st == k).values].mean():+6.2f} (n{int((m_ & (st == k).values).sum())})' for k in range(3)))
