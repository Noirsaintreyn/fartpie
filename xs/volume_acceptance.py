"""Nov 10 2026: VOLUME / ACCEPTANCE vs REJECTION: does volume stepping in, and price ACCEPTING beyond the noise band (or beyond the prior-day value area), separate good breakouts from failed ones? DECLARED BEFORE RESULTS.
Data: ET 5m NQ (nz_core), engine = noise-area momentum (k 1.0, 30m decisions, stop 40), cost 0.5, 2014-02..2026-01-30, periods 2014-21 | 2022-26. All features are read at the SIGNAL bar (the last completed bar before the entry open), nothing later.
FEATURES (per trade, in the trade direction):
  rv      candle volume: signal-bar volume / mean volume of the same time of day over the prior 14 sessions.      rv3  mean of that ratio over the last 3 bars (volume stepping IN).
  wick    rejection wick on the signal bar: share of the bar's range that lies beyond the close against the trade (0 = closed on the extreme, 1 = closed on the opposite end).
  acc     acceptance: consecutive 5m closes beyond the band ending at the signal bar (1 = just broke).     vb  share of the session's volume so far that traded with a close beyond the band.
  vae     acceptance beyond the PRIOR-DAY value area: the signal close is beyond the prior day's volume-profile VAH (long) / VAL (short), 2018+ only through the profile from xs/noise_context.
TEST 1 pts per trade by third of each feature (cuts from 2014-21), spread top minus bottom third, PASS = same sign and |t| >= 2 in BOTH periods.   TEST 2 the filtered engine: keep only trades that pass the rule (acc >= 2 closes, rv >= 1.0, ...): pts/day, std, mean/std, trades kept.
TEST 3 NEW ENGINE 'failure to accept': a close beyond the band on volume (rv >= 1.3) followed within 3 bars by a close back inside = rejection; fade at the next open (target = VWAP, stop = the break extreme + 5 pts, flat 15:55): pts/day both periods.
TEST 4 the vector trades (Rule A, V15a, live tables with the 30-minute-move skip): rv / wick of the signal bar (the bar before the entry) by third."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
import nz_core as Z
from nz_core import A, OPEN, PC, SIG, SI, ALLD, DAYS, GRID, NB, noise, COST
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
V = pd.DataFrame({d: A[d][0][4] for d in ALLD}).T; VM = V.rolling(14, min_periods=10).mean().shift(1); RV = (V / VM).values; DI = {d: i for i, d in enumerate(ALLD)}
def bands(d, k=1.0):
    s = SIG[SI[d]] * k; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; return max(op, pc) * (1 + s), min(op, pc) * (1 - s)
def feats(d, jb, dr):
    o, h, l, c, v, vw = A[d][0]; i = DI[d]; ub, lb = bands(d)
    if jb < 0 or np.isnan(c[jb]) or not np.isfinite(ub[jb]): return None
    rv = RV[i][jb]; rv3 = np.nanmean(RV[i][max(0, jb - 2):jb + 1]); rng = h[jb] - l[jb]
    wick = ((h[jb] - c[jb]) / rng if dr > 0 else (c[jb] - l[jb]) / rng) if rng > 0 else np.nan
    beyond = (c > ub) if dr > 0 else (c < lb); beyond = np.nan_to_num(beyond.astype(float)); acc = 0
    for q in range(jb, -1, -1):
        if beyond[q]: acc += 1
        else: break
    tot = np.nansum(v[:jb + 1]); vb = np.nansum(v[:jb + 1] * beyond[:jb + 1]) / tot if tot > 0 else np.nan
    return dict(rv=rv, rv3=rv3, wick=wick, acc=acc, vb=vb)
rows = []
for d in DAYS:
    for (dd, em, pos, pts, mae) in noise(d):
        f = feats(d, em - 1, pos)
        if f: rows.append(dict(day=d, dir=pos, pts=pts, m=GRID[em], **f))
T = pd.DataFrame(rows); T['day'] = pd.to_datetime(T.day)
# prior-day value area acceptance (profile from noise_context, 2018+)
import importlib, io, contextlib
with contextlib.redirect_stdout(io.StringIO()): import noise_context as C
def vae(d, jb, dr):
    i = ALLD.index(d)
    if i == 0 or ALLD[i - 1] not in C.PR: return np.nan
    p = C.PR[ALLD[i - 1]]; cl = A[d][0][3][jb]; return float(cl > p['vah'] if dr > 0 else cl < p['val'])
T['vae'] = [vae(r.day, int(r.m // 5 - 114 - 1), r.dir) if r.day >= pd.Timestamp('2018-02-01') else np.nan for r in T.itertuples()]
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
PER = (('14-21', lambda x: x < '2022-01-01'), ('22-26', lambda x: x >= '2022-01-01'))
print(f'noise trades with features: {len(T)}; pts/trade 14-21 {T[T.day < "2022-01-01"].pts.mean():+.2f} | 22-26 {T[T.day >= "2022-01-01"].pts.mean():+.2f}\n')
print('=== TEST 1: pts per trade by third of each feature (bottom | middle | top), 14-21 // 22-26; spread top-bottom, t ===')
CUT = {c: T[T.day < '2022-01-01'][c].quantile([1 / 3, 2 / 3]).values for c in ('rv', 'rv3', 'wick', 'acc', 'vb')}
def table(T_, label, cols):
    for c in cols:
        q = CUT[c]; b = np.where(T_[c] <= q[0], 0, np.where(T_[c] > q[1], 2, 1)); out = []; sp = []
        for lab, f in PER:
            m = f(T_.day).values; g = [T_.pts[m & (b == k)] for k in (0, 1, 2)]; out.append(' | '.join(f'{x.mean():+6.1f}' for x in g)); sp.append((g[2].mean() - g[0].mean(), tt(g[2], g[0])))
        ok = np.sign(sp[0][0]) == np.sign(sp[1][0]) and all(np.isfinite(s[1]) and abs(s[1]) >= 2 for s in sp)
        print(f'  {label:<8}{c:<5} {out[0]} // {out[1]} | top-bottom {sp[0][0]:+6.1f} (t {sp[0][1]:+5.2f}) // {sp[1][0]:+6.1f} (t {sp[1][1]:+5.2f}){"   <== PASS" if ok else ""}', flush=True)
table(T, 'all', ('rv', 'rv3', 'wick', 'acc', 'vb'))
table(T[T.dir > 0], 'long', ('rv', 'rv3', 'wick', 'acc', 'vb')); table(T[T.dir < 0], 'short', ('rv', 'rv3', 'wick', 'acc', 'vb'))
for lab, f in PER:
    x = T[f(T.day) & T.vae.notna()]; a, b = x[x.vae == 1].pts, x[x.vae == 0].pts
    print(f'  vae (beyond prior-day value area) {lab}: accepted {a.mean():+6.1f} (n{len(a)}) vs inside {b.mean():+6.1f} (n{len(b)}), t {tt(a, b):+5.2f}')
print('\n=== TEST 2: the filtered noise engine (pts per day over all days; trades kept) ===')
D = pd.DatetimeIndex(DAYS)
def eng(mask, label):
    s = T[mask].groupby('day').pts.sum().reindex(D).fillna(0.0); o = []
    for lab, f in PER:
        x = s[f(s.index)]; o.append(f'{lab}: {x.mean():+5.2f}/d std {x.std():5.1f} m/s {x.mean() / x.std():.3f} (t {x.mean() / (x.std() / np.sqrt(len(x))):+4.2f})')
    print(f'  {label:<34} kept {mask.mean():4.0%} | ' + ' | '.join(o), flush=True)
eng(T.pts == T.pts, 'all trades (base)'); eng(T.acc >= 2, 'accepted: >= 2 closes beyond'); eng(T.acc >= 3, 'accepted: >= 3 closes beyond'); eng(T.rv >= 1.0, 'rv >= 1.0'); eng(T.rv >= 1.3, 'rv >= 1.3'); eng(T.rv3 >= 1.0, 'rv3 >= 1.0 (volume stepping in)')
eng(T.wick <= .25, 'closed near the extreme (wick <= .25)'); eng((T.rv >= 1.0) & (T.wick <= .25), 'rv >= 1 and wick <= .25'); eng((T.acc >= 2) & (T.rv3 >= 1.0), 'acc >= 2 and rv3 >= 1'); eng(T.vb >= CUT['vb'][1], 'top third volume-beyond share')
eng(T.rv < 1.0, 'rv < 1.0 (thin breaks, for reference)'); eng(T.acc == 1, 'first-bar break only (for reference)')
print('\n=== TEST 3: failure to accept (rejection) fade engine ===')
def fta(d, rvmin=1.3, within=3):
    o, h, l, c, v, vw = A[d][0]; i = DI[d]; ub, lb = bands(d); res = []; used = False
    for j in range(8, NB - 8):
        if used or np.isnan(c[j]) or not np.isfinite(ub[j]): continue
        for dr, brk, back in ((1, c[j] > ub[j], lambda q: c[q] < ub[q]), (-1, c[j] < lb[j], lambda q: c[q] > lb[q])):
            if not brk or not np.isfinite(RV[i][j]) or RV[i][j] < rvmin: continue
            for q in range(j + 1, min(j + within + 1, NB - 2)):
                if np.isnan(c[q]) or np.isnan(o[q + 1]): continue
                if back(q):
                    e = o[q + 1]; ext = np.nanmax(h[j:q + 1]) if dr > 0 else np.nanmin(l[j:q + 1]); stp = ext + dr * 5.0; px = None
                    for r in range(q + 1, NB):
                        if np.isnan(c[r]): continue
                        if (dr > 0 and h[r] >= stp) or (dr < 0 and l[r] <= stp): px = stp; break
                        if (dr > 0 and c[r] <= vw[r]) or (dr < 0 and c[r] >= vw[r]): px = c[r]; break
                    if px is None: px = c[~np.isnan(c)][-1]
                    res.append((d, -dr, (px - e) * (-dr) - COST)); used = True; break
            if used: break
    return res
for rvmin, within in ((1.3, 3), (1.0, 3), (1.6, 3), (1.3, 6)):
    R = pd.DataFrame([r for d in DAYS for r in fta(d, rvmin, within)], columns=['day', 'dir', 'pts']); R['day'] = pd.to_datetime(R.day); s = R.groupby('day').pts.sum().reindex(D).fillna(0.0); o = []
    for lab, f in PER:
        x = s[f(s.index)]; o.append(f'{lab}: {x.mean():+5.2f}/d (t {x.mean() / (x.std() / np.sqrt(len(x))):+5.2f})')
    print(f'  rv >= {rvmin}, back inside within {within} bars: n {len(R)} | ' + ' | '.join(o) + f' | per trade {R.pts.mean():+.1f}', flush=True)
print('\n=== TEST 4: vector trades (Rule A, V15a): signal-bar volume and wick by third ===')
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); BM = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb')))
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n not in ('RA', 'V15a'): continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); B_ = BM[n].copy(); B_['day'] = pd.to_datetime(B_.day).dt.normalize(); B_ = B_[B_.sz > 0][['day', 'm']]; t = t.merge(B_, on=['day', 'm']); t = t[(t.day >= '2014-02-01') & (t.day <= '2026-01-30')]
    fr = []
    for r in t.itertuples():
        jb = int((r.m - 570) // 5) - 1; f = feats(r.day, jb, int(r.dir)) if r.day in DI else None
        fr.append(f if f else dict(rv=np.nan, rv3=np.nan, wick=np.nan, acc=np.nan, vb=np.nan))
    t = pd.concat([t.reset_index(drop=True), pd.DataFrame(fr)], axis=1); t = t[t.rv.notna()]
    cut = {c: t[t.day < '2022-01-01'][c].quantile([1 / 3, 2 / 3]).values for c in ('rv', 'rv3', 'wick')}
    for c in ('rv', 'rv3', 'wick'):
        q = cut[c]; b = np.where(t[c] <= q[0], 0, np.where(t[c] > q[1], 2, 1)); out = []; sp = []
        for lab, f in PER:
            m = f(t.day).values; g = [t.pts[m & (b == k)] for k in (0, 1, 2)]; out.append(' | '.join(f'{x.mean():+6.1f} (n{len(x)})' for x in g)); sp.append((g[2].mean() - g[0].mean(), tt(g[2], g[0])))
        ok = np.sign(sp[0][0]) == np.sign(sp[1][0]) and all(np.isfinite(s[1]) and abs(s[1]) >= 2 for s in sp)
        print(f'  {n:<5}{c:<5} {out[0]} // {out[1]} | top-bottom t {sp[0][1]:+5.2f} // {sp[1][1]:+5.2f}{"   <== PASS" if ok else ""}', flush=True)
