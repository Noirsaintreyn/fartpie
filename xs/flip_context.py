"""Nov 10 2026: FLIP CONTEXT - time of day, where the flip sits against the prior daily / weekly / monthly / 52-week VOLUME PROFILE, volatility at the flip, flip-bar volume. Find the BAD pockets (a floor) and test cutting them. DECLARED BEFORE RESULTS.
Trades: the live S6 V2 trades with the pipeline mv30 skip (sz > 0), engines RA, M1, R1, RULES, V15a (RA / V15a carry a direction; the others do not, so direction-dependent features run on those two only). ET 5m NQ (nz_core). Signal bar = last completed 5m bar before the entry.
Volume profiles (RTH bars, volume placed at each bar's typical price, relative bin width 0.04% of price): DAILY = prior session; WEEKLY = prior full calendar week; MONTHLY = prior full calendar month; 52W = the last 252 sessions to the prior close.
Each gives POC and the 70% value area (VAL, VAH).  Features:
  loc_X     where the signal close sits: above VAH / inside / below VAL, for X in d, w, m, y.   dir_X (RA, V15a): BEYOND = long above VAH or short below VAL (accepting new value in the trade direction), INSIDE, AGAINST.
  dpoc_X    |signal close - POC| / ADR20 (how far from the fair price).     nstack  number of the four VAs the close is beyond in the trade direction (0..4).
  volr      realised vol of the last 12 5m returns / its mean at the same time of day over the prior 14 sessions (vol at the flip).     fv  flip-bar volume / same-time 14-day mean.    fv3 the last 3 bars.
  tod       entry time: 09:35-10:29 / 10:30-11:59 / 12:00-13:59 / 14:00+.    side (RA, V15a) long / short.
TEST 1 for every categorical bucket (vs the rest of that engine's trades) and every third of each continuous feature (thirds cut on 2014-21 / 2018-21 for RA): same sign and |t| >= 2 in BOTH periods and n >= 30 per period = PASS.
NULL: the same tests on the engine's points shuffled 200 times (within period), to report how many PASS flags appear by chance - read the real list against that.
TEST 2 (only for real PASS flags that are a BAD pocket, negative in both periods): cut that bucket - engine points per trade, trades cut, system weighted points / day, daily std, mean / std, worst month, both periods."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
import nz_core as Z
from nz_core import A, ALLD, DAYS, GRID, NB
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; DI = {d: i for i, d in enumerate(ALLD)}
TP = {d: (A[d][0][1] + A[d][0][2] + A[d][0][3]) / 3 for d in ALLD}; VO = {d: np.nan_to_num(A[d][0][4]) for d in ALLD}
def profile(days):
    tp = np.concatenate([TP[d] for d in days]); v = np.concatenate([VO[d] for d in days]); ok = ~np.isnan(tp) & (v > 0)
    if ok.sum() < 60: return None
    tp, v = tp[ok], v[ok]; bw = max(.25, round(.0004 * np.median(tp) / .25) * .25); lo = tp.min(); idx = ((tp - lo) // bw).astype(int); h = np.bincount(idx, weights=v); poc = int(np.argmax(h)); a = b = poc; acc = h[poc]; tot = h.sum()
    while acc < .7 * tot and (a > 0 or b < len(h) - 1):
        up = h[b + 1] if b < len(h) - 1 else -1; dn = h[a - 1] if a > 0 else -1
        if up >= dn: b += 1; acc += max(up, 0)
        else: a -= 1; acc += max(dn, 0)
    px = lambda i: lo + (i + .5) * bw; return px(poc), px(b), px(a)
wk = pd.Series({d: d.isocalendar()[:2] for d in ALLD}); mo = pd.Series({d: (d.year, d.month) for d in ALLD}); LV = {}
for i, d in enumerate(ALLD):
    if i < 260: continue
    prev = ALLD[i - 1]; LV[d] = dict(d=profile([prev]), y=profile(ALLD[i - 252:i]))
# weekly / monthly: prior full week / month
bywk = {}; bymo = {}
for d in ALLD: bywk.setdefault(wk[d], []).append(d); bymo.setdefault(mo[d], []).append(d)
wkeys = sorted(bywk); mkeys = sorted(bymo)
for i, d in enumerate(ALLD):
    if d not in LV: continue
    kw = wkeys.index(wk[d]); km = mkeys.index(mo[d])
    LV[d]['w'] = profile(bywk[wkeys[kw - 1]]) if kw > 0 else None; LV[d]['m'] = profile(bymo[mkeys[km - 1]]) if km > 0 else None
RNG = pd.Series({d: np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2]) for d in ALLD}); ADR = RNG.shift(1).rolling(20).mean()
V_ = pd.DataFrame({d: A[d][0][4] for d in ALLD}).T; FVR = (V_ / V_.rolling(14, min_periods=10).mean().shift(1)).values
RV12 = np.full((len(ALLD), NB), np.nan)
for i, d in enumerate(ALLD):
    c = A[d][0][3]; r = np.diff(np.log(c), prepend=np.nan)
    for j in range(12, NB):
        w_ = r[j - 11:j + 1]; RV12[i, j] = np.nanstd(w_) if np.isfinite(w_).sum() >= 8 else np.nan
VOLR = RV12 / pd.DataFrame(RV12).rolling(14, min_periods=10).mean().shift(1).values
def feat(d, m, dr):
    jb = int((m - 570) // 5) - 1
    if d not in LV or jb < 0 or d not in DI: return None
    i = DI[d]; cl = A[d][0][3][jb]
    if np.isnan(cl): return None
    f = dict(volr=VOLR[i, jb], fv=FVR[i, jb], fv3=np.nanmean(FVR[i, max(0, jb - 2):jb + 1]), tod=0 if m < 630 else 1 if m < 720 else 2 if m < 840 else 3); st = 0
    for X in 'dwmy':
        p = LV[d].get(X)
        if p is None: f['loc_' + X] = None; f['dir_' + X] = None; f['dpoc_' + X] = np.nan; continue
        poc, vah, val = p; loc = 'above' if cl > vah else 'below' if cl < val else 'inside'; f['loc_' + X] = loc; f['dpoc_' + X] = abs(cl - poc) / ADR[d]
        if dr != 0:
            beyond = (cl > vah) if dr > 0 else (cl < val); against = (cl < val) if dr > 0 else (cl > vah); f['dir_' + X] = 'beyond' if beyond else 'against' if against else 'inside'; st += int(beyond)
        else: f['dir_' + X] = None
    f['nstack'] = st if dr != 0 else np.nan; return f
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); BM = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb'))); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'RULES': 1.0, 'V15a': .3, 'ON+': .4, 'ON-': .1}
TR = {}
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n in ('ON+', 'ON-'): continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); B_ = BM[n].copy(); B_['day'] = pd.to_datetime(B_.day).dt.normalize(); B_ = B_[B_.sz > 0][['day', 'm', 'sz']].rename(columns={'sz': 'szb'}); t = t.merge(B_, on=['day', 'm']); t['sz'] = t.szb
    t = t[(t.day >= '2018-02-01') & (t.day <= '2026-01-30')].reset_index(drop=True); dirs = t.dir.values if 'dir' in t.columns else np.zeros(len(t))
    F = [feat(d, int(mm_), int(dr)) for d, mm_, dr in zip(t.day, t.m, dirs)]; ok = [f is not None for f in F]
    t = t[ok].reset_index(drop=True); t = pd.concat([t, pd.DataFrame([f for f in F if f])], axis=1); t['side'] = np.where(dirs[ok] > 0, 'long', np.where(dirs[ok] < 0, 'short', None)); TR[n] = t
print({k: len(v) for k, v in TR.items()}, flush=True)
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
PER = (('18-21', lambda x: x < '2022-01-01'), ('22-26', lambda x: x >= '2022-01-01'))
CONT = ('volr', 'fv', 'fv3', 'dpoc_d', 'dpoc_w', 'dpoc_m', 'dpoc_y', 'nstack'); CAT = ('tod', 'loc_d', 'loc_w', 'loc_m', 'loc_y', 'dir_d', 'dir_w', 'dir_m', 'dir_y', 'side')
def flags(t, pts):
    """returns {name: ((d1,t1,n1),(d2,t2,n2))} for every bucket; PASS decided by caller"""
    out = {}; e = PER[0][1](t.day).values; l = PER[1][1](t.day).values
    for c in CAT:
        col = t[c]
        if col.isna().all(): continue
        for k in col.dropna().unique():
            res = []
            for mm in (e, l):
                a, b = pts[mm & (col == k).values], pts[mm & (col != k).values & col.notna().values]; res.append((a.mean() - b.mean(), tt(a, b), len(a)))
            out[f'{c}={k}'] = tuple(res)
    for c in CONT:
        col = t[c]
        if col.isna().all(): continue
        q = col[e].quantile([1 / 3, 2 / 3]).values
        if q[0] == q[1]: continue
        for nm, sel in (('low', col <= q[0]), ('high', col > q[1])):
            res = []
            for mm in (e, l):
                a, b = pts[mm & sel.values], pts[mm & ~sel.values & col.notna().values]; res.append((a.mean() - b.mean(), tt(a, b), len(a)))
            out[f'{c} {nm} third'] = tuple(res)
    return out
import os; THR = float(os.environ.get('THR', 2))
def passed(r): return np.sign(r[0][0]) == np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1]) >= THR and x[2] >= 30 for x in r)
rng = np.random.default_rng(7); REAL = {}
print('\n=== TEST 1: PASS flags per engine (bucket vs the rest of that engine; spread in pts/trade, t; early // late) with the chance rate from 200 shuffles ===')
for n, t in TR.items():
    fl = flags(t, t.pts); ps = {k: v for k, v in fl.items() if passed(v)}; REAL[n] = ps
    null = []
    for _ in range(200):
        sh = t.pts.copy(); e = PER[0][1](t.day).values
        for mm in (e, ~e): sh[mm] = rng.permutation(sh[mm].values)
        null.append(sum(passed(v) for v in flags(t, sh).values()))
    print(f'\n  {n}: {len(t)} trades, {len(fl)} bucket tests, real PASS flags {len(ps)} vs chance {np.mean(null):.1f} (95th pct {np.percentile(null, 95):.0f})', flush=True)
    for k, v in sorted(ps.items(), key=lambda kv: -abs(kv[1][0][1]) - abs(kv[1][1][1])):
        print(f'     {k:<28} {v[0][0]:+7.1f} (t {v[0][1]:+5.2f}, n{v[0][2]}) // {v[1][0]:+7.1f} (t {v[1][1]:+5.2f}, n{v[1][2]})', flush=True)
print('\n=== TOD x engine: pts per trade (n), early // late ===')
for n, t in TR.items():
    o = []
    for lab, f in PER:
        x = t[f(t.day)]; o.append(' | '.join(f'{x[x.tod == k].pts.mean():+6.1f} ({(x.tod == k).sum()})' for k in range(4)))
    print(f'  {n:<6} 09:35-10:29 | 10:30-11:59 | 12:00-13:59 | 14:00+ :  {o[0]} // {o[1]}')
for n in ('RA', 'V15a'):
    t = TR[n]; print(f'\n=== {n}: pts per trade by VP direction class (beyond | inside | against), early // late ===')
    for X, nm in (('d', 'daily'), ('w', 'weekly'), ('m', 'monthly'), ('y', '52-week')):
        o = []
        for lab, f in PER:
            x = t[f(t.day)]; o.append(' | '.join(f'{x[x["dir_" + X] == k].pts.mean():+6.1f} ({(x["dir_" + X] == k).sum()})' for k in ('beyond', 'inside', 'against')))
        print(f'  {nm:<8} {o[0]} // {o[1]}')
    o = []
    for lab, f in PER:
        x = t[f(t.day)]; o.append(' | '.join(f'{x[x.nstack == k].pts.mean():+6.1f} ({(x.nstack == k).sum()})' for k in range(5)))
    print(f'  nstack 0..4 beyond: {o[0]} // {o[1]}')
print('\n=== TEST 2: cutting a consistently BAD pocket (negative in both periods; real PASS flags only) ===', flush=True)
DAYS_ = pd.DatetimeIndex([d for d in DAYS if d >= pd.Timestamp('2018-02-01')])
def system(cut):
    s = pd.Series(0.0, index=DAYS_)
    for n, t in TR.items():
        keep = np.ones(len(t), bool)
        if n in cut: keep = ~cut[n](t)
        s = s.add((t.pts * t.sz * UW[n] * keep).groupby(t.day.values).sum().reindex(DAYS_).fillna(0.0), fill_value=0)
    return s
def rep(lab, cut):
    s = system(cut); o = []
    for nm, f in PER:
        x = s[f(s.index)]; mo_ = x.groupby(x.index.to_period('M')).sum(); o.append(f'{nm}: {x.mean():+5.2f}/d std {x.std():5.1f} m/s {x.mean() / x.std():.3f} worstM {mo_.min():+5.0f}')
    print(f'  {lab:<40}' + ' | '.join(o), flush=True)
rep('base', {})
def mk(c, k):
    if c in CONT or ' third' in c: return None
    col, val = c.split('=')
    return (lambda t, col=col, val=val: (t[col].astype(str) == val).values)
for n, ps in REAL.items():
    for k, v in ps.items():
        if v[0][0] < 0 and v[1][0] < 0 and '=' in k:
            col, val = k.split('=')
            rep(f'cut {n}: {k} (n {v[0][2]}+{v[1][2]})', {n: (lambda t, col=col, val=val: (t[col].astype(str) == val).values)})
        elif v[0][0] < 0 and v[1][0] < 0 and ' third' in k:
            c, nm = k.split(' ')[0], k.split(' ')[1]
            def fn(t, c=c, nm=nm, n=n):
                q = t[c][PER[0][1](t.day).values].quantile([1 / 3, 2 / 3]).values; return (t[c] <= q[0]).values if nm == 'low' else (t[c] > q[1]).values
            rep(f'cut {n}: {k} (n {v[0][2]}+{v[1][2]})', {n: fn})
