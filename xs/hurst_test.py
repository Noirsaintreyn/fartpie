"""Nov 10 2026: HURST AND HURST VARIANTS (variogram / structure-function, rescaled range R/S, detrended fluctuation DFA, variance ratio, efficiency ratio) AS A REGIME FILTER FOR THE ENGINES. DECLARED BEFORE RESULTS.
Does a persistence measure known at the signal predict (a) the day's trendiness, (b) the engines' trade points?  ET 5m NQ (nz_core), RTH, 2014-02..2026-01-30; early 2014-21 | late 2022-26.
MEASURES (all use closed bars only, overnight gaps removed from the paths):
  hv5  variogram Hurst: slope of log std of lag-k price differences, k = 2..20, on the prior 5 sessions' 5m path (~390 bars)      (the form in the THOD / bot sigma).
  hr5  R/S Hurst on the same returns (window sizes 8..128).      hd5  DFA exponent (order 1, scales 8..64) on the same returns.
  hD   daily variogram Hurst, last 100 daily closes, k = 2..20.      hdD  daily DFA (scales 4..25) on the last 100 daily returns.
  vr   Lo-MacKinlay variance ratio VR(6) on the last 72 5m returns up to the signal bar (>1 trending, <1 mean reverting).      er  Kaufman efficiency ratio over the last 24 bars.
PART A: the day's directional efficiency |close - open| / (high - low) and the noise engine's points that day, by third of each measure at 09:30 (hv5, hr5, hd5, hD, hdD), early // late; PASS = same sign and |t| >= 2 both periods.
PART B: per engine (RA, V15a, M1, R1, RULES, noise) points per trade by third of each measure (daily measures at 09:30, vr / er at the signal bar): PASS rule as before; chance rate from 200 shuffles.
PART C: if any PASS is a bad pocket, the system-level cut (mean/std, worst month)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
import nz_core as Z
from nz_core import A, ALLD, DAYS, GRID, NB, OPEN, CLOSE
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; DI = {d: i for i, d in enumerate(ALLD)}
R5 = {}
for d in ALLD:
    o, h, l, c, v, vw = A[d][0]; r = np.diff(np.log(c[~np.isnan(c)]), prepend=np.log(o[~np.isnan(o)][0])); r[0] = np.log(c[~np.isnan(c)][0] / o[~np.isnan(o)][0]); R5[d] = r
def variogram(x, lags=range(2, 21)):
    p = np.cumsum(x); s = [np.std(p[k:] - p[:-k]) for k in lags]
    return np.polyfit(np.log(list(lags)), np.log(s), 1)[0] if min(s) > 0 else np.nan
def rs(x, sizes=(8, 12, 16, 24, 32, 48, 64, 96, 128)):
    out = []
    for n in sizes:
        k = len(x) // n
        if k < 2: continue
        vals = []
        for i in range(k):
            w = x[i * n:(i + 1) * n]; z = np.cumsum(w - w.mean()); sd = w.std()
            if sd > 0: vals.append((z.max() - z.min()) / sd)
        if vals: out.append((n, np.mean(vals)))
    return np.polyfit(np.log([a for a, _ in out]), np.log([b for _, b in out]), 1)[0] if len(out) >= 4 else np.nan
def dfa(x, scales=(8, 12, 16, 24, 32, 48, 64)):
    y = np.cumsum(x - x.mean()); F = []
    for s in scales:
        k = len(y) // s
        if k < 2: continue
        t = np.arange(s); rm = []
        for i in range(k):
            w = y[i * s:(i + 1) * s]; cf = np.polyfit(t, w, 1); rm.append(np.mean((w - np.polyval(cf, t)) ** 2))
        F.append((s, np.sqrt(np.mean(rm))))
    return np.polyfit(np.log([a for a, _ in F]), np.log([b for _, b in F]), 1)[0] if len(F) >= 4 else np.nan
DC = pd.Series({d: CLOSE[d] for d in ALLD}); DR = np.log(DC).diff().values; M = {}
for i, d in enumerate(ALLD):
    if i < 130: continue
    x = np.concatenate([R5[ALLD[q]] for q in range(i - 5, i)]); lc = np.log(DC.values[i - 100:i]); dr = DR[i - 99:i + 0]
    M[d] = dict(hv5=variogram(x), hr5=rs(x), hd5=dfa(x), hD=variogram(np.diff(lc), range(2, 21)) if False else np.polyfit(np.log(range(2, 21)), np.log([np.std(lc[k:] - lc[:-k]) for k in range(2, 21)]), 1)[0], hdD=dfa(dr[-99:], (4, 6, 8, 12, 16, 25)))
MD = pd.DataFrame(M).T; MD.index = pd.DatetimeIndex(MD.index)
print('daily measures built:', len(MD), '| corr among measures:'); print(MD.corr().round(2).to_string(), flush=True)
def local(d, jb):
    r = R5[d]; k = jb + 1; x = r[max(0, k - 72):k]
    if len(x) < 24: return np.nan, np.nan
    v1 = x.var(); q = 6; y = np.convolve(x, np.ones(q), 'valid'); vr = y.var() / (q * v1) if v1 > 0 else np.nan
    c = np.cumsum(r)[:k]; w = c[-min(25, len(c)):]; er = abs(w[-1] - w[0]) / np.abs(np.diff(w)).sum() if len(w) > 2 and np.abs(np.diff(w)).sum() > 0 else np.nan
    return vr, er
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
PER = (('14-21', lambda x: x < '2022-01-01'), ('22-26', lambda x: x >= '2022-01-01')); CD = ('hv5', 'hr5', 'hd5', 'hD', 'hdD')
def thirds(col, e): q = col[e].quantile([1 / 3, 2 / 3]).values; return col <= q[0], col > q[1]
print('\n=== PART A: day-level. Efficiency |c-o|/(h-l) and noise-engine pts/day by third (low | mid | high), early // late; spread high-low, t ===')
EF = pd.Series({d: abs(CLOSE[d] - OPEN[d]) / max(np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2]), 1e-9) for d in ALLD}); EF = EF.reindex(MD.index)
nz = pd.DataFrame([r for d in DAYS for r in Z.noise(d)], columns=['day', 'j', 'dir', 'pts', 'mae']); nz['day'] = pd.to_datetime(nz.day); NZD = nz.groupby('day').pts.sum().reindex(pd.DatetimeIndex(DAYS)).fillna(0.0).reindex(MD.index)
for nm, y in (('efficiency', EF), ('noise pts/day', NZD)):
    for c in CD:
        ok = MD[c].notna() & y.notna(); e = PER[0][1](MD.index) & ok.values; l_ = PER[1][1](MD.index) & ok.values; res = []; txt = []
        for lab, mm in (('e', e), ('l', l_)):
            lo, hi = thirds(MD[c], e); a, b = y[mm & hi.values], y[mm & lo.values]; mid = y[mm & ~hi.values & ~lo.values]
            txt.append(f'{b.mean():+7.3f} | {mid.mean():+7.3f} | {a.mean():+7.3f}'); res.append((a.mean() - b.mean(), tt(a, b)))
        ok2 = np.sign(res[0][0]) == np.sign(res[1][0]) and all(abs(r[1]) >= 2 for r in res)
        print(f'  {nm:<14}{c:<5} {txt[0]} // {txt[1]} | spread {res[0][0]:+.3f} (t {res[0][1]:+5.2f}) // {res[1][0]:+.3f} (t {res[1][1]:+5.2f}){"   <== PASS" if ok2 else ""}', flush=True)
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); BM = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb'))); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'RULES': 1.0, 'V15a': .3}; TR = {}
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n in ('ON+', 'ON-'): continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); B_ = BM[n].copy(); B_['day'] = pd.to_datetime(B_.day).dt.normalize(); B_ = B_[B_.sz > 0][['day', 'm', 'sz']].rename(columns={'sz': 'szb'}); t = t.merge(B_, on=['day', 'm']); t['sz'] = t.szb
    TR[n] = t[(t.day >= '2014-02-01') & (t.day <= '2026-01-30')].reset_index(drop=True)
nz['m'] = GRID[nz.j.values.astype(int)]; nz['sz'] = 1.0; TR['noise'] = nz[nz.day >= '2014-02-01'].reset_index(drop=True)
for n, t in TR.items():
    t = t.join(MD, on='day'); vr, er = zip(*[local(d, int((m - 570) // 5) - 1) if d in DI else (np.nan, np.nan) for d, m in zip(t.day, t.m)]); t['vr'] = vr; t['er'] = er; TR[n] = t[t.hv5.notna()]
CB = CD + ('vr', 'er')
def flags(t, pts):
    out = {}; e = PER[0][1](t.day).values; l_ = PER[1][1](t.day).values
    for c in CB:
        col = t[c]
        if col.isna().mean() > .5: continue
        lo, hi = thirds(col, e)
        for nm, sel in (('low', lo), ('high', hi)):
            res = []
            for mm in (e, l_):
                a, b = pts[mm & sel.values], pts[mm & ~sel.values & col.notna().values]; res.append((a.mean() - b.mean(), tt(a, b), len(a)))
            out[f'{c} {nm} third'] = tuple(res)
    return out
def passed(r): return np.sign(r[0][0]) == np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1]) >= 2 and x[2] >= 30 for x in r)
rng = np.random.default_rng(11); REAL = {}
print('\n=== PART B: trade points by third of each measure (thirds cut on 2014-21); PASS flags per engine vs the chance rate ===', flush=True)
for n, t in TR.items():
    fl = flags(t, t.pts); ps = {k: v for k, v in fl.items() if passed(v)}; REAL[n] = ps; null = []; e = PER[0][1](t.day).values
    for _ in range(200):
        sh = t.pts.copy()
        for mm in (e, ~e): sh[mm] = rng.permutation(sh[mm].values)
        null.append(sum(passed(v) for v in flags(t, sh).values()))
    print(f'  {n:<6} {len(t):5d} trades, {len(fl)} tests, real PASS {len(ps)} vs chance {np.mean(null):.1f} (95th pct {np.percentile(null, 95):.0f})', flush=True)
    for k, v in sorted(ps.items(), key=lambda kv: -abs(kv[1][0][1]) - abs(kv[1][1][1])): print(f'        {k:<22} {v[0][0]:+7.1f} (t {v[0][1]:+5.2f}, n{v[0][2]}) // {v[1][0]:+7.1f} (t {v[1][1]:+5.2f}, n{v[1][2]})', flush=True)
for n in ('M1', 'RULES', 'V15a', 'noise'):
    t = TR[n]; print(f'\n  {n}: pts per trade by third of hv5 / hd5 / vr (low | mid | high), early // late')
    for c in ('hv5', 'hd5', 'vr'):
        e = PER[0][1](t.day).values; lo, hi = thirds(t[c], e); o = []
        for mm in (e, ~e): g = [t.pts[mm & lo.values], t.pts[mm & ~lo.values & ~hi.values & t[c].notna().values], t.pts[mm & hi.values]]; o.append(' | '.join(f'{x.mean():+6.1f}' for x in g))
        print(f'     {c:<4} {o[0]} // {o[1]}')
