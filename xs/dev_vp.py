"""Nov 10 2026: LOWER-TIMEFRAME (DEVELOPING) VOLUME PROFILE CONTEXT ON NOISE-BAND AND ORB BREAKOUTS. DECLARED BEFORE RESULTS.
Engines: NZ (noise-area momentum, nz_core.noise, 40-pt stop) and ORB (30-min opening range, first close beyond the range, stop at the opposite side, exit at the close; one trade per day).
At the signal bar (closed bars only) build the DEVELOPING SESSION profile (volume spread uniformly over each 5m bar's high-low range, 0.25-pt bins) and the LAST-HOUR profile (12 bars). Features, signed with the trade direction where relevant (in noise-sigma units s = open x sigma_end):
  pocd  (close - session POC)/s x dir;  pocd1h  same on the 1-hour POC;  pocmig  (POC now - POC 12 bars ago)/s x dir;  vaw (VAH - VAL)/s (70% value area);  outva  +1 close beyond the value area in the trade direction, 0 inside, -1 beyond it against;  dens  volume density at the close price / mean density of the session profile (low = air pocket / LVN, high = HVN);  dens1h same on the last hour;  vabreak  1 when the breakout bar closes outside the value area in the trade direction.
Test: points per trade by third of each continuous feature (cuts on 2014-21), PASS = same sign and |t| >= 2 in 2014-21 and 2022-26, n >= 30; chance rate from 200 shuffles. Then the best consistent feature (if any) as a filter / x1.25 / x0.6 on the long-only NZ leg (the funded add-on): pts/day, std, mean/std both periods."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, DAYS, GRID, NB, OPEN, PC, SIG, SI
C = .5; BIN = .25
def profile(o, h, l, c, v, hi_idx):
    lo = np.nanmin(l[:hi_idx + 1]); hi = np.nanmax(h[:hi_idx + 1]); n = int((hi - lo) / BIN) + 2; prof = np.zeros(n)
    for j in range(hi_idx + 1):
        if np.isnan(v[j]) or np.isnan(h[j]): continue
        a, b = int((l[j] - lo) / BIN), int((h[j] - lo) / BIN); prof[a:b + 1] += v[j] / (b - a + 1)
    return prof, lo
def stats(prof, lo, price):
    if prof.sum() == 0: return None
    p = int(np.argmax(prof)); poc = lo + (p + .5) * BIN; tot = prof.sum(); a = b = p; got = prof[p]
    while got < .7 * tot and (a > 0 or b < len(prof) - 1):
        up = prof[b + 1] if b + 1 < len(prof) else -1; dn = prof[a - 1] if a > 0 else -1
        if up >= dn: b += 1; got += prof[b]
        else: a -= 1; got += prof[a]
    k = min(max(int((price - lo) / BIN), 0), len(prof) - 1); return poc, lo + (b + 1) * BIN, lo + a * BIN, prof[k] / (prof[prof > 0].mean() + 1e-9)
def feats(d, jb, pos):
    o, h, l, c, v, vw = A[d][0]; sg = SIG[SI[d]]; s_ = OPEN[d] * sg[-1]
    if jb < 14 or not np.isfinite(s_) or np.isnan(c[jb]): return None
    P, lo = profile(o, h, l, c, v, jb); st = stats(P, lo, c[jb]); P1, lo1 = profile(o[jb - 11:], h[jb - 11:], l[jb - 11:], c[jb - 11:], v[jb - 11:], 11); s1 = stats(P1, lo1, c[jb])
    P0, lo0 = profile(o, h, l, c, v, jb - 12); s0 = stats(P0, lo0, c[jb - 12])
    if st is None or s1 is None or s0 is None: return None
    poc, vah, val, dens = st; poc1, _, _, dens1 = s1; out = 1.0 if (pos > 0 and c[jb] > vah) or (pos < 0 and c[jb] < val) else (-1.0 if (pos > 0 and c[jb] < val) or (pos < 0 and c[jb] > vah) else 0.0)
    return dict(pocd=pos * (c[jb] - poc) / s_, pocd1h=pos * (c[jb] - poc1) / s_, pocmig=pos * (poc - s0[0]) / s_, vaw=(vah - val) / s_, outva=out, dens=dens, dens1h=dens1)
rows = []
for d in DAYS:
    o, h, l, c, v, vw = A[d][0]
    for (dd, em, pos, pts, mae) in Z.noise(d):
        f = feats(d, em - 1, pos)
        if f: rows.append(dict(eng='NZ', day=d, pos=pos, pts=pts, **f))
    if len(o) < NB or np.isnan(o[0]) or np.isnan(c[NB - 1]): continue
    hh, ll = np.nanmax(h[:6]), np.nanmin(l[:6])
    for j in range(6, NB - 2):
        if np.isnan(c[j]) or np.isnan(o[j + 1]): continue
        pos = 1 if c[j] > hh else -1 if c[j] < ll else 0
        if pos == 0: continue
        e = o[j + 1]; st_ = ll if pos > 0 else hh; px = c[NB - 1]
        for q in range(j + 1, NB):
            if np.isnan(c[q]): continue
            if (pos > 0 and l[q] <= st_) or (pos < 0 and h[q] >= st_): px = st_; break
        f = feats(d, j, pos)
        if f: rows.append(dict(eng='ORB', day=d, pos=pos, pts=(px - e) * pos - C, **f))
        break
R = pd.DataFrame(rows); R['day'] = pd.to_datetime(R.day); print({k: len(g) for k, g in R.groupby('eng')}, flush=True)
FE = ['pocd', 'pocd1h', 'pocmig', 'vaw', 'dens', 'dens1h', 'outva']
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
def flags(t, pts):
    out = {}; e = (t.day < '2022-01-01').values; l_ = ~e
    for c in FE:
        col = t[c]
        if c == 'outva': groups = {'beyond VA with': col == 1, 'inside VA': col == 0, 'beyond VA against': col == -1}
        else: q = col[e].quantile([1 / 3, 2 / 3]).values; groups = {'low third': col <= q[0], 'high third': col > q[1]}
        for nm, sel in groups.items():
            res = []
            for mm in (e, l_):
                a, b = pts[mm & sel.values], pts[mm & ~sel.values]; res.append((a.mean() - b.mean(), tt(a, b), int((mm & sel.values).sum())))
            out[f'{c} {nm}'] = tuple(res)
    return out
def passed(r): return np.sign(r[0][0]) == np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1]) >= 2 and x[2] >= 30 for x in r)
rng = np.random.default_rng(3)
for eng in ('NZ', 'ORB'):
    for sub, mask in (('all trades', np.ones(len(R), bool)), ('long only', (R.pos > 0).values)):
        t = R[(R.eng == eng).values & mask].reset_index(drop=True); fl = flags(t, t.pts); ps = {k: v for k, v in fl.items() if passed(v)}; e = (t.day < '2022-01-01').values; null = []
        for _ in range(200):
            sh = t.pts.copy()
            for mm in (e, ~e): sh[mm] = rng.permutation(sh[mm].values)
            null.append(sum(passed(v) for v in flags(t, sh).values()))
        print(f'\\n{eng} {sub}: {len(t)} trades, {len(fl)} tests, real PASS {len(ps)} vs chance {np.mean(null):.1f} (95th pct {np.percentile(null, 95):.0f})')
        for k, v in sorted(fl.items(), key=lambda kv: -abs(kv[1][0][1]) - abs(kv[1][1][1]))[:6]: print(f'    {k:<26} {v[0][0]:+7.1f} (t {v[0][1]:+5.2f}, n{v[0][2]}) | {v[1][0]:+7.1f} (t {v[1][1]:+5.2f}, n{v[1][2]}){"   <== PASS" if passed(v) else ""}')
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/devvp_trades.pkl')
