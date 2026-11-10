"""Nov 10 2026: WHY DO OPENING-RANGE AND NOISE-BAND BREAKOUTS WORK, AND HOW DO THEY RELATE TO THE VECTOR TRADES (RA and V15a, the two vector engines)? DECLARED BEFORE RESULTS.
Real NQ 5m (09:30-15:55 ET, bar-open stamps). At each vector entry (bar open m, direction d) the state is read from the LAST COMPLETED bar (close of bar m-5), so nothing from the entry bar or later is used.
  ORB state    opening range = 09:30-09:45 (first 3 bars). with = price already beyond the OR in the trade direction | inside | against = beyond the OR on the other side.
               ORB whip = price has already broken the OR in the trade direction and is back inside it.
  Noise band   sigma(t) = mean over the prior 14 sessions of |close(t)/open - 1| at that time of day; UB = max(open, prior close) x (1 + sigma), LB = min(open, prior close) x (1 - sigma). Same three states.
  VWAP side, OR width / ADR20, band half-width (sigma at entry), distance beyond/inside the band in sigma units.
  Outcome = trade points (live table V2, 1 contract after costs), win %, "bad" = trade ended <= -40 pts (stop or worse).
Rule: a state matters only if the gap between its best and worst bucket has the same sign and |t| >= 2 in BOTH periods (RA 2018-21 | 2022-26; V15a 2014-21 | 2022-26).
PART C: standalone baseline - ORB30 close-break (first 5m close beyond the 09:30-10:00 range, 10:00-11:30, stop = opposite side of the range, flat 12:00 / 15:55), and how often it agrees with the vector direction."""
import sys, os, numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
_eh = open('xs/nq_edge_hunt.py').read(); exec(_eh[:_eh.index("def days(d):")])
d = load(); mod = d.index.hour * 60 + d.index.minute; d = d[(mod >= 570) & (mod < 960)].copy(); d['day'] = d.index.normalize(); d['mod'] = d.index.hour * 60 + d.index.minute
d['tp'] = (d.h + d.l + d.c) / 3; d['pv'] = d.tp * d.v; d['cv'] = d.groupby('day').v.cumsum(); d['vw'] = d.groupby('day').pv.cumsum() / d.cv.replace(0, np.nan)
G = {k: g.set_index('mod') for k, g in d.groupby('day')}; DK = sorted(G); pos = {k: i for i, k in enumerate(DK)}
O = pd.Series({k: G[k].o.iloc[0] for k in DK}); C = pd.Series({k: G[k].c.iloc[-1] for k in DK}); sym = pd.Series({k: G[k].sym.iloc[0] for k in DK})
PC = C.shift(1).where(sym == sym.shift(1), np.nan); RNG = pd.Series({k: G[k].h.max() - G[k].l.min() for k in DK}); ADR = RNG.shift(1).rolling(20).mean()
MODS = list(range(570, 960, 5)); AB = pd.DataFrame({k: (G[k].c.reindex(MODS) / O[k] - 1).abs() for k in DK}).T                       # days x time of day
SIG = AB.rolling(14, min_periods=10).mean().shift(1)
def state(day, m, dr):
    g = G.get(day)
    if g is None or (m - 5) not in g.index or day not in SIG.index: return None
    lc = g.c[m - 5]; vw = g.vw[m - 5]; early = g[(g.index >= 570) & (g.index < 585)]
    if len(early) < 3: return None
    orh, orl = early.h.max(), early.l.min(); sg = SIG.loc[day, m - 5]
    if not np.isfinite(sg): return None
    o = O[day]; pc = PC[day] if np.isfinite(PC[day]) else o; ub, lb = max(o, pc) * (1 + sg), min(o, pc) * (1 - sg)
    orb = 'with' if (lc > orh if dr > 0 else lc < orl) else ('against' if (lc < orl if dr > 0 else lc > orh) else 'inside')
    prior = g[(g.index >= 585) & (g.index <= m - 5)]; broke = ((prior.c > orh).any() if dr > 0 else (prior.c < orl).any()); whip = bool(broke and orb == 'inside')
    nb = 'beyond' if (lc > ub if dr > 0 else lc < lb) else ('opposite' if (lc < lb if dr > 0 else lc > ub) else 'inside')
    dist = ((lc - ub) if dr > 0 else (lb - lc)) / (o * sg)                                                                        # >0 beyond the band in the trade direction (sigma units)
    return dict(orb=orb, whip=whip, nb=nb, dist=dist, vw=int(dr * (lc - vw) > 0), orw=(orh - orl) / ADR[day], sig=sg, dvw=dr * (lc - vw) / ADR[day])
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); R = []
print('AFTER THE PIPELINE mv30 SKIP' if os.environ.get('SKIP') == '1' else 'ALL V2 TRADES (before the mv30 skip)')
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n not in ('RA', 'V15a'): continue
    t = t[t.sz > 0].copy(); t['day'] = pd.to_datetime(t.day).dt.normalize()
    if os.environ.get('SKIP') == '1':                                                  # the pipeline's own mv30 skip (30-min move into the flip, bottom third out) applied
        B_ = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb')))[n]; B_['day'] = pd.to_datetime(B_.day).dt.normalize(); keep = set(zip(B_[B_.sz > 0].day, B_[B_.sz > 0].m)); t = t[[(a, b) in keep for a, b in zip(t.day, t.m)]]
    for r in t.itertuples():
        if r.day not in pos or r.day > pd.Timestamp('2026-01-30'): continue
        s = state(r.day, int(r.m), int(r.dir))
        if s: R.append(dict(eng=n, day=r.day, m=int(r.m), dir=int(r.dir), pts=r.pts, mae=r.mae, **s))
X = pd.DataFrame(R); X['bad'] = X.pts <= -40; X['win'] = X.pts > 0
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
print(f'vector trades with a state: {X.groupby("eng").size().to_dict()}\n')
for eng, cut in (('RA', '2022-01-01'), ('V15a', '2022-01-01')):
    x = X[X.eng == eng]; E = x.day < cut
    print(f'=== {eng}: points per trade (win % | bad %) by state; early period n | late period n ===')
    def block(col, cats=None):
        out = []
        for v in (cats or sorted(x[col].unique(), key=str)):
            a, b = x[(x[col] == v) & E], x[(x[col] == v) & ~E]
            out.append(f'{v}: {a.pts.mean():+6.1f} ({a.win.mean():.0%}|{a.bad.mean():.0%}, n {len(a)}) / {b.pts.mean():+6.1f} ({b.win.mean():.0%}|{b.bad.mean():.0%}, n {len(b)})')
        print(f'  {col:<5} ' + ' || '.join(out))
    for col in ('orb', 'nb', 'vw', 'whip'): block(col)
    print('  spreads (best bucket minus worst bucket; same sign and |t| >= 2 in both periods passes):')
    for col in ('orb', 'nb', 'vw', 'whip'):
        res = []
        for lab, mm in (('early', E), ('late', ~E)):
            m_ = x[mm].groupby(col).pts.mean(); hi, lo = m_.idxmax(), m_.idxmin(); res.append((hi, lo, m_[hi] - m_[lo], tt(x[mm][x[mm][col] == hi].pts, x[mm][x[mm][col] == lo].pts)))
        ok = res[0][0] == res[1][0] and res[0][1] == res[1][1] and all(abs(r[3]) >= 2 for r in res)
        print(f'    {col:<5} early {res[0][0]} over {res[0][1]} {res[0][2]:+6.1f} (t {res[0][3]:+5.2f}) | late {res[1][0]} over {res[1][1]} {res[1][2]:+6.1f} (t {res[1][3]:+5.2f}){"   <== PASS" if ok else ""}')
    for col in ('dist', 'orw', 'sig', 'dvw'):
        res = []
        for lab, mm in (('early', E), ('late', ~E)):
            q = x[mm][col].rank(pct=True); hi, lo = x[mm].pts[q > 2 / 3], x[mm].pts[q <= 1 / 3]; res.append((hi.mean() - lo.mean(), tt(hi, lo)))
        ok = np.sign(res[0][0]) == np.sign(res[1][0]) and all(abs(r[1]) >= 2 for r in res)
        print(f'    {col:<5} top third minus bottom third: early {res[0][0]:+6.1f} (t {res[0][1]:+5.2f}) | late {res[1][0]:+6.1f} (t {res[1][1]:+5.2f}){"   <== PASS" if ok else ""}')
    print()
print('=== PART C: standalone ORB30 close-break on real NQ 5m, 1 contract, cost 1 pt ===', flush=True)
rows = []
for k in DK:
    g = G[k]; e = g[(g.index >= 570) & (g.index < 600)]
    if len(e) < 5 or k < pd.Timestamp('2014-02-01') or k > pd.Timestamp('2026-01-30'): continue
    orh, orl = e.h.max(), e.l.min(); late = g[(g.index >= 600) & (g.index <= 690)]; ent = None
    for mm, b in late.iterrows():
        if b.c > orh: ent = (1, mm, g.o.get(mm + 5, np.nan)); break
        if b.c < orl: ent = (-1, mm, g.o.get(mm + 5, np.nan)); break
    if ent is None or not np.isfinite(ent[2]): continue
    dr, mm, ep = ent; st = orl if dr > 0 else orh; res = {}
    for ex in (720, 955):
        px = None
        for m2, b in g[(g.index >= mm + 5) & (g.index <= ex)].iterrows():
            if (b.l <= st if dr > 0 else b.h >= st): px = st; break
        if px is None: px = g.c[g[g.index <= ex].index[-1]]
        res[ex] = (px - ep) * dr - 1.0
    rows.append(dict(day=k, dir=dr, m=mm, risk=abs(ep - st), p12=res[720], pcl=res[955]))
B = pd.DataFrame(rows).set_index('day'); A = B.index < '2022-01-01'
for lab, mm in (('2014-21', A), ('2022-26', ~A)):
    b = B[mm]; print(f'  {lab}: {len(b)} trades ({len(b) / (mm.sum() and 1):.0f}) | flat 12:00 {b.p12.mean():+6.2f} pts/trade (win {np.mean(b.p12 > 0):.0%}) | flat close {b.pcl.mean():+6.2f} | avg risk {b.risk.mean():.0f} pts')
Vd = X.groupby('day').dir.first(); J = B.join(Vd.rename('vd'), how='inner')
if len(J):
    ag = J.dir == J.vd; print(f'  days with an ORB30 break and a vector entry: {len(J)}; same direction {ag.mean():.0%}')
    for lab, mm in (('2014-21', J.index < '2022-01-01'), ('2022-26', J.index >= '2022-01-01')):
        a, b = J[mm & ag], J[mm & ~ag]; print(f'    {lab}: ORB30 agrees with vector {a.p12.mean():+6.1f} (n {len(a)}) vs disagrees {b.p12.mean():+6.1f} (n {len(b)}), t {tt(a.p12, b.p12):+5.2f}')
X.to_pickle(f'{SP}/orb_noise_vector{os.environ.get("SKIP", "")}.pkl')
