"""Nov 10 2026: DOES POSITIVE GAMMA / THE PRIOR-DAY VOLUME PROFILE / TPO STRUCTURE DICTATE WHEN THE NOISE-AREA BREAKOUT WORKS (AND HOW BIG THE NOISE IS)? DECLARED BEFORE RESULTS.
Engine: nz_core.noise() (band k 1.0, 30-min decisions, stop 40, both sides), 1 NQ, cost 0.5, ET clock. Periods: gamma needs the QQQ GEX file (2018-01 ..), so ALL tests use 2018-21 | 2022-Jan26.
Features, all known at 09:30 (prior-day data only):
  GAMMA   net_gex of QQQ at the prior close (xs/ext/gex/QQQ.csv). pos = net_gex > 0; gp = rolling 1y percentile.
  VP      prior RTH session volume profile: each 5m bar's volume spread evenly over its high-low in 2.5-pt bins. POC, value area (70% of volume, expanded from the POC).
  TPO     30-min periods, every period counts once in each bin its high-low touches; TPO-POC (most letters; ties -> nearest the middle), 70% TPO value area, single prints (bins touched by one period) as a share of the range.
  loc     today's open vs the prior VP value area: above / inside / below.   dpoc = |open - prior POC| / (open x sigma_ref) with sigma_ref = last noise sigma at 15:55 (noise units).
  vaw     VP value-area width / ADR20.   pocpos = (POC - low) / range (0 bottom-heavy .. 1 top-heavy).   sp = single-print share.   bal = value-area width / range.
TEST 1 (does it dictate NOISE size?)  y = log(today's RTH range / open) - log(mean of the last 14 days' range / open). Top third minus bottom third of each continuous feature, categories for pos / loc.
TEST 2 (does it dictate the ENGINE?)  daily noise-engine points (days with no trade = 0) by bucket; PASS = same sign and |t| >= 2 in both periods (top minus bottom third, or best minus worst category).
TEST 3  at each noise entry, where the entry bar sits vs the prior VP value area / POC (with = beyond the VA in the trade direction, inside, against) - points per trade.
TEST 4  filtered engine: trade only inside the buckets that pass (if any) and report pts/day, daily std, mean/std by period.
Selection discipline: buckets are fixed up front (thirds of the 2018-21 distribution applied to both periods, so no look-ahead into 2022+)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs')
import nz_core as Z
from nz_core import A, OPEN, CLOSE, SYM, PC, SIG, SI, ALLD, DAYS, GRID, NB, pos_of, noise
U = '/home/claude/fartpie'; BIN = 2.5
GX = pd.read_csv(f'{U}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); GX.index = GX.index.normalize()
GP = GX.rolling(252, min_periods=120).rank(pct=True)
def profile(d):
    a, _ = A[d]; o, h, l, c, v, vw = a; ok = ~np.isnan(c)
    if ok.sum() < 60: return None
    lo, hi = np.nanmin(l), np.nanmax(h); n = int(np.ceil((hi - lo) / BIN)) + 1; vol = np.zeros(n); tpo = np.zeros(n)
    for j in np.where(ok)[0]:
        i0, i1 = int((l[j] - lo) // BIN), int((h[j] - lo) // BIN); vol[i0:i1 + 1] += v[j] / (i1 - i0 + 1)
    per = (GRID // 30)
    for p in np.unique(per[ok]):
        js = np.where(ok & (per == p))[0]; i0, i1 = int((np.nanmin(l[js]) - lo) // BIN), int((np.nanmax(h[js]) - lo) // BIN); tpo[i0:i1 + 1] += 1
    def va(arr, poc):
        tot = arr.sum(); a_, b_ = poc, poc; acc = arr[poc]
        while acc < .7 * tot and (a_ > 0 or b_ < n - 1):
            up = arr[b_ + 1] + (arr[b_ + 2] if b_ + 2 < n else 0) if b_ < n - 1 else -1; dn = arr[a_ - 1] + (arr[a_ - 2] if a_ >= 2 else 0) if a_ > 0 else -1
            if up >= dn: b_ = min(b_ + 2, n - 1); acc += max(up, 0)
            else: a_ = max(a_ - 2, 0); acc += max(dn, 0)
        return a_, b_
    pv = int(np.argmax(vol)); vl, vh = va(vol, pv)
    mx = tpo.max(); cand = np.where(tpo == mx)[0]; pt = int(cand[np.argmin(np.abs(cand - n / 2))]); tl, th = va(tpo, pt)
    px = lambda i: lo + (i + .5) * BIN
    return dict(poc=px(pv), vah=px(vh), val=px(vl), tpoc=px(pt), tvah=px(th), tval=px(tl), rng=hi - lo, pocpos=(px(pv) - lo) / max(hi - lo, 1), sp=float((tpo == 1).sum() / n), bal=(px(vh) - px(vl)) / max(hi - lo, 1), sym=SYM[d])
PR = {}
for d in ALLD:
    p = profile(d)
    if p: PR[d] = p
RNG = pd.Series({d: (np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2])) for d in ALLD}); ADR = RNG.shift(1).rolling(20).mean()
RR = pd.Series({d: RNG[d] / OPEN[d] for d in ALLD}); RR14 = RR.shift(1).rolling(14, min_periods=10).mean()
rows = []
for i, d in enumerate(ALLD):
    if d not in DAYS or i == 0: continue
    pd_ = ALLD[i - 1]
    if pd_ not in PR or PR[pd_]['sym'] != SYM[d]: continue
    p = PR[pd_]; op = OPEN[d]; sref = SIG[SI[d]][-1] if np.isfinite(SIG[SI[d]][-1]) else np.nan
    gi = GX.index.searchsorted(d) - 1
    g = GX.iloc[gi] if gi >= 0 and GX.index[gi] >= d - pd.Timedelta(days=5) else np.nan; gpv = GP.iloc[gi] if gi >= 0 and np.isfinite(g) else np.nan
    rows.append(dict(day=d, gex=g, pos=(g > 0) if np.isfinite(g) else np.nan, gp=gpv, loc=('above' if op > p['vah'] else 'below' if op < p['val'] else 'inside'),
                     tloc=('above' if op > p['tvah'] else 'below' if op < p['tval'] else 'inside'), dpoc=abs(op - p['poc']) / (op * sref) if np.isfinite(sref) else np.nan,
                     dtpoc=abs(op - p['tpoc']) / (op * sref) if np.isfinite(sref) else np.nan, vaw=(p['vah'] - p['val']) / ADR[d], pocpos=p['pocpos'], sp=p['sp'], bal=p['bal'],
                     y=np.log(RR[d] / RR14[d]) if np.isfinite(RR14[d]) else np.nan))
F = pd.DataFrame(rows).set_index('day'); F = F[(F.index >= '2018-02-01') & F.gex.notna()]
print(f'feature days {len(F)} ({F.index.min().date()} .. {F.index.max().date()}); positive-gamma days {F.pos.mean():.0%}\n')
PER = (('18-21', lambda ix: ix < '2022-01-01'), ('22-26', lambda ix: ix >= '2022-01-01'))
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
R = [r for d in DAYS for r in noise(d)]; T = pd.DataFrame(R, columns=['day', 'j', 'dir', 'pts', 'mae']); T['day'] = pd.to_datetime(T.day)
S = T.groupby('day').pts.sum().reindex(F.index).fillna(0.0); SL = T[T.dir > 0].groupby('day').pts.sum().reindex(F.index).fillna(0.0); SS = T[T.dir < 0].groupby('day').pts.sum().reindex(F.index).fillna(0.0)
CUT = {c: F[F.index < '2022-01-01'][c].quantile([1 / 3, 2 / 3]).values for c in ('gp', 'dpoc', 'dtpoc', 'vaw', 'pocpos', 'sp', 'bal')}
def bucket(c):
    q = CUT[c]; return pd.Series(np.where(F[c] <= q[0], 0, np.where(F[c] > q[1], 2, 1)), index=F.index).where(F[c].notna())
def spread(vals, label, cats=None, fmt='{:+6.2f}'):
    out, ok, parts = [], True, []
    res = []
    for lab, f in PER:
        m = f(F.index); x = vals[m]
        if cats is None: b = BK[label]; hi, lo = x[b[m] == 2], x[b[m] == 0]; hn, ln = 'top', 'bot'
        else:
            g = {k: x[(CAT[label][m] == k)] for k in cats}; ms = {k: v.mean() for k, v in g.items() if len(v) > 8}; hn, ln = max(ms, key=ms.get), min(ms, key=ms.get); hi, lo = g[hn], g[ln]
        res.append((hn, ln, hi.mean() - lo.mean(), tt(hi, lo)))
    ok = res[0][0] == res[1][0] and res[0][1] == res[1][1] and all(np.isfinite(r[3]) and abs(r[3]) >= 2 for r in res)
    return ' | '.join(f'{lab} {r[0]} over {r[1]} {r[2]:+7.2f} (t {r[3]:+5.2f})' for (lab, _), r in zip(PER, res)) + ('   <== PASS' if ok else '')
BK = {c: bucket(c) for c in CUT}; CAT = {'pos': F.pos.astype(float), 'loc': F['loc'], 'tloc': F.tloc}
print('=== TEST 1: does it dictate NOISE size? y = log(range / trailing-14d range); top third minus bottom third (positive = bigger day than the sigma model says) ===')
for c in CUT: print(f'  {c:<6}', spread(F.y, c))
print('  pos   ', spread(F.y, 'pos', cats=[0.0, 1.0])); print('  loc   ', spread(F.y, 'loc', cats=['above', 'inside', 'below'])); print('  tloc  ', spread(F.y, 'tloc', cats=['above', 'inside', 'below']))
for nm, v in (('BOTH SIDES', S), ('LONG', SL), ('SHORT', SS)):
    print(f'\n=== TEST 2: noise engine pts / day by bucket, {nm} ===')
    for c in CUT: print(f'  {c:<6}', spread(v, c))
    print('  pos   ', spread(v, 'pos', cats=[0.0, 1.0])); print('  loc   ', spread(v, 'loc', cats=['above', 'inside', 'below'])); print('  tloc  ', spread(v, 'tloc', cats=['above', 'inside', 'below']))
print('\n  bucket means (both sides, pts/day; 18-21 | 22-26):')
for c in CUT:
    print(f'  {c:<6}', ' ; '.join(f'{k}: ' + ' | '.join(f'{S[f(F.index) & (BK[c] == k)].mean():+5.2f}' for _, f in PER) for k in (0, 1, 2)))
for c, cats in (('pos', [0.0, 1.0]), ('loc', ['above', 'inside', 'below']), ('tloc', ['above', 'inside', 'below'])):
    print(f'  {c:<6}', ' ; '.join(f'{k}: ' + ' | '.join(f'{S[f(F.index) & (CAT[c] == k)].mean():+5.2f} (n {int((f(F.index) & (CAT[c] == k)).sum())})' for _, f in PER) for k in cats))
print('\n=== TEST 3: entry location vs the prior-day value area (points per trade) ===')
T['loc3'] = ''
for k, r in T.iterrows():
    d = r.day; i = ALLD.index(d) if d in SI else None
    if i is None or i == 0 or ALLD[i - 1] not in PR or PR[ALLD[i - 1]]['sym'] != SYM[d]: T.at[k, 'loc3'] = None; continue
    p = PR[ALLD[i - 1]]; px = A[d][0][0][int(r.j)]; vh, vl = p['vah'], p['val']
    T.at[k, 'loc3'] = 'with' if (px > vh if r.dir > 0 else px < vl) else ('against' if (px < vl if r.dir > 0 else px > vh) else 'inside')
T = T[T.loc3.notna() & (T.day >= '2018-02-01')]
for lab, f in PER:
    m = f(T.day); print(f'  {lab}: ' + ' | '.join(f'{k} {T[m & (T.loc3 == k)].pts.mean():+6.1f} (n {int((m & (T.loc3 == k)).sum())})' for k in ('with', 'inside', 'against')))
print('\n=== TEST 4: the strongest single filters, applied (engine pts/day, daily std, mean/std; 18-21 | 22-26). Chosen only from TEST 2 passes. ===')
def show(lab, mask):
    x = S.where(mask, 0.0); print(f'  {lab:<34} ' + ' | '.join(f'{f_}: {x[f(F.index)].mean():+5.2f} std {x[f(F.index)].std():5.1f} m/s {x[f(F.index)].mean() / x[f(F.index)].std():5.3f} (trades on {mask[f(F.index)].mean():.0%})' for f_, f in PER))
show('all days', pd.Series(True, index=F.index))
for c in CUT:
    for k in (0, 2): show(f'{c} {"bottom" if k == 0 else "top"} third only', BK[c] == k)
show('positive gamma only', F.pos == True); show('negative gamma only', F.pos == False)
for k in ('above', 'inside', 'below'): show(f'open {k} prior VA only', F['loc'] == k)
