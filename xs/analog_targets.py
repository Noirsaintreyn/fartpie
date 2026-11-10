"""Nov 10 2026: ANALOG (nearest-neighbour) NOISE BANDS, INSPIRED BY THE WAVE / ODDS SCREENSHOTS. DECLARED BEFORE RESULTS.
The engine's band half-width per time of day is sigma_t = mean over the prior 14 sessions of |close_t/open - 1|. Replace it with (V1) the median, (V2) a trimmed mean that drops the 2 largest of the 14 (outlier removal), (V3) 30-session mean, and (V4..V6) an ANALOG profile: for day d find the k=30 earlier days with the closest standardised 09:30 context (log HAR d/w/m range, gap/ADR, |prior return|, log of the 14-day sigma at the close, VIX, QQQ ATM IV, GEX pct, DEX pct; Gaussian kernel weights, past days only, >= 250 days of history) and use their weighted mean (V4), weighted 60th percentile (V5), weighted 75th percentile (V6) of |close_t/open - 1| at each time of day. Everything else in the engine (entries, VWAP trailing exit, 40-pt stop, 30-min decisions) is unchanged.
Score: engine pts/day (cost 0.5), t, mean/std, 2018-21 | 2022-26. Also the forecast quality: corr (log) of each band's end-of-day value with the day's actual |close/open - 1|. PASS for a variant = beats the baseline mean/std in BOTH periods."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, DAYS, NB, OPEN, CLOSE, PC, AB, SI
U = '/home/claude/fartpie'
vx = pd.read_parquet(f'{U}/data/yahoo/daily_v/VIX.parquet'); vx = vx['Close'] if 'Close' in vx else vx.iloc[:, 0]; vx.index = pd.to_datetime(vx.index).normalize()
G1 = pd.read_csv(f'{U}/options_greeks_daily_QQQ.csv', parse_dates=['date']).set_index('date').sort_index(); G1.index = G1.index.normalize()
OWN = pd.read_csv(f'{U}/xs/ext/gex/QQQ.csv', parse_dates=['date']).set_index('date').net_gex.sort_index(); OWN.index = OWN.index.normalize()
ix = pd.DatetimeIndex(ALLD); rng = pd.Series({d: np.nanmax(A[d][0][1]) - np.nanmin(A[d][0][2]) for d in ALLD}); lr = np.log(rng / OPEN)
pct = lambda s: s.rolling(252, min_periods=120).rank(pct=True); sg14 = pd.Series(Z.SIG[:, -1], index=ix)
F = pd.DataFrame(index=ix); F['d'] = lr.shift(1); F['w'] = np.log(rng.rolling(5).mean().shift(1) / OPEN); F['m'] = np.log(rng.rolling(22).mean().shift(1) / OPEN); F['gap'] = ((OPEN - PC) / rng.rolling(20).mean().shift(1)).abs()
F['pr'] = (CLOSE / OPEN - 1).abs().shift(1); F['sg'] = np.log(sg14); F['vix'] = np.log(vx.shift(1).reindex(ix, method='ffill')); F['iv'] = np.log(G1.atm_iv).shift(1).reindex(ix, method='ffill')
F['gex'] = pct(OWN).shift(1).reindex(ix, method='ffill'); F['dex'] = pct(G1.call_dex + G1.put_dex).shift(1).reindex(ix, method='ffill')
Zf = ((F - F.median()) / F.std()).fillna(0.0).clip(-4, 4).values; ABv = AB.values; end_abs = np.abs(CLOSE / OPEN - 1).values
def wquant(vals, w, q):
    o = np.argsort(vals, axis=0); sv = np.take_along_axis(vals, o, axis=0); sw = w[o]; cw = np.cumsum(sw, axis=0) / sw.sum(axis=0); return np.array([sv[np.searchsorted(cw[:, j], q), j] for j in range(vals.shape[1])])

N = len(ix); K = 30; hi_ = np.array([np.nanmax(A[d][0][1]) for d in ALLD]) / OPEN.values - 1; lo_ = 1 - np.array([np.nanmin(A[d][0][2]) for d in ALLD]) / OPEN.values
QU = {q: np.full(N, np.nan) for q in (.25, .5, .75)}; QD = {q: np.full(N, np.nan) for q in (.25, .5, .75)}; ok_row = ~np.isnan(ABv).any(axis=1)
def wq(v, w, q):
    o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum(); return v[o][np.searchsorted(cw, q)]
for i in range(250, N):
    cand = np.where(ok_row[:i])[0]; d = np.sqrt(((Zf[cand] - Zf[i]) ** 2).sum(axis=1)); nn = np.argsort(d)[:K]; dd = d[nn]; h = np.median(dd) + 1e-9; w = np.exp(-(dd / h) ** 2); idx = cand[nn]
    for q in (.25, .5, .75): QU[q][i] = wq(hi_[idx], w, q); QD[q][i] = wq(lo_[idx], w, q)
print('analog excursions built', flush=True)
COST = Z.COST; GRID = Z.GRID
def noise2(d, tpq=None, tpmult=None, skipq=None, stop=40.0, step=30, start=600, end=930):
    a, _ = A[d]; o, h, l, c, v, vw = a; s = Z.SIG[SI[d]]; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; i = SI[d]; rows = []; pos = 0; e = em = None; worst = 0.0; pend = None; tpL = tpS = None
    if tpq is not None: tpL, tpS = op * (1 + QU[tpq][i]), op * (1 - QD[tpq][i])
    elif tpmult is not None and np.isfinite(s[-1]): tpL, tpS = op * (1 + tpmult * s[-1]), op * (1 - tpmult * s[-1])
    sk = (op * (1 + QU[skipq][i]), op * (1 - QD[skipq][i])) if skipq is not None else None
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None: pos, e, em, worst = pend, o[j], j, 0.0; pend = None
        if pos != 0:
            lo = (l[j] - e) if pos > 0 else (e - h[j]); worst = min(worst, lo)
            if stop is not None and lo <= -stop: rows.append((d, em, pos, -stop - COST, worst - COST)); pos = 0; continue
            tgt = tpL if pos > 0 else tpS
            if tgt is not None and np.isfinite(tgt) and ((pos > 0 and h[j] >= tgt > e) or (pos < 0 and l[j] <= tgt < e)): rows.append((d, em, pos, (tgt - e) * pos - COST, worst - COST)); pos = 0; continue
        if j == NB - 1 or np.isnan(o[j + 1]):
            if j == NB - 1 and pos != 0: rows.append((d, em, pos, (c[j] - e) * pos - COST, worst - COST)); pos = 0
            continue
        t = GRID[j]
        if (t + 5) % step != 0 or t + 5 < start or t + 5 > end or not np.isfinite(s[j]): continue
        ub, lb = max(op, pc) * (1 + s[j]), min(op, pc) * (1 - s[j]); nx = o[j + 1]
        if pos == 1 and c[j] < max(ub, vw[j]): rows.append((d, em, 1, nx - e - COST, worst - COST)); pos = 0
        elif pos == -1 and c[j] > min(lb, vw[j]): rows.append((d, em, -1, e - nx - COST, worst - COST)); pos = 0
        if pos == 0 and c[j] > ub and (sk is None or c[j] < sk[0]): pend = 1
        elif pos == 0 and c[j] < lb and (sk is None or c[j] > sk[1]): pend = -1
    return rows
ALLDAYS = [d for d in DAYS if d >= pd.Timestamp('2018-02-01') and SI[d] >= 250]; AD = pd.DatetimeIndex(ALLDAYS); base = None
def score(label, **kw):
    global base
    T = pd.DataFrame([r for d in ALLDAYS for r in noise2(d, **kw)], columns=['d', 'm', 'pos', 'pts', 'mae']); s = T.groupby('d').pts.sum().reindex(AD).fillna(0.0); out = []; ms = []
    for nm, f in (('18-21', AD < '2022-01-01'), ('22-26', AD >= '2022-01-01')):
        x = s[f]; t = x.mean() / (x.std() / np.sqrt(len(x))); out.append(f'{nm}: {x.mean():+5.2f}/d t {t:+5.2f} std {x.std():5.1f} m/s {x.mean() / x.std():.3f}'); ms.append(x.mean() / x.std())
    if base is None: base = ms
    print(f'  {label:<44} ' + ' | '.join(out) + f' | trades {len(T)} avg {T.pts.mean():+.1f}' + ('   <== beats baseline both' if ms[0] > base[0] and ms[1] > base[1] else ''), flush=True)
print('\nnoise engine with exits / skips from the analog excursion distribution (target = analog quantile of the day\'s high / low from the open)')
score('baseline (no target)')
for q in (.5, .75): score(f'TP at analog q{int(q*100)} excursion', tpq=q)
score('TP at analog q25 excursion', tpq=.25)
for m in (1.0, 1.5, 2.0): score(f'CONTROL: TP at {m} x fixed sigma_end', tpmult=m)
for q in (.5, .75): score(f'skip entry beyond analog q{int(q*100)} (no room)', skipq=q)
score('TP q75 + skip q75', tpq=.75, skipq=.75); score('TP q50 + skip q75', tpq=.5, skipq=.75)
