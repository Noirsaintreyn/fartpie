"""Nov 10 2026: (1) CVD PROXIED FROM 5m OHLCV ON 12 YEARS, (2) ES-ALREADY-FLIPPED CONFIRMATION FOR NQ FLIPS. DECLARED BEFORE RESULTS.
(1) bar delta proxy = volume x (2 (close - low)/(high - low) - 1) (close-location), and the sign-of-bar proxy = volume x sign(close - open). VALIDATION: correlation of each proxy with the REAL MNQ 5m delta (sum of 1-min signed volume, Dec 2024 - Nov 2025). If the proxy tracks real delta (r >= 0.5), run the order-flow features on the full noise trade set: d5 (signal bar proxy delta / volume, aligned with the trade), d30 (6 bars), dsess (session cumulative / session volume), div (new session extreme in the trade direction with negative aligned d30). Thirds / PASS rule: same sign and |t| >= 2 in 2014-21 and 2022-26, n >= 30; chance from 200 shuffles.
(2) For every NQ noise trade, ES band state at the signal bar in the trade direction: WITH (ES already outside its band in the same direction), INSIDE, AGAINST; also ES flipped EARLIER than NQ (ES with at signal bar and NQ's first break of the day later than ES's). Points per trade by ES state, both periods; sizing test on the long-only leg and on both sides: x1.5 when ES with, x0.5 otherwise vs x1.0 baseline (mean/std, pts/day both periods)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, DAYS, GRID, NB, OPEN, PC, SIG, SI
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; ES = pickle.load(open(f'{SP}/es_state.pkl', 'rb'))
def proxy(d):
    o, h, l, c, v, vw = A[d][0]; rg = np.where(h > l, h - l, np.nan); return np.nan_to_num(v * (2 * (c - l) / rg - 1)), np.nan_to_num(v * np.sign(c - o)), np.nan_to_num(v)
# ---- validation vs real delta
O = pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet'); O['dt'] = pd.to_datetime(O.datetime).dt.tz_convert('America/New_York').dt.tz_localize(None); O['day'] = O.dt.dt.normalize(); O['m'] = O.dt.dt.hour * 60 + O.dt.dt.minute
O = O[(O.m >= 570) & (O.m < 960)]; O['bar'] = (O.m - 570) // 5; real = O.groupby(['day', 'bar']).signed_vol.sum(); R1, R2 = [], []; P1, P2 = [], []
for d in sorted(set(O.day)):
    if d not in A: continue
    a, b, v = proxy(d); r = real.loc[d].reindex(range(NB)).fillna(0).values; P1 += list(a); P2 += list(b); R1 += list(r)
print(f'validation: corr(real 5m delta, close-location proxy) {np.corrcoef(R1, P1)[0, 1]:+.2f}; corr(real, sign-of-bar proxy) {np.corrcoef(R1, P2)[0, 1]:+.2f}  ({len(R1)} bars)', flush=True)
# ---- trades
rows = []
for d in DAYS:
    a, b, v = proxy(d); cs = np.cumsum(a); cv = np.cumsum(v); o_, h_, l_, c_, v_, vw_ = A[d][0]; es = ES.get(d); firstbrk = None
    for (dd, em, pos, pts, mae) in Z.noise(d):
        jb = em - 1
        if jb < 7: continue
        d5 = pos * a[jb] / max(v[jb], 1); d30 = pos * a[jb - 5:jb + 1].sum() / max(v[jb - 5:jb + 1].sum(), 1); dsess = pos * cs[jb] / max(cv[jb], 1)
        newx = (pos > 0 and h_[jb] >= np.nanmax(h_[:jb + 1])) or (pos < 0 and l_[jb] <= np.nanmin(l_[:jb + 1]))
        st = (es[jb] * pos) if es is not None and len(es) == NB else np.nan; esearly = np.nan
        if es is not None and len(es) == NB:
            ix = np.where(es * pos > 0)[0]; esearly = float(len(ix) > 0 and ix[0] < jb and es[jb] * pos > 0)
        rows.append(dict(day=d, pos=pos, pts=pts, d5=d5, d30=d30, dsess=dsess, div=float(newx and d30 < 0), es=st, esearly=esearly))
T = pd.DataFrame(rows); T['day'] = pd.to_datetime(T.day); print(len(T), 'noise trades', flush=True)
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
def flags(t, pts):
    out = {}; e = (t.day < '2022-01-01').values; l_ = ~e
    for c in ('d5', 'd30', 'dsess'):
        col = t[c]; q = col[e].quantile([1 / 3, 2 / 3]).values
        for nm, sel in (('low third', col <= q[0]), ('high third', col > q[1])):
            out[f'{c} {nm}'] = tuple((pts[mm & sel.values].mean() - pts[mm & ~sel.values].mean(), tt(pts[mm & sel.values], pts[mm & ~sel.values]), int((mm & sel.values).sum())) for mm in (e, l_))
    sel = (t['div'] == 1).values; out['divergence'] = tuple((pts[mm & sel].mean() - pts[mm & ~sel].mean(), tt(pts[mm & sel], pts[mm & ~sel]), int((mm & sel).sum())) for mm in (e, l_)); return out
def passed(r): return np.sign(r[0][0]) == np.sign(r[1][0]) and all(np.isfinite(x[1]) and abs(x[1]) >= 2 and x[2] >= 30 for x in r)
rng = np.random.default_rng(4)
for lab, t in (('all trades', T), ('long only', T[T.pos > 0].reset_index(drop=True))):
    fl = flags(t, t.pts); ps = {k: v for k, v in fl.items() if passed(v)}; e = (t.day < '2022-01-01').values; null = []
    for _ in range(200):
        sh = t.pts.copy()
        for mm in (e, ~e): sh[mm] = rng.permutation(sh[mm].values)
        null.append(sum(passed(v) for v in flags(t, sh).values()))
    print(f'\\nCVD-proxy features, {lab}: {len(t)} trades, {len(fl)} tests, real PASS {len(ps)} vs chance {np.mean(null):.1f} (95th pct {np.percentile(null, 95):.0f})')
    for k, v in sorted(fl.items(), key=lambda kv: -abs(kv[1][0][1]) - abs(kv[1][1][1]))[:5]: print(f'    {k:<16} {v[0][0]:+7.1f} (t {v[0][1]:+5.2f}, n{v[0][2]}) | {v[1][0]:+7.1f} (t {v[1][1]:+5.2f}, n{v[1][2]}){"   <== PASS" if passed(v) else ""}')
print('\\n=== ES confirmation: NQ noise trades by ES band state at the signal bar (in the trade direction) ===')
for lab, t in (('all', T.dropna(subset=['es'])), ('long only', T[T.pos > 0].dropna(subset=['es']))):
    print(f'  {lab}:')
    for k, nm in ((1, 'ES with (already outside same way)'), (0, 'ES inside'), (-1, 'ES against')):
        a = [t[(t.es == k) & (t.day < '2022-01-01')].pts, t[(t.es == k) & (t.day >= '2022-01-01')].pts]; print(f'    {nm:<36} {a[0].mean():+6.1f} (n{len(a[0])}) | {a[1].mean():+6.1f} (n{len(a[1])})')
    for f_, nm in ((t.es == 1, 'ES with'), (t.esearly == 1, 'ES flipped EARLIER than the signal bar')):
        r = [(t[f_ & m].pts.mean() - t[~f_ & m].pts.mean(), tt(t[f_ & m].pts, t[~f_ & m].pts)) for m in (t.day < '2022-01-01', t.day >= '2022-01-01')]
        ok = np.sign(r[0][0]) == np.sign(r[1][0]) and all(abs(x[1]) >= 2 for x in r); print(f'    {nm}: with minus without {r[0][0]:+.1f} (t {r[0][1]:+.2f}) | {r[1][0]:+.1f} (t {r[1][1]:+.2f}){"   <== PASS" if ok else ""}')
    for nm, fw, fo in (('x1.5 ES with / x0.5 otherwise', 1.5, .5), ('x1.5 ES with / x1.0 otherwise', 1.5, 1.0), ('x1.0 with / x0.5 otherwise', 1.0, .5)):
        o = []
        for p_, m in (('14-21', t.day < '2022-01-01'), ('22-26', t.day >= '2022-01-01')):
            x = t[m]; w = np.where(x.es == 1, fw, fo); s = (x.pts * w).groupby(x.day).sum().reindex(pd.DatetimeIndex(sorted(T.day[(T.day < '2022-01-01') if p_ == '14-21' else (T.day >= '2022-01-01')].unique()))).fillna(0); b0 = x.pts.groupby(x.day).sum().reindex(s.index).fillna(0); o.append(f'{p_}: {s.mean():+5.2f}/d m/s {s.mean() / s.std():.3f} (base {b0.mean():+5.2f}/d {b0.mean() / b0.std():.3f})')
        print(f'    sizing {nm:<30} ' + ' | '.join(o))
