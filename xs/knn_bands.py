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
N = len(ix); K = 30; mean_p = np.full((N, NB), np.nan); q60 = np.full((N, NB), np.nan); q75 = np.full((N, NB), np.nan); med = np.full((N, NB), np.nan); trim = np.full((N, NB), np.nan); m30 = np.full((N, NB), np.nan)
ok_row = ~np.isnan(ABv).any(axis=1)
for i in range(N):
    lo14 = max(0, i - 14); blk = ABv[lo14:i]
    if len(blk) >= 10: med[i] = np.nanmedian(blk, axis=0); s = np.sort(blk, axis=0); trim[i] = np.nanmean(s[:max(len(blk) - 2, 1)], axis=0)
    if i >= 30: m30[i] = np.nanmean(ABv[i - 30:i], axis=0)
    if i < 250: continue
    cand = np.where(ok_row[:i])[0]; d = np.sqrt(((Zf[cand] - Zf[i]) ** 2).sum(axis=1)); nn = np.argsort(d)[:K]; dd = d[nn]; h = np.median(dd) + 1e-9; w = np.exp(-(dd / h) ** 2); idx = cand[nn]; vals = ABv[idx]
    mean_p[i] = (vals * w[:, None]).sum(axis=0) / w.sum(); q60[i] = wquant(vals, w, .6); q75[i] = wquant(vals, w, .75)
print('bands built', flush=True)
VARS = {'V0 mean14 (current)': Z.SIG.copy(), 'V1 median14': med, 'V2 trimmed14 (drop 2 largest)': trim, 'V3 mean30': m30, 'V4 analog mean (k30)': mean_p, 'V5 analog q60': q60, 'V6 analog q75': q75}
ALLDAYS = [d for d in DAYS if d >= pd.Timestamp('2018-02-01') and SI[d] >= 250]; AD = pd.DatetimeIndex(ALLDAYS)
def score(label, S, ref):
    Z.SIG = S; rows = [r for d in ALLDAYS for r in Z.noise(d)]; T = pd.DataFrame(rows, columns=['d', 'm', 'pos', 'pts', 'mae']); s = T.groupby('d').pts.sum().reindex(AD).fillna(0.0); out = []; ms = []
    for nm, f in (('18-21', AD < '2022-01-01'), ('22-26', AD >= '2022-01-01')):
        x = s[f]; t = x.mean() / (x.std() / np.sqrt(len(x))); out.append(f'{nm}: {x.mean():+5.2f}/d t {t:+5.2f} std {x.std():5.1f} m/s {x.mean() / x.std():.3f}'); ms.append(x.mean() / x.std())
    ok = ref is not None and ms[0] > ref[0] and ms[1] > ref[1]
    fc = np.corrcoef(np.log(np.where(np.isfinite(S[[SI[d] for d in ALLDAYS], -1]) & (S[[SI[d] for d in ALLDAYS], -1] > 0), S[[SI[d] for d in ALLDAYS], -1], np.nan))[~np.isnan(S[[SI[d] for d in ALLDAYS], -1])], np.log(end_abs[[SI[d] for d in ALLDAYS]] + 1e-6)[~np.isnan(S[[SI[d] for d in ALLDAYS], -1])])[0, 1]
    print(f'  {label:<32} ' + ' | '.join(out) + f' | trades {len(T)} | corr(log band_end, log |day move|) {fc:+.2f}' + ('   <== beats baseline both' if ok else ''), flush=True); return ms
ref = None
for lab, S in VARS.items():
    ms = score(lab, S, ref)
    if ref is None: ref = ms
