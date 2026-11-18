"""Nov 18 2026 report 305. A GAP in everything tested so far: market INTERNALS. Day type (trend vs rotation) is not predictable from VIX/IV/gamma/gap/ER/PCA (reports 252-257, 270: AUC <= .57) and the algo is a trend-day engine (285: M1 +36 / V15a +61 / RA +26 pts on TREND days, -2..-25 on MR days).
THEORY (declared before running). The index is a basket; an index trend-day needs many constituents moving together (basket/program/ETF-creation flow, macro flow), a rotation day is stock-specific flow that nets out. Physics analogue: an Ising/Kuramoto ORDER PARAMETER -- aggregate alignment of the components.
 C1h = |mean ret| / mean |ret| over the 8 largest NQ names from 09:30 to 10:30 (1 = all move together).  Hypotheses (directions fixed): (H1) high C1h -> higher efficiency of the REST of the day (10:30-16:00) and larger rest-of-day range/ADR; (H2) the rest-of-day return continues the sign of the first-hour composite, more so when C1h is high; (H3) ex-ante, yesterday's full-day coherence and its 5-day mean predict today's algo pts/day (no look-ahead).
 Why it should work: basket flow is persistent and cannot be arbitraged away intraday (creation/redemption lags, benchmark rebalancing); why it would stop: 8 names ~ 40% of NQ weight, so one earnings mover (NVDA) can fake coherence; algos that trade baskets compress the edge; COVID-style macro days are coherent but whipsaw.
 H4 (separate): early participation: NQ volume 09:30-10:30 relative to its 20d median predicts rest-of-day range/ADR (size) but not efficiency (direction), as in 257 for IV.
Split: train 2019-21 | test 2018 + 2022-Aug 2026. Spearman and AUC (top third vs bottom third efficiency), day-level."""
import numpy as np, pandas as pd, glob, pickle, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize()
g = N.groupby('day'); rows = []
for d, x in g:
    x = x.sort_values('m')
    if len(x) < 70 or x.m.iloc[0] != 570: continue
    o = x.open.iloc[0]; a = x[x.m <= 625]; r = x[x.m > 625]
    if len(r) < 40 or len(a) < 12: continue
    p = a.close.iloc[-1]; e = x.close.iloc[-1]; path = r.close.diff().abs().sum() + abs(r.close.iloc[0] - p)
    rows.append(dict(day=d, o=o, p1030=p, end=e, hi=x.high.max(), lo=x.low.min(), fh=p - o, rest=e - p, eff=abs(e - p) / path if path > 0 else np.nan, rrange=r.high.max() - r.low.min(), vol1h=a.volume.sum(), rng=x.high.max() - x.low.min(), net=e - o, pathd=x.close.diff().abs().sum()))
D = pd.DataFrame(rows).set_index('day'); D['adr'] = D.rng.rolling(14).mean().shift(1); D['rr'] = D.rrange / D.adr; D['rest_n'] = D.rest / D.adr; D['rvol'] = D.vol1h / D.vol1h.rolling(20).median().shift(1); D['effday'] = D.net.abs() / D.pathd
# --- megacap internals (15m, ET)
NAMES = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AVGO', 'TSLA']; R1 = {}; RD = {}
for t in NAMES:
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{t}_15m_*')[0]; H = pd.read_parquet(f); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H['m'] = H.dt.dt.hour * 60 + H.dt.dt.minute; H['day'] = H.dt.dt.normalize(); H = H[(H.m >= 570) & (H.m <= 945)]
    o = H[H.m == 570].set_index('day').open; c1 = H[H.m == 615].set_index('day').close; cl = H[H.m == 945].set_index('day').close
    R1[t] = c1 / o - 1; RD[t] = cl / o - 1
R1 = pd.DataFrame(R1).dropna(); RD = pd.DataFrame(RD).dropna()
coh = lambda R: R.mean(axis=1).abs() / R.abs().mean(axis=1).replace(0, np.nan)
D = D.join(pd.DataFrame({'C1h': coh(R1), 'comp1h': R1.mean(axis=1), 'agree': R1.apply(np.sign).mean(axis=1).abs(), 'Cday': coh(RD), 'compday': RD.mean(axis=1)}), how='inner')
D['Cday_y'] = D.Cday.shift(1); D['C5'] = D.Cday.rolling(5).mean().shift(1); D = D.dropna(subset=['eff', 'rr', 'adr', 'C1h', 'rvol'])
D['tr'] = (D.index >= '2019-01-01') & (D.index < '2022-01-01'); D['te'] = ~D.tr
print(f'days {len(D)} train {int(D.tr.sum())} test {int(D.te.sum())}  | mean C1h {D.C1h.mean():.2f}  mean Cday {D.Cday.mean():.2f}')
def sp(a, b, m): z = D[m][[a, b]].dropna(); r, p = spearmanr(z[a], z[b]); return f'{r:+.3f}(p{p:.2f})'
def auc(f, m):
    z = D[m][[f, 'eff']].dropna(); lo, hi = z.eff.quantile([1 / 3, 2 / 3]); y = z[(z.eff <= lo) | (z.eff >= hi)]; return roc_auc_score((y.eff >= hi).astype(int), y[f])
print('\n[H1] first-hour features vs REST-of-day (10:30-16:00)  | Spearman  train | test    AUC(TREND vs MR rest-of-day) train | test')
for f in ('C1h', 'agree', 'rvol'):
    print(f'  {f:6s} -> eff {sp(f, "eff", D.tr)} | {sp(f, "eff", D.te)}   AUC {auc(f, D.tr):.3f} | {auc(f, D.te):.3f}   -> range/ADR {sp(f, "rr", D.tr)} | {sp(f, "rr", D.te)}')
print('\n[H2] rest-of-day return (in ADR) continuing the first-hour composite sign, by C1h tercile (cuts: train)  mean in sign units, t')
q = D.loc[D.tr, 'C1h'].quantile([1 / 3, 2 / 3]).values
for lab, m in (('train', D.tr), ('test ', D.te)):
    z = D[m].copy(); z['b'] = np.where(z.C1h <= q[0], 'low', np.where(z.C1h >= q[1], 'high', 'mid')); z['x'] = np.sign(z.comp1h) * z.rest_n; out = []
    for b in ('low', 'mid', 'high'): s = z[z.b == b].x; out.append(f'{b} {s.mean():+.3f} (t{s.mean() / (s.std() / np.sqrt(len(s))):+.1f}, n{len(s)})')
    print(' ', lab, ' | '.join(out))
# --- H3 ex-ante vs algo P&L
L = pickle.load(open(SP + 'legs287_ext.pkl', 'rb')); P = {}
for k in ('RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1'):
    v = L[k].copy(); v['day'] = pd.to_datetime(v.day).dt.normalize(); P[k] = v.groupby('day').pts.sum()
P = pd.DataFrame(P).fillna(0); P['ALL'] = P.sum(axis=1); P = P.reindex(D.index).fillna(0); D2 = D.join(P)
print('\n[H3] EX-ANTE: yesterday Cday and 5-day mean C vs today\'s pts/day (Spearman train | test), plus pts/day by tercile (cuts train)')
for f in ('Cday_y', 'C5'):
    qq = D2.loc[D2.tr, f].quantile([1 / 3, 2 / 3]).values
    for k in ('ALL', 'M1', 'V15a', 'RA', 'RULES'):
        z = D2.dropna(subset=[f]); a = z[z.tr]; b = z[z.te]; ra = spearmanr(a[f], a[k])[0]; rb = spearmanr(b[f], b[k])[0]
        tb = lambda w: ' / '.join(f'{w[(w[f] <= qq[0])][k].mean():+.1f}' if i == 0 else (f'{w[(w[f] >= qq[1])][k].mean():+.1f}' if i == 2 else f'{w[(w[f] > qq[0]) & (w[f] < qq[1])][k].mean():+.1f}') for i in range(3))
        print(f'  {f:7s} {k:6s} rho {ra:+.3f} | {rb:+.3f}   pts/day low/mid/high  train {tb(a)}  | test {tb(b)}')
D2.to_parquet(SP + 'd305.parquet')
