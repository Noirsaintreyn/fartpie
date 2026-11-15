"""Nov 15 2026 CAN THE OUTLIER FEATURES TRANSFER TO LARGE CAPS (Rey). Declared before results. Mechanism from the small-cap work: tails need thin supply and high volatility, so the prior is that they do NOT transfer to large caps (little tail to catch); the question is whether the same three features (ATRP = ATR/price, EXT = close/60d low - 1, turnover) still RANK large-cap flips on the mean and on the >= +10% flag. PART 1 daily flips, LARGE-CAP = 20d median dollar volume >= $100M and price >= $10 (572 train / 1,515 test flips), also MID = $20-100M and price >= $10 for contrast: AUC of each feature for the >= +10% flag (BIG is too rare), Spearman IC to net return, and quintile means of the OS score (report_118 train-ECDF scale) on TRAIN 2019-21 / TEST 2022-26; declared sign + on all three; lead bar: top-quintile minus bottom-quintile positive on BOTH eras and AUC deviates >= 0.04 in the declared direction on both. PART 2 the 1H large-cap flips (report_110 set: 7,190 flips, 132 names, 2023-12..2026-11, entry at the flip-bar close, 10bp, hold to the 1H bear flip): join ATRP and EXT from the daily bars at the prior close (turnover not available), report net and drift-excess (trade - names' per-bar drift x bars held) by ATRP, EXT and ATRP+EXT rank-sum terciles, TRAIN <= 2024-12-31 / TEST 2025+, with day-clustered t of top minus bottom tercile. PART 3 one slot-design check on the large-cap 1H flips: queue vs first-come at N = 10, 10% each, ranked by ATRP+EXT within the weekly pool is NOT run unless PART 2 shows a lead."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, rankdata
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; Of = pd.read_pickle(X + 'outlier_features.pkl'); Of['LTURN'] = np.log(Of.TURN.clip(lower=1e-9)); tr = Of[~Of.test]
def ecdf(col): s = np.sort(tr[col].dropna().values); return lambda x: np.searchsorted(s, x, side='right') / len(s)
Of['OS'] = np.where(Of[['ATRP', 'EXT', 'LTURN']].notna().all(axis=1), ecdf('ATRP')(Of.ATRP.values) + ecdf('EXT')(Of.EXT.values) + ecdf('LTURN')(Of.LTURN.values), np.nan); Of['n'] = Of.r - .003; Of['w10'] = Of.n >= .1
def auc(x, y):
    m = x.notna(); x = x[m]; y = y[m].astype(bool)
    if y.sum() < 5: return np.nan
    r = rankdata(x); return (r[y.values].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
for lab, U in (('LARGE-CAP (>= $100M, px >= $10)', Of[(Of.dv >= 1e8) & (Of.px >= 10)]), ('MID (>= $20M < $100M, px >= $10)', Of[(Of.dv >= 2e7) & (Of.dv < 1e8) & (Of.px >= 10)])):
    print(f'\n######## {lab} ########')
    for era, z in (('TRAIN', U[~U.test]), ('TEST ', U[U.test])):
        print(f' {era} n {len(z)} mean {z.n.mean()*100:+.2f}% win10 {z.w10.mean()*100:.1f}% | ' + ' '.join(f'{f}: AUC10 {auc(z[f], z.w10):.3f} IC {spearmanr(z[f], z.n, nan_policy="omit")[0]:+.3f}' for f in ('ATRP', 'EXT', 'LTURN', 'OS')))
        zz = z.dropna(subset=['OS']); zz = zz.assign(q=pd.qcut(zz.OS, 5, labels=False, duplicates='drop')); print('       OS quintile 1..5 mean net: ' + ' '.join(f'{zz[zz.q==k].n.mean()*100:+6.2f}%(n{(zz.q==k).sum()})' for k in range(5)) + ' | win10: ' + ' '.join(f'{zz[zz.q==k].w10.mean()*100:4.1f}%' for k in range(5)))
H = pd.read_pickle(X + 'lc_1h.pkl'); D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); fx = {}
for tk in H.tk.unique():
    q = D.get(tk)
    if q is None: continue
    q = q[['high', 'low', 'close']].copy(); q.index = pd.to_datetime(q.index).tz_localize(None).normalize(); q = q[~q.index.duplicated()]; c = q.close; pc = c.shift(); tr_ = pd.concat([q.high - q.low, (q.high - pc).abs(), (q.low - pc).abs()], axis=1).max(axis=1); fx[tk] = pd.DataFrame(dict(ATRP=(tr_.rolling(14).mean() / c).shift(1), EXT=(c / q.low.rolling(60).min() - 1).shift(1)))
rows = []
for tk, g in H.groupby('tk'):
    f = fx.get(tk)
    if f is None: continue
    j = f.reindex(g.d.values); rows.append(pd.concat([g.reset_index(drop=True), j.reset_index(drop=True)], axis=1))
H = pd.concat(rows).dropna(subset=['ATRP', 'EXT']); H['ex'] = H.v - H.drift * H.hv; H['era'] = np.where(H.d <= '2024-12-31', 'TRAIN', 'TEST'); print(f'\n######## PART 2: 1H large-cap flips with daily ATRP/EXT: n {len(H)} ########')
def cl_(z, col, hi_, lo_): g = z.assign(x=z[col]).groupby(z.d.astype(str)).apply(lambda w: w[w.s >= hi_].x.mean() - w[w.s <= lo_].x.mean()).dropna(); return g.mean() * 100, g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 5 else np.nan
for era in ('TRAIN', 'TEST'):
    z = H[H.era == era].copy(); z['s'] = (z.ATRP.rank(pct=True) + z.EXT.rank(pct=True)); print(f'\n {era} n {len(z)} mean net {z.v.mean()*100:+.2f}% excess {z.ex.mean()*100:+.2f}%')
    for col in ('ATRP', 'EXT', 's'):
        q1, q2 = z[col].quantile([1 / 3, 2 / 3]); t = z[z[col] >= q2]; b = z[z[col] <= q1]; z2 = z.assign(s2=z[col]); m, tt = cl_(z2.assign(s=z2.s2), 'v', q2, q1); print(f'  {col:5s} top tercile n {len(t)} net {t.v.mean()*100:+6.2f}% ex {t.ex.mean()*100:+6.2f}% win10 {(t.v>=.1).mean()*100:4.1f}% | bottom n {len(b)} net {b.v.mean()*100:+6.2f}% ex {b.ex.mean()*100:+6.2f}% win10 {(b.v>=.1).mean()*100:4.1f}% | top-bottom (day-clustered) {m:+.2f}pp t {tt:+.2f}')
