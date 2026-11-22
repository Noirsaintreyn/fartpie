"""Nov 22 2026 report 358. Rey: can ATRP, HAR or Yang-Zhang improve the slot ranking, or a MIX of them (the ranked list idea)?  Declared before running.
Pick order among same-bar flips only (6 slots, 8% breaker, 2+2bp, 54 names; live order = ratio, train 33.8 | test 17.5, 23.6%/-23.5%/2.60).  All features use data up to the PRIOR close.
 ATRP = Wilder ATR14 / close (daily bars built from the 15m RTH cache).  YZ = Yang-Zhang 20d volatility (overnight, open-close, Rogers-Satchell, k = 0.34/(1.34+(n+1)/(n-1))).
 HAR = fixed-weight HAR-RV forecast, NO fitting: sqrt(0.3 RVd + 0.4 mean(RV,5d) + 0.3 mean(RV,22d)), RV = sum of squared 15m log returns per day.
 Each as LEVEL and as RELATIVE (own value / own trailing 252d median), HIGH-first and LOW-first.
 MIXES (percentile ranks across all candidate flips, equal weights): M1 = mean(ATRP, YZ, HAR) levels; M2 = M1 + ratio rank (ratio high-first); M3 = mean of the three RELATIVE vols + ratio; each direction.  The ratio order stays the tie-break inside a mix.
THEORY: report 136 found high-ATRP first hurts large caps (not the small-cap supply regime); the HAR/YZ forecasts are cleaner volatility estimates, so if volatility has any ranking value in this book they should show it; low-first = calm persistent trenders, high-first = bigger excursions.
Measured: CAGR / DD / Sharpe, train | test CAGR, mean taken-trade return."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
_s = open('/home/claude/fartpie/xs/report_349_swap_scaleout.py').read(); exec(_s[:_s.index("TK, _ = sim_swap(B0, None)")])
feat = []
for tk in B0.tk.unique():
    f = sorted(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index()
    H['d'] = H.index.normalize(); H['lr2'] = np.log(H.close / H.close.shift()).where(H.d == H.d.shift(), np.nan) ** 2
    Dd = H.groupby('d').agg(o=('open', 'first'), h=('high', 'max'), l=('low', 'min'), c=('close', 'last'), rv=('lr2', 'sum'))
    pc = Dd.c.shift(); tr = np.maximum(Dd.h - Dd.l, np.maximum((Dd.h - pc).abs(), (Dd.l - pc).abs())); atr = tr.ewm(alpha=1 / 14, adjust=False).mean(); Dd['atrp'] = atr / Dd.c
    n = 20; k = .34 / (1.34 + (n + 1) / (n - 1)); ov = np.log(Dd.o / pc); oc = np.log(Dd.c / Dd.o); rs = np.log(Dd.h / Dd.c) * np.log(Dd.h / Dd.o) + np.log(Dd.l / Dd.c) * np.log(Dd.l / Dd.o)
    Dd['yz'] = np.sqrt(ov.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean())
    Dd['har'] = np.sqrt(.3 * Dd.rv + .4 * Dd.rv.rolling(5).mean() + .3 * Dd.rv.rolling(22).mean())
    for c in ('atrp', 'yz', 'har'): Dd[c + '_r'] = Dd[c] / Dd[c].rolling(252, min_periods=100).median()
    Dd = Dd[['atrp', 'yz', 'har', 'atrp_r', 'yz_r', 'har_r']].shift(1); Dd['tk'] = tk; feat.append(Dd.reset_index())
Fd = pd.concat(feat); B0['d'] = B0.entry.dt.normalize(); B0 = B0.merge(Fd, on=['tk', 'd'], how='left')
print('feature coverage', {c: f'{B0[c].notna().mean():.0%}' for c in ('atrp', 'yz', 'har', 'atrp_r', 'yz_r', 'har_r')})
P = lambda c: B0[c].rank(pct=True).fillna(.5)
B0['m1'] = (P('atrp') + P('yz') + P('har')) / 3; B0['m2'] = (B0.m1 + B0.ratio.rank(pct=True)) / 2; B0['m3'] = ((P('atrp_r') + P('yz_r') + P('har_r')) / 3 + B0.ratio.rank(pct=True)) / 2
def run(col, hi):
    global order
    def order(Bs):
        k = (-Bs[col].fillna(Bs[col].median()).values if hi else Bs[col].fillna(Bs[col].median()).values) if col else Bs.brank.values + 1e-6 * Bs.ratio.rank(ascending=False).values
        return Bs.assign(_k=k).sort_values(['entry', '_k', 'tk'], kind='stable')
    TK, _ = sim_swap(B0, None); return TK
def line(lab, TK): show(lab, TK); print(f'{"":22s} mean taken-trade {np.mean([t[3] for t in TK]):+.3%}')
line('LIVE (ratio order)', run(None, True))
for col in ('atrp', 'yz', 'har', 'atrp_r', 'yz_r', 'har_r', 'm1', 'm2', 'm3'):
    for hi in (True, False): line(f'{col} {"HIGH" if hi else "LOW"} first', run(col, hi))
