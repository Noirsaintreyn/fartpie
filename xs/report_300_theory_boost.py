"""Nov 18 2026 report 300 (Rey: redo the spot-boost list with signals that have a THEORY and were NOT found by looking at this data; no peeking at 2023-26). PRE-REGISTERED before any run:
Boost signals (each +K spots up the ratio-ranked list when on; K = 2 fixed a priori, K sweep reported only as sensitivity):
  TREND   close > 200-day moving average                                  (time-series momentum / Faber: persistent trends; fails in whipsaw/range markets, sharp reversals)
  MOM     12-1 month return in the top third of the 54 names that day     (Jegadeesh-Titman; fails at momentum crashes / junk rallies / rotations)
  HI52    close >= 90% of 252-day high                                    (George-Hwang: anchoring on the 52w high; fails when the high is a blow-off)
  SIZE    50d $ volume in the top third of the 54                         (institutional participation, liquidity; fails when megacaps de-rate e.g. 2022, late 2026)
Not used: flip-bar volume/range, 1h alignment, room-to-stop (found by examining our own results). Reported: CAGR/DD 2019-22 | 2023-26, per-year CAGR delta vs live, one-signal ablations (drop-one and only-one)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
A = pd.read_parquet(SP + 'flips295.parquet'); W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W = W[W.sp & ~W.h1 & (W['var'] == 'base+sticky')][['tk', 'entry', 'ratio']]
B = A.merge(W, on=['tk', 'entry']); B = B[(B.ratio >= 1.5) & (B.entry >= '2019-01-01') & (B.exit < END)].copy(); B['day'] = B.entry.dt.normalize(); B['h1'] = False; FE = []
for tk in sorted(B.tk.unique()):
    Y = pd.read_parquet(f'data/yahoo/daily_v/{tk}.parquet'); Y.index = pd.DatetimeIndex(Y.index).tz_localize(None).normalize(); c = Y.close; dv = c * Y.volume
    f = pd.DataFrame({'trend': c > c.rolling(200, min_periods=150).mean(), 'mom': np.log(c.shift(21) / c.shift(252)), 'h52b': c / c.rolling(252, min_periods=150).max() >= .9, 'ldv': np.log(dv.rolling(50).mean())}).shift(1); f['tk'] = tk; f['day'] = f.index; FE.append(f)
FE = pd.concat(FE); FE['momp'] = FE.groupby('day').mom.rank(pct=True); FE['ldvp'] = FE.groupby('day').ldv.rank(pct=True)
B = B.merge(FE[['tk', 'day', 'trend', 'momp', 'h52b', 'ldvp']], on=['tk', 'day'], how='left').dropna(subset=['momp', 'ldvp', 'trend']); B['TREND'] = B.trend.astype(bool); B['MOM'] = B.momp >= 2 / 3; B['HI52'] = B.h52b.astype(bool); B['SIZE'] = B.ldvp >= 2 / 3
O = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RK = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1).rank(axis=1, ascending=False, method='min'); B['brank'] = [RK[t].asof(d) if t in RK.columns else np.nan for t, d in zip(B.tk, B.day)]; B = B.dropna(subset=['brank']).copy()
print('trades', len(B), '| signal rates:', {k: round(B[k].mean(), 2) for k in ('TREND', 'MOM', 'HI52', 'SIZE')})
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    return ' | '.join(out)
def run(sigs, K):
    ns = B[list(sigs)].sum(axis=1) if sigs else 0; X = B.assign(_k=(B.brank - K * ns).values + 1e-6 * B.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable'); global T0; T0 = X; tk_, _ = taken(); s_, w_ = daily(tk_, 6); return s_
yr = lambda s: ((1 + s).groupby(s.index.year).prod() - 1)
base = run((), 0); allS = ('TREND', 'MOM', 'HI52', 'SIZE'); main = run(allS, 2)
print('\nPRE-REGISTERED: all four signals, K=2'); print(f'  live ratio order        {st2(base)}'); print(f'  + 4-signal boost K=2    {st2(main)}')
d = (yr(main) - yr(base)) * 100; yb = yr(base) * 100; print('  per-year return, live -> boosted (delta, pts): ' + ' '.join(f'{y}:{yb[y]:+.0f}->{yb[y] + d[y]:+.0f}({d[y]:+.1f})' for y in d.index)); print(f'  years improved {int((d > 0).sum())} of {len(d)}')
print('\nsensitivity K (all four signals):'); [print(f'  K={K}  {st2(run(allS, K))}', flush=True) for K in (1, 3, 5)]
print('\nablation, K=2: only one signal / drop one signal'); 
for s in allS: print(f'  only {s:6s} {st2(run((s,), 2))}   | drop {s:6s} {st2(run(tuple(x for x in allS if x != s), 2))}', flush=True)
