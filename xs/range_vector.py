"""Oct 26 2026: does vector / time-of-day context sharpen the adaptive expected range?
Daily: at every RTH 15m close 10:00-15:30, with the RTH low so far `lo` (set at tl):
  ctx 'flip after low'  = 15m vector is bull and its latest bull flip came after the low was set (the low 'confirmed' by the vector)
  ctx '1h flip after low' = same with the 1h vector;  'no flip' = everything else with price above the low
Outcomes: does the low hold to the close; the final RTH high above the low, in units of today's HAR budget (1.7 sigma, what the site uses).
Models for the final high, fit on 2019-21, scored on 2022-26 (median abs error in budget units, and 80% band coverage):
  site  = max(high so far, low + 1 budget)      tod = bucketed by time of day      tod+vec = by time of day x vector context
Weekly: the same at each day's close Mon-Thu with the week's low, the 1h vector, weekly HAR budget.
python3 xs/range_vector.py BARS_DIR"""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv = sys.argv[:2] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0])
BARS = sys.argv[1]
def har(x):  # x = log(log range) series -> next-period sigma
    return (np.exp(-0.6186 + 0.2939 * x + 0.3661 * x.rolling(5).mean() + 0.1783 * x.rolling(22).mean()) / 1.665).shift(1)
DR, WR = [], []
for f in sorted(glob.glob(f'{BARS}/*_15m_*.parquet')):
    tk = os.path.basename(f).split('_15m_')[0]
    if tk in ('SPY', 'QQQ'): continue
    H = load(f); reg, s1 = states(H, vec); H['r15'], H['r1'] = reg, s1
    H['f15'] = (H.r15 == 1) & (H.r15.shift(1) != 1); H['f1'] = (H.r1 == 1) & (pd.Series(s1).shift(1).values != 1)
    R = H[(H.m >= 570) & (H.m < 960)].copy(); R['d'] = R.dt.dt.normalize()
    D = R.groupby('d').agg(h=('high', 'max'), l=('low', 'min'))
    D['b'] = 1.7 * har(np.log(np.log(D.h / D.l).clip(lower=1e-5)))
    # weekly
    D['w'] = D.index.to_period('W'); Wk = D.groupby('w').agg(h=('h', 'max'), l=('l', 'min'))
    Wk['b'] = 1.7 * har(np.log(np.log(Wk.h / Wk.l).clip(lower=1e-5)))
    for d, g in R.groupby('d'):
        if d < pd.Timestamp('2019-01-01') or len(g) < 24 or not D.b.get(d, np.nan) > 0: continue
        b = D.b[d]; lo_ = g.low.cummin().values; hi_ = g.high.cummax().values; m = g.m.values
        tl = pd.Series(np.where(g.low.values <= lo_, m, np.nan)).ffill().values
        lf15 = pd.Series(np.where(g.f15.values, m, np.nan)).ffill().values; lf1 = pd.Series(np.where(g.f1.values, m, np.nan)).ffill().values
        HH, LL = hi_[-1], lo_[-1]
        for k in range(1, len(g) - 1):
            if not 600 <= m[k] + 15 <= 930: continue
            c15 = g.r15.values[k] == 1 and lf15[k] >= tl[k]; c1 = g.r1.values[k] == 1 and lf1[k] >= tl[k]
            DR.append((tk, d, m[k] + 15, c15, c1, LL >= lo_[k], np.log(HH / lo_[k]) / b, np.log(hi_[k] / lo_[k]) / b, (m[k] - tl[k]) // 15 + 1))
    # weekly checkpoints: each day's close Mon-Thu
    E = R.groupby('d').agg(lo=('low', 'min'), hi=('high', 'max'), r1=('r1', 'last'), f1=('f1', 'any'))
    E['w'] = E.index.to_period('W')
    for w, g in E.groupby('w'):
        if g.index[0] < pd.Timestamp('2019-01-01') or len(g) < 4 or not Wk.b.get(w, np.nan) > 0: continue
        b = Wk.b[w]; lo_ = g.lo.cummin().values; hi_ = g.hi.cummax().values; HH, LL = hi_[-1], lo_[-1]
        tl = pd.Series(np.where(g.lo.values <= lo_, np.arange(len(g)), np.nan)).ffill().values
        lf = pd.Series(np.where(g.f1.values, np.arange(len(g)), np.nan)).ffill().values
        for k in range(len(g) - 1):
            c1 = g.r1.values[k] == 1 and lf[k] >= tl[k]
            WR.append((tk, g.index[k], g.index[k].dayofweek, c1, LL >= lo_[k], np.log(HH / lo_[k]) / b, np.log(hi_[k] / lo_[k]) / b))
    print(tk, flush=True)
X = pd.DataFrame(DR, columns=['tk', 'd', 't', 'c15', 'c1', 'holds', 'ext', 'sofar', 'age']); X.to_parquet('xs/ext/range_vector_daily.parquet')
Y = pd.DataFrame(WR, columns=['tk', 'd', 'dow', 'c1', 'holds', 'ext', 'sofar']); Y.to_parquet('xs/ext/range_vector_weekly.parquet')
