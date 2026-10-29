"""Oct 28 2026: a second return stream. (1) the Mid/Small book (monthly backtest series on the site) blended with the screener book;
(2) other trend models' books (daily MTM, same names) blended with the vector book. Return/DD per half; correlation."""
import json, numpy as np, pandas as pd
S = json.load(open('/home/claude/fartpie/live_data/site_series.json'))
m = lambda k: pd.Series(S[k]['b'], index=pd.PeriodIndex(S[k]['d'], freq='M')).pct_change()
A = pd.DataFrame({'scr': m('scr'), 'mid': m('mid')}).dropna()
def stats(r, per):
    eq = (1 + r).cumprod(); y = len(r) / per; c = eq.iloc[-1] ** (1 / y) - 1; dd = (eq / eq.cummax() - 1).min(); return c, dd
print(f'1) monthly, {A.index[0]}..{A.index[-1]}  corr scr/mid {A.scr.corr(A.mid):.2f}  (2020-21 {A.loc[:"2021"].corr().iloc[0, 1]:.2f}, 2022-26 {A.loc["2022":].corr().iloc[0, 1]:.2f})')
for w in (0, .2, .3, .5, 1.0):
    r = (1 - w) * A.scr + w * A.mid
    print(f'  {1 - w:.0%} screener / {w:.0%} mid   ' + '   '.join(f'{h}: {c:+.1%}/yr DD {d:.0%} ratio {c / -d:.2f}' for h, (c, d) in (('2020-21', stats(r.loc[:'2021'], 12)), ('2022-26', stats(r.loc['2022':], 12)))))
for w in (.3, .5):
    r = A.scr + w * A.mid
    print(f'  100% screener + {w:.0%} mid (leverage {1 + w:.1f}x) ' + '   '.join(f'{h}: {c:+.1%}/yr DD {d:.0%} ratio {c / -d:.2f}' for h, (c, d) in (('2020-21', stats(r.loc[:'2021'], 12)), ('2022-26', stats(r.loc['2022':], 12)))))
D = pd.read_parquet('/tmp/claude-0/ml/vt/series.parquet'); v = D['tm:vec']
print(f'\n2) daily trend-model books vs the vector book (corr, then 50/50 blend)  vector alone: ' + '  '.join(f'{h} {c:+.1%}/{d:.0%}' for h, (c, d) in (('19-21', stats(v.loc[:'2021'], 252)), ('22-26', stats(v.loc['2022':], 252)))))
for c_ in D.columns:
    if c_ == 'tm:vec': continue
    r = .5 * v + .5 * D[c_]
    print(f'  {c_:<20} corr {v.corr(D[c_]):.2f}   50/50: ' + '  '.join(f'{h} {c:+.1%}/{d:.0%}' for h, (c, d) in (('19-21', stats(r.loc[:'2021'], 252)), ('22-26', stats(r.loc['2022':], 252)))))
