"""Nov 18 2026 report 333. Does a plain 5-minute opening-range breakout on NQ look like the +0.4R/trade profile Rey was shown (another person's strategy; rules unknown)? Declared before running. GENERIC engine, no fitted parameters:
first RTH 5m candle (09:30-09:35 ET) high/low = range (R = candle range). Long stop-entry at the high / short at the low on the first later 5m bar that touches it (both sides in one bar -> skip), stop at the opposite end (1R), NO target, exit at the 15:55 bar open. One trade per day. Cost 0.75 NQ pt round trip (1 tick each side + commission) and a stressed 2.0 pt. Train 2019-21 | test 2018 + 2022-26.
Why it should work: the first candle sets the day's auction extremes; breaks with 1R stop capture trend days (our algo is a trend-day engine, report 285/307).  Why it would fail: stop-entry fills on a 5m bar are optimistic (entry price assumed exactly at the level), small candles make costs a big share of R, and 5m-bar data cannot tell stop-then-entry ordering inside a bar (treated as stopped = pessimistic)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize(); rows = []
for d, x in N.groupby('day'):
    x = x.sort_values('m')
    if len(x) < 70 or x.m.iloc[0] != 570: continue
    hi, lo = x.high.iloc[0], x.low.iloc[0]; rg = hi - lo
    if rg < 2: continue
    rest = x.iloc[1:]; ex = rest[rest.m == 955].open.iloc[0] if (rest.m == 955).any() else rest.close.iloc[-1]; res = None
    for _, b in rest.iterrows():
        up, dn = b.high >= hi, b.low <= lo
        if up and dn: res = None; break
        if up or dn:
            dr = 1 if up else -1; ent = hi if up else lo; stop = lo if up else hi; later = rest[rest.m > b.m]; xp = ex
            if (dr == 1 and b.low <= stop) or (dr == -1 and b.high >= stop): xp = stop      # same-bar stop (pessimistic)
            else:
                for _, c in later.iterrows():
                    if (dr == 1 and c.low <= stop) or (dr == -1 and c.high >= stop): xp = stop; break
            res = dict(day=d, dir=dr, rg=rg, pts=(xp - ent) * dr); break
    if res: rows.append(res)
T = pd.DataFrame(rows).set_index('day'); T['tr'] = (T.index >= '2019-01-01') & (T.index < '2022-01-01')
def rep(c):
    r = (T.pts - c) / T.rg
    f = lambda m: f'{r[m].mean():+.3f}R (t{r[m].mean() / (r[m].std() / np.sqrt(m.sum())):+.1f}) win {100 * (r[m] > 0).mean():.0f}% n{int(m.sum())} pts/trade {(T.pts[m] - c).mean():+.1f}'
    print(f'cost {c} pts: train {f(T.tr)} | test {f(~T.tr)}'); return r
r0 = rep(.75); rep(2.0)
print('\nby year, R/trade at 0.75 pts cost:', r0.groupby(T.index.year).mean().round(2).to_dict()); print('median candle range pts:', round(T.rg.median(), 1), '| cost as % of median R: ', round(.75 / T.rg.median() * 100, 1), '%')
print('top 5% of trades share of total R:', round(r0.sort_values(ascending=False).head(int(len(r0) * .05)).sum() / r0.sum(), 2))
