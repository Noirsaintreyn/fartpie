"""Nov 23 2026 report 385. Rey: 'top names also lead the indexes; what happens if you drop them from SPY/QQQ'.
Same top-5 (NVDA CMG AAPL FDX BA) removed from (a) an equal-weight daily-close hold of the 54-name set, (b) a dollar-volume-weighted (60d trailing, proxy for cap weight) hold of the 54 names, (c) the live vector book (report 384). Compare % of CAGR lost.
SPY/QQQ constituent weights are not in the workspace; (b) is a proxy only. SPY/QQQ buy-and-hold shown for scale. Daily closes from 15m RTH."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
U = '/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/'; b = pd.read_parquet('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/b302.parquet'); names = sorted(b.tk.unique()); top5 = ['NVDA', 'CMG', 'AAPL', 'FDX', 'BA']
def daily(tk):
    f = sorted(glob.glob(U + f'{tk}_15m_2018-01-01_*.parquet'))[-1]; H = pd.read_parquet(f); H.index = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); H = H.sort_index(); m = H.index.hour * 60 + H.index.minute; H = H[(m >= 570) & (m < 960)]
    g = H.groupby(H.index.normalize()); return g.close.last(), (H.close * H.volume).groupby(H.index.normalize()).sum()
C, DV = {}, {}
for tk in names + ['SPY', 'QQQ']:
    try: C[tk], DV[tk] = daily(tk)
    except Exception as e: print('skip', tk, e)
C = pd.DataFrame(C); DV = pd.DataFrame(DV); C = C[C.index >= '2018-06-01']; R = C.pct_change(); R[R.abs() > .5] = np.nan; DV = DV.reindex(C.index)
def cagr(s): e = (1 + s.fillna(0)).cumprod(); y = (s.index[-1] - s.index[0]).days / 365.25; return e.iloc[-1] ** (1 / y) - 1
def sh(s): return s.mean() / s.std() * 252 ** .5
def basket(cols, w):
    W = w[cols].shift(1); W = W.div(W.sum(1), axis=0); return (R[cols] * W).sum(1, min_count=1)
nm = [n for n in names if n in R.columns]; ew = pd.DataFrame(1.0, index=R.index, index_=None) if False else pd.DataFrame(1.0, index=R.index, columns=nm)
ew = ew.where(R[nm].notna()); dvw = DV[nm].rolling(60, min_periods=20).mean()
for lab, w in (('equal-weight 54', ew), ('dollar-volume-weight 54 (cap proxy)', dvw)):
    a = basket(nm, w); c = basket([n for n in nm if n not in top5], w)
    for h, msk in (('all', a.index >= '2019-01-01'), ('train', (a.index >= '2019-01-01') & (a.index < '2022-01-01')), ('test', a.index >= '2022-01-01')):
        print('%-36s %-5s with top5 CAGR %5.1f%% Sh %.2f | without %5.1f%% Sh %.2f | CAGR lost %.0f%%' % (lab, h, cagr(a[msk]) * 100, sh(a[msk]), cagr(c[msk]) * 100, sh(c[msk]), (1 - cagr(c[msk]) / cagr(a[msk])) * 100))
for t in ('SPY', 'QQQ'):
    for h, msk in (('all', R.index >= '2019-01-01'), ('test', R.index >= '2022-01-01')): print('%s buy&hold %-5s CAGR %.1f%% Sh %.2f' % (t, h, cagr(R[t][msk]) * 100, sh(R[t][msk])))
print('vector book (report 384): with top5 23.6%% Sh 2.60 | without 14.4%% Sh 1.75 | CAGR lost %.0f%%' % ((1 - 14.4 / 23.6) * 100))
