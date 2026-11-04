"""Nov 4 2026: context ranking of vector entries. DECLARED BEFORE RESULTS.
Every one of the 17,447 backtest entries is tagged with what was known at entry (prior close):
  stock vol    rv20 (20d daily std) tercile within year; rvrel = rv20 / own 252d median (hot/normal/quiet)
  earnings     days to the next report (<=3 / 4-10 / >10), days since the last one (<=5 / else)  (Yahoo dates)
  market       SPY above its 200d; VIX/VIX3M > 1 (stress); VIX 1y percentile tercile; SPY 20d return sign
  sector       hand map of the 54 names;  season: calendar month, quarter
Tests (all in 2019-21 and 2022-26, ratio >= 1.5 universe, trades the vector signals):
  1 cell table: avg trade, win, share of trades >= +15%, n.  A cell 'holds' if its avg trade is on the same side of the period mean in BOTH periods
    by >= 0.25 pt, n >= 120 per period.
  2 persistence: correlation of ticker / sector / month mean return between the two periods.
  3 walk-forward ticker score: mean of that ticker's earlier closed trades (>= 15), top vs bottom half next-trade return.
  4 track (15m vs 1h-track) x market regime."""
import sys, os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
X['d'] = X.entry.dt.normalize(); X['per'] = np.where(X.entry.dt.year <= 2021, '19-21', '22-26')
SEC = dict(AAPL='tech', MSFT='tech', ORCL='tech', ADBE='tech', CRM='tech', IBM='tech', PANW='tech', CRWD='tech', PLTR='tech', APP='tech', UBER='tech', ABNB='cons', NFLX='comm', META='comm', GOOGL='comm', DIS='comm',
           NVDA='semi', AMD='semi', AVGO='semi', MU='semi', LRCX='semi', QCOM='semi', SMCI='semi', AMZN='cons', TSLA='cons', BKNG='cons', CMG='cons', SBUX='cons', MCD='cons', LULU='cons', CVNA='cons', WYNN='cons', PENN='cons',
           HD='cons', TGT='stap', WMT='stap', COST='stap', DLTR='stap', JPM='fin', GS='fin', C='fin', PYPL='fin', COIN='fin', HOOD='fin', LLY='hlth', UNH='hlth', BA='indu', CAT='indu', FDX='indu', UPS='indu', LMT='indu',
           GEV='indu', OXY='energy', ENPH='energy')
X['sec'] = X.tk.map(SEC); print('unmapped:', X[X.sec.isna()].tk.unique())
# stock vol
rvl = {}
for tk in X.tk.unique():
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    H = load(f[0]); H = H[(H.m >= 570) & (H.m < 960)]; c = H.groupby(H.dt.dt.normalize()).close.last()
    rv = c.pct_change().rolling(20).std().shift(1); rvl[tk] = pd.DataFrame({'rv': rv, 'rvrel': rv / rv.rolling(252, min_periods=60).median()})
X = pd.concat([pd.merge_asof(g.sort_values('d'), rvl[t].rename_axis('d_').pipe(lambda z: z.set_axis(z.index.astype('datetime64[ns]'))), left_on='d', right_index=True) for t, g in X.groupby('tk')]).sort_values('entry').reset_index(drop=True)
X['rvt'] = X.groupby(X.d.dt.year).rv.transform(lambda s: pd.qcut(s, 3, labels=['quiet', 'mid', 'hot'], duplicates='drop'))
X['rvrel_b'] = pd.cut(X.rvrel, [0, .8, 1.25, 99], labels=['quiet-vs-own', 'normal', 'hot-vs-own'])
# earnings
dn, dl = [], []
for t, g in X.groupby('tk'):
    E = pd.read_csv(f'/tmp/claude-0/ml/earn/{t}.csv'); ed = pd.to_datetime(E['Earnings Date'].str[:10]).sort_values().values
    a = g.d.values; i = np.searchsorted(ed, a, side='left'); nx = np.where(i < len(ed), ed[np.minimum(i, len(ed) - 1)], np.datetime64('NaT'))
    pv = np.where(i > 0, ed[np.maximum(i - 1, 0)], np.datetime64('NaT'))
    dn.append(pd.Series((nx - a) / np.timedelta64(1, 'D'), g.index)); dl.append(pd.Series((a - pv) / np.timedelta64(1, 'D'), g.index))
X['dnext'] = pd.concat(dn); X['dlast'] = pd.concat(dl)
X['earn_b'] = pd.cut(X.dnext, [-1, 3, 10, 1e4], labels=['<=3d to report', '4-10d', '>10d'])
X['post_b'] = np.where(X.dlast <= 5, 'just reported (<=5d)', 'other')
# market
P = pd.read_parquet('/tmp/claude-0/ml/macro/px.parquet').ffill(); P.index = pd.to_datetime(P.index).astype('datetime64[ns]')
M = pd.DataFrame({'spy200': P.SPY > P.SPY.rolling(200).mean(), 'vts': P['^VIX'] / P['^VIX3M'], 'vixp': P['^VIX'].rolling(252, min_periods=60).rank(pct=True),
                  'spy20': P.SPY.pct_change(20)}).shift(1).dropna(subset=['vts'])
X = pd.merge_asof(X.sort_values('d'), M, left_on='d', right_index=True).sort_values('entry').reset_index(drop=True)
X['mkt'] = np.where(X.spy200.astype(bool), 'SPY>200d', 'SPY<200d'); X['stress'] = np.where(X.vts > 1, 'VIX stress', 'VIX calm')
X['vixt'] = pd.cut(X.vixp, [-1, .33, .67, 2], labels=['VIX low', 'VIX mid', 'VIX high'])
X['mom'] = np.where(X.spy20 > 0, 'SPY 20d up', 'SPY 20d down'); X['month'] = X.entry.dt.month
X['regime'] = np.where(X.spy200.astype(bool) & (X.vts <= 1), 'calm bull', np.where(X.spy200.astype(bool), 'bull+stress', np.where(X.vts <= 1, 'bear calm', 'bear+stress')))
X.to_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet')
U = X[(X.ratio >= 1.5)].copy(); U['big'] = (U.ret >= .15)
print('universe trades', len(U), {p: round(U[U.per == p].ret.mean() * 100, 3) for p in ('19-21', '22-26')})
mu = {p: U[U.per == p].ret.mean() for p in ('19-21', '22-26')}
HOLD = []
def table(col, title, minn=120):
    print(f'\n{title}   (avg trade | win | >=15% share | n)   19-21 || 22-26   * = same side of period mean by >= .25pt in both')
    for v, g in U.groupby(col, observed=True):
        out = []; ds = []
        for p in ('19-21', '22-26'):
            z = g[g.per == p]; out.append(f'{z.ret.mean()*100:+.2f}% | {z.ret.gt(0).mean():.0%} | {z.big.mean():.1%} | n{len(z)}' if len(z) else 'n0'); ds.append((z.ret.mean() - mu[p]) * 100 if len(z) >= minn else np.nan)
        ok = all(not np.isnan(d) and abs(d) >= .25 for d in ds) and np.sign(ds[0]) == np.sign(ds[1])
        if ok: HOLD.append((title, v, ds))
        print(f'  {str(v):<24}{out[0]:<34}||  {out[1]:<34}{"*" if ok else ""}')
for col, t in (('rvt', 'stock vol tercile (within year)'), ('rvrel_b', 'stock vol vs its own history'), ('earn_b', 'days to next report'), ('post_b', 'days since last report'),
               ('mkt', 'SPY vs 200d'), ('stress', 'VIX term structure'), ('vixt', 'VIX 1y percentile'), ('mom', 'SPY 20d'), ('regime', 'market regime'), ('sec', 'sector'), ('month', 'calendar month'), ('h1', 'h1 track')):
    table(col, t)
print('\nSECTOR x REGIME (avg trade, n)')
for s, g in U.groupby('sec'):
    row = []
    for r in ('calm bull', 'bull+stress', 'bear calm', 'bear+stress'):
        z = g[g.regime == r]; row.append(f'{r}: {z.ret.mean()*100:+.2f}% n{len(z)}')
    print(f'  {s:<7}', ' | '.join(row))
print('\nPERSISTENCE: correlation of mean return between 19-21 and 22-26 (cells with n >= 20 in both)')
for col in ('tk', 'sec', 'month'):
    a = U[U.per == '19-21'].groupby(col).ret.agg(['mean', 'size']); b = U[U.per == '22-26'].groupby(col).ret.agg(['mean', 'size'])
    j = a.join(b, lsuffix='a', rsuffix='b').dropna(); j = j[(j.sizea >= 20) & (j.sizeb >= 20)]
    from scipy.stats import spearmanr
    r = spearmanr(j.meana, j.meanb); print(f'  {col:<6} cells {len(j):>3}  spearman {r[0]:+.2f}  p {r[1]:.2f}')
# walk-forward ticker score
U = U.sort_values('entry').reset_index(drop=True); sc = np.full(len(U), np.nan)
for t, g in U.groupby('tk'):
    for i in g.index:
        pr = g[(g.exit < U.entry[i])]
        if len(pr) >= 15: sc[i] = pr.ret.mean()
U['sc'] = sc; W = U.dropna(subset=['sc'])
print('\nWALK-FORWARD ticker score (mean of its own earlier closed trades): next-trade avg by score half (within year)')
W['half'] = W.groupby(W.entry.dt.year).sc.transform(lambda s: np.where(s >= s.median(), 'top half', 'bottom half'))
for p in ('19-21', '22-26'):
    z = W[W.per == p]; print(' ', p, {h: f'{g.ret.mean()*100:+.2f}% n{len(g)}' for h, g in z.groupby('half')})
# regime x track
print('\nTRACK x REGIME (avg trade, n)')
for r, g in U.groupby('regime'):
    print(f'  {r:<12}', ' | '.join(f'{"1h" if h else "15m"} {p}: {z.ret.mean()*100:+.2f}% n{len(z)}' for h in (False, True) for p in ('19-21', '22-26') for z in [g[(g.h1 == h) & (g.per == p)]]))
print('\nCELLS THAT HOLD IN BOTH PERIODS:'); [print(' ', t, v, [round(x, 2) for x in d]) for t, v, d in HOLD]
