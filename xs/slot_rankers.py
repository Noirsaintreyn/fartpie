"""Oct 26 2026: which signal gets a slot - rankers from the literature, as the tie-break among signals that arrive together (the live
book uses the options ratio; random ordering halves 2022-26, so ranking matters). All traits from daily closes up to the prior session.
 fip     frog-in-the-pan information discreteness (Da, Gurun, Warachka 2014): sign(12-1m return) x (% down days - % up days); lower = smoother trend
 h52     nearness to the 52-week high (George & Hwang 2004): close / 252-day max
 rmom    residual momentum (Blitz, Huij, Martens 2011): 12-1m return net of SPY beta, / residual volatility
 mom     raw 12-1m momentum
 ivol    60-day residual volatility vs SPY (low-vol anomaly: lower ranked first)
 vr      variance ratio VR(5) over the last year (> 1 = returns persist, trend-friendly)
 LTR     learning to rank (Poh, Lim, Zohren, Roberts 2020): LightGBM LambdaRank on groups of simultaneous signals, walk-forward by year
Each enters as ratio + w x percentile (w = 0.5, 1), or alone. Luck band from xs/book_lab.py noise: ~+51..+60% (2019-21), ~+24..+30% (2022-26)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec'].reset_index(drop=True)
P = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/close_px_sp.parquet').pivot_table(index='date', columns='ticker', values='close').sort_index()
P = P[[c for c in P.columns if c in set(L.tk)]]
spy = pd.read_parquet('/mnt/user-data/uploads/fartpie/etf_daily/SPY.parquet'); spy = spy.set_index(pd.to_datetime(spy.datetime).dt.tz_localize(None).dt.normalize()).close
spy = spy[~spy.index.duplicated(keep='last')].reindex(P.index).ffill()
r = np.log(P).diff(); m = np.log(spy).diff()
ret121 = np.log(P.shift(21) / P.shift(252))
pos = (r > 0).astype(float).where(r.notna()).shift(21).rolling(231, min_periods=150).mean(); neg = (r < 0).astype(float).where(r.notna()).shift(21).rolling(231, min_periods=150).mean()
TR = {'fip': -(np.sign(ret121) * (neg - pos)), 'h52': P / P.rolling(252, min_periods=150).max(), 'mom': ret121}
mr = lambda x: x.rolling(252, min_periods=150).mean()
beta = (mr(r.mul(m, axis=0)).sub(mr(r).mul(mr(m), axis=0))).div(m.rolling(252, min_periods=150).var(ddof=0), axis=0)
res = r - beta.shift(1).mul(m, axis=0); TR['rmom'] = res.shift(21).rolling(231, min_periods=150).sum() / res.shift(21).rolling(231, min_periods=150).std()
TR['ivol'] = -res.rolling(60, min_periods=40).std()
r5 = np.log(P).diff(5); TR['vr'] = r5.rolling(252, min_periods=150).var() / (5 * r.rolling(252, min_periods=150).var())
# value known at the prior close for each trade
dd = L.d.values; idx = P.index.values; k = np.searchsorted(idx, dd) - 1
for nm, X in TR.items():
    A = X.values; col = {c: i for i, c in enumerate(X.columns)}
    L[nm] = [A[kk, col[t]] if (kk >= 0 and t in col) else np.nan for kk, t in zip(k, L.tk)]
print('coverage:', {c: round(L[c].notna().mean(), 3) for c in TR})
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
Y = [(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01')) for y in range(2019, 2027)]; base = [sim(L, a, b)[0] for a, b in Y]; B20 = None
def show(nm, Z):
    yr = [sim(Z, a, b)[0] if ((Z.entry >= a) & (Z.entry < b)).any() else np.nan for a, b in Y]; R = [sim(Z, a, b) for a, b in PER]
    flag = '  <-- beats the luck band in both' if R[0][0] > .60 and R[1][0] > .30 else ''
    print(f'  {nm:<44}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R) + f'  beats live {sum(x > y + 1e-9 for x, y in zip(yr, base) if x == x)}/8{flag}', flush=True)
print(f'{"":<46}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live (options ratio)', L)
for c in (['rmom', 'ivol'] if len(sys.argv) > 1 else TR):
    pr = L[c].rank(pct=True).fillna(.5)
    for w in (.5, 1.0):
        z = L.copy(); z['ratio'] = z.ratio + w * pr; show(f'ratio + {w} x {c}', z)
    z = L.copy(); z['ratio'] = pr; show(f'{c} alone', z)
# learning to rank on simultaneous groups
F = ['ratio', 'h1'] + list(TR); L['h1'] = L.h1.astype(float); L['ltr'] = np.nan
for yy in range(2020, 2027):
    tr = L[L.exit < pd.Timestamp(f'{yy}-01-01')]; g = tr.groupby('entry').filter(lambda x: len(x) >= 2).sort_values('entry')
    lab = g.groupby('entry').ret.rank(method='first').astype(int) - 1; grp = g.groupby('entry', sort=False).size().values
    mdl = lgb.LGBMRanker(objective='lambdarank', n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=100, verbose=-1, label_gain=list(range(64)))
    mdl.fit(g[F], lab.clip(upper=63), group=grp); te = L.entry.dt.year == yy; L.loc[te, 'ltr'] = mdl.predict(L.loc[te, F])
B = L[L.entry >= '2020-01-01'].copy(); print('\n(2020+, walk-forward learning-to-rank)'); show('live', B)
z = B.copy(); z['ratio'] = z.ltr; show('LambdaRank alone', z)
z = B.copy(); z['ratio'] = z.ratio + z.ltr.rank(pct=True); show('ratio + LambdaRank', z)
