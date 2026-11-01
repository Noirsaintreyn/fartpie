"""Nov 1 2026: ML ranker for the slots, judged fairly. Every flip (not just admitted ones) with everything known at entry:
 vector states (15m RTH/ext, 5m, 1h, 4h, daily), flip quality (clv, wick, beyond, relvol), base (dur, width, durpct), noise band (nb),
 options ratio, earnings distance, SPY downtrend, SPY distance from its 200d, QQQ 20d return, VIX level & term structure proxy (VIX/VIX3M n/a: VIX only),
 time of day, 1h vs 15m track.
Walk-forward by year: train on flips that EXITED before Jan 1 of year Y, predict year Y (2020-2026 out of sample). LightGBM (small, regularized)
and a ridge on percentile ranks. Target = trade return (winsorized). Book: ML score as the tie-break vs the live ratio.
Fair yardstick: each ranker run 40 times with the same random 5% of flips dropped per run -> median and how often ML beats live."""
import os, glob, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
os.chdir('/tmp/claude-0/ml'); exec(open('/home/claude/fartpie/xs/book_mtm.py').read().split("if __name__")[0])
src = open('/home/claude/fartpie/xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
import sys; sys.path.insert(0, '/home/claude/fartpie'); cwd = os.getcwd(); os.chdir('/home/claude/fartpie'); import live_screener as LS; os.chdir(cwd)
L = pd.read_parquet('/tmp/claude-0/ml/flipq.parquet').sort_values('entry').reset_index(drop=True); L['d'] = L.entry.dt.normalize()
E = '/mnt/user-data/uploads/fartpie/etf_daily'
def dly(t):
    x = pd.read_parquet(f'{E}/{t}.parquet'); x['dt'] = pd.to_datetime(x.datetime).dt.normalize(); return x.set_index('dt')
S = dly('SPY'); Sv = S.reset_index()[['dt', 'open', 'high', 'low', 'close']].assign(volume=1.0); V = LS.vec(Sv)
mk = pd.DataFrame({'dn': ((V.reg.values == -1) & (S.close.values < S.close.rolling(200).mean().values)).astype(float),
                   'spy200': (S.close / S.close.rolling(200).mean() - 1).values}, index=S.index)
Qd = dly('QQQ'); mk['qqq20'] = (Qd.close / Qd.close.shift(20) - 1).reindex(mk.index)
if os.path.exists(f'{E}/VIX.parquet'): mk['vix'] = dly('VIX').close.reindex(mk.index)
mk = mk.shift(1)                                                                 # known at the prior close
L = L.join(mk.reindex(L.d.values).reset_index(drop=True))
src2 = open('/home/claude/fartpie/xs/pre_earn_flip.py').read(); exec(src2[src2.index('k = np.full'):src2.index("L['k'] = k")]); L['k_earn'] = np.minimum(k, 30)
L['tod'] = L.entry.dt.hour * 60 + L.entry.dt.minute; L['h1f'] = L.h1.astype(float)
X_COLS = ['ratio', 'v15r', 'v15x', 'v5', 'v1h', 'v4h', 'vD', 'clv', 'uwick', 'beyond', 'relvol', 'dur', 'width', 'durpct', 'nb', 'k_earn', 'dn', 'spy200', 'qqq20', 'tod', 'h1f'] + (['vix'] if 'vix' in L else [])
y = L.ret.clip(-.15, .25)
L['p_lgb'] = np.nan; L['p_rdg'] = np.nan
for Y in range(2020, 2027):
    tr = L.exit < pd.Timestamp(f'{Y}-01-01'); te = L.entry.dt.year == Y
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=.02, num_leaves=8, min_child_samples=300, subsample=.8, subsample_freq=1, colsample_bytree=.7, reg_lambda=5, verbose=-1)
    m.fit(L.loc[tr, X_COLS], y[tr]); L.loc[te, 'p_lgb'] = m.predict(L.loc[te, X_COLS])
    R = L[X_COLS].rank(pct=True).fillna(.5); r = Ridge(alpha=50).fit(R[tr], y[tr]); L.loc[te, 'p_rdg'] = r.predict(R[te])
    print(Y, 'trained on', int(tr.sum()), flush=True)
imp = pd.Series(m.feature_importances_, X_COLS).sort_values(ascending=False); print('LightGBM importance (last fold):', imp.head(10).to_dict())
O = L[L.entry.dt.year >= 2020].copy(); O['h'] = np.where(O.entry.dt.year <= 2021, '2020-21', '2022-26')
print('\nA) all flips, out of sample: avg trade % by prediction quintile (q5 = best predicted)')
for p in ('p_lgb', 'p_rdg', 'ratio'):
    q = O.groupby('h')[p].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)) + 1
    t = (O.groupby(['h', q]).ret.mean() * 100).unstack().round(2); print(f'  {p}:'); print(t.to_string())
P = [(np.datetime64('2020-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
pr = lambda s: s.groupby(L.entry.dt.year).rank(pct=True)
KEYS = {'live (ratio)': L.ratio, 'LightGBM': L.p_lgb, 'ridge': L.p_rdg, 'ratio + LightGBM': pr(L.ratio) + pr(L.p_lgb), 'ratio + ridge': pr(L.ratio) + pr(L.p_rdg)}
rng = np.random.default_rng(7); DROPS = [rng.random(len(L)) < .05 for _ in range(40)]
res = {}
for nm, key in KEYS.items():
    Z0 = L.copy(); Z0['ratio'] = key.values
    res[nm] = np.array([[sim(Z0[~dmask], a, b)[0] for a, b in P] for dmask in [np.zeros(len(L), bool)] + DROPS])
    r = res[nm]; print(f'  {nm:<18} exact {r[0, 0]:+.1%} / {r[0, 1]:+.1%}   median of 40 perturbed runs {np.median(r[1:, 0]):+.1%} / {np.median(r[1:, 1]):+.1%}   (10-90%: {np.percentile(r[1:, 1], 10):+.1%}..{np.percentile(r[1:, 1], 90):+.1%})', flush=True)
base = res['live (ratio)'][1:]
for nm in list(KEYS)[1:]:
    w = (res[nm][1:] > base).mean(axis=0); print(f'  {nm:<18} beats live in the same perturbed run: 2020-21 {w[0]:.0%}, 2022-26 {w[1]:.0%}')
