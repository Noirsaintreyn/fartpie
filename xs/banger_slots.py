"""Oct 27 2026: 6 regular slots + up to 2 extra slots (1/6 each, up to 133% invested) that open ONLY for 'banger' signals arriving while
the 6 are full. Banger filters (all known at entry): SPY downtrend (daily vector bear + under 200d); the out-of-sample conditions score
(xs/edge_combo_oos.py) top third; options ratio >= its 2019-21 90th pct; walk-forward LightGBM score (xs/meta_rank.py) top 20%; combos."""
import numpy as np, pandas as pd, glob
from collections import deque
import lightgbm as lgb
ms = open('xs/market_signal.py').read(); exec(ms[:ms.index("PER = [('2016-06-01'")]); MK = OUT
exec(open('xs/edge_combo_oos.py').read().split("def show2")[0])
S = MK['SPY']; L['down'] = ((S.vec_d < 0) & (S.sma200 < 0)).astype(float).reindex(L.d.values, method='ffill').values > 0
L['oos_hi'] = L.oos > t2; rq = L[L.inbook & (L.half == '19-21')].ratio.quantile(.9); L['ratio_hi'] = L.ratio >= rq
mr = open('xs/meta_rank.py').read(); X = ['ratio', 'h1i', 'tmin', 'dow', 'age1h', 'dage', 'dstate', 'mkt20', 'sec20', 'res20', 'stk20', 'cp5', 'dse', 'rse', 'used', 'both', 'wk', 'age']
F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/fx/*.parquet')).drop(columns=['exit', 'ret'])
RV = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/rv/*.parquet')); RV = RV[RV['var'] == 'live'][['tk', 'h1', 'entry', 'used', 'both', 'wk', 'age']]
M = L[['tk', 'h1', 'entry', 'exit', 'ret', 'ratio']].merge(F.drop(columns=[c for c in ('dage', 'mkt20') if c in L.columns and False]).drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left').merge(RV.drop_duplicates(['tk', 'h1', 'entry']), on=['tk', 'h1', 'entry'], how='left')
M['tmin'] = M.entry.dt.hour * 60 + M.entry.dt.minute; M['dow'] = M.entry.dt.dayofweek; M['h1i'] = M.h1.astype(int)
for c in ('both', 'wk'): M[c] = M[c].astype(float)
M['ml'] = np.nan; pc = dict(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1)
for Y in range(2020, 2027):
    trn = M[M.exit < pd.Timestamp(f'{Y}-01-01')]; te = M.entry.dt.year == Y
    mdl = lgb.LGBMRegressor(**pc).fit(trn[X], trn.ret.clip(-.1, .1)); M.loc[te, 'ml'] = mdl.predict(M.loc[te, X]); M.loc[te, 'mlcut'] = np.quantile(mdl.predict(trn[X]), .8)
L['ml_hi'] = (M.ml > M.mlcut).values
def sim_x(Z, a, b, ok, n=6, extra=2, brk=.08):
    Z = Z.assign(ok=ok)[(Z.entry >= a) & (Z.entry < b)].sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; xt_ = []
    for e, x, r_, k_ in zip(Z.entry.values, Z.exit.values, Z.ret.values, Z.ok.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr, w in done: real += np.log1p(rr * w); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n + extra or (len(op) >= n and not k_): continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        if len(op) >= n: xt_.append(r_)
        op.append(x); pend.append((x, r_, 1 / n))
    for xt, rr, w in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * w); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); eq = np.exp(s.groupby(s.index.normalize()).last()); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min(), xt_
P2 = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2020-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2020-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show3(nm, ok, extra=2):
    R_ = [sim_x(L, a, b, ok, extra=extra) for a, b in P2]; ex = R_[2][2]
    print(f'  {nm:<46}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d, _ in R_) + '  ret/DD ' + ' '.join(f'{c / -d:.1f}' for c, d, _ in R_[:2]) +
          f'  extra-slot trades {len(ex)}, avg {np.mean(ex) * 100 if ex else 0:+.2f}%', flush=True)
print(f'{"(2020+, ML is walk-forward)":<48}{"2020-21":>16}{"2022-26":>16}{"2020-26":>16}{"2026 YTD":>16}')
none = np.zeros(len(L), bool); show3('live: 6 slots', none, extra=0)
show3('control: 7th/8th open to ANY signal', ~none)
for nm, ok in (('SPY downtrend', L.down.values), ('conditions score top third', L.oos_hi.values), ('options ratio top 10%', L.ratio_hi.values), ('ML score top 20%', L.ml_hi.values),
               ('downtrend AND score top third', (L.down & L.oos_hi).values), ('downtrend OR score top third', (L.down | L.oos_hi).values),
               ('2+ of: downtrend, score, ratio, ML', (L.down.astype(int) + L.oos_hi + L.ratio_hi + L.ml_hi).values >= 2)):
    show3(f'7th/8th only for: {nm}', ok)
