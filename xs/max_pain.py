"""Oct 25 2026: does max pain pull QQQ toward it into expiration? (data/qqq_options_greeks.db, 2019-2025)
For each expiration E: open interest on D = 1 and 3 trading days before E; max pain = the strike that minimizes total option-holder payoff at expiry.
Test: does QQQ move from its close on D toward max pain by E's close? Directional hit rate and the mean move in max pain's direction."""
import sqlite3, numpy as np, pandas as pd
c = sqlite3.connect('file:data/qqq_options_greeks.db?mode=ro', uri=True)
U = pd.read_sql("select date, close from underlying_prices where symbol='QQQ'", c, parse_dates=['date']).set_index('date').close
days = U.index
ex = [pd.Timestamp(x[0]) for x in c.execute("select distinct expiration from options_data where expiration >= '2019-01-01' and expiration <= '2025-12-12'")]
rows = []
for E in ex:
    if E not in days: continue
    k = days.get_loc(E)
    for lag in (1, 3):
        if k - lag < 0: continue
        D = days[k - lag]
        O = pd.read_sql("select strike, type, open_interest from options_data where expiration=? and date=?", c, params=(E.strftime('%Y-%m-%d'), D.strftime('%Y-%m-%d')))
        if O.open_interest.sum() < 1000: continue
        K = np.sort(O.strike.unique()); cl = O[O.type == 'call']; pt = O[O.type == 'put']
        pay = [((np.maximum(s - cl.strike, 0) * cl.open_interest).sum() + (np.maximum(pt.strike - s, 0) * pt.open_interest).sum()) for s in K]
        mp = K[int(np.argmin(pay))]; s0, s1 = U[D], U[E]
        rows.append(dict(E=E, lag=lag, monthly=E.day >= 15 and E.day <= 21 and E.dayofweek == 4, s0=s0, s1=s1, mp=mp, dist=(mp - s0) / s0, mv=(s1 - s0) / s0))
X = pd.DataFrame(rows); X = X[X.dist.abs() > 1e-4]
X['toward'] = np.sign(X.dist) * X.mv; X['hit'] = X.toward > 0
X['closer'] = (X.s1 - X.mp).abs() < (X.s0 - X.mp).abs()
X['per'] = np.where(X.E.dt.year >= 2022, '22-25', '19-21')
def t(g): return pd.Series({'n': len(g), 'hit %': g.hit.mean() * 100, 'closer %': g.closer.mean() * 100, 'move toward MP (bp)': g.toward.mean() * 1e4,
                            't': g.toward.mean() / (g.toward.std() / np.sqrt(len(g))), 'avg |dist| %': g.dist.abs().mean() * 100})
print(X.groupby(['lag', 'per']).apply(t).round(2).to_string()); print(); print(X.groupby(['lag', 'monthly']).apply(t).round(2).to_string())
big = X[X.dist.abs() > .01]; print('\nwhen max pain is > 1% away:'); print(big.groupby(['lag', 'per']).apply(t).round(2).to_string())
