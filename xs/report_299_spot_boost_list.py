"""Nov 18 2026 report 299 (Rey: a running ranked list - base rank = options ratio, strong aligned signals move a name UP a few spots). DECLARED BEFORE RESULTS.
Base rank: each day, rank the 54 names by 252d-median options ratio (1 = highest). Signals (from report 298, thresholds from train): $-volume percentile >= 2/3 | 20d momentum >= train upper third | 1h vector bull at the flip | tight stop (room <= train median).
Effective rank = base rank - K x (number of signals on). K = 0 (live), 1, 2, 3, 5 spots per signal. Flips in the same bar take slots in effective-rank order. 6 slots, 2bp, CAGR/DD 2019-22 | 2023-26. Also: per-trade return of the top-6 effective rank vs the rest."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
_s = open('/home/claude/fartpie/xs/scr_slots.py').read(); exec(_s[:_s.index("if __name__ == '__main__':")])
exec(open('/home/claude/fartpie/xs/report_298_confluence.py').read().split("B['score'] =")[0].split("SP = ")[1].join(["SP = ", ""]) if False else "")
src = open('/home/claude/fartpie/xs/report_298_confluence.py').read(); exec(src[src.index("SP = "):src.index("B['score'] =")])
O = pd.read_csv('/mnt/user-data/uploads/fartpie/xs/ext/optvol_ratio_live.csv', parse_dates=['date']); O = O[O.sh > 0]; O['r'] = O.opt_contracts * 100 / O.sh
RAT = O.pivot_table(index='date', columns='tk', values='r').sort_index().rolling(252, min_periods=60).median().shift(1); RK = RAT.rank(axis=1, ascending=False, method='min')
B['brank'] = [RK[t].asof(d) if t in RK.columns else np.nan for t, d in zip(B.tk, B.day)]; B = B.dropna(subset=['brank']).copy(); B['nsig'] = B[['S_ldv', 'S_mom', 'S_h1', 'S_room']].sum(axis=1); print('trades', len(B), '| median base rank', B.brank.median())
def st2(s):
    out = []
    for a, b in (('2019-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')):
        z = s[(s.index >= a) & (s.index < b)]; e = (1 + z).cumprod(); y = (z.index[-1] - z.index[0]).days / 365.25; out.append(f'{e.iloc[-1] ** (1 / y) - 1:6.1%} / DD {(e / e.cummax() - 1).min():6.1%}')
    return ' | '.join(out)
print('\nspots per signal | CAGR/DD 2019-22 | 2023-26 | all-period top-6-effective vs rest per-trade return (only crowded bars)')
Bc = B.copy(); Bc['n'] = Bc.groupby('entry').tk.transform('size')
for K in (0, 1, 2, 3, 5, 10):
    X = B.assign(_k=(B.brank - K * B.nsig).values + 1e-6 * B.ratio.rank(ascending=False).values).sort_values(['entry', '_k', 'tk'], kind='stable'); T0 = X; tk_, _ = taken(); s_, w_ = daily(tk_, 6)
    Z = Bc.assign(_k=(Bc.brank - K * Bc.nsig)); Z = Z[Z.n >= 7]; Z['rk'] = Z.groupby('entry')._k.rank(method='first'); top = Z[Z.rk <= 3].ret.mean(); rest = Z[Z.rk > 3].ret.mean()
    print(f'  K={K:2d}  {st2(s_)} | crowded bars (>=7): top-3 {top:+.2%} vs rest {rest:+.2%}', flush=True)
