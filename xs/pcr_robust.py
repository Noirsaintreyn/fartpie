import numpy as np, pandas as pd, glob, os
exec(open('xs/ext/vt/an.py').read().split("print(f'{\"\":<46}")[0])
O = {os.path.basename(f)[:-4]: pd.read_csv(f, parse_dates=['date']).set_index('date') for f in glob.glob('/mnt/user-data/uploads/fartpie/xs/ext/opt_summary/*.csv')}
pan = lambda c: pd.DataFrame({t: d[c] for t, d in O.items()}).sort_index()
spot = pan('spot'); r1 = spot.pct_change(fill_method=None); pcr = pan('put_vol') / pan('call_vol')
def ls(sig, q=1/3, lag=1, freq=4, tc=5e-4):
    days = spot.index[spot.index.dayofweek == freq] if freq is not None else spot.index
    s = sig.reindex(days); r = s.rank(axis=1, pct=True); L = (r <= q).astype(float); S = (r > 1 - q).astype(float)
    L = L.div(L.sum(axis=1), axis=0); S = S.div(S.sum(axis=1), axis=0); Wt = (L - S).reindex(spot.index).ffill().shift(lag)
    turn = Wt.diff().abs().sum(axis=1); return (Wt * r1).sum(axis=1) - turn * tc
print(f'{"":<46}{"2019-21":>15}{"2022-26":>15}{"2026 YTD":>15}')
for nm, s in (('PCR tercile, weekly (base)', ls(pcr)), ('PCR quintile', ls(pcr, .2)), ('PCR, 1 extra day lag', ls(pcr, lag=2)), ('PCR 5-day avg', ls(pcr.rolling(5).mean())),
              ('PCR vs own 60d median', ls(pcr / pcr.rolling(60).median())), ('PCR Wednesday rebalance', ls(pcr, freq=2)), ('PCR 15 bp costs', ls(pcr, tc=15e-4))):
    J = s.reindex(live.index); ok = J.notna(); line(f'{nm} (corr {live[ok].corr(J[ok]):+.2f})', s[s.index >= '2018-06-01'])
    line('   0.8 live + 0.2 of it x2 (same gross)', .8 * live + .4 * J.fillna(0))
