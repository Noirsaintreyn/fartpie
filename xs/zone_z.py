"""Nov 1 2026: long zone vs short zone by z-score (prior close). z = (close - EMA_n) / std(close, n), for SPY and for the stock itself.
Avg short trade (bear flips) and avg long trade (bull flips, the trades the live book takes / all flips) by z bucket; then short book gated on
'over-extended' (high z) and the combined book."""
import os, glob, numpy as np, pandas as pd
exec(open('/home/claude/fartpie/xs/short_an.py').read().split("Sh['h'] = np.where(")[0].replace("print(", "(lambda *a, **k: None)("))
def z(c, n): return ((c - c.ewm(span=n, adjust=False).mean()) / c.rolling(n).std()).shift(1)
mz = pd.DataFrame({'mz20': z(S.close, 20), 'mz50': z(S.close, 50)}, index=S.index)
SZ = {}
for tk in set(Sh.tk) | set(Ls.tk):
    f = glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')
    if not f: continue
    H = pd.read_parquet(f[0], columns=['datetime', 'close']); H['dt'] = pd.to_datetime(H.datetime, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m = H.dt.dt.hour * 60 + H.dt.dt.minute; c = H[(m >= 570) & (m < 960)].groupby(H.dt.dt.normalize()).close.last()
    SZ[tk] = pd.DataFrame({'sz20': z(c, 20), 'sz50': z(c, 50)})
def attach(T):
    T = T.copy(); T['d'] = T.entry.dt.normalize(); T = T.join(mz.reindex(T.d.values).reset_index(drop=True).set_index(T.index))
    for c in ('sz20', 'sz50'): T[c] = [SZ[t][c].asof(d) if t in SZ else np.nan for t, d in zip(T.tk, T.d)]
    T['h'] = np.where(T.entry.dt.year <= 2021, '2019-21', '2022-26'); return T
Sh = attach(Sh); Ls = attach(Ls)
TKL = taken(Ls).merge(Ls[['tk', 'entry', 'mz20', 'mz50', 'sz20', 'sz50', 'h']], on=['tk', 'entry'])
bins = [-99, -2, -1, 0, 1, 2, 99]; lab = ['<-2', '-2..-1', '-1..0', '0..1', '1..2', '>2']
for c in ('mz20', 'mz50', 'sz20', 'sz50'):
    print(f'\n{c}: avg trade %  [shorts (all bear flips) | longs the book takes]')
    a = Sh.groupby([pd.cut(Sh[c], bins, labels=lab), 'h']).ret.agg(['mean', 'size']).unstack('h'); b = TKL.groupby([pd.cut(TKL[c], bins, labels=lab), 'h']).ret.mean().unstack('h')
    for k in lab:
        try: print(f'  {k:>7}  short {a.loc[k, ("mean", "2019-21")] * 100:+.2f}/{a.loc[k, ("mean", "2022-26")] * 100:+.2f} (n {int(a.loc[k, ("size", "2019-21")])}/{int(a.loc[k, ("size", "2022-26")])})   long {b.loc[k, "2019-21"] * 100:+.2f}/{b.loc[k, "2022-26"] * 100:+.2f}')
        except (KeyError, ValueError): pass
print('\nshort book gated on over-extension (return/yr, DD, return per DD)')
G = {'SPY z20 > 1': Sh.mz20 > 1, 'SPY z20 > 2': Sh.mz20 > 2, 'SPY z50 > 1.5': Sh.mz50 > 1.5, 'stock z20 > 1': Sh.sz20 > 1, 'stock z20 > 2': Sh.sz20 > 2,
     'stock z50 > 2': Sh.sz50 > 2, 'stock z20 > 1 & SPY z20 > 1': (Sh.sz20 > 1) & (Sh.mz20 > 1), 'stock z50 > 1.5 & SPY z50 > 1.5': (Sh.sz50 > 1.5) & (Sh.mz50 > 1.5)}
LE = simq(Ls); idx = pd.bdate_range('2019-01-01', LE.index.max()); le = LE.reindex(idx, method='ffill').fillna(0).diff().fillna(0)
print(f'  long book alone                    {fmt(stat(LE))}')
for nm, g in G.items():
    g = g.fillna(False).astype(bool); se_eq = simq(Sh[g]); print(f'  short only when {nm:<26}{fmt(stat(se_eq))}  (n {int(g.sum())})')
    se = se_eq.reindex(idx, method='ffill').fillna(0).diff().fillna(0); print(f'      long + 50% of it: {fmt(stat((le + .5 * se).cumsum()))}   corr {le.corr(se):+.2f}', flush=True)
