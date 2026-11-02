"""Nov 2 2026: informed-options signals from per-contract ThetaData EOD (39 underlyings, 2018-26; ~23 overlap the book's universe).
Day-d EOD values are used for entries on the NEXT session; OI reported the morning of r is used for entries on r.
 st_abn     short-dated (<= 14 days) OTM (> 2% above spot) call volume vs its own 60-day median   (Augustin, Brenner & Subrahmanyam)
 st_share   that volume's share of all call volume, vs its 60-day median
 st_cp      short-dated OTM call volume / short-dated OTM put volume (log), vs 60-day median
 sgn_net    net buying proxy: sum(volume x sign(last trade - mid)) calls minus puts, / total volume, vs 60-day mean
 oi_build   1-day change in open interest of short-dated OTM calls / its 20-day average OI
 oi_c5      5-day % change in total call OI minus 5-day % change in put OI
Tests on the book restricted to names with ThetaData coverage (so missing data can't drive results): quintiles, then tie-break, 40 matched runs."""
import os, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
E = pd.read_csv('/mnt/user-data/uploads/fartpie/data/theta_feat/all_eod.csv', parse_dates=['d']).sort_values(['tk', 'd'])
O = pd.read_csv('/mnt/user-data/uploads/fartpie/data/theta_feat/all_oi.csv', parse_dates=['r']).sort_values(['tk', 'r'])
g = E.groupby('tk'); med = lambda s, n: s.groupby(E.tk).transform(lambda x: x.rolling(n, min_periods=20).median().shift(1))
E['st_abn'] = np.log((E.st_otm_c + 1) / (med(E.st_otm_c, 60) + 1))
sh = E.st_otm_c / E.cvol.replace(0, np.nan); E['st_share'] = sh - sh.groupby(E.tk).transform(lambda x: x.rolling(60, min_periods=20).median().shift(1))
cp = np.log((E.st_otm_c + 1) / (E.st_otm_p + 1)); E['st_cp'] = cp - cp.groupby(E.tk).transform(lambda x: x.rolling(60, min_periods=20).median().shift(1))
sn = (E.sgn_c - E.sgn_p) / (E.cvol + E.pvol).replace(0, np.nan); E['sgn_net'] = sn - sn.groupby(E.tk).transform(lambda x: x.rolling(60, min_periods=20).mean().shift(1))
O['oi_build'] = O.groupby('tk').oi_st_otm_c.diff() / O.groupby('tk').oi_st_otm_c.transform(lambda x: x.rolling(20, min_periods=10).mean().shift(1))
O['oi_c5'] = O.groupby('tk').oi_c.pct_change(5) - O.groupby('tk').oi_p.pct_change(5)
KE, KO = ['st_abn', 'st_share', 'st_cp', 'sgn_net'], ['oi_build', 'oi_c5']; K = KE + KO
Ls = Ls.reset_index(drop=True); Ls['d'] = Ls.entry.dt.normalize(); cov = set(E.tk) & set(Ls.tk); Lc = Ls[Ls.tk.isin(cov)].reset_index(drop=True)
def attach(Z, F, dcol, cols, strict):
    out = pd.DataFrame(np.nan, index=Z.index, columns=cols)
    for tk, gz in Z.groupby('tk'):
        f = F[F.tk == tk]
        if not len(f): continue
        k = np.searchsorted(f[dcol].values, gz.d.values, side='left' if strict else 'right') - 1
        v = f[cols].values; ok = k >= 0; out.loc[gz.index[ok], cols] = v[k[ok]]
    return out
Lc[KE] = attach(Lc, E, 'd', KE, True).values; Lc[KO] = attach(Lc, O, 'r', KO, False).values
print(f'names with coverage in the book universe: {len(cov)}; flips {len(Lc)} of {len(Ls)}')
TK = taken(Lc).merge(Lc[['tk', 'entry'] + K], on=['tk', 'entry']); TK['h'] = np.where(TK.entry.dt.year <= 2021, '2019-21', '2022-26')
print('A) trades the covered-name book takes: avg trade % by quintile (q1 low ... q5 high)')
for c in K:
    T2 = TK[TK[c].notna()]; q = pd.qcut(T2[c].rank(method='first'), 5, labels=False) + 1; t = (T2.groupby([q, 'h']).ret.mean() * 100).unstack('h')
    print(f'  {c:<9} ' + '  '.join(f'q{i}: {t.loc[i, "2019-21"]:+.2f}/{t.loc[i, "2022-26"]:+.2f}' for i in t.index) + f'   (cover {TK[c].notna().mean():.0%})', flush=True)
pr = lambda s: s.groupby(Lc.entry.dt.year).rank(pct=True)
rng = np.random.default_rng(89); DR = [rng.random(len(Lc)) < .05 if i else np.zeros(len(Lc), bool) for i in range(41)]
def ev(key_):
    Z0 = Lc.copy(); Z0['ratio'] = key_.fillna(key_.median()).values
    return np.array([[v_ for cd in stat(dret(simw(Z0[~dl])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in DR])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<22} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}'
    print(s, flush=True)
print('\nB) tie-break on the covered-name book, 40 matched runs, 2019-21 / 2022-26'); B0 = ev(Lc.ratio); line('ratio (live rule)', B0)
for c in K: line(f'{c} high first', ev(Lc[c]), B0); line(f'ratio + {c}', ev(pr(Lc.ratio) + pr(Lc[c]).fillna(.5)), B0)
