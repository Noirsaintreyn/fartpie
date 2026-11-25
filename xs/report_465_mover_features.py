"""Report 465 (Nov 24 2026): WHAT DO THE BIG MOVERS HAVE IN COMMON? Rey: not just fundamentals - cash, sector rotation, deals/news, overnight gap/volume, ATRP, volatility, macro; backtest the biggest movers over all the years and see which features are prevalent.
DECLARED BEFORE RESULTS (to keep it from being a fishing trip): unit = low-priced daily bull vector flips 2019-2026Q1 (structure_small, px < $20, n ~8.4k; every flip is an entry we could take). Targets: BIG = flip return >= +30% (primary), r >= +10% (secondary). ~60 features in 7 declared families (technical/volatility, overnight gap + volume, sector rotation, cash/runway, fundamentals, deals/news/insider, macro).
Test per feature: AUC of the feature for the target, computed separately on TRAIN (2019-21) and TEST (2022+); direction is read from TRAIN and must hold in TEST. PASS: |AUC-.5| >= .04 in both halves, same side, test Mann-Whitney z >= 3.3 (about 60 features x 2 targets, Bonferroni). Also report each family's best feature and the share of the biggest movers (top 5% of flip returns) in the feature's top third vs overall.
Features are all known at the flip close (8-K known-by rule from report 464, SEC facts filed <= flip date, macro lagged one day). Missing = excluded from that feature's AUC. Overnight PRE-MARKET volume is not in any data we hold; the overnight GAP (open vs prior close on the flip day) and the volume surge (5d/60d) stand in."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src464 = open('xs/report_464_catalyst_flips.py').read(); exec(src464[:src464.index("def mp_(a, b):")])
src463 = open('xs/report_463_lowcap_fund_macro_flips.py').read(); pre = src463[src463.index("def series("):src463.index("D = O[(O.px < 20)")]
pre = pre.replace("loga=np.log(a) if a > 0 else np.nan))", "loga=np.log(a) if a > 0 else np.nan, cash=c, assets=a))")
names = {os.path.basename(f)[:-5] for f in glob.glob('data/fund/sec_small/*.json')} if False else None
import os, glob, json
names = {os.path.basename(f)[:-5] for f in glob.glob('data/fund/sec_small/*.json')}; REV = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax']
exec(pre)
parts = []
for tk, g in O.groupby('tk'):
    F = fund_table(tk) if tk in names else None
    if F is None or F.empty: parts.append(g.assign(stale=np.nan)); continue
    parts.append(pd.merge_asof(g.sort_values('d'), F, left_on='d', right_on='filed', direction='backward', allow_exact_matches=False).assign(stale=lambda x: (x.d - x.filed).dt.days))
O = pd.concat(parts, ignore_index=True)
bad = O.stale > 250
for c in ['rev_yoy', 'ni_margin', 'loss', 'runway', 'debt_a', 'eq_a', 'dil', 'loga', 'cash', 'assets']:
    if c in O: O.loc[bad, c] = np.nan
O['cash_assets'] = O.cash / O.assets; O['cash_mcap'] = O.cash / np.exp(O.MCAP.astype(float)); O['log_runway'] = np.log1p(O.runway)
# macro (annual-horizon) + sector rotation via the sector ETFs
M = pd.DataFrame({'d10_12m': fr.DGS10 - fr.DGS10.shift(252), 'curve': fr.T10Y2Y, 'usd_12m': fr.DTWEXBGS.pct_change(252), 'oil_12m': fr.DCOILWTICO.pct_change(252), 'vix': fr.VIXCLS, 'gold_proxy_usd_3m': fr.DTWEXBGS.pct_change(63)}).shift(1)
M = M.reindex(M.index.union(O.d.unique())).ffill().reindex(O.d.values)
for c in M: O[c] = M[c].values
etf = {'XLE': [13, 29], 'XLB': [10, 12, 14, 33, 26, 24, 32], 'XLF': [60, 61, 62, 63, 64], 'XLV': [28, 80, 38, 83], 'XLK': [35, 36, 37, 73, 7], 'XLC': [48, 78], 'XLU': [49], 'XLRE': [65, 67], 'XLP': [20, 54, 21, 2], 'XLI': [15, 16, 17, 34, 40, 42, 45, 47], 'XLY': [50, 51, 52, 53, 55, 56, 57, 58, 59, 70, 75, 79]}
E_ = list(etf); mpx = mp[E_ + ['SPY']]; rs20 = (mpx[E_].pct_change(20).sub(mpx.SPY.pct_change(20), axis=0)); rs60 = (mpx[E_].pct_change(60).sub(mpx.SPY.pct_change(60), axis=0)); rk20 = rs20.rank(axis=1, pct=True); rk60 = rs60.rank(axis=1, pct=True)
for nm, T in (('sect_rs20', rs20), ('sect_rs60', rs60), ('sect_rank20', rk20), ('sect_rank60', rk60)):
    T = T.shift(1); T = T.reindex(T.index.union(O.d.unique())).ffill(); v = np.full(len(O), np.nan); s2 = pd.to_numeric(O.sec, errors='coerce').values
    for i, (s_, d_) in enumerate(zip(s2, O.d.values)):
        for e, l in etf.items():
            if s_ in l: v[i] = T.at[d_, e]; break
    O[nm] = v
O['sect_leader'] = (O.sect_rank20 >= .75).astype(float).where(O.sect_rank20.notna())
O['n8k30'] = 0.0; K30 = K.groupby('ticker').known.apply(lambda s: s.values.astype('datetime64[ns]')).to_dict()
O['n8k30'] = [float(((K30[t] <= d) & (K30[t] > d - np.timedelta64(30, 'D'))).sum()) if t in K30 else np.nan for t, d in zip(O.tk.values, O.d.values)]
for c in ('DEAL', 'PR', 'EARN', 'NEG', 'INS'): O[c] = O[c].astype(float)
O['tr'] = (O.d.dt.year <= 2021).values
FAM = {'technical/volatility': ['ATRP', 'VR', 'EXT', 'R60', 'BLOW', 'LMC', 'LTURN', 'LDV', 'LPX', 'PEERS', 'IDIO', 'OS', 'TS12', 'TS1', 'MAC', 'TTR', 'ENR', 'SKEW', 'KURT', 'PANIC', 'MKT126', 'MVR', 'CONT', 'F1', 'F2', 'F4', 'F6', 'F8', 'F9', 'F10', 'F11', 'r20'],
       'overnight gap + volume': ['F3', 'F5', 'VOLX'], 'sector rotation': ['sect_rs20', 'sect_rs60', 'sect_rank20', 'sect_rank60', 'sect_leader'], 'cash/runway': ['cash_assets', 'cash_mcap', 'log_runway', 'loss'],
       'fundamentals': ['rev_yoy', 'ni_margin', 'debt_a', 'eq_a', 'dil', 'loga'], 'deals/news/insider': ['DEAL', 'PR', 'EARN', 'NEG', 'INS', 'n8k30'], 'macro (annual horizon)': ['d10_12m', 'curve', 'usd_12m', 'oil_12m', 'vix', 'TAIL']}
from scipy.stats import rankdata
def auc(x, y):
    m = ~np.isnan(x); x, y = x[m], y[m]
    if y.sum() < 15 or (1 - y).sum() < 15 or np.nanstd(x) == 0: return np.nan, np.nan, int(m.sum())
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1; a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0); se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0)); return a, (a - .5) / se, int(m.sum())
rows = []
top5 = O.r >= O.r.quantile(.95)
for fam, fs in FAM.items():
    for f in fs:
        if f not in O or O[f].notna().sum() < 300: continue
        x = O[f].astype(float).values
        for tgt, y in (('BIG>=30%', (O.r >= .3).values.astype(float)), ('r>=10%', (O.r >= .1).values.astype(float))):
            a_tr, z_tr, n_tr = auc(x[O.tr.values], y[O.tr.values]); a_te, z_te, n_te = auc(x[~O.tr.values], y[~O.tr.values])
            rows.append(dict(family=fam, feat=f, target=tgt, auc_tr=a_tr, auc_te=a_te, z_te=z_te, n_te=n_te, same=(a_tr - .5) * (a_te - .5) > 0))
R = pd.DataFrame(rows); R['PASS'] = R.same & ((R.auc_tr - .5).abs() >= .04) & ((R.auc_te - .5).abs() >= .04) & (R.z_te.abs() >= 3.3)
print('flips', len(O), '| train', int(O.tr.sum()), 'test', int((~O.tr).sum()), '| BIG rate train', f"{(O[O.tr].r >= .3).mean():.1%}", 'test', f"{(O[~O.tr].r >= .3).mean():.1%}", '| features tested', R.feat.nunique())
for tgt in ('BIG>=30%', 'r>=10%'):
    S = R[R.target == tgt].copy(); S['gap'] = (S.auc_te - .5).abs(); print(f'\n=== target {tgt}: top features by |test AUC-.5| (AUC>.5 = higher feature value -> more big movers; train | test, test z, same side?) ===')
    for r in S.sort_values('gap', ascending=False).head(16).itertuples(): print(f'  {r.family:24s} {r.feat:12s} train {r.auc_tr:.3f} | test {r.auc_te:.3f}  z {r.z_te:+5.1f}  n {r.n_te:5d}  {"same" if r.same else "FLIPS"}  {"PASS" if r.PASS else ""}')
    print('  best by family:'); 
    for fam, g in S.groupby('family'): b = g.sort_values('gap', ascending=False).iloc[0]; print(f'    {fam:24s} {b.feat:12s} train {b.auc_tr:.3f} test {b.auc_te:.3f} z {b.z_te:+.1f}')
    print('  PASS:', ', '.join(S[S.PASS].feat) or 'none')
