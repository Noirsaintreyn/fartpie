"""Report 472 (Nov 24 2026): Rey - (a) FINRA short-sale volume and (b) insider open-market buying, INSIDE the high-vol ATRP base group, px<20/<10/<5.
DECLARED BEFORE RESULTS. Base group = top 30% base5 score (train-cut per price bucket), train 2019-21 / test 2022+, targets REACH (PEAK>=+10%) and REAL (r>=+10%).
(a) FINRA daily short-sale volume (TotalVolume, ShortVolume): SVR = short/total on the flip day; SVR_CHG = SVR minus its prior-20d mean; SVR_5 = prior-5d mean SVR; SVR_Z = z of flip-day SVR vs prior 60d. Mechanism: short-sale volume is mostly market-maker hedging of buyer flow; a jump in SVR on the flip day = unusual buyer demand being hedged (declared sign: higher = more REACH); the opposite reading (shorts pressing) is the control. Pass rule: same side train/test, |AUC-.5|>=.04 both, test z>=3.3.
(b) Insider open-market purchases (Form 4 code P, filed before the flip day): INS30 (any in 30d), INS90, INS10 (last 10d), CLUSTER (>=2 distinct insiders in 30d), OFFICER (a P by an Officer/CEO/CFO in 30d), VAL30 (dollar value, log) , BIGBUY (value >= $100k in 30d). Mechanism: insiders buy with private information about runway/catalysts and cluster buys carry the most. Sign: more/larger = more REACH. Era tables (2019-21/2022/2023-26) with n; PASS for a flag = beats no-flag on REACH in every era with n>=8, pooled z>=2."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
O = pd.read_pickle(SP + 'structure_small.pkl'); O['d'] = pd.to_datetime(O.d).astype('datetime64[ns]'); O = O[O.px < 20].copy().reset_index(drop=True)
O['era3'] = np.select([O.d.dt.year <= 2021, O.d.dt.year == 2022], ['2019-21', '2022', ], '2023-26'); O['tr'] = (O.d.dt.year <= 2021).values
# ---- FINRA ----
F = pd.read_pickle(SP + 'finra_sv_wide.pkl'); F['Date'] = pd.to_datetime(F.Date); F = F[F.TotalVolume > 0].copy(); F['svr'] = F.ShortVolume / F.TotalVolume
fx = {t: x.sort_values('Date').set_index('Date').svr for t, x in F.groupby('Symbol')}
c = {k: np.full(len(O), np.nan) for k in ('SVR', 'SVR_CHG', 'SVR_5', 'SVR_Z')}
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    s = fx.get(tk)
    if s is None or d not in s.index: continue
    j = s.index.get_loc(d)
    if j < 25: continue
    v = s.iloc[j]; p20 = s.iloc[j - 20:j]; p60 = s.iloc[max(0, j - 60):j]; c['SVR'][i] = v; c['SVR_CHG'][i] = v - p20.mean(); c['SVR_5'][i] = s.iloc[j - 5:j].mean(); c['SVR_Z'][i] = (v - p60.mean()) / p60.std() if p60.std() > 0 and len(p60) >= 30 else np.nan
for k, v in c.items(): O[k] = v
# ---- insiders ----
I = pd.read_pickle(SP + 'insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P'].copy(); I['fd'] = pd.to_datetime(I.fd).astype('datetime64[ns]'); I['val'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce'); I['off'] = I.RPTOWNER_RELATIONSHIP.astype(str).str.contains('Officer|CEO|CFO|President', case=False)
ix = {t: x.sort_values('fd') for t, x in I.groupby('ISSUERTRADINGSYMBOL')}
for k in ('INS30', 'INS90', 'INS10', 'CLUSTER', 'OFFICER', 'BIGBUY'): O[k] = 0.0
O['VAL30'] = np.nan
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    x = ix.get(tk)
    if x is None: continue
    m = x[(x.fd.values < d) & (x.fd.values >= d - np.timedelta64(90, 'D'))]
    if m.empty: continue
    O.at[i, 'INS90'] = 1.0; w = m[m.fd.values >= d - np.timedelta64(30, 'D')]
    if len(w):
        O.at[i, 'INS30'] = 1.0; O.at[i, 'CLUSTER'] = float(w.RPTOWNERCIK.nunique() >= 2); O.at[i, 'OFFICER'] = float(w.off.any()); v = w.val.sum(); O.at[i, 'VAL30'] = np.log10(max(v, 1)); O.at[i, 'BIGBUY'] = float(v >= 1e5)
    if (m.fd.values >= d - np.timedelta64(10, 'D')).any(): O.at[i, 'INS10'] = 1.0
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
def auc(x, y):
    m = ~np.isnan(x); x, y = x[m], y[m]; p, n = y.sum(), (1 - y).sum()
    if p < 10 or n < 10: return np.nan, np.nan
    r = pd.Series(x).rank().values; a = (r[y == 1].sum() - p * (p + 1) / 2) / (p * n); se = np.sqrt((p + n + 1) / (12 * p * n)); return a, (a - .5) / se
def pz(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5: return np.nan
    p1, p2 = a.mean(), b.mean(); p = (a.sum() + b.sum()) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); return (p1 - p2) / se if se > 0 else np.nan
for lab, cap in (('px<20', 20), ('px<10', 10), ('px<5', 5)):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut]
    print(f'\n################ {lab}: base group {len(G)} (train {G.tr.sum()}, test {(~G.tr).sum()}) | FINRA coverage {G.SVR.notna().mean():.0%} | REACH base test {(G[~G.tr].PEAK >= .1).mean():.1%} REAL {(G[~G.tr].r >= .1).mean():.1%}')
    for tgt, y in (('REACH', (G.PEAK >= .1).values.astype(float)), ('REAL', (G.r >= .1).values.astype(float))):
        print(f'  == {tgt}: AUC train / test (z) | n test | PASS')
        for f in ('SVR', 'SVR_CHG', 'SVR_5', 'SVR_Z', 'INS30', 'INS90', 'INS10', 'CLUSTER', 'OFFICER', 'VAL30', 'BIGBUY'):
            x = G[f].values.astype(float); t = G.tr.values; a1, z1 = auc(x[t], y[t]); a2, z2 = auc(x[~t], y[~t]); n2 = int((~np.isnan(x[~t])).sum())
            ps = a1 == a1 and a2 == a2 and (a1 - .5) * (a2 - .5) > 0 and abs(a1 - .5) >= .04 and abs(a2 - .5) >= .04 and abs(z2) >= 3.3
            print(f'    {f:8s} {a1:.3f} / {a2:.3f} (z {z2:+.1f}) n {n2:5d} {"PASS" if ps else ""}')
    # FINRA quartiles of SVR_CHG and SVR on REACH
    for f in ('SVR', 'SVR_CHG'):
        q = pd.qcut(G[f].rank(method='first'), 4, labels=False, duplicates='drop'); print(f'  {f} quartile REACH (Q1 low..Q4 high): ' + ' '.join(f'{(G[q == k].PEAK >= .1).mean():.0%}' for k in range(4)) + ' | test only: ' + ' '.join(f'{(G[(q == k) & ~G.tr].PEAK >= .1).mean():.0%}' for k in range(4)))
    print('  insider flag lift on REACH by era 2019-21 / 2022 / 2023-26 [n flagged]:')
    rest = G[G.INS90 == 0]
    for f in ('INS30', 'INS90', 'INS10', 'CLUSTER', 'OFFICER', 'BIGBUY'):
        a_ = G[G[f] == 1]; s = ''
        for e in ('2019-21', '2022', '2023-26'):
            x = a_[a_.era3 == e]; y = G[(G.era3 == e) & (G.INS90 == 0)]; s += f' {(x.PEAK >= .1).mean() * 100:3.0f}% v {(y.PEAK >= .1).mean() * 100:3.0f}% [{len(x)}] /' if len(x) else ' - /'
        print(f'    {f:8s} n {len(a_):3d} | REACH {(a_.PEAK >= .1).mean():.0%} v no-insider-90d {(rest.PEAK >= .1).mean():.0%} (z {pz((a_.PEAK >= .1).astype(float), (rest.PEAK >= .1).astype(float)):+.1f}) | REAL {(a_.r >= .1).mean():.0%} v {(rest.r >= .1).mean():.0%} (z {pz((a_.r >= .1).astype(float), (rest.r >= .1).astype(float)):+.1f}) | median r {a_.r.median() * 100:+.1f}% v {rest.r.median() * 100:+.1f}% |{s}')
