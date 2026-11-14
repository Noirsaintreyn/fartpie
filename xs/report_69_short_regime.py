"""Nov 14 2026 WHY DID HIGH-SHORT-INTEREST FLIPS WORK IN 2025-26 AND NOT BEFORE? (Rey: explain before trashing). Declared before results. Part A DESCRIBE: DTC>=10 flips by half-year vs all flips that half (n, mean net, HUGE per 100, lift), the biggest 2025-26 winners with industry, and concentration (top-5 share of HUGE and of return). Part B MARKET LEVEL: build the daily equal-weight basket of names with published days-to-cover>=10 (membership known the prior day, +8d publication lag) and compare its return to the whole universe by half-year: if the basket led only in 2025-26 that is a market-wide 'most-shorted' regime, not a flip effect. Part C MECHANISM GATE (one hypothesis, one threshold, fixed now): squeeze flips pay when the high-short basket is LEADING the universe (basket 40d return minus universe 40d return > 0 at the flip bar). Test DTC>=10 & LEAD vs DTC>=10 & not-LEAD vs all-flips-LEAD (is it short-specific or just 'speculative leadership'?), per era and per half-year. Pass = LEAD version has HUGE lift>=1.3 and mean above base in both eras with n>=30. Data: wide daily flips (report_64), FINRA short interest 2018-10..2026."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003; P['d0'] = pd.to_datetime(P.d0)
D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); cl = pd.DataFrame({tk: B.close[~B.index.duplicated()] for tk, B in D.items()}).sort_index(); cl = cl[cl.index >= '2018-09-01']
sic = pd.read_csv(SC + 'sic.csv'); P = P.merge(sic[['tk', 'sic', 'desc']], on='tk', how='left')
si = pd.read_pickle(SC + 'shortint.pkl'); si['avail'] = pd.to_datetime(si.settlementDate) + pd.Timedelta(days=8); si = si[si.symbolCode.isin(cl.columns)]
dtc = pd.DataFrame(np.nan, index=cl.index, columns=cl.columns)
for tk, g in si.groupby('symbolCode'):
    s = g.sort_values('avail').drop_duplicates('avail').set_index('avail').daysToCoverQuantity; s = s[~s.index.duplicated()]; dtc[tk] = s.reindex(s.index.union(cl.index)).ffill(limit=20).reindex(cl.index)
# flip-level DTC (latest published as of d0)
P['DTC'] = [dtc.at[d, tk] if (d in dtc.index and tk in dtc.columns) else np.nan for d, tk in zip(P.d0, P.tk)]; P['SQ10'] = P.DTC >= 10; P['hy'] = P.t.dt.year.astype(str) + 'H' + ((P.t.dt.month > 6) + 1).astype(str)
ret = cl.pct_change(fill_method=None).clip(-0.5, 1.0); hi = (dtc >= 10).shift(1).fillna(False); lo = (dtc < 10).shift(1).fillna(False) & dtc.shift(1).notna()
bk = ret.where(hi).mean(axis=1); un = ret.mean(axis=1); lw = ret.where(lo).mean(axis=1)
print('PART A: DTC>=10 flips vs all flips by half-year (flips with DTC data only)'); print(f'  {"half":7s} {"n_all":>6s} {"mean_all":>8s} {"HUGE/100":>8s} | {"n_SQ10":>6s} {"mean":>7s} {"HUGE/100":>8s} {"lift":>5s}')
for h, g in P[P.DTC.notna()].groupby('hy'):
    s = g[g.SQ10]; print(f'  {h:7s} {len(g):6d} {g.rn.mean()*100:+8.2f} {g.HUGE.mean()*100:8.2f} | {len(s):6d} {s.rn.mean()*100 if len(s) else np.nan:+7.2f} {s.HUGE.mean()*100 if len(s) else np.nan:8.2f} {s.HUGE.mean()/max(g.HUGE.mean(),1e-9) if len(s) else np.nan:5.2f}')
z = P[P.SQ10 & (P.t >= '2025')]; print(f'\n  2025-26 SQ10 flips {len(z)}; HUGE {int(z.HUGE.sum())}; top-5 trades share of summed return {z.nlargest(5,"ret").ret.sum()/z.ret.sum()*100:.0f}%; unique tickers {z.tk.nunique()}')
print('  top 15 SQ10 winners 2025-26:'); [print(f'    {r.tk:6s} {r.t.date()} ret {r.ret*100:+7.0f}% DTC {r.DTC:5.0f} BV {r.BV:4.1f} | {str(r.desc)[:42]}') for r in z.nlargest(15, 'ret').itertuples()]
print('\nPART B: equal-weight daily basket, DTC>=10 names vs DTC<10 names vs universe (annualised mean daily return x252, %), by half-year')
for h in sorted(set(bk.index.year.astype(str) + 'H' + ((bk.index.month > 6) + 1).astype(str))):
    m = (bk.index.year.astype(str) + 'H' + ((bk.index.month > 6) + 1).astype(str)) == h
    if m.sum() < 20: continue
    print(f'  {h}: high-short {bk[m].mean()*252*100:+7.1f} | low-short {lw[m].mean()*252*100:+7.1f} | universe {un[m].mean()*252*100:+7.1f} | high minus low {(bk[m]-lw[m]).mean()*252*100:+7.1f}')
lead = ((1 + bk.fillna(0)).rolling(40).apply(np.prod, raw=True) - (1 + un.fillna(0)).rolling(40).apply(np.prod, raw=True)); P['LEAD'] = lead.reindex(P.d0).values > 0
print('\nPART C: mechanism gate LEAD = high-short basket 40d return > universe 40d return at flip bar')
def mc(S, m, base):
    s = S[m & base]; r = S[~m & base]; a = s.groupby(s.t.dt.to_period('M')).rn.mean(); b = r.groupby(r.t.dt.to_period('M')).rn.mean(); dm = (a - b).dropna(); return dm.mean() / (dm.std() / np.sqrt(len(dm))) if len(dm) > 5 else np.nan
for en, a, b in (('2019-21', '2019', '2022'), ('2022-24', '2022', '2025'), ('2025-26Q1', '2025', '2026-04')):
    S = P[(P.t >= a) & (P.t < b) & P.DTC.notna()].reset_index(drop=True); base = S.HUGE.mean(); print(f'--- {en}: n {len(S)} base mean {S.rn.mean()*100:+.2f}% HUGE/100 {base*100:.2f} | LEAD share of flip days {S.LEAD.mean()*100:.0f}%')
    for lab, m in (('SQ10 & LEAD', S.SQ10 & S.LEAD), ('SQ10 & not LEAD', S.SQ10 & ~S.LEAD), ('all flips & LEAD', S.LEAD), ('all flips & not LEAD', ~S.LEAD), ('SQ10 & LEAD & BV>=1.5', S.SQ10 & S.LEAD & (S.BV >= 1.5)), ('LEAD & BV>=1.5 (no SI)', S.LEAD & (S.BV >= 1.5))):
        s = S[m]
        if len(s) < 15: print(f'    {lab:26s} n {len(s)}'); continue
        print(f'    {lab:26s} n {len(s):5d} mean {s.rn.mean()*100:+7.2f}% med {s.rn.median()*100:+6.2f} HUGE/100 {s.HUGE.mean()*100:5.2f} lift {s.HUGE.mean()/base:4.2f} ex-top3 {np.sort(s.rn.values)[:-3].mean()*100:+6.2f} t-vs-rest {mc(S, m, pd.Series(True, index=S.index)):+.1f}')
