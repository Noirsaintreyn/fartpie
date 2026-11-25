"""Report 495 (Nov 25 2026): Rey - 'analyze 8-Ks and insider buys, see if that leads to something else; we want to know why and if it is real'.
DECLARED BEFORE RESULTS. Population: ALL daily vector flips with px<5 (n~1500, 2019-Mar 2026), NOT the 30% live group - 4x the events. Outcomes: HIT (PEAK>=+10% above next-open entry before the vector exit) and r (net return at vector exit, winsorised 1/99%). Train 2019-21, test 2022+ (2018 has no flips).
Control for what the flag may merely proxy: regress outcome on flag + ATRP tercile x log-market-cap tercile dummies (cuts frozen on train), SE clustered by month. A flag is only a LEAD if its stratified effect has the SAME SIGN in train and test and |t|>=2 in the pooled sample; REAL if also |t|>=2 in test alone with n_flag>=30 test events AND a mechanism read below. 8-K items are read as 'known' timestamps (filing accepted before the flip close), window = prior 7 days; insiders window = 30 days.
Hypotheses with mechanism (declared, two-sided unless noted):
 INSIDER (informed buying, Cohen-Malloy-Pomorski): FIRST (buyer's first buy at the issuer in 24+ months) +; OPP (not routine) +; ROUT (routine buyer, uninformative) 0 (placebo); BIG (>=$100k by one buyer) +; CLUS (>=2 buyers) +.
 8-K ITEMS: 3.02 unregistered equity sale (dilution supply overhang) -; 1.02/1.03/2.05/3.01/4.02 (termination, bankruptcy, exit cost, delisting, restatement) -; 2.03 (new debt/financing) -; 1.01 (material agreement: mixed, deals for real revenue vs financings) two-sided; 5.02 (officer/director change) two-sided; 2.02 earnings 0/two-sided; 7.01/8.01 press release/other event two-sided; 5.07 shareholder vote 0 (placebo); 9.01 only exhibits 0 (placebo).
Also a mechanism read: for any lead, compare what it changes (ATRP, size, next-day gap, days to peak, share reaching before day 3) and whether the flagged flips cluster in few tickers/months (cluster concentration: top-5 tickers' share of flagged events, and the effect leaving each top ticker out)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
src = open('xs/report_489_informed_insiders_early_earnings.py').read(); exec(src[:src.index("B = {'ATRP'")]) if "B = {'ATRP'" in src else exec(src[:src.index("for win, nm in")])
it = K['items'].astype(str).str.split(',')
def kflag(codes):
    m = it.apply(lambda l: any(x in l for x in codes)); k = K[m][['ticker', 'known']].drop_duplicates(); g = {t: np.sort(x.known.values) for t, x in k.groupby('ticker')}; out = np.zeros(len(O), bool)
    for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
        a = g.get(tk)
        if a is not None: out[i] = ((a <= d) & (a > d - np.timedelta64(7, 'D'))).any()
    return out
only9 = it.apply(lambda l: set(l) <= {'9.01'}); k9 = K[only9][['ticker', 'known']].drop_duplicates(); g9 = {t: np.sort(x.known.values) for t, x in k9.groupby('ticker')}
O['I_9only'] = [((g9[t] <= d) & (g9[t] > d - np.timedelta64(7, 'D'))).any() if t in g9 else False for t, d in zip(O.tk.values, O.d.values)]
for nm, cs in (('I_302', ['3.02']), ('I_BADEXIT', ['1.02', '1.03', '2.05', '3.01', '4.02']), ('I_203', ['2.03']), ('I_101', ['1.01']), ('I_502', ['5.02']), ('I_202', ['2.02']), ('I_PR', ['7.01', '8.01']), ('I_507', ['5.07'])): O[nm] = kflag(cs)
O['hit'] = (O.PEAK >= .10).astype(float); lo, hi = O.r.quantile([.01, .99]); O['rw'] = O.r.clip(lo, hi)
S = O[(O.px < 5) & O.ATRP.notna() & O.LMC.notna()].copy(); tr = S.tr
S['a3'] = pd.cut(S.ATRP, [-np.inf, *np.quantile(S.ATRP[tr], [1/3, 2/3]), np.inf], labels=False); S['c3'] = pd.cut(S.LMC, [-np.inf, *np.quantile(S.LMC[tr], [1/3, 2/3]), np.inf], labels=False); S['mo_'] = S.d.dt.to_period('M').astype(str)
print(f'population px<5: n {len(S)} train {int(tr.sum())} test {int((~tr).sum())} | base reach train {S[tr].hit.mean():.0%} test {S[~tr].hit.mean():.0%} | mean rw train {S[tr].rw.mean():+.1%} test {S[~tr].rw.mean():+.1%}')
def eff(D, flag, y):
    if D[flag].sum() < 5: return (np.nan, np.nan, int(D[flag].sum()))
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3) + C(c3)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]}); k = [n for n in m.params.index if n.startswith(flag)][0]
    return (m.params[k], m.tvalues[k], int(D[flag].sum()))
flags = ['FIRST', 'INS_OPP', 'INS_ROUT', 'BIG', 'CLUS', 'OFF', 'DIR', 'I_302', 'I_BADEXIT', 'I_203', 'I_101', 'I_502', 'I_202', 'I_PR', 'I_507', 'I_9only']
print(f"{'flag':10s} | {'n tr/te':9s} | raw reach flag vs rest (test)    | ADJ reach effect: train  test  all (t) | ADJ mean r effect: train  test  all (t) | verdict")
rows = []
for f in flags:
    tr_, te_ = S[S.tr], S[~S.tr]; a = [eff(D, f, 'hit') for D in (tr_, te_, S)]; b = [eff(D, f, 'rw') for D in (tr_, te_, S)]
    raw = f"{te_[te_[f]].hit.mean():.0%} vs {te_[~te_[f]].hit.mean():.0%}" if te_[f].sum() else 'n/a'
    same = lambda z: (np.sign(z[0][0]) == np.sign(z[1][0])) and abs(z[2][1]) >= 2
    v = 'LEAD' if (same(a) or same(b)) else '-'
    if v == 'LEAD' and te_[f].sum() >= 30 and (abs(a[1][1]) >= 2 or abs(b[1][1]) >= 2): v = 'REAL?'
    print(f"{f:10s} | {a[0][2]:3d}/{a[1][2]:3d}  | {raw:30s} | {a[0][0]:+.2f} {a[1][0]:+.2f} {a[2][0]:+.2f} (t {a[2][1]:+.1f}) | {b[0][0]:+.3f} {b[1][0]:+.3f} {b[2][0]:+.3f} (t {b[2][1]:+.1f}) | {v}"); rows.append((f, v))
leads = [f for f, v in rows if v != '-']
print('\nLEADS:', leads)
for f in leads:
    X = S[S[f]]; R = S[~S[f]]; tks = X.tk.value_counts()
    print(f"\n-- {f}: n {len(X)}, tickers {X.tk.nunique()}, top5 tickers share {tks.head(5).sum() / len(X):.0%}; ATRP med {X.ATRP.median():.3f} vs {R.ATRP.median():.3f}; mcap(log) {X.LMC.median():.2f} vs {R.LMC.median():.2f}; PEAK med {X.PEAK.median():+.1%} vs {R.PEAK.median():+.1%}; era reach 19-21 {X[X.tr].hit.mean():.0%} | 2022 {X[X.d.dt.year==2022].hit.mean():.0%} (n {int((X.d.dt.year==2022).sum())}) | 23-26 {X[X.d.dt.year>=2023].hit.mean():.0%} (n {int((X.d.dt.year>=2023).sum())})")
    print('   leave-top-ticker-out ADJ reach effect: ' + ', '.join(f"{t}: {eff(S[S.tk != t], f, 'hit')[0]:+.2f}" for t in tks.head(4).index))
