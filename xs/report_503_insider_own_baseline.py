"""Report 503 (Nov 26 2026): Rey - 'instead of saying normal for small caps, analyze if it is normal for the STOCK: how much is an outlier that indicates a future move, how much is everyday transactions or noise'.
DECLARED BEFORE RESULTS. Population: all daily vector flips px<5 (n~1500), same as 496; outcomes HIT (PEAK>=+10% above next-open entry before the vector exit) and r (winsorised 1/99%). Train 2019-21 / test 2022+ (flips start 2019; no 2018 flips exist), ATRP x size tercile controls, month-clustered SE.
Data: every open-market Form 4 buy (P) and sale (S) 2018-Q1 2026, ext/insider_PS_all.pkl. Window = filings in the 30d up to the flip close.
OWN BASELINE per stock = the same stock's activity in the prior 24 months (data filed before the window starts): avg $ per 30d, and number of active months.
Classes, per side (buy / sell), mutually exclusive:
  NONE     no trade in the window
  NEW      trades in the window but NONE by anyone at the issuer in the prior 24m (dormant stock wakes up)
  ELEV     the stock does trade, and window $ >= 3x its own prior-24m average per-30d $
  ROUTINE  the stock does trade and window $ < 3x its own average (everyday flow)
Mechanisms (declared): BUY NEW/ELEV + (informed cluster/unusual conviction, Cohen-Malloy-Pomorski: routine is the uninformative part); BUY ROUTINE 0 (placebo).
 SELL ROUTINE 0 (10b5-1 plans, comp-related, diversification: the everyday part); SELL ELEV - and SELL NEW - (unusual distribution / supply overhang), directional guess only.
Also compared head to head: the earlier small-cap-wide size cut (window $ as % of market cap >= top-10% cut of buys, frozen on train) vs the own-baseline ELEV. LEAD = same sign train and test and pooled |t|>=2; REAL? = also |t|>=2 in test alone with >=30 test events.
Then: share of insider-active flips that fall in each class (how much is noise), and the dollar thresholds that separate the classes."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
src = open('xs/report_489_informed_insiders_early_earnings.py').read(); exec(src[:src.index("B = {'ATRP'")]) if "B = {'ATRP'" in src else exec(src[:src.index("for win, nm in")])
O['hit'] = (O.PEAK >= .10).astype(float); lo, hi = O.r.quantile([.01, .99]); O['rw'] = O.r.clip(lo, hi)
S = O[(O.px < 5) & O.ATRP.notna() & O.LMC.notna()].copy().reset_index(drop=True); tr = S.tr
S['a3'] = pd.cut(S.ATRP, [-np.inf, *np.quantile(S.ATRP[tr], [1/3, 2/3]), np.inf], labels=False); S['c3'] = pd.cut(S.LMC, [-np.inf, *np.quantile(S.LMC[tr], [1/3, 2/3]), np.inf], labels=False); S['mo_'] = S.d.dt.to_period('M').astype(str)
A = pd.read_pickle(SP + 'ext/insider_PS_all.pkl'); A['d'] = pd.to_datetime(A.FILING_DATE, format='%d-%b-%Y', errors='coerce'); A = A.dropna(subset=['d', 'ISSUERTRADINGSYMBOL'])
A['usd'] = pd.to_numeric(A.TRANS_SHARES, errors='coerce') * pd.to_numeric(A.TRANS_PRICEPERSHARE, errors='coerce'); A = A.dropna(subset=['usd']); A = A[A.usd > 0]
A['side'] = np.where((A.TRANS_CODE == 'P') & (A.TRANS_ACQUIRED_DISP_CD == 'A'), 'buy', np.where((A.TRANS_CODE == 'S') & (A.TRANS_ACQUIRED_DISP_CD == 'D'), 'sell', 'x')); A = A[A.side != 'x']
G = {(k[0], k[1]): v for k, v in A.groupby(['ISSUERTRADINGSYMBOL', 'side'])}
sh = pd.read_pickle(SP + 'ext/shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed); SH = {t: x.sort_values('filed') for t, x in sh.groupby('tk')}
rows = []
for r in S.itertuples():
    d = r.d; o = dict(); q = SH.get(r.tk); cap = (q[q.filed <= d].val.iloc[-1] * r.px) if q is not None and (q.filed <= d).any() else np.nan
    for side in ('buy', 'sell'):
        x = G.get((r.tk, side)); w = pd.DataFrame()
        if x is not None: w = x[(x.d > d - pd.Timedelta(days=30)) & (x.d <= d)]; base = x[(x.d <= d - pd.Timedelta(days=30)) & (x.d > d - pd.Timedelta(days=30 + 730))]
        else: base = pd.DataFrame()
        u = w.usd.sum() if len(w) else 0.; avg = base.usd.sum() / (730 / 30) if len(base) else 0.
        c = 'NONE' if u == 0 else ('NEW' if len(base) == 0 else ('ELEV' if u >= 3 * avg else 'ROUTINE'))
        o[side + '_c'] = c; o[side + '_usd'] = u; o[side + '_avg'] = avg; o[side + '_pcap'] = u / cap * 100 if cap == cap and cap > 0 else np.nan
    rows.append(o)
S = pd.concat([S, pd.DataFrame(rows)], axis=1); tr = S.tr
for side in ('buy', 'sell'):
    for c in ('NEW', 'ELEV', 'ROUTINE'): S[f'{side[0].upper()}_{c}'] = S[side + '_c'] == c
    S[f'{side[0].upper()}_ANY'] = S[side + '_c'] != 'NONE'
cut = S[tr & (S.buy_usd > 0)].buy_pcap.quantile(.9); S['B_BIGCAP'] = S.buy_pcap >= cut; cut_s = S[tr & (S.sell_usd > 0)].sell_pcap.quantile(.9); S['S_BIGCAP'] = S.sell_pcap >= cut_s
print(f'population px<5: n {len(S)} train {int(tr.sum())} test {int((~tr).sum())} | base reach train {S[tr].hit.mean():.0%} test {S[~tr].hit.mean():.0%} | mean rw train {S[tr].rw.mean():+.1%} test {S[~tr].rw.mean():+.1%}; BIGCAP cuts (train top-10% of buys) {cut:.3f}% cap, sells {cut_s:.3f}%')
print('\nHOW MUCH IS NOISE (share of flips by class, whole sample)')
for side in ('buy', 'sell'): print(f'  {side}: ' + ' | '.join(f"{c} {int((S[side + '_c'] == c).sum())} ({(S[side + '_c'] == c).mean():.1%})" for c in ('NONE', 'NEW', 'ELEV', 'ROUTINE')) + f"  | of insider-active: ROUTINE {(S[S[side + '_c'].isin(['NEW', 'ELEV', 'ROUTINE'])][side + '_c'] == 'ROUTINE').mean():.0%}")
def eff(D, flag, y):
    if D[flag].sum() < 5: return (np.nan, np.nan, int(D[flag].sum()))
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3) + C(c3)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]}); k = [n for n in m.params.index if n.startswith(flag)][0]
    return (m.params[k], m.tvalues[k], int(D[flag].sum()))
flags = ['B_ANY', 'B_NEW', 'B_ELEV', 'B_ROUTINE', 'B_BIGCAP', 'S_ANY', 'S_NEW', 'S_ELEV', 'S_ROUTINE', 'S_BIGCAP']
print(f"\n{'flag':10s} | {'n tr/te':9s} | raw reach flag vs rest (test)    | ADJ reach effect: train  test  all (t) | ADJ mean r effect: train  test  all (t) | verdict")
rows = []
for f in flags:
    tr_, te_ = S[S.tr], S[~S.tr]; a = [eff(D, f, 'hit') for D in (tr_, te_, S)]; b = [eff(D, f, 'rw') for D in (tr_, te_, S)]
    raw = f"{te_[te_[f]].hit.mean():.0%} vs {te_[~te_[f]].hit.mean():.0%}" if te_[f].sum() else 'n/a'
    same = lambda z: (np.sign(z[0][0]) == np.sign(z[1][0])) and abs(z[2][1]) >= 2
    v = 'LEAD' if (same(a) or same(b)) else '-'
    if v == 'LEAD' and te_[f].sum() >= 30 and (abs(a[1][1]) >= 2 or abs(b[1][1]) >= 2): v = 'REAL?'
    print(f"{f:10s} | {a[0][2]:3d}/{a[1][2]:3d}  | {raw:30s} | {a[0][0]:+.2f} {a[1][0]:+.2f} {a[2][0]:+.2f} (t {a[2][1]:+.1f}) | {b[0][0]:+.3f} {b[1][0]:+.3f} {b[2][0]:+.3f} (t {b[2][1]:+.1f}) | {v}"); rows.append((f, v))
print('\nDOLLAR SIZES by class (median window $ | median own avg per-30d $)')
for side in ('buy', 'sell'):
    for c in ('NEW', 'ELEV', 'ROUTINE'): X = S[S[side + '_c'] == c]; print(f"  {side} {c:8s} n {len(X):4d}  window ${X[side + '_usd'].median():>12,.0f}  own avg ${X[side + '_avg'].median():>12,.0f}  %cap med {X[side + '_pcap'].median():.2f}")
leads = [f for f, v in rows if v != '-']; print('\nLEADS:', leads)
for f in leads:
    X = S[S[f]]; R = S[~S[f]]; tks = X.tk.value_counts()
    print(f"\n-- {f}: n {len(X)}, tickers {X.tk.nunique()}, top5 tickers share {tks.head(5).sum() / len(X):.0%}; ATRP med {X.ATRP.median():.3f} vs {R.ATRP.median():.3f}; PEAK med {X.PEAK.median():+.1%} vs {R.PEAK.median():+.1%}; reach by era 2019-21 {X[X.tr].hit.mean():.0%} 2022+ {X[~X.tr].hit.mean():.0%}")
    print('   leave-top-ticker-out ADJ reach effect: ' + ', '.join(f"{t}: {eff(S[S.tk != t], f, 'hit')[0]:+.2f}" for t in tks.head(4).index))
S.to_pickle(SP + 'ext/insider_own_baseline_flips.pkl')
