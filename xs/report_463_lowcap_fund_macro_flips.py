"""Report 463 (Nov 24 2026): can quarterly fundamentals + annual-horizon macro, with a few simple models, pick the low-priced flips that make a big move (>=10%)? Rey: "10 stocks, one could go >10%; get the entry off the flip".
DECLARED BEFORE RESULTS. Mechanism: lower-priced/smaller names are less efficiently priced and fragile, so slow-moving balance-sheet/growth facts (improving quarterly revenue growth, runway/solvency, low dilution) and the funding backdrop (credit/rates 12-month change) shift how often a vector flip runs >=10%.
Sample: xs structure_small flips (daily bull vector flips, 2019-2026Q1), entry px < $10 and name has SEC facts (data/fund/sec_small): n ~2.7k, 317 names; robustness px < $20. Target: flip return r >= +10% (primary) and >= +30%. Split: train 2019-21, test 2022+ (data starts 2019: no 2018 half, flagged).
Missing fundamentals other than rev_yoy/size/loss are filled with the TRAIN median. Features (point in time: facts filed <= flip date): FUND = rev_yoy (latest quarter vs same quarter a year earlier), ni_margin (latest quarter, clipped), loss flag, runway (cash / quarterly loss, capped 12), debt/assets, equity/assets, share dilution YoY, log assets. MACRO = 12-month changes known the day before: 10y yield, dollar and WTI 12m %, curve (10y-2y), VIX level (HY spreads dropped: the FRED series only covers ~3 years, 70% missing before). OS = the live outlier score (technical, frozen).
Models (fit on train only, ONE spec each, no tuning): M0 OS logit; M1 FUND logit; M2 FUND+MACRO logit; M3 OS+FUND+MACRO logit; M4 HGB depth 2 on OS+FUND+MACRO (min leaf 100). Report test AUC, top-quintile (train-frozen cut) P(r>=10%), mean/median/win of top quintile vs rest, month-clustered t, 2022 vs 2023+ table, and a 'top 10 flips per month' basket (hit rate, mean, median). PASS (declared): beats M0 on test AUC by >= .02 AND top-quintile P(r>=10%) above rest in BOTH test eras AND month-clustered t >= 2.2 (4 models x 2 targets, Bonferroni).
Caveat declared up front: macro features have ~7 independent annual observations in 2019-26, so they can only describe regimes, not be fitted; survivorship (today's small-cap list) flatters distress."""
import json, glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d).astype('datetime64[ns]')
names = {os.path.basename(f)[:-5] for f in glob.glob('data/fund/sec_small/*.json')}
REV = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax']
def series(us, tags, dur):
    rows = []
    for t in tags:
        for e in (us.get(t, {}).get('units', {}).get('USD', []) if t in us else []):
            if 'filed' not in e: continue
            if dur:
                if 'start' not in e: continue
                n = (pd.Timestamp(e['end']) - pd.Timestamp(e['start'])).days
                if not 80 <= n <= 100: continue
            rows.append((pd.Timestamp(e['end']), pd.Timestamp(e['filed']), e['val']))
    if not rows: return pd.DataFrame(columns=['end', 'filed', 'val'])
    d = pd.DataFrame(rows, columns=['end', 'filed', 'val']).sort_values(['end', 'filed']).drop_duplicates('end', keep='first'); return d
def fund_table(tk):
    j = json.load(open(f'data/fund/sec_small/{tk}.json')); us = j['facts'].get('us-gaap', {}); dei = j['facts'].get('dei', {})
    rev = series(us, REV, True); ni = series(us, ['NetIncomeLoss'], True); cash = series(us, ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'], False)
    assets = series(us, ['Assets'], False); eq = series(us, ['StockholdersEquity'], False); debt = series(us, ['LongTermDebt', 'LongTermDebtNoncurrent', 'DebtAndCapitalLeaseObligations'], False)
    sh = series(us, ['WeightedAverageNumberOfDilutedSharesOutstanding'], True)
    if sh.empty:
        sh = pd.DataFrame([(pd.Timestamp(e['end']), pd.Timestamp(e['filed']), e['val']) for e in dei.get('EntityCommonStockSharesOutstanding', {}).get('units', {}).get('shares', []) if 'filed' in e], columns=['end', 'filed', 'val'])
    if rev.empty or assets.empty: return None
    R = rev.set_index('end'); fl = {}
    def asof(tab, end): 
        x = tab[tab.end <= end]; return x.val.iloc[-1] if len(x) else np.nan
    out = []
    for end, r in R.iterrows():
        prev = R[(R.index >= end - pd.Timedelta(days=380)) & (R.index <= end - pd.Timedelta(days=350))]; yoy = r.val / prev.val.iloc[0] - 1 if len(prev) and prev.val.iloc[0] > 0 else np.nan
        n_ = ni[ni.end == end].val; n_ = n_.iloc[0] if len(n_) else np.nan; a = asof(assets, end); c = asof(cash, end); e = asof(eq, end); dbt = asof(debt, end)
        s_now = asof(sh, end); s_old = asof(sh, end - pd.Timedelta(days=365))
        out.append(dict(filed=r.filed, rev_yoy=np.clip(yoy, -1, 5), ni_margin=np.clip(n_ / r.val, -5, 1) if r.val > 0 else np.nan, loss=float(n_ < 0), runway=min(c / -n_, 12) if n_ < 0 and c == c else 12.0,
                        debt_a=np.clip(dbt / a, 0, 3) if a > 0 and dbt == dbt else 0.0, eq_a=np.clip(e / a, -2, 1) if a > 0 and e == e else np.nan, dil=np.clip(s_now / s_old - 1, -.5, 3) if s_old > 0 and s_now == s_now else np.nan, loga=np.log(a) if a > 0 else np.nan))
    o = pd.DataFrame(out); o['filed'] = pd.to_datetime(o.filed).astype('datetime64[ns]'); return o.sort_values('filed')
D = O[(O.px < 20) & O.tk.isin(names)].copy().reset_index(drop=True); parts = []
for tk, g in D.groupby('tk'):
    F = fund_table(tk)
    if F is None or F.empty: continue
    parts.append(pd.merge_asof(g.sort_values('d'), F, left_on='d', right_on='filed', direction='backward', allow_exact_matches=False).assign(stale=lambda x: (x.d - x.filed).dt.days))
D = pd.concat(parts); D = D[D.stale.notna() & (D.stale <= 200)]
fr = pd.read_parquet(SP + 'fred.parquet'); fr.index = pd.to_datetime(fr.index).astype('datetime64[ns]'); fr = fr.ffill()
M = pd.DataFrame({'d10_12m': fr.DGS10 - fr.DGS10.shift(252), 'curve': fr.T10Y2Y, 'usd_12m': fr.DTWEXBGS.pct_change(252), 'oil_12m': fr.DCOILWTICO.pct_change(252), 'vix': fr.VIXCLS}).shift(1)
Mr = M.reset_index(); Mr.columns = ['dm'] + list(M.columns); Mr['dm'] = pd.to_datetime(Mr.dm).astype('datetime64[ns]'); D['d'] = D.d.astype('datetime64[ns]'); D = pd.merge_asof(D.sort_values('d'), Mr, left_on='d', right_on='dm', direction='backward')
FUND = ['rev_yoy', 'ni_margin', 'loss', 'runway', 'debt_a', 'eq_a', 'dil', 'loga']; MAC = ['d10_12m', 'curve', 'usd_12m', 'oil_12m', 'vix']; OSF = ['OS']
print('NaN rates before filtering:', D[FUND + MAC + OSF].isna().mean().round(2).to_dict(), 'rows', len(D)); D = D.dropna(subset=['rev_yoy', 'loga', 'loss'] + MAC + OSF).copy(); med = D[D.d.dt.year <= 2021][FUND].median()
for c in FUND: D[c] = D[c].fillna(med[c])
D = D.copy(); D['mon'] = D.d.dt.strftime('%Y-%m'); D['tr'] = D.d.dt.year <= 2021
print('flips with fundamentals+macro:', len(D), 'names', D.tk.nunique(), '| px<10:', int((D.px < 10).sum()), '| train', int(D.tr.sum()), 'test', int((~D.tr).sum()))
def mt(x, y):
    a = x.groupby('mon').r.mean(); b = y.groupby('mon').r.mean(); d = (a - b).dropna(); return d.mean() * 100, (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan)
def run(S, label):
    S = S.copy(); tr, te = S[S.tr], S[~S.tr]; print(f'\n##### {label}: n {len(S)} (train {len(tr)}, test {len(te)}) | base P(r>=10%) train {(tr.r >= .1).mean():.1%} test {(te.r >= .1).mean():.1%}; P(r>=30%) {(tr.r >= .3).mean():.1%} / {(te.r >= .3).mean():.1%}; test median {te.r.median() * 100:+.1f}% win {(te.r > 0).mean():.0%}')
    specs = {'M0 OS': OSF, 'M1 FUND': FUND, 'M2 FUND+MACRO': FUND + MAC, 'M3 OS+FUND+MACRO': OSF + FUND + MAC}
    for thr in (.10, .30):
        y = (S.r >= thr).values; ytr, yte = y[S.tr.values], y[~S.tr.values]; print(f'--- target r >= {thr:.0%}')
        models = {k: make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=500)) for k in specs}; models['M4 HGB OS+FUND+MACRO'] = HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=.05, min_samples_leaf=100, random_state=0); specs['M4 HGB OS+FUND+MACRO'] = OSF + FUND + MAC
        for k, m in models.items():
            m.fit(tr[specs[k]].values, ytr); ptr = m.predict_proba(tr[specs[k]].values)[:, 1]; pte = m.predict_proba(te[specs[k]].values)[:, 1]; cut = np.quantile(ptr, .8); top = te[pte >= cut]; rest = te[pte < cut]
            auc = roc_auc_score(yte, pte); d_, t_ = mt(top, rest); era = ' '.join(f"{e}: top {(x[x.era == e].r >= thr).mean():.0%} vs rest {(y2[y2.era == e].r >= thr).mean():.0%}" for e, x, y2 in [(e, top, rest) for e in sorted(te.era.unique())])
            te2 = te.assign(p=pte); bk = te2.sort_values('p', ascending=False).groupby('mon').head(10); print(f'{k:24s} test AUC {auc:.3f} | top-quintile n {len(top)} P(hit) {(top.r >= thr).mean():.1%} vs rest {(rest.r >= thr).mean():.1%} | mean {top.r.mean() * 100:+.1f}% med {top.r.median() * 100:+.1f}% win {(top.r > 0).mean():.0%} (rest mean {rest.r.mean() * 100:+.1f}%) month-paired diff {d_:+.1f}pp t {t_:.2f} | {era} | top10/mo basket n {len(bk)} hit {(bk.r >= thr).mean():.1%} mean {bk.r.mean() * 100:+.1f}% med {bk.r.median() * 100:+.1f}%')
run(D[D.px < 10], 'px < $10'); run(D, 'px < $20 (robustness)')
