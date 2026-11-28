"""Report 509 (Nov 27 2026): small-cap supply events vs sub-$5 flip outcome. Declared BEFORE results.
(2) ATM / offering supply: SUPP = a 424B5 or 424B4 prospectus (offering executed) filed in the prior 90d; SHELF = an S-3 filed in the prior 365d (capacity to sell). Mechanism: an issuer selling into strength caps the excursion. Prediction: SUPP lowers REACH and median r (sign -); SHELF weaker (-).
(3) Russell reconstitution window: signal day 06-15..07-01. Mechanism: forced index flows in small caps. Two-sided, no prediction (Bonferroni).
(4) Nasdaq/NYSE deficiency notice: 8-K item 3.01 in the prior 60d. Mechanism: forced reverse-split / delisting pressure. Prediction: worse median r / left tail (sign -), REACH neutral.
Unit: sub-$5 daily flips (flip_history.csv), ATRP tercile + month-clustered SE, train 2019-21 / test 2022+, filings strictly before the signal day. Pass = same sign both eras + pooled |t|>=2."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
F = pd.read_parquet(SP + 'flip_filings.parquet'); F['filed'] = pd.to_datetime(F.filed)
H = pd.read_csv('flip_history.csv', parse_dates=['d']); H['d'] = H.d.astype('datetime64[ns]')
by = {k: g.sort_values('filed') for k, g in F.groupby('tk')}
def cnt(tk, d, forms, days, item=None):
    g = by.get(tk); 
    if g is None: return np.nan
    m = g[(g.filed < d) & (g.filed >= d - pd.Timedelta(days=days)) & g.form.isin(forms)]
    if item: m = m[m['items'].astype(str).str.contains(item, regex=False)]
    return len(m)
H['SUPP'] = [cnt(t, d, ['424B5', '424B4'], 90) > 0 for t, d in zip(H.tk, H.d)]
H['SHELF'] = [cnt(t, d, ['S-3', 'S-3/A', 'S-3ASR'], 365) > 0 for t, d in zip(H.tk, H.d)]
H['DEF'] = [cnt(t, d, ['8-K'], 60, '3.01') > 0 for t, d in zip(H.tk, H.d)]
H['RUS'] = [((d.month == 6 and d.day >= 15) or (d.month == 7 and d.day <= 1)) for d in H.d]
H['mon'] = H.d.dt.strftime('%Y-%m'); H['tr'] = H.d < '2022-01-01'; H['era'] = np.where(H.tr, 'train 19-21', 'test 22-26')
q = H[H.tr].ATRP.quantile([1 / 3, 2 / 3]).values; H['a3'] = np.digitize(H.ATRP, q); lo, hi = H.r.quantile([.01, .99]); H['rw'] = H.r.clip(lo, hi)
print('flips', len(H), ' SUPP', int(H.SUPP.sum()), ' SHELF', int(H.SHELF.sum()), ' DEF(3.01)', int(H.DEF.sum()), ' RUS', int(H.RUS.sum()), '\n')
def eff(D, flag, y):
    if D[flag].sum() < 8 or (~D[flag]).sum() < 8: return None
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mon)[0]}); k = [p for p in m.params.index if p.startswith(flag)][0]
    return m.params[k] * 100, m.tvalues[k]
for flag in ('SUPP', 'SHELF', 'DEF', 'RUS'):
    print(f'== {flag}')
    for era, D in list(H.groupby('era')) + [('pooled', H)]:
        r1 = eff(D, flag, 'hit'); r2 = eff(D, flag, 'rw'); n1 = int(D[flag].sum()); a = D[D[flag]]; b = D[~D[flag]]
        print(f'  {era:12} n flagged {n1:4} | REACH {100 * a.hit.mean():.0f}% vs {100 * b.hit.mean():.0f}%  adj {r1[0]:+.1f}pp (t {r1[1]:+.2f}) | r(w) median {100 * a.rw.median():+.1f} vs {100 * b.rw.median():+.1f}  adj mean {r2[0]:+.1f}pp (t {r2[1]:+.2f})' if r1 else f'  {era:12} n flagged {n1:4} too few')
