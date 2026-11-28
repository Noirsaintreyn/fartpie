"""Report 515b: tests for report 515 (declared in report_515_runs_build.py). Train 2019-21 / test 2018 + 2022+. Month-clustered t on the within-ATRP-bucket lift."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression; from sklearn.metrics import roc_auc_score; from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
P = pd.read_pickle(X + 'runs515_panel.pkl'); P = P[P.y.notna() & P.ATRP.notna() & P.r20.notna()].copy()
P['yr'] = P.d.dt.year; P['era'] = np.select([P.yr <= 2021, P.yr == 2022], ['2019-21', '2022'], '2023-26'); P.loc[P.yr == 2018, 'era'] = '2018'; P['mon'] = P.d.dt.strftime('%Y-%m')
TR = (P.yr >= 2019) & (P.yr <= 2021); TE = ~TR
print(f'panel {len(P):,} stock-days, {P.tk.nunique()} names, base P(double<=40d) = {P.y.mean():.3f}; train {TR.sum():,} test {TE.sum():,}')
print('by era base:', P.groupby('era').y.mean().round(3).to_dict())
# ---------- episodes (cluster y=1 days per ticker) and trigger-day catalysts
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K['filed'] = pd.to_datetime(K.filed); K['acc'] = pd.to_datetime(K.accept, utc=True, errors='coerce')
K['known'] = pd.to_datetime(np.where(K.acc.dt.hour < 20, K.filed, K.filed + pd.Timedelta(days=1))); KG = {t: g for t, g in K.groupby('ticker')}
EP = []
for tk, g in P[P.y == 1].groupby('tk'):
    g = g.sort_values('d'); last = None
    q = D[tk].dropna(subset=['close']); q.index = pd.DatetimeIndex(q.index).astype('datetime64[ns]'); c = q.close
    for _, r in g.iterrows():
        if last is not None and (r.d - last).days < 60: continue
        i = c.index.get_loc(r.d); w = c.iloc[i:i + 41]; dbl = np.argmax(w.values >= 2 * w.values[0]); seg = w.iloc[:dbl + 1]; rr = seg.pct_change().fillna(0); td = rr.idxmax(); gap = q.open.loc[td] / c.loc[:td].iloc[-2] - 1 if len(seg) > 1 else np.nan
        last = seg.index[-1]; kk = KG.get(tk); items = ''
        if kk is not None:
            z = kk[(kk.known >= td - pd.Timedelta(days=1)) & (kk.known <= td)]; items = ','.join(sorted(set(','.join(z['items'].astype(str)).split(',')))) if len(z) else ''
        EP.append(dict(tk=tk, start=r.d.date(), px=round(r.c, 2), days_to_2x=int(dbl), peak40=round(r.fmax, 2), trig_day=td.date(), trig_ret=round(rr.loc[td], 2), trig_gap=round(gap, 2), items=items, ATRP=round(r.ATRP, 3), era=r.era))
E = pd.DataFrame(EP); E['cat'] = np.select([E['items'] == '', E['items'].str.contains('1.01|2.01'), E['items'].str.contains('2.02'), E['items'].str.contains('7.01|8.01'), E['items'].str.contains('3.02|1.02|1.03|2.05')], ['none', 'deal', 'earn', 'pr', 'neg'], 'other')
E.to_csv('xs/report_515_episodes.csv', index=False); print(f'\nEPISODES (first 100%+ run per 60d, deduped): {len(E)} across {E.tk.nunique()} names')
cov = KG.keys(); E['has8k'] = E.tk.isin(cov)
print('trigger-day 8-K category (names with 8-K coverage):'); print(E[E.has8k].groupby('era').cat.value_counts(normalize=True).unstack().round(2).to_string())
# base rate: P(8-K on a random day of the same stocks), for the same categories
Pc = P[P.tk.isin(cov)]; print('base rate (random stock-day 8-K filed that day or prior day):', {c: round(float(Pc[c].mean()), 3) for c in ('DEAL', 'PR', 'EARN', 'NEG')}, '(7-day windows)')
print('trigger-day gap (open vs prior close) median', E.trig_gap.median().round(3), '| trig_ret median', E.trig_ret.median(), '| days_to_2x median', E.days_to_2x.median())
print('years:', E.groupby(E.start.astype(str).str[:4]).size().to_dict())
# ---------- bucketed within-ATRP lift
P['ab'] = pd.qcut(P.ATRP.rank(method='first'), 5, labels=False)
thr = {}
def top(col, q=.67):
    t = P.loc[TR, col].quantile(q); thr[col] = t; return (P[col] > t).astype(int)
FE = {'CAT_DEAL': P.DEAL, 'CAT_PR': P.PR, 'CAT_EARN': P.EARN, 'CAT_NEG': P.NEG, 'CAT_ANY': P.ANY, 'DTC_hi': top('DTC'), 'SVR5_hi': top('SVR5'), 'VX5_hi': top('VX5'), 'DD252_deep': 1 - top('DD252', .33).where(P.DD252.notna(), np.nan) if False else (P.DD252 < P.loc[TR, 'DD252'].quantile(.33)).astype(int),
      'r20_up': (P.r20 > .2).astype(int), 'r20_dn': (P.r20 < -.2).astype(int), 'DVOL_low': (P.DVOL < P.loc[TR, 'DVOL'].quantile(.33)).astype(int), 'VIX_hi': top('VIX'), 'VIX_lo': (P.VIX < P.loc[TR, 'VIX'].quantile(.33)).astype(int), 'dVIX_up': (P.dVIX20 > 0).astype(int), 'HY_wide': top('HY'), 'USD_up': (P.USD20 > 0).astype(int), 'NFCI_loose': (P.NFCI < P.loc[TR, 'NFCI'].quantile(.33)).astype(int), 'curve_inv': (P.T10Y2Y < 0).astype(int),
      'breadth_hi': top('breadth'), 'idx_up': (P.idx_reg > 0).astype(int) if P.idx_reg.nunique() > 1 else 0, 'WAVE_hi': top('WAVE')}
def lift(flag, mask):
    d = P[mask].assign(f=np.asarray(flag)[mask.values]); d = d[d.f.isin([0, 1])]
    out = []
    for (m, b), g in d.groupby(['mon', 'ab']):
        a, c = g[g.f == 1].y, g[g.f == 0].y
        if len(a) >= 3 and len(c) >= 3: out.append((m, b, a.mean() - c.mean(), len(a)))
    if not out: return np.nan, np.nan, np.nan, 0
    o = pd.DataFrame(out, columns=['m', 'b', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); t = mm.mean() / (mm.std() / np.sqrt(len(mm))) if len(mm) > 2 else np.nan
    return d[d.f == 1].y.mean(), d[d.f == 0].y.mean(), mm.mean(), t
rows = []
for k, v in FE.items():
    v = pd.Series(v, index=P.index) if not isinstance(v, pd.Series) else v
    r = dict(feature=k, share=round(float(v.mean()), 2))
    for nm, mk in (('train', TR), ('2018', P.yr == 2018), ('2022', P.yr == 2022), ('2023-26', P.yr >= 2023)):
        p1, p0, d, t = lift(v, mk); r[nm + '_p1'] = round(p1, 3); r[nm + '_p0'] = round(p0, 3); r[nm + '_lift_in_ATRP'] = round(d, 3); r[nm + '_t'] = round(t, 1)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv('xs/report_515_feature_lifts.csv', index=False)
print('\nWITHIN-ATRP-QUINTILE LIFT in P(double<=40d) (flag minus no-flag, month-clustered t):'); print(R[['feature', 'share', 'train_p1', 'train_p0', 'train_lift_in_ATRP', 'train_t', '2018_lift_in_ATRP', '2018_t', '2022_lift_in_ATRP', '2022_t', '2023-26_lift_in_ATRP', '2023-26_t']].to_string(index=False))
print('ATRP quintile P(y):', P.groupby('ab').y.mean().round(3).to_dict(), '| ATRP bucket edges', P.groupby('ab').ATRP.median().round(3).to_dict())
# ---------- logistic: does anything add to ATRP out of sample
def design(cols):
    Z = P[cols].copy();
    for c in cols: Z[c + '_na'] = Z[c].isna().astype(int)
    Z = Z.fillna(Z[TR.values].median()).fillna(0); return Z
def auc_fit(cols, name):
    Z = design(cols); sc = StandardScaler().fit(Z[TR.values]); m = LogisticRegression(C=.5, max_iter=300).fit(sc.transform(Z[TR.values]), P.y[TR]); pr = m.predict_proba(sc.transform(Z))[:, 1]
    res = {'set': name, 'train_auc': round(roc_auc_score(P.y[TR], pr[TR.values]), 3)}
    for nm, mk in (('2018', P.yr == 2018), ('2022', P.yr == 2022), ('2023-26', P.yr >= 2023)): res[nm] = round(roc_auc_score(P.y[mk], pr[mk.values]), 3)
    return res, pr
base = ['ATRP']; sets = {'ATRP only': base, '+tech': base + ['r20', 'DD252', 'VX5', 'DVOL'], '+catalyst 8-K': base + ['DEAL', 'PR', 'EARN', 'NEG'], '+short (DTC,SVR5)': base + ['DTC', 'SVR5'],
        '+macro': base + ['VIX', 'dVIX20', 'HY', 'dHY20', 'USD20', 'T10Y2Y', 'NFCI', 'breadth', 'idx_dd'], '+wave': base + ['WAVE'], 'ALL': base + ['r20', 'DD252', 'VX5', 'DVOL', 'DEAL', 'PR', 'EARN', 'NEG', 'DTC', 'SVR5', 'VIX', 'dVIX20', 'HY', 'dHY20', 'USD20', 'T10Y2Y', 'NFCI', 'breadth', 'idx_dd', 'WAVE']}
A = []
for nm, cols in sets.items(): r, _ = auc_fit(cols, nm); A.append(r)
print('\nOUT-OF-SAMPLE AUC for "doubles within 40d" (train 2019-21; random=0.5):'); print(pd.DataFrame(A).to_string(index=False))
# ---------- HMM (a) stock-level, (b) market-level; trained 2019-21, forward filter only
def ffilter(m, Xs):
    B = np.exp(m._compute_log_likelihood(Xs)); a = np.zeros((len(Xs), m.n_components)); p = m.startprob_ * B[0]; a[0] = p / p.sum()
    for t in range(1, len(Xs)): p = (a[t - 1] @ m.transmat_) * B[t]; a[t] = p / max(p.sum(), 1e-300)
    return a
S = P[['tk', 'd', 'ret1', 'lv', 'rg']].replace([np.inf, -np.inf], np.nan).dropna().sort_values(['tk', 'd']); tr = S[(S.d.dt.year >= 2019) & (S.d.dt.year <= 2021)]
lens = tr.groupby('tk').size().values; hm = GaussianHMM(3, covariance_type='diag', n_iter=40, random_state=0).fit(tr[['ret1', 'lv', 'rg']].values, lens)
mu = hm.means_; order = np.argsort(mu[:, 2]); print('\nHMM-a stock states (ret1, logvolratio, logrange) means, ordered by range:\n', np.round(mu[order], 3), '\nself-persistence', np.round(np.diag(hm.transmat_)[order], 2))
post = []
for tk, g in S.groupby('tk'): a = ffilter(hm, g[['ret1', 'lv', 'rg']].values); post.append(pd.DataFrame(a[:, order], index=g.index, columns=['s0', 's1', 's2']))
Q = pd.concat(post); P = P.join(Q); P['hs'] = Q.values.argmax(1) if False else np.nan; P.loc[Q.index, 'hs'] = Q.values.argmax(1)
print('P(double<=40d) by stock HMM state (0 = calm ... 2 = high-range):'); print(P.groupby(['hs', 'era']).y.mean().unstack().round(3).to_string()); print('share of days:', P.hs.value_counts(normalize=True).sort_index().round(2).to_dict())
for nm, cols in (('ATRP + HMM-a posterior', base + ['s0', 's1', 's2']),): r, _ = auc_fit(cols, nm); print(r)
# market-level HMM
MK = P.groupby('d').agg(EWR=('ret', 'mean'), DISP=('ret', 'std'), dV=('dVIX20', 'first'), n=('tk', 'size')).dropna(); MK = MK[MK.n >= 30]; MK['EWR'] = MK.EWR.clip(-.1, .1)
mt = MK[(MK.index.year >= 2019) & (MK.index.year <= 2021)]; hm2 = GaussianHMM(3, covariance_type='diag', n_iter=60, random_state=1).fit(mt[['EWR', 'DISP', 'dV']].values); o2 = np.argsort(hm2.means_[:, 1])
print('\nHMM-b market states (EWR, DISP, dVIX20) means ordered by dispersion:\n', np.round(hm2.means_[o2], 3), '\nself-persistence', np.round(np.diag(hm2.transmat_)[o2], 2))
a2 = ffilter(hm2, MK[['EWR', 'DISP', 'dV']].values)[:, o2]; MK['ms'] = a2.argmax(1); P['ms'] = P.d.map(MK.ms)
print('P(double<=40d) by MARKET hidden state:'); print(P.groupby(['ms', 'era']).y.mean().unstack().round(3).to_string()); print('share of days:', P.ms.value_counts(normalize=True).sort_index().round(2).to_dict())
for c in ('hs', 'ms'): FE[c + '_top'] = (P[c] == 2).astype(int)
for k in ('hs_top', 'ms_top'):
    r = dict(feature=k)
    for nm, mk in (('train', TR), ('2018', P.yr == 2018), ('2022', P.yr == 2022), ('2023-26', P.yr >= 2023)): p1, p0, d, t = lift(FE[k], mk); r[nm] = f'{d:+.3f} (t {t:.1f})'
    print('within-ATRP lift', r)
# ---------- wave / macro by era
print('\nP(double<=40d) by WAVE tercile (share of sub-$5 names up >30% in 20d) x era:'); P['wt'] = pd.qcut(P.WAVE.rank(method='first'), 3, labels=['low', 'mid', 'high']); print(P.groupby(['wt', 'era']).y.mean().unstack().round(3).to_string())
P['vt'] = pd.qcut(P.VIX.rank(method='first'), 3, labels=['lo', 'mid', 'hi']); print('by VIX tercile x era:'); print(P.groupby(['vt', 'era']).y.mean().unstack().round(3).to_string())
m = P.groupby('mon').agg(y=('y', 'mean'), wave=('WAVE', 'mean'), vix=('VIX', 'mean'), hy=('HY', 'mean')); print('\nmonth-level corr of mean P(y) with: ', m.corr()['y'].round(2).to_dict())
print('top 8 months by P(double):', m.y.sort_values(ascending=False).head(8).round(2).to_dict())
P.to_pickle(X + 'runs515_panel2.pkl')
