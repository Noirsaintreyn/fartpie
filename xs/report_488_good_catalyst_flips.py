"""Report 488 (Nov 25 2026): Rey - 'we are looking for a classification on volatile stocks, nothing explains why a move could be happening; cherry-pick the flips after GOOD catalysts and see if those are good.' Same group as the live alert: 4-feature base5 (ATRP, F11, LTURN, -LMC; train-ECDF top 30% of each price bucket), daily flips, test 2022+, px<20/<10/<5.
DECLARED BEFORE RESULTS. GOOD catalysts known at the flip close (reports 464/481 definitions, plus two new): EARNP = earnings 8-K (2.02) in the prior 8 days AND its 2-day price reaction >= +5%; DEAL = 8-K 1.01/2.01 in prior 7d; INS = open-market insider buy (Form 4 P) in prior 30d; ACT = new SCHEDULE 13D (activist/owner >5%, not 13G) in prior 30d; SPIKE = a day with return >= +15% on >= 3x volume inside the prior 7 sessions BEFORE the flip day (price-confirmed news; excludes the flip day); GOODANY = any of EARNP, DEAL, INS, ACT, SPIKE. Control: NEGBAD = 8-K 3.01/1.02/1.03/2.05/2.06/4.02 in prior 7d (expected worse).
Mechanism: a vector flip is price-only; a flip that arrives WITH an identifiable positive reason (a deal, an earnings beat that re-rates, insiders buying, an activist) has a holder base re-pricing the stock, so the move should extend further and fail less.
Read per flag vs rest-of-group: n, REACH, mean, median, win, p10, era (2022 | 2023-26), month-clustered paired t of mean gap. PASS (as report 464) = flag beats no-flag on mean AND REACH in BOTH eras AND t >= 2.2 (7 flags, Bonferroni). Also read GOODANY n so the 'cherry-picked' set size is honest."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
O['tr'] = O.d.dt.year <= 2021; W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl'); O = O.reset_index(drop=True)
# EARNP: 2-day price reaction of earnings 8-Ks
E_ = K[K.EARN][['ticker', 'known']].drop_duplicates(); react = []
for tk, x in E_.groupby('ticker'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index; c = df.close.values
    for kd in x.known:
        j = ix.searchsorted(pd.Timestamp(kd))
        if 1 <= j < len(ix) - 2: react.append((tk, ix[j], c[j + 1] / c[j - 1] - 1))      # known date -> next-day close vs the close before
R_ = pd.DataFrame(react, columns=['ticker', 'ed', 'rx']); R_ = R_[R_.rx >= .05]; ge = {t: x.ed.values.astype('datetime64[ns]') for t, x in R_.groupby('ticker')}
O['EARNP'] = [((ge[t] <= d) & (ge[t] > d - np.timedelta64(8, 'D'))).any() if t in ge else False for t, d in zip(O.tk.values, O.d.values)]
D13 = pd.read_csv(SP + 'ext/13d_wide.csv'); D13 = D13[D13.form.astype(str).str.upper().str.startswith('SC 13D') | D13.form.astype(str).str.upper().str.startswith('SCHEDULE 13D')]; D13['f'] = pd.to_datetime(D13.filed).astype('datetime64[ns]'); g13 = {t: x.f.values for t, x in D13.groupby('ticker')}
O['ACT'] = [((g13[t] <= d) & (g13[t] > d - np.timedelta64(30, 'D'))).any() if t in g13 else False for t, d in zip(O.tk.values, O.d.values)]
sp = np.zeros(len(O), bool)
for tk, g in O.groupby('tk'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index; c = df.close.values; v = df.volume.values.astype(float); rr = np.r_[np.nan, c[1:] / c[:-1] - 1]; vm = pd.Series(v).rolling(20).mean().shift(1).values; hit = (rr >= .15) & (v >= 3 * vm)
    for i, d in zip(g.index, g.d):
        j = ix.searchsorted(pd.Timestamp(d))
        if j < 10 or j >= len(ix) or ix[j] != pd.Timestamp(d): continue
        sp[i] = hit[j - 7:j].any()
O['SPIKE'] = sp; it2 = K['items'].astype(str).str.split(','); K['NEGBAD'] = it2.apply(lambda l: any(x in l for x in ('3.01', '1.02', '1.03', '2.05', '2.06', '4.02'))); O['NEGBAD'] = flag('NEGBAD')
O['GOODANY'] = O.EARNP | O.DEAL | O.INS | O.ACT | O.SPIKE
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+6.1%} med {x.r.median():+.1%} p10 {x.r.quantile(.1):+.0%}' if len(x) else 'n 0'
for cap in (20, 10, 5):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[(~S.tr) & (S.s_ >= cut)].copy(); G['mo'] = G.d.dt.to_period('M').astype(str)
    print(f'\n######## sub-${cap} 4-feature group test 2022+: {st(G)}')
    for f in ('EARNP', 'DEAL', 'INS', 'ACT', 'SPIKE', 'GOODANY', 'NEGBAD'):
        a = G[G[f]]; b = G[~G[f]]
        if len(a) < 5: print(f'  {f:8s} n {len(a)} (too few)'); continue
        ma = a.groupby('mo').r.mean(); mb = b.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b; t = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 else np.nan
        e = lambda x, y: x[x.d.dt.year == 2022].r.mean() if y == 22 else x[x.d.dt.year >= 2023].r.mean()
        print(f'  {f:8s} FLAG {st(a)} | REST {st(b)} | gap {d.mean():+.1%} t {t:+.2f} | 2022 {e(a, 22):+.1%} vs {e(b, 22):+.1%} (n {(a.d.dt.year == 2022).sum()}) | 2023-26 {e(a, 23):+.1%} vs {e(b, 23):+.1%}')
