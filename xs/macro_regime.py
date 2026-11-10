"""Nov 10 2026: GROWTH x INFLATION QUADRANTS + HMM on fundamental inputs (FRED), as an exposure gate. DECLARED BEFORE RESULTS.
Inputs with publication lags (monthly series +45d, weekly claims +5d, daily yields +1d):
  growth G = mean of expanding z-scores of: INDPRO 3m annualized change, -ICSA (4wk avg) 13-week % change, 10y-3m curve, PAYEMS 3m annualized change
  inflation I = mean of expanding z-scores of: CPI 3m annualized change, 5y breakeven (T5YIE, from 2003)
  quadrant at each month-end: growth rising/falling (G vs 3 months earlier) x inflation rising/falling (I vs 3 months earlier)
  HMM: 3-state Gaussian HMM on monthly [G, I], refit every January on data to date, FILTERED (forward-only) probabilities -> most likely state
Tests, per asset (SPY, QQQ, TLT, GLD monthly since 2003; NQ S6 daily-by-month; screener monthly book 2019-26):
  (1) mean monthly return / Sharpe by quadrant and by HMM state, first half vs second half: does the ranking hold?
  (2) walk-forward gate: scale exposure x0.5 in a regime whose expanding-window mean (>=24 obs) is negative, else x1; Sharpe, CAGR, max DD vs ungated.
PASS = gate improves Sharpe in BOTH halves."""
import numpy as np, pandas as pd, json, pickle, warnings, io, requests; warnings.filterwarnings('ignore')
def fred(s):
    t = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}', timeout=30).text
    d = pd.read_csv(io.StringIO(t)); d.columns = ['d', 'v']; d['d'] = pd.to_datetime(d.d); d['v'] = pd.to_numeric(d.v, errors='coerce'); return d.dropna().set_index('d').v
IND, ICS, CRV, BE, CPI, PAY = (fred(x) for x in ('INDPRO', 'ICSA', 'T10Y3M', 'T5YIE', 'CPIAUCSL', 'PAYEMS'))
ME = pd.date_range('2001-01-31', '2026-10-31', freq='ME')
def asof(s, lag):                                   # latest value known at each month-end, given a publication lag in days
    s2 = s.copy(); s2.index = s2.index + pd.Timedelta(days=lag); return s2.reindex(ME, method='ffill')
def mon(s, lag, n): x = s.pct_change(n) * (12 / n); return asof(x, 45)
g_ind = asof(IND.pct_change(3) * 4, 45); g_pay = asof(PAY.pct_change(3) * 4, 45)
icw = ICS.rolling(4).mean(); g_ics = asof(-(icw.pct_change(13)), 5); g_crv = asof(CRV, 1)
i_cpi = asof(CPI.pct_change(3) * 4, 45); i_be = asof(BE, 1)
def ez(x): return (x - x.expanding(min_periods=24).mean()) / x.expanding(min_periods=24).std()
G = pd.concat([ez(g_ind), ez(g_ics), ez(g_crv), ez(g_pay)], axis=1).mean(axis=1); I = pd.concat([ez(i_cpi), ez(i_be)], axis=1).mean(axis=1)
Q = pd.DataFrame({'G': G, 'I': I}).dropna()
Q['grow'] = np.where(Q.G > Q.G.shift(3), 'G↑', 'G↓'); Q['inf'] = np.where(Q.I > Q.I.shift(3), 'I↑', 'I↓'); Q['quad'] = Q.grow + Q.inf; Q = Q.iloc[3:]
# ---- HMM filtered
from hmmlearn.hmm import GaussianHMM
def filt(model, X):
    ll = model._compute_log_likelihood(X); a = np.log(model.transmat_ + 1e-12); al = np.zeros_like(ll); al[0] = np.log(model.startprob_ + 1e-12) + ll[0]
    for t in range(1, len(X)): al[t] = ll[t] + np.logaddexp.reduce(al[t - 1][:, None] + a, axis=0)
    p = np.exp(al - al.max(axis=1, keepdims=True)); return p / p.sum(axis=1, keepdims=True)
X = Q[['G', 'I']].values; hs = pd.Series(index=Q.index, dtype=float); 
for y in range(2008, 2027):
    tr = Q.index < pd.Timestamp(f'{y}-01-01'); te = (Q.index >= pd.Timestamp(f'{y}-01-01')) & (Q.index < pd.Timestamp(f'{y+1}-01-01'))
    if tr.sum() < 60 or te.sum() == 0: continue
    m = GaussianHMM(3, covariance_type='diag', n_iter=200, random_state=1).fit(X[tr]); order = np.argsort(m.means_[:, 0])      # label states by growth mean (0 weakest)
    P = filt(m, X[tr | te]); st = P.argmax(axis=1)[-te.sum():]; hs[te] = np.array([list(order).index(s) for s in st])
Q['hmm'] = hs.map({0: 'weak', 1: 'mid', 2: 'strong'})
# ---- assets (monthly)
import yfinance as yf
A = {}
for tk in ('SPY', 'QQQ', 'TLT', 'GLD'):
    d = yf.download(tk, start='2002-06-01', progress=False, auto_adjust=True); d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]; A[tk] = d.Close.resample('ME').last().pct_change()
# NQ S6 (sz-weighted, non-overnight) per month
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
O, N = pickle.load(open(f'{SP}/s6.pkl', 'rb')); a = []
for nm, (df, _, _) in zip(N, O['V2']):
    if nm.startswith('ON'): continue
    df = df.copy(); df['day'] = pd.to_datetime(df.day).dt.normalize(); df['w'] = df.pts * df.sz; a.append(df[['day', 'w']])
S6 = pd.concat(a).groupby('day').w.sum(); S6 = S6[S6.index >= '2018-02-01']
NQm = S6.resample('ME').sum()                                         # points per month (1 contract); treated as the "return"
scr = json.load(open('/home/claude/fartpie/live_data/site_series.json'))['scr']; sb = pd.Series(scr['b'], index=pd.PeriodIndex(scr['d'], freq='M').to_timestamp('M')).pct_change()
ASSETS = {**A, 'NQ S6 pts': NQm, 'Screener': sb}
def sh(x): return x.mean() / x.std() * np.sqrt(12) if len(x) > 6 and x.std() > 0 else np.nan
for lab in ('quad', 'hmm'):
    print(f'\n=== by {lab.upper()} (monthly mean, annualized Sharpe; regime known at the prior month-end) ===')
    for an, r in ASSETS.items():
        R = r.dropna(); R = R[R.index >= '2008-01-01'] if an in A else R
        L = Q[lab].shift(1).reindex(R.index); mid = R.index[len(R) // 2]; rows = []
        for k in sorted(L.dropna().unique()):
            x1 = R[(L == k) & (R.index < mid)]; x2 = R[(L == k) & (R.index >= mid)]
            f = (lambda v: f'{v.mean()*100:+.2f}%/S{sh(v):+.1f} n{len(v)}') if an in A or an == 'Screener' else (lambda v: f'{v.mean():+.0f}pts n{len(v)}')
            rows.append(f'{k}: {f(x1)} | {f(x2)}')
        print(f' {an:10} (split {mid.date()})  ' + '  ;  '.join(rows))
# ---- walk-forward gate
print('\n=== WALK-FORWARD GATE: x0.5 in a regime whose expanding mean is negative (>=24 obs) ===')
for lab in ('quad', 'hmm'):
    for an, r in ASSETS.items():
        R = r.dropna(); R = R[R.index >= '2008-01-01'] if an in A else R
        L = Q[lab].shift(1).reindex(R.index); sc = pd.Series(1.0, index=R.index)
        for i, t in enumerate(R.index):
            k = L.iloc[i]
            if isinstance(k, str):
                h = R.iloc[:i][L.iloc[:i] == k]
                if len(h) >= 24 and h.mean() < 0: sc.iloc[i] = 0.5
        g = R * sc; mid = R.index[len(R) // 2]
        def cg(x): e = (1 + x).cumprod() if an in A or an == 'Screener' else x.cumsum(); return (e / e.cummax() - 1).min() if an in A or an == 'Screener' else (e - e.cummax()).min()
        print(f' {lab:4} {an:10} Sharpe base/gated: 1st half {sh(R[R.index<mid]):+.2f}/{sh(g[g.index<mid]):+.2f}  2nd half {sh(R[R.index>=mid]):+.2f}/{sh(g[g.index>=mid]):+.2f}  | maxDD {cg(R):.2f}->{cg(g):.2f} | exposure cut in {(sc<1).mean():.0%} of months')
print('\nCurrent regime (latest month-end):', Q.iloc[-1][['G', 'I', 'quad', 'hmm']].to_dict())
Q.to_csv('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/macro_regime.csv')
