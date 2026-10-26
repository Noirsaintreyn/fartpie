"""HMM / GMM market regime paired with the vector (Oct 25 2026). Regime models on SPY daily (yfinance 2000 -> now):
  HMM  2-state Gaussian HMM on [daily return, 5-day realized vol], expanding refit every 63 sessions, FORWARD-filtered probabilities
       (no smoothing: P(stress) at a close uses only data up to that close), stress = the higher-volatility state
  GMM  2-component Gaussian mixture on the same features, same refit schedule, P(stress) from the latest day's features
  JM3  3-state HMM (calm / normal / stress) with the same protocol
Known at the prior close for each entry. Rules on the live book: skip new entries in stress; half size in stress; and a 'calm only' version."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/batch_oct25.py').read().split("PART = os.getenv('PART', 'ALL')")[0])
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
S = yf.download('SPY', start='2000-01-01', progress=False, auto_adjust=True)['Close'].squeeze().dropna()
r = np.log(S).diff(); v = r.rolling(5).std(); F = pd.concat([r, v], axis=1).dropna() * 100; X = F.values; idx = F.index
def fwd_filter(m, X):
    """forward-filtered state probabilities (alpha recursion, normalized)"""
    from scipy.stats import multivariate_normal as mvn
    K = m.n_components; A = m.transmat_; pi = m.startprob_
    B = np.column_stack([mvn(m.means_[k], m.covars_[k]).pdf(X) for k in range(K)]) + 1e-300
    out = np.zeros((len(X), K)); a = pi * B[0]; a /= a.sum(); out[0] = a
    for t in range(1, len(X)): a = (a @ A) * B[t]; a /= a.sum(); out[t] = a
    return out
def walk(kind, K=2):
    P = pd.DataFrame(np.nan, index=idx, columns=['stress', 'calm']); start = 1500; step = 63
    for s in range(start, len(X), step):
        tr = X[:s]
        if kind == 'gmm':
            m = GaussianMixture(K, covariance_type='full', random_state=0, n_init=3).fit(tr); hi = int(np.argmax(m.covariances_[:, 1, 1])); lo = int(np.argmin(m.covariances_[:, 1, 1]))
            pr = m.predict_proba(X[s:s + step])
        else:
            m = GaussianHMM(K, covariance_type='full', n_iter=200, random_state=0).fit(tr); hi = int(np.argmax(m.covars_[:, 1, 1])); lo = int(np.argmin(m.covars_[:, 1, 1]))
            pr = fwd_filter(m, X[:s + step])[s:s + step]                          # filter runs through history; row t uses data <= t only
        P.iloc[s:s + step, 0] = pr[:, hi]; P.iloc[s:s + step, 1] = pr[:, lo]
    return P.shift(1)                                                            # known at the prior close
R = {nm: walk(k, K) for nm, k, K in (('HMM', 'hmm', 2), ('GMM', 'gmm', 2), ('HMM3', 'hmm', 3))}
d = T.entry.dt.normalize()
for nm, PP in R.items():
    T[nm + "_s"] = PP.stress.reindex(d.values, method='ffill').values; T[nm + "_c"] = PP.calm.reindex(d.values, method='ffill').values
    st = PP.stress.dropna(); print(f"{nm}: stress share of days since 2019 {(st[st.index >= '2019-01-01'] > .5).mean():.0%} | now P(stress) {st.iloc[-1]:.2f}")
    g = T.groupby(np.where(T[nm + '_s'] > .5, 'stress', 'calm')).ret.agg(['mean', 'count']); g['mean'] *= 100; print(g.round(3).to_string())
head('HMM / GMM regime paired with the vector')
for nm in R:
    s = T[nm + '_s']
    show(f'{nm}: no new entries in stress', T[~(s > .5)])
    T['w'] = np.where(s > .5, .5, 1.0); show(f'{nm}: half size in stress', T, size='w')
    show(f'{nm}: only when P(calm) > 0.5', T[T[nm + '_c'] > .5])
