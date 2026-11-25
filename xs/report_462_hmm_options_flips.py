"""Report 462 (Nov 24 2026): HMM on OPTIONS features, applied to vector flips (every flip is a bull vector state; the HMM says what positioning regime it fires into).
DECLARED BEFORE RESULTS. Mechanism: flips follow through more when the options market is in a stressed/hedged regime (high put/call volume, steep skew, dealers short gamma) because hedges unwind and dealer hedging chases the move (reports 457-461: put/call volume and negative gamma were the live leads).
Data: the 23 gated-flip names with theta features (data/theta_feat2), daily, ALL features prior close (shifted one day, own-trailing-252d ranks already in prep): r_pc_vol, r_skew, r_charm, r_iv30, tanh(gex / trailing-252d std of gex). Flips: xs/ext/wide_T.parquet (S&P, ratio >= 1.5), 15m and 1h separately.
HMM: Gaussian, diag cov, K=3 (declared) fit on TRAIN days only (2019-2021, all 23 names concatenated as separate sequences); states for every day come from the FORWARD (filtered) pass only (no smoothing = no look-ahead). Flip state = argmax filtered prob of its prior-close day.
Test (declared): pick the state with the highest mean flip return on TRAIN (n >= 150); PASS needs, in the test set (2018 + 2022+): that state beats the other states by month-paired t >= 2 AND the same sign in train and test AND beats a circular-shift null (state series shifted by a random offset within each name, 300 draws, one-sided p < .05). Then by era table. K=2 is run as one declared alternative (Bonferroni: PASS needs t >= 2.2 and p < .025)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from hmmlearn.hmm import GaussianHMM
src = open('xs/report_457_optfeat_flips.py').read().split("NAMES = dict")[0].split('"""')[2]; exec(src)
rng = np.random.default_rng(7)
FEATS = ['r_pc_vol', 'r_skew', 'r_charm', 'r_iv30', 'gz']
X = {}
for tk in TK:
    r = pd.read_csv(UP + f'data/theta_feat2/{tk}.csv', parse_dates=['date']).set_index('date').sort_index()
    F = FE.xs(tk, level='tk')[['r_pc_vol', 'r_skew', 'r_charm', 'r_iv30']].copy(); sd = r.gex.rolling(252, min_periods=60).std(); F['gz'] = np.tanh(r.gex.shift(1) / sd.shift(1)).reindex(F.index)
    X[tk] = F.dropna()
trn = {tk: f[(f.index.year >= 2019) & (f.index.year <= 2021)] for tk, f in X.items()}
def fit(K):
    A = np.vstack([f.values for f in trn.values() if len(f) > 60]); L = [len(f) for f in trn.values() if len(f) > 60]
    best = None
    for s in range(5):
        m = GaussianHMM(K, covariance_type='diag', n_iter=200, random_state=s).fit(A, L)
        sc = m.score(A, L)
        if best is None or sc > best[0]: best = (sc, m)
    return best[1]
def filtered(m, f):
    """forward (filtering) pass only: P(state_t | x_1..t). No smoothing, so no look-ahead."""
    lp = m._compute_log_likelihood(f.values); lp = lp - lp.max(1, keepdims=True); B = np.exp(lp); A = m.transmat_; a = m.startprob_ * B[0]; a /= a.sum(); out = [a.argmax()]
    for t in range(1, len(B)):
        a = (a @ A) * B[t]; a /= a.sum(); out.append(a.argmax())
    return pd.Series(out, index=f.index)
def paired(a, b):
    d = (a.groupby('mon').ret.mean() - b.groupby('mon').ret.mean()).dropna(); return d.mean() * 1e4, (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan)
def run(K, h1):
    m = fit(K); S = {tk: filtered(m, f) for tk, f in X.items()}; D = W[W.h1 == h1].copy()
    D['st'] = [S[t].get(d, np.nan) if t in S else np.nan for t, d in zip(D.tk, D.day)]; D = D.dropna(subset=['st']); D['st'] = D.st.astype(int)
    tr, te = D[D.half == 'train'], D[D.half == 'test']
    mu = {k: tr[tr.st == k].ret.mean() for k in range(K) if (tr.st == k).sum() >= 150}
    if not mu: print('no state with n>=150'); return
    b = max(mu, key=mu.get)
    print(f'\n=== K={K} {"1h" if h1 else "15m"}: flips {len(D)} (train {len(tr)}, test {len(te)}) | state means on TRAIN (bp, n): ' + ' '.join(f'{k}:{tr[tr.st == k].ret.mean() * 1e4:+.0f}({(tr.st == k).sum()})' for k in range(K)) + f' -> chosen state {b}')
    print('  state centers (ranks pc_vol, skew, charm, iv30; gz): ' + ' | '.join(f'{k}: ' + ' '.join(f'{v:.2f}' for v in m.means_[k]) for k in range(K)))
    sp = lambda x: paired(x[x.st == b], x[x.st != b])
    s_tr, s_te = sp(tr), sp(te); print(f'  chosen-vs-rest: train {s_tr[0]:+.0f}bp t {s_tr[1]:.2f} | TEST {s_te[0]:+.0f}bp t {s_te[1]:.2f}  (n state {int((te.st == b).sum())}, rest {int((te.st != b).sum())})  win {(te[te.st == b].ret > 0).mean():.0%} vs {(te[te.st != b].ret > 0).mean():.0%}')
    null = []
    for _ in range(300):
        Sh = {tk: pd.Series(np.roll(s.values, rng.integers(20, max(21, len(s) - 20))), index=s.index) for tk, s in S.items()}
        T2 = D.copy(); T2['st'] = [Sh[t].get(d, np.nan) for t, d in zip(T2.tk, T2.day)]; T2 = T2.dropna(subset=['st']); a, c = T2[T2.half == 'train'], T2[T2.half == 'test']
        mu2 = {k: a[a.st == k].ret.mean() for k in range(K) if (a.st == k).sum() >= 150}
        if mu2: b2 = max(mu2, key=mu2.get); null.append(paired(c[c.st == b2], c[c.st != b2])[0])
    p = (np.array(null) >= s_te[0]).mean(); print(f'  circular-shift null p (test spread) = {p:.3f}   eras (state {b} vs rest, bp): ' + ' '.join(f'{e} {D[(D.era == e) & (D.st == b)].ret.mean() * 1e4:+.0f}/{D[(D.era == e) & (D.st != b)].ret.mean() * 1e4:+.0f}' for e in ['18-19', '20-21', '22', '23-26']))
    ok = s_te[1] >= (2.2 if K == 2 else 2) and s_tr[0] > 0 and s_te[0] > 0 and p < (.025 if K == 2 else .05); print('  PASS' if ok else '  no pass')
for K in (3, 2):
    for h1 in (False, True): run(K, h1)
